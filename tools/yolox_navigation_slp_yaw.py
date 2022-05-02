#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# <editor-fold desc="import modules">
from __future__ import print_function
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil
import math
import sys

sys.path.append(r'D:/YOLOX')
from loguru import logger
import cv2
import torch
from yolox.data.data_augment import preproc, sliding_window
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import pyzed.sl as sl
import argparse
import os
import time
import numpy as np


# </editor-fold>


def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:  # Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "-con",
        "--connect",
        default="/dev/ttyUSB0",
        help="The Port to connect to the drone."
    )

    return parser


# Drone FRD movement
def goto_position_target_local_frd(front, right, down):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b0000111111111000,  # type_mask (only positions enabled)
        front, right, down,  # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0,  # x, y, z velocity in m/s  (not used)
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    # send command to vehicle
    vehicle.send_mavlink(msg)


def condition_yaw(heading, direction=1, relative=True):  # (private)
    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle

    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,  # confirmation
        heading,  # param 1, yaw in degrees
        0,  # param 2, yaw speed deg/s
        direction,  # param 3, direction -1 ccw, 1 cw
        is_relative,  # param 4, relative offset 1, absolute angle 0
        0, 0, 0)  # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)


def goto_point(front, right, down):
    goto_position_target_local_frd(front, right, down)
    acc = True
    while True:
        if vehicle.airspeed > 0.05 and acc is True:  # if it just start moving
            acc = False
        if vehicle.airspeed < 0.05 and acc is False:  # if it is stoping
            break


def cam2uav(theta, xyz):  # for gimbal (private)
    theta = -2 * math.pi * theta / 360

    transform = [(xyz[2] * math.cos(theta)) - (xyz[1] * math.sin(theta)),
                 xyz[0],
                 (xyz[2] * math.sin(theta)) + (xyz[1] * math.cos(theta))]

    return transform


def land():
    condition_yaw(heading=90, relative=False)  # UAV face to the absolutely direction.
    time.sleep(3)
    vehicle.mode = VehicleMode("LAND")
    # Aruco Marker detection.


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            decoder=None,
            device="gpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        image = img
        # Prediction with sliding window.
        (winW, winH) = (exp.test_size[0], exp.test_size[1])
        if image.shape[0] == 1080:
            i = 0
            # 1920*1080 (W*H) y : 440 x : 640
            t0 = time.time()
            for (x, y, window) in sliding_window(image, ystepSize=440, xstepSize=640,
                                                 windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                # cv2.imshow("cropwindow", window)
                # cv2.waitKey(0)
                img, ratio = preproc(window, self.test_size, self.rgb_means, self.std)
                img = torch.from_numpy(img).unsqueeze(0)
                if self.device == "gpu":
                    img = img.cuda()

                with torch.no_grad():
                    outputs = self.model(img)  # detection
                    if i == 0:
                        new_outputs = outputs
                        i += 1
                    else:
                        # The bounding box of sliding image need to return the real position.
                        # outputs[:, :, :4] = (x center, y center, w, h)
                        outputs[:, :, 0] = torch.add(outputs[:, :, 0], x)
                        outputs[:, :, 1] = torch.add(outputs[:, :, 1], y)
                        new_outputs = torch.cat((new_outputs, outputs), 1)  # (1, 50400, 6)
            # print(new_outputs.tolist())
            if self.decoder is not None:
                new_outputs = self.decoder(new_outputs, dtype=outputs.type())
                print("Pass through decoder.")
            outputs = postprocess(
                new_outputs, self.num_classes, self.confthre, self.nmsthre
            )
            print(outputs)
            if outputs[0] is None:
                pass
            elif len(outputs[0]) == 2:
                li_outputs = []
                temp = torch.empty(1, 7)
                temp[0][0] = torch.min(outputs[0][0, 0], outputs[0][1, 0])
                temp[0][1] = torch.min(outputs[0][0, 1], outputs[0][1, 1])
                temp[0][2] = torch.max(outputs[0][0, 2], outputs[0][1, 2])
                temp[0][3] = torch.max(outputs[0][0, 3], outputs[0][1, 3])
                temp[0][4] = torch.add(outputs[0][0, 4], outputs[0][1, 4]) / 2
                temp[0][5] = torch.add(outputs[0][0, 5], outputs[0][1, 5]) / 2
                temp[0][6] = torch.add(outputs[0][0, 6], outputs[0][1, 6]) / 2
                li_outputs.append(temp)
                outputs = li_outputs

            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            print(outputs)

        elif image.shape[0] == 1242:
            # 2208*1242 (W*H) y : 602 x : 522
            i = 0
            t0 = time.time()
            for (x, y, window) in sliding_window(image, ystepSize=602, xstepSize=522,
                                                 windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                    # cv2.imshow("cropwindow", window)
                    # cv2.waitKey(0)
                img, ratio = preproc(window, self.test_size, self.rgb_means, self.std)
                img = torch.from_numpy(img).unsqueeze(0)
                if self.device == "gpu":
                    img = img.cuda()

                with torch.no_grad():
                    outputs = self.model(img)  # detection
                    if i == 0:
                        new_outputs = outputs
                        i += 1
                    else:
                        # The bounding box of sliding image need to return the real position.
                        # outputs[:, :, :4] = (x center, y center, w, h)
                        outputs[:, :, 0] = torch.add(outputs[:, :, 0], x)
                        outputs[:, :, 1] = torch.add(outputs[:, :, 1], y)
                        new_outputs = torch.cat((new_outputs, outputs), 1)  # (1, 50400, 6)
                # print(new_outputs.tolist())
            if self.decoder is not None:
                new_outputs = self.decoder(new_outputs, dtype=outputs.type())
                print("Pass through decoder.")
            outputs = postprocess(
                new_outputs, self.num_classes, self.confthre, self.nmsthre
            )
            print(outputs)
            if outputs[0] is None:
                pass
            elif len(outputs[0]) == 2:
                li_outputs = []
                temp = torch.empty(1, 7)
                temp[0][0] = torch.min(outputs[0][0, 0], outputs[0][1, 0])
                temp[0][1] = torch.min(outputs[0][0, 1], outputs[0][1, 1])
                temp[0][2] = torch.max(outputs[0][0, 2], outputs[0][1, 2])
                temp[0][3] = torch.max(outputs[0][0, 3], outputs[0][1, 3])
                temp[0][4] = torch.add(outputs[0][0, 4], outputs[0][1, 4]) / 2
                temp[0][5] = torch.add(outputs[0][0, 5], outputs[0][1, 5]) / 2
                temp[0][6] = torch.add(outputs[0][0, 6], outputs[0][1, 6]) / 2
                li_outputs.append(temp)
                outputs = li_outputs

            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            print(outputs)

        else:
            i = 0
            t0 = time.time()
            for (x, y, window) in sliding_window(image, ypadding=40, ystepSize=41, xstepSize=320,
                                                 windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                    # cv2.imshow("cropwindow", window)
                    # cv2.waitKey(0)
                img, ratio = preproc(window, self.test_size, self.rgb_means, self.std)
                img = torch.from_numpy(img).unsqueeze(0)
                if self.device == "gpu":
                    img = img.cuda()

                with torch.no_grad():
                    outputs = self.model(img)  # detection
                    if i == 0:
                        new_outputs = outputs
                        i += 1
                    else:
                        # The bounding box of sliding image need to return the real position.
                        # outputs[:, :, :4] = (x center, y center, w, h)
                        outputs[:, :, 0] = torch.add(outputs[:, :, 0], x)
                        outputs[:, :, 1] = torch.add(outputs[:, :, 1], y)
                        new_outputs = torch.cat((new_outputs, outputs), 1)  # (1, 50400, 6)
                # print(new_outputs.tolist())
            if self.decoder is not None:
                new_outputs = self.decoder(new_outputs, dtype=outputs.type())
                print("Pass through decoder.")
            outputs = postprocess(
                new_outputs, self.num_classes, self.confthre, self.nmsthre
            )
            print(outputs)
            if outputs[0] is None:
                pass
            elif len(outputs[0]) == 2:
                li_outputs = []
                temp = torch.empty(1, 7)
                temp[0][0] = torch.min(outputs[0][0, 0], outputs[0][1, 0])
                temp[0][1] = torch.min(outputs[0][0, 1], outputs[0][1, 1])
                temp[0][2] = torch.max(outputs[0][0, 2], outputs[0][1, 2])
                temp[0][3] = torch.max(outputs[0][0, 3], outputs[0][1, 3])
                temp[0][4] = torch.add(outputs[0][0, 4], outputs[0][1, 4]) / 2
                temp[0][5] = torch.add(outputs[0][0, 5], outputs[0][1, 5]) / 2
                temp[0][6] = torch.add(outputs[0][0, 6], outputs[0][1, 6]) / 2
                li_outputs.append(temp)
                outputs = li_outputs

            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            print(outputs)

        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        # ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        # bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def zed_demo(predictor, vis_folder, current_time, args):
    # Create a zed Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.depth_minimum_distance = 0.3  # Set the minimum depth perception distance to 0.3m
    init_params.depth_maximum_distance = 40  # Set the maximum depth perception distance to 40m
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width / 2
    image_size.height = image_size.height / 2
    # Declare your sl.Mat matrices
    im = sl.Mat(image_size.width, image_size.height)
    point_cloud = sl.Mat()
    depth = sl.Mat(image_size.width, image_size.height)

    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "zed_camera.mp4")
    logger.info(f"video save_path is {save_path}")

    # err = zed.enable_recording(save_path, sl.SVO_COMPRESSION_MODE.H264)
    # if err != sl.ERROR_CODE.SUCCESS:
    #     print(repr(err))
    #     zed.close()
    #     exit(1)
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        1,
        (image_size.width, image_size.height)
    )
    r = 0  # 用於計算未辨識到的次數
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(im, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

            im_ocv = im.get_data()
            # start_time = time.time()
            outputs, img_info = predictor.inference(im_ocv[:, :, :3])  # BGRA, BGR = [:, :, :3]
            print(outputs)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            # end_time = time.time()
            # fps = 1 / (end_time-start_time)
            if outputs[0] is not None:
                px = torch.add(outputs[0][0], outputs[0][2]) / 2
                py = torch.add(outputs[0][1], outputs[1][3]) / 2
                perr, point_cloud_value = point_cloud.get_value(px, py)
                derr, depth_value = depth.get_value(px, py)
                print("The point coordinate is x:{}, y:{}, z:{}".format(point_cloud_value[0], point_cloud_value[1],
                                                                        point_cloud_value[2]), end="\r")
                print("Distance to Camera at ({0}, {1}): {2} m".format(px, py, depth_value), end="\r")
                if depth_value > 20:  # The distance to target far over 20 meters drone go front 10 down 10.
                    while px - (im.get_width() / 2) > 20 or px - (im.get_width() / 2) < 20:  # target to center of view
                        print("The center pixel to target center:{}".format(px - (im.get_width() / 2)))
                        condition_yaw(heading=2)
                    goto_point(10, 0, 10)

                elif depth_value <= 20:
                    angle = vehicle.gimbal  # Get camera angle,
                    uav_point = cam2uav(angle, point_cloud_value)  # transfer cam frame to uav frame
                    goto_point(uav_point[0], uav_point[1], 0)  # Go to top of landing platform.
                    vehicle.gimbal.rotate(-90, 0, 0)  # Gimbal 鏡頭朝下
                    time.sleep(1)
                    # 切換LAND模式, 降落 Precision landing
                    zed.close()
                    break
            else:
                r += 1
                if r == 1:
                    vehicle.gimbal.rotate(-45, 0, 0)
                    print("Gimbal rotate to -45 degree.")
                elif r == 2:
                    vehicle.gimbal.rotate(-90, 0, 0)
                    print("Gimbal rotate to -90 degree.")
                elif r == 3:
                    vehicle.gimbal.rotate(0, 0, 0)
                    condition_yaw(heading=90)
                    print("Gimbal rotate to 0 degree and UAV turn clock wise 90 degree.")
                    r = 0

            if args.save_result:
                out.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                print("Stop the detection.")
                print("Close the vehicle connection.")
                vehicle.close()
                break
        else:
            break

    # zed.disable_recording()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    model.cuda()
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device)
    current_time = time.localtime()

    zed_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    print("Connecting to the Drone...")
    # Connect to the Vehicle (in this case a UDP endpoint)
    vehicle = connect(args["connect"], wait_ready=True, baud=57600)

    print("Successfully connect to the drone.")
    print(" GPS: {}" .format(vehicle.gps_0))
    print(" Global Location: {}".format(vehicle.location.global_frame))
    print(" Attitude: {}".format(vehicle.attitude))
    print(" Battery: {}".format(vehicle.battery))
    print("Gimbal status: {}".format(vehicle.gimbal))
    print(" System status: {}".format(vehicle.system_status.state))
    print(" Mode: {}".format(vehicle.mode.name))
    arm_and_takeoff(30)  # takeoff to 30 meters high.
    main(exp, args)  # detection
    land()
    vehicle.close()
