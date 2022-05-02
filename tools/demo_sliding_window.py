#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# <editor-fold desc="import modules">
from loguru import logger

import cv2

import torch
import sys

sys.path.append(r'D:/YOLOX')
from yolox.data.data_augment import preproc, sliding_window
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, setup_logger
import pyzed.sl as sl
import argparse
import os
import time
import numpy as np

# </editor-fold>

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
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
        default="cpu",
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
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
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
                logger.info("Pass through decoder.")
            outputs = postprocess(
                new_outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info(outputs)
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
            logger.info(outputs)

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
                logger.info("Pass through decoder.")
            outputs = postprocess(
                new_outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info(outputs)
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
            logger.info(outputs)

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
                logger.info("Pass through decoder.")
            outputs = postprocess(
                new_outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info(outputs)
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
            logger.info(outputs)

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


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def zed_demo(predictor, vis_folder, current_time, args):
    # Create a zed Camera object
    if args.demo == "zed":
        zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logger.error(repr(status))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution

    im = sl.Mat()
    point_cloud = sl.Mat()

    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "zed_camera.mp4")
    logger.info(f"video save_path is {save_path}")

    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        1,
        (int(image_size.width), int(image_size.height),)
    )
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(im, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            im_ocv = im.get_data()
            start_time = time.time()
            outputs, img_info = predictor.inference(im_ocv[:, :, :3])
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            logger.info("FPS : {}".format(fps))
            if outputs[0] is not None:
                logger.info("The target has detect : {}".format(outputs[0]))
                px = torch.add(outputs[0][0], outputs[0][2]) / 2
                py = torch.add(outputs[0][1], outputs[1][3]) / 2
                perr, point_cloud_value = point_cloud.get_value(px, py)
                logger.info(
                    "The point coordinate is x:{}, y:{}, z:{}".format(point_cloud_value[0], point_cloud_value[1],
                                                                      point_cloud_value[2]))
            if args.save_result:
                out.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    current_time = time.localtime()

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)
        log_save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        setup_logger(
            log_save_folder,
            distributed_rank=0,
            filename="Detection_log.txt",
            mode="a",
        )

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
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

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device)
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "zed":
        zed_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
