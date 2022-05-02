import cv2
import cv2.aruco as aruco
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, required=True, help="path to output image containing Aruco tag.")
args = vars(ap.parse_args())

# --- Define Tag
id_to_find = 1
marker_size = 15  # - [cm]

# --- Define the aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
parameters = aruco.DetectorParameters_create()

# --- Get the camera calibration path
# calib_path = ""
# camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
# camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

# --- 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0


# --- Capture the videocamera (this may also be a video or a picture)
cap = cv2.VideoCapture(0)
# -- Set the camera size as the one it was calibrated with. 2K resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)

# -- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args["output"], fourcc, 5.0, (2208, 1242))

while True:

    # -- Read the camera frame
    ret, frame = cap.read()
    # Extract left and right images from side-by-side
    left_right_image = np.split(frame, 2, axis=1)
    # -- Convert in gray scale
    gray = cv2.cvtColor(left_right_image[1], cv2.COLOR_BGR2GRAY)  # remember, OpenCV stores color images in BGR.
    frame = left_right_image[1]
    # -- Find all the aruco markers in the image
    # corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
    #                                              cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)

    if ids is not None and ids[0] == id_to_find:
        # -- ret = [rvec, tvec, ?]
        # -- array of rotation and position of each marker in camera frame
        # -- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
        # -- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
        # ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size)

        # -- Unpack the output, get only the first
        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        # -- Draw the detected marker and put a reference frame over it
        aruco.drawDetectedMarkers(frame, corners)
        # aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)
        aruco.drawAxis(frame, rvec, tvec, 10)

        # -- Print the tag position in camera frame
        str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f" % (tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print("MARKER Position x=%2.2f  y=%2.2f  z=%2.2f" % (tvec[0] / 100, tvec[1] / 100, tvec[2] / 100))

    # --- Display the frame
    cv2.imshow('frame', frame)
    out.write(frame)
    # --- use 'q' to quit
    key = cv2.waitKey(200) & 0xFF
    if key == ord('q'):
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        break
