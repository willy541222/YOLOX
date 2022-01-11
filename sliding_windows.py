import argparse
import os.path
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--resize", nargs='+', default=None, help="The size to resize the image")
args = vars(ap.parse_args())

image_name = os.path.basename(args["image"])
image = cv2.imread(args["image"])
print(type(image))


if args["resize"] != None:
    image = cv2.resize(image, (int(args["resize"][0]), int(args["resize"][1])))

# define sliding Windows.
(winW, winH) = (640, 640)


def sliding_window(img, ystepsize, xstepsize, windowsize, ypadding=0):
    for y in range(ypadding, img.shape[0], ystepsize):
        for x in range(0, img.shape[1], xstepsize):
            # yield the current window
            yield x, y, img[y:y + windowsize[1], x:x + windowsize[0]]
    return


if image.shape[0] == 1080:
    # 1920*1080 (W*H) y : 440 x : 640
    i = 0
    for (x, y, window) in sliding_window(image, ystepsize=440, xstepsize=640, windowsize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        img_crop = clone[y:y + winH, x:x + winW]
        print(img_crop.shape)
        cv2.imshow("Window", clone)
        cv2.imwrite("{}_{}.jpg".format(image_name[:-4], i), window)
        i += 1
        cv2.waitKey(1)
        time.sleep(1)

elif image.shape[0] == 1242:
    # 2208*1242 (W*H) y : 602 x : 522
    i = 0
    for (x, y, window) in sliding_window(image, ystepsize=602, xstepsize=522, windowsize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        img_crop = clone[y:y + winH, x:x + winW]
        print(img_crop.shape)
        cv2.imshow("Window", clone)
        cv2.imwrite("{}_{}.jpg".format(image_name[:-4], i), window)
        i += 1
        cv2.waitKey(1)
        time.sleep(1)

else:
    i = 0
    for (x, y, window) in sliding_window(image, ypadding=40, ystepsize=41, xstepsize=320, windowsize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        img_crop = clone[y:y + winH, x:x + winW]
        print(img_crop.shape)
        cv2.imshow("Window", clone)
        cv2.imwrite("{}_{}.jpg".format(image_name[:-4], i), window)
        i += 1
        cv2.waitKey(1)
        time.sleep(1)
