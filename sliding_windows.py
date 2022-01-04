import argparse
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--resize", nargs='+', help= "The size to resize the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
print(type(image))
if args["resize"] != None:
    image = cv2.resize(image, (int(args["resize"][0]), int(args["resize"][1])))

# define sliding Windows.
(winW, winH) = (640, 640)
def sliding_window(image, ystepSize, xstepSize, windowSize, ypadding=0):
    for y in range(ypadding, image.shape[0], ystepSize):
        for x in range(0, image.shape[1], xstepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    return

if image.shape[0] == 1080:
    # 1920*1080 (W*H) y : 440 x : 640
    for (x, y, window) in sliding_window(image, ystepSize=440, xstepSize=640, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        img_crop = clone[y:y + winH,x:x + winW]
        print(img_crop.shape)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(1)

elif image.shape[0] == 1242:
    # 2208*1242 (W*H) y : 602 x : 522
    for (x, y, window) in sliding_window(image, ystepSize=602, xstepSize=522, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(1)

else :
    for (x, y, window) in sliding_window(image, ypadding=40 ,ystepSize=41 , xstepSize=320, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(1)
