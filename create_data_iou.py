import os
import xml.etree.ElementTree as ET
import cv2

image_filepath = ''
bb_filepath = ''
savepath = ''
(winW, winH) = (640, 640)


# Read annotations file

def read_content(bb_file):
    tree = ET.parse(bb_file)
    root = tree.getroot()
    list_with_all_boxes = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


# Calculate weight and height.

def cal_w_h(bb_filename):
    filename, boxes = read_content(bb_filename)

    w = boxes[2] - boxes[0]
    h = boxes[4] - boxes[1]

    return w, h, boxes


# Find the corresponding annotation file name.

def find_bbfile(img_name):
    for bb_filename in bb_filepath:
        if img_name[:-4] == bb_filename[:-4]:
            w, h, boxes = cal_w_h(bb_filename)
            break
    return w, h, boxes


# Sliding window

def sliding_window(image, ystepSize, xstepSize, windowSize, ypadding=0, xpadding=0):
    for y in range(ypadding, image.shape[0], ystepSize):
        for x in range(xpadding, image.shape[1], xstepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    return


# Calculate iou

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def save_img(img_filepath):
    for img_name in img_filepath:
        w, h, gt_boxes = find_bbfile(img_name)
        img = cv2.imread(img_name)
        for (x, y, window) in sliding_window(img, ystepSize=1, xstepSize=1, windowSize=(winW, winH)):
            seven_index, eight_index, nine_index, six_index = 0, 0, 0, 0
            for (bb_x, bb_y, bb_window) in sliding_window(window, ystepSize=1, xstepSize=1, windowSize=(w, h)):
                bboxes = [bb_x, bb_y, bb_x + w, bb_y + h]
                iou_output = bb_iou(gt_boxes, bboxes)
                if 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], seven_index), window)
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], eight_index), window)
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], eight_index), window)
                    nine_index += 1
                elif 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], six_index), window)
                    six_index += 1
                else:
                    continue


def main():
    save_img(image_filepath)
    return


if __name__ == "__main__":
    main()
