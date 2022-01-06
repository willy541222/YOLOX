import os
from lxml.etree import Element, SubElement
import xml.etree.ElementTree as ET
import cv2

image_filepath = 'D:/YOLOX/Original_dataset/images/'
bb_filepath = 'D:/YOLOX/Original_dataset/ann/'
image_filepath_list = os.listdir(image_filepath)
bb_filepath_list = os.listdir(bb_filepath)
savepath = 'D:/YOLOX/IOU_dataset'
(winW, winH) = (640, 640)


# To make the annotation file looks pretty.
# Reference https://vae-0118.github.io/2017/11/06/Python%E4%B8%ADXML%E7%9A%84%E8%AF%BB%E5%86%99%E6%80%BB%E7%BB%93/

def __indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# Read annotations file

def write_annotation(im_filename, ann_xmin, ann_xmax, ann_ymin, ann_ymax, ann_filename):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = im_filename
    node_path = SubElement(node_root, 'path')
    node_path.text = im_filename

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'CHOU, WEI-LIN'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '640'

    node_height = SubElement(node_size, 'height')
    node_height.text = '640'

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_seg = SubElement(node_root, 'segmented')
    node_seg.text = '0'

    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'landing platform'
    node_difficult = SubElement(node_object, 'pose')
    node_difficult.text = 'Unspecified'
    node_difficult = SubElement(node_object, 'truncated')
    node_difficult.text = '0'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_difficult = SubElement(node_object, 'occluded')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(ann_xmin)
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(ann_ymin)
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(ann_xmax)
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(ann_ymax)

    tree = ET.ElementTree(node_root)
    __indent(node_root)  # 增加换行符
    try:
        tree.write(ann_filename, short_empty_elements=False)
    except Exception as e:
        print('Could not save file: {}'.format(e))
        return False
    return


def read_content(bb_file):
    tree = ET.parse(bb_file)
    root = tree.getroot()
    list_with_all_boxes = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text
        xmin = int(boxes.find("bndbox/xmin").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        ymax = int(boxes.find("bndbox/ymax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


# Find the corresponding annotation file name.

def find_bbfile(img_name):
    for bb_filename in bb_filepath_list:
        if img_name[:-4] == bb_filename[:-4]:
            filename, boxes = read_content('{}{}'.format(bb_filepath, bb_filename))
    return boxes


# Sliding window

def sliding_window(image, ystepSize, xstepSize, windowSize, ypadding=0, xpadding=0):
    for y in range(ypadding, image.shape[0], ystepSize):
        for x in range(xpadding, image.shape[1], xstepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]
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
    if float(boxAArea + boxBArea - interArea) != 0:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0
    # return the intersection over union value
    return iou


def main(img_filepath):
    for img_name in img_filepath:
        gt_boxes = find_bbfile(img_name)
        img = cv2.imread(image_filepath + "/" + img_name)
        one_index, six_index, seven_index, eight_index, nine_index = 0, 0, 0, 0, 0
        print(img_name)
        if not os.path.exists('{}/{}'.format(savepath, img_name[:-4])):
            os.makedirs('{}/{}'.format(savepath, img_name[:-4]))
            os.makedirs('{}/{}/1'.format(savepath, img_name[:-4]))
            os.makedirs('{}/{}/06'.format(savepath, img_name[:-4]))
            os.makedirs('{}/{}/07'.format(savepath, img_name[:-4]))
            os.makedirs('{}/{}/08'.format(savepath, img_name[:-4]))
            os.makedirs('{}/{}/09'.format(savepath, img_name[:-4]))

        for (x, y, window) in sliding_window(img, ystepSize=5, xstepSize=5, windowSize=(winW, winH)):
            if x <= gt_boxes[0][0] and x + winW >= gt_boxes[0][2] and y <= gt_boxes[0][1] and y + winH >= gt_boxes[0][3]:
                bboxes = [gt_boxes[0][0], gt_boxes[0][1], gt_boxes[0][2], gt_boxes[0][3]]
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                cv2.imwrite('{}/{}/1/1_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], one_index), window)
                write_annotation(im_filename="1_{}_{}.jpg".format(img_name[:-4], one_index),
                                 ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                 ann_ymax=ann_boxes[3],
                                 ann_filename='{}/{}/1/1_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], one_index))
                one_index += 1

            elif x < gt_boxes[0][0] and x + winW < gt_boxes[0][2] and y < gt_boxes[0][1] and y + winH < gt_boxes[0][3]:
                bboxes = [gt_boxes[0][0], gt_boxes[0][1], x + winW, y + winH]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue

            elif x > gt_boxes[0][0] and x + winW > gt_boxes[0][2] and y > gt_boxes[0][1] and y + winH > gt_boxes[0][3]:
                bboxes = [x, y, gt_boxes[0][2], gt_boxes[0][3]]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue

            elif x < gt_boxes[0][0] and x + winW > gt_boxes[0][2] and y > gt_boxes[0][1] and y + winH > gt_boxes[0][3]:
                bboxes = [gt_boxes[0][0], y, x + winW, gt_boxes[0][3]]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue

            elif x > gt_boxes[0][0] and x + winW > gt_boxes[0][2] and y < gt_boxes[0][1] and y + winH < gt_boxes[0][3]:
                bboxes = [x, gt_boxes[0][1], gt_boxes[0][2], y + winH]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue

            elif x <= gt_boxes[0][0] and x + winW > gt_boxes[0][2] and y < gt_boxes[0][1] and y + winH < gt_boxes[0][3]:
                bboxes = [gt_boxes[0][0], gt_boxes[0][1], gt_boxes[0][2], y + winH]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue

            elif x < gt_boxes[0][0] and x + winW < gt_boxes[0][2] and y <= gt_boxes[0][1] and y + winH > gt_boxes[0][3]:
                bboxes = [gt_boxes[0][0], gt_boxes[0][1], x + winW, gt_boxes[0][3]]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue

            elif x <= gt_boxes[0][0] and x + winW > gt_boxes[0][2] and y > gt_boxes[0][1] and y + winH > gt_boxes[0][3]:
                bboxes = [gt_boxes[0][0], y, gt_boxes[0][2], gt_boxes[0][3]]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue

            elif x > gt_boxes[0][0] and x + winW > gt_boxes[0][2] and y <= gt_boxes[0][1] and y + winH > gt_boxes[0][3]:
                bboxes = [x, gt_boxes[0][1], gt_boxes[0][2], gt_boxes[0][3]]
                iou_output = bb_iou(gt_boxes[0], bboxes)
                print(iou_output)
                ann_boxes = [0, 0, 0, 0]
                ann_boxes[0] = bboxes[0] - x
                ann_boxes[2] = bboxes[2] - x
                ann_boxes[1] = bboxes[1] - y
                ann_boxes[3] = bboxes[3] - y
                if 0.7 >= iou_output >= 0.6:
                    cv2.imwrite('{}/{}/06/06_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], six_index), window)
                    write_annotation(im_filename="06_{}_{}.jpg".format(img_name[:-4], six_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/06/06_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], six_index))
                    six_index += 1
                elif 0.8 >= iou_output >= 0.7:
                    cv2.imwrite('{}/{}/07/07_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], seven_index), window)
                    write_annotation(im_filename="07_{}_{}.jpg".format(img_name[:-4], seven_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/07/07_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], seven_index))
                    seven_index += 1
                elif 0.9 >= iou_output >= 0.8:
                    cv2.imwrite('{}/{}/08/08_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], eight_index), window)
                    write_annotation(im_filename="08_{}_{}.jpg".format(img_name[:-4], eight_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/08/08_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], eight_index))
                    eight_index += 1
                elif 1 >= iou_output >= 0.9:
                    cv2.imwrite('{}/{}/09/09_{}_{}.jpg'.format(savepath, img_name[:-4], img_name[:-4], nine_index), window)
                    write_annotation(im_filename="09_{}_{}.jpg".format(img_name[:-4], nine_index),
                                     ann_xmin=ann_boxes[0], ann_xmax=ann_boxes[2], ann_ymin=ann_boxes[1],
                                     ann_ymax=ann_boxes[3],
                                     ann_filename='{}/{}/09/09_{}_{}.xml'.format(savepath, img_name[:-4], img_name[:-4], nine_index))
                    nine_index += 1
                else:
                    continue
            else:
                print("No property image ! ")
                continue


if __name__ == "__main__":
    main(image_filepath_list)
