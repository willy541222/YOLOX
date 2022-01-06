import os
from lxml.etree import Element, SubElement
import xml.etree.ElementTree as ET
import cv2

# bb_filepath = 'D:/YOLOX/Landing-platform.v2-resize1280-720.voc/valid/old/'
# bb_filepath_list = os.listdir(bb_filepath)
# savepath = 'D:/YOLOX/Landing-platform.v2-resize1280-720.voc/valid/XML/'
#
# if not os.path.exists(savepath):
#     os.makedirs(savepath)


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
    node_width.text = '3840'

    node_height = SubElement(node_size, 'height')
    node_height.text = '2160'

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
    filename = None
    for boxes in root.iter('object'):
        filename = root.find('filename').text
        xmin = int(boxes.find("bndbox/xmin").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        ymax = int(boxes.find("bndbox/ymax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    print(filename)
    return filename, list_with_all_boxes[0]


def check():
    ch_fname, ch_bboxes = read_content("D:/YOLOX/IOU_dataset/1176/07/07_1176_696.xml")
    image = cv2.imread("D:/YOLOX/IOU_dataset/1176/07/07_1176_696.jpg")
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.rectangle(image, (ch_bboxes[0], ch_bboxes[1]), (ch_bboxes[2], ch_bboxes[3]), (0, 255, 0), 2)
    cv2.imshow("output", image)
    cv2.waitKey(0)


def main():
    for file in bb_filepath_list:
        fname, bboxes = read_content(bb_filepath + file)
        fname = fname.split('.')
        if bboxes is None:
            continue
        else:
            an_xmin = bboxes[0] * 3
            an_ymin = bboxes[1] * 3
            an_xmax = bboxes[2] * 3
            an_ymax = bboxes[3] * 3
            print(an_xmin, an_xmax, an_ymin, an_ymax)
            an_fname = "{}{}.xml".format(savepath, fname[0][:-4])
            print(an_fname)
            write_annotation(fname[0][:-4] + ".jpg", an_xmin, an_xmax, an_ymin, an_ymax, an_fname)


if __name__ == "__main__":
    # main()
    check()
