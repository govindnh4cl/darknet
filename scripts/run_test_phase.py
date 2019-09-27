"""
This script takes in the input images + ground truth labels and creates two files:
    1. Ground truth labels (with some transformation)
    2. Detected labels (by running object-detection on each input image)

It also, dumps the labeled images with both gt and detected boxes

Motivation: After we have these two files, we can use this repo https://github.com/rafaelpadilla/Object-Detection-Metrics   
to get a quant metric for our results (not part of this script)
"""
import sys, os, shutil
import cv2
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb

assert sys.version_info >= (3,)  # Make sure we're using Python3 
dn.set_gpu(0)

cfg_path = b"cfg/yolov3-tiny.person.cfg"
weights_path = b"backup/yolov3-tiny_130000.weights"
metadata_path = b"cfg/coco.person.data"


print('Cfg paths: {:} {:} {:}'.format(cfg_path, weights_path, metadata_path))
assert os.path.exists(cfg_path)
assert os.path.exists(weights_path)
assert os.path.exists(metadata_path)

net = dn.load_net(cfg_path, weights_path, 0)
meta = dn.load_meta(metadata_path)


def draw_bboxes(img, boxes, color=(255, 255, 255)):
    img_ht, img_wd, _ = img.shape

    for person in boxes:
        box = person['box']
        xtl = int(max(0, box[0] - box[2] / 2))
        ytl = int(max(0, box[1] - box[3] / 2))
        xbr = int(min(img_wd - 1, box[0] + box[2] / 2))
        ybr = int(min(img_ht - 1, box[1] + box[3] / 2))

        cv2.rectangle(img, (xtl, ytl), (xbr, ybr), color=color, thickness=2)

    return


def main():
    # --------------- Configurable parameters start -----------
    img_extn = '.png'
    label_extn = '.txt'
    img_list_path = '/home/govind/work/projects/ivs1+_dir/darknet_dir/darknet/external/ivs_val.txt'
    img_label_dir = '/home/govind/work/projects/ivs1+_dir/darknet_dir/data/val/labels'
    # We would create a new directory where 3 things: labeled images, transformed gt, detected labels will be dumped
    output_dir = '/home/govind/work/projects/ivs1+_dir/darknet_dir/darknet/external/output'
    # --------------- Configurable parameters end -----------

    # Sanity checks
    assert os.path.exists(img_list_path)
    assert os.path.exists(img_label_dir)

    # Get a list of image paths
    with open(img_list_path, 'r') as fp:
        img_list = [x.rstrip('\n') for x in fp.readlines() if x.endswith(img_extn + '\n')]
    assert len(img_list) != 0  # Sanity check

    # Create necessary directories
    os.makedirs(output_dir)
    out_gt_dir = os.path.join(output_dir, 'gt_labels')
    out_det_dir = os.path.join(output_dir, 'det_labels')
    out_img_dir = os.path.join(output_dir, 'labeled_images')
    os.makedirs(out_gt_dir, exist_ok=True)
    os.makedirs(out_det_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    # Loop over all image samples
    for i, img_path in enumerate(img_list):
        print('\rProcessing image {:d} of {:d}'.format(i, len(img_list)), end='')
        assert os.path.exists(img_path)  # Image must exist
        img_name = os.path.basename(img_path)
        label_file = img_name.replace(img_extn, label_extn)
        label_file_path = os.path.join(img_label_dir, label_file)
        assert os.path.exists(label_file_path)  # Check if gt label exists

        img = cv2.imread(img_path)  # Read image
        assert img is not None  # Verify that image-read was successful
        img_ht, img_wd, _ = img.shape

        # --------- Create gt label file ---------
        # Read gt for this image sample
        with open(label_file_path, 'r') as fp:
            gt_lines = fp.readlines()

        gt_boxes = []
        with open(os.path.join(out_gt_dir, label_file), 'w') as fp:  # Create a new label file
            out_str = ''

            for line in gt_lines:  # Loop over all boxes
                # We should bring the bboxes back to their absolute scale
                entries = line.rstrip('\n').split(' ')
                entries[1], entries[3] = str(int(float(entries[1]) * img_wd)), str(int(float(entries[3]) * img_wd))
                entries[2], entries[4] = str(int(float(entries[2]) * img_ht)), str(int(float(entries[4]) * img_ht))

                out_str += ' '.join(entries) + '\n'
                gt_boxes.append({'box': [int(x) for x in entries[1:5]]})

            fp.write(out_str)  # Write all boxes to output label file

        # --------- Create detected label file ---------
        boxes = []  # Would hold bounding boxes
        det_result = dn.detect(net, meta, dn.load_image(img_path.encode('ascii'), 0, 0))  # Perform object detection
        for box in det_result:  # Loop over all detected bounding boxes
            boxes.append({'conf': box[1], 'box': box[2]})  # Coordinates are x_mid, y_mid, wd, ht

        with open(os.path.join(out_det_dir, label_file), 'w') as fp:  # Create a new label file
            out_str = ''
            for person in boxes:  # Loop over all boxes
                box_str = [str(int(x)) for x in person['box']]
                out_str += ' '.join(['0'] + [str(person['conf'])] + box_str + ['\n'])  # '0' for 'person' class ID

            fp.write(out_str)  # Write all boxes to output label file

        # --------- Dump labeled image ---------
        draw_bboxes(img, gt_boxes, color=(0, 255, 0))  # Draw gt bboxes
        draw_bboxes(img, boxes, color=(0, 0, 255))  # Draw detected bboxes
        cv2.imwrite(os.path.join(out_img_dir, img_name), img)

    print("Done")

    return


if __name__ == '__main__':
    main()

