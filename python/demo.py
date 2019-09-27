# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
import cv2
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb

assert sys.version_info >= (3,)  # Make sure we're using Python3 
dn.set_gpu(0)

if 0:
    cfg_path = b"cfg/yolov3-voc.person.cfg"
    weights_path = b"backup/yolov3-voc_900.weights"
    metadata_path = b"cfg/voc.person.data"
elif 1:
    cfg_path = b"cfg/yolov3-tiny.person.cfg"
    weights_path = b"backup/yolov3-tiny.backup"
    metadata_path = b"cfg/coco.person.data"    
elif 0:  # COCO pre-trained model
    cfg_path = b"cfg/yolov3.cfg"
    weights_path = b"weights/yolov3.weights"
    metadata_path = b"cfg/coco.data"  

print('Cfg paths: {:} {:} {:}'.format(cfg_path, weights_path, metadata_path))

net = dn.load_net(cfg_path, weights_path, 0)
meta = dn.load_meta(metadata_path)

images = [
    # b"data/dog.jpg",
    # b"data/eagle.jpg",
    # b"data/giraffe.jpg",
    # b"data/horses.jpg",
    # b"data/kite.jpg",
    # b"data/person.jpg",
    # b"data/scream.jpg",
    # b"data/2_person.jpg",
    b"/home/govind/work/projects/ivs1+_dir/darknet_dir/data/val/images/vlcsnap-2019-09-23-11h01m35s190.png",
]

results_dir = 'results'
assert os.path.exists(results_dir)
r = None


def draw_bboxes(img_path, boxes):
    img_path = img_path.decode('ascii')
    img_name = os.path.basename(img_path)
    if len(boxes) == 0:
        print('No bboxes found for {:s}.'.format(img_name))
        return

    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    assert img is not None
    img_ht, img_wd, _ = img.shape

    for person in boxes:
        box = person['box']
        xtl = int(max(0, box[0] - box[2] / 2))
        ytl = int(max(0, box[1] - box[3] / 2))
        xbr = int(min(img_wd - 1, box[0] + box[2] / 2))
        ybr = int(min(img_ht - 1, box[1] + box[3] / 2))

        cv2.rectangle(img, (xtl, ytl), (xbr, ybr), color=(0, 0, 255), thickness=2)

    out_img_path = os.path.join(results_dir, os.path.basename(img_path))
    cv2.imwrite(out_img_path, img)
    print('Dumped labeled image: {:s}'.format(out_img_path))


for image_path in images:
    boxes = []
    det_result = dn.detect(net, meta, dn.load_image(image_path, 0, 0))
    for box in det_result:  # Loop over all detected bounding boxes
        boxes.append({'conf': box[1], 'box': box[2]})
    
    print(boxes)
    draw_bboxes(image_path, boxes)



