# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb

assert sys.version_info >= (3,)  # Make sure we're using Python3 
dn.set_gpu(0)

if 1:
    cfg_path = b"cfg/yolov3-voc.person.cfg"
    weights_path = b"backup/yolov3-voc_900.weights"
    metadata_path = b"cfg/voc.person.data"
elif 0:  # COCO pre-trained model
    cfg_path = b"cfg/yolov3.cfg"
    weights_path = b"weights/yolov3.weights"
    metadata_path = b"cfg/coco.data"  

print('Cfg paths: {:} {:} {:}'.format(cfg_path, weights_path, metadata_path))

net = dn.load_net(cfg_path, weights_path, 0)
meta = dn.load_meta(metadata_path)

print(dn.detect(net, meta, dn.load_image(b"data/dog.jpg", 0, 0)))
print(dn.detect(net, meta, dn.load_image(b"data/eagle.jpg", 0, 0)))
print(dn.detect(net, meta, dn.load_image(b"data/giraffe.jpg", 0, 0)))
print(dn.detect(net, meta, dn.load_image(b"data/horses.jpg", 0, 0)))
print(dn.detect(net, meta, dn.load_image(b"data/kite.jpg", 0, 0)))
print(dn.detect(net, meta, dn.load_image(b"data/person.jpg", 0, 0)))
print(dn.detect(net, meta, dn.load_image(b"data/scream.jpg", 0, 0)))
print(dn.detect(net, meta, dn.load_image(b"data/2_person.jpg", 0, 0)))

