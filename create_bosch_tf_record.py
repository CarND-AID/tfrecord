#!/usr/bin/env python

import io
import os.path
import hashlib
import random
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import PIL
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

MAP = {
    'Green': 'green',
    'GreenLeft': 'green',
    'GreenRight': 'green',
    'GreenStraight': 'green',
    'GreenStraightLeft': 'green',
    'GreenStraightRight': 'green',
    'Red': 'red',
    'RedLeft': 'red',
    'RedRight': 'red',
    'RedStraight': 'red',
    'RedStraightLeft': 'red',
    'Yellow': 'yellow'
}

flags = tf.app.flags
flags.DEFINE_string('yaml', '', 'Root directory to Bosch dataset.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'bosch_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

width = None
height = None

def process_frame(label_map_dict, frame):
    global width
    global height

    filename = frame['path']
    if not os.path.exists(filename):
        tf.logging.error("File %s not found", filename)
        return

    with tf.gfile.GFile(filename, 'rb') as img:
        encoded_png = img.read()

    png = PIL.Image.open(io.BytesIO(encoded_png))
    if png.format != 'PNG':
        tf.logging.error("File %s has unexpeted image format '%s'", filename, png.format)
        return

    if width is None and height is None:
        width = png.width
        height = png.height
        tf.logging.info('Expected image size: %dx%d', width, height)

    if width != png.width or height != png.height:
        tf.logging.error('File %s has unexpected size', filename)
        return

    key = hashlib.sha256(encoded_png).hexdigest()

    labels = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classes = []
    for bb in frame['boxes']:
        if bb['label'] not in MAP:
            continue

        labels.append(label_map_dict[MAP[bb['label']]])
        xmin.append(bb['x_min']/width)
        xmax.append(bb['x_max']/width)
        ymin.append(bb['y_min']/height)
        ymax.append(bb['y_max']/height)
        classes.append(bb['label'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes),
        'image/object/class/label': dataset_util.int64_list_feature(labels)
    }))
    return example

def main(_):
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    with tf.gfile.GFile(FLAGS.yaml, 'r') as f:
        data = yaml.load(f)
    random.shuffle(data)

    with tf.python_io.TFRecordWriter(FLAGS.output_path) as writer:
        for d in data:
            e = process_frame(label_map_dict, d)
            writer.write(e.SerializeToString())

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
