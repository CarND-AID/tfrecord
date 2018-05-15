#!/usr/bin/env python

import itertools
import io
import os
import os.path
import random

import tensorflow as tf
import numpy as np

import cv2

flags = tf.app.flags
flags.DEFINE_string('input_record', '', 'TFRecord to process')
flags.DEFINE_string('output_path', '', 'Path to output images')
flags.DEFINE_integer('num_samples', 50, 'Number of samples')
# TODO: read labels and create color encoding
FLAGS = flags.FLAGS

COLOR_MAP = { 1: (0, 255, 0),
              2: (0, 0, 255) }

def sample(name, num_samples):
    # TODO: try to avoid scan through record
    num_records = sum(1 for _ in tf.python_io.tf_record_iterator(name))
    tf.logging.info("Sampling %d records from %s", num_records, name)
    return sorted(random.sample(range(num_records), num_samples))

def sampled(iterable, samples):
    cnt = 0
    for e in iterable:
        cnt += 1
        if cnt - 1 not in samples:
            continue
        yield e

def main(_):
    if not os.path.exists(FLAGS.input_record):
        raise ValueError(FLAGS.input_record + " does not exist")
    if FLAGS.output_path == '':
        raise ValueError("Output path not specified")
    os.makedirs(FLAGS.output_path, exist_ok=True)

    records = tf.python_io.tf_record_iterator(FLAGS.input_record)
    for record in sampled(records, sample(FLAGS.input_record, FLAGS.num_samples)):
        example = tf.train.Example()
        example.ParseFromString(record)

        h = example.features.feature['image/height'].int64_list.value[0]
        w = example.features.feature['image/width'].int64_list.value[0]

        bytes = example.features.feature['image/encoded'].bytes_list.value[0]
        img = np.fromstring(bytes, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        del bytes
        assert(list(img.shape) == [h, w, 3])

        xmn = example.features.feature['image/object/bbox/xmin'].float_list.value
        xmx = example.features.feature['image/object/bbox/xmax'].float_list.value
        ymn = example.features.feature['image/object/bbox/ymin'].float_list.value
        ymx = example.features.feature['image/object/bbox/ymax'].float_list.value
        cls = example.features.feature['image/object/class/label'].int64_list.value
        txt = example.features.feature['image/object/class/text'].bytes_list.value
        n = example.features.feature['image/filename'].bytes_list.value[0].decode('utf8')

        assert(len(xmn) == len(xmx))
        assert(len(xmx) == len(ymn))
        assert(len(ymn) == len(ymx))
        assert(len(ymx) == len(cls))
        assert(len(cls) == len(txt))

        for i in range(len(txt)):
            cv2.rectangle(img,
                          (int(w*xmn[i]), int(h*ymn[i])),
                          (int(w*xmx[i]), int(h*ymx[i])),
                          COLOR_MAP[cls[i]],
                          2)
            cv2.putText(img,
                        txt[i].decode('utf8'),
                        (int(w*xmn[i]), int(h*ymn[i])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, # scale
                        COLOR_MAP[cls[i]],
                        2, # thickness
                        cv2.LINE_AA)
        cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(n)), img)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
