#!/usr/bin/env python

import io
import os.path
import itertools
import hashlib

import PIL
import tensorflow as tf

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('input_records', '', 'Comma separated list of input recordsx')
flags.DEFINE_string('output_record', '', 'Path to output TFRecord')
flags.DEFINE_string('output_shape', None, 'Output record image shape. Comma separated integers w,h')
# TODO: support label remapping
FLAGS = flags.FLAGS

def passthrough(w, h, bytes):
    return w, h, bytes

def reshape(w, h, bytes, shape):
    img = PIL.Image.open(io.BytesIO(bytes))
    fmt = img.format
    img = img.crop((0, 0, *shape))
    out = io.BytesIO()
    img.save(out, fmt)
    return shape[0], shape[1], out.getvalue()

def process(record, converter):
    example = tf.train.Example()
    example.ParseFromString(record)

    w = example.features.feature['image/width'].int64_list.value[0]
    h = example.features.feature['image/height'].int64_list.value[0]
    fmt = example.features.feature['image/format'].bytes_list.value[0].decode('utf8')
    img = example.features.feature['image/encoded'].bytes_list.value[0]

    nw, nh, encoded = converter(w, h, img)
    assert(PIL.Image.open(io.BytesIO(encoded)).format.lower() == fmt.lower())
    if nw == w and nh == h and all(encoded != img):
        return example

    key = hashlib.sha256(encoded).hexdigest()

    ymn = example.features.feature['image/object/bbox/ymin'].float_list.value
    ymx = example.features.feature['image/object/bbox/ymax'].float_list.value
    assert(len(ymn) == 0 or nh >= h*max(max(ymn), max(ymx)))

    example.features.feature['image/width'].CopyFrom(dataset_util.int64_feature(nw))
    example.features.feature['image/height'].CopyFrom(dataset_util.int64_feature(nh))
    example.features.feature['image/width'].CopyFrom(dataset_util.int64_feature(nw))
    example.features.feature['image/key/sha256'].CopyFrom(dataset_util.bytes_feature(key.encode('utf8')))
    example.features.feature['image/encoded'].CopyFrom(dataset_util.bytes_feature(encoded))
    example.features.feature['image/object/bbox/ymin'].CopyFrom(
        dataset_util.float_list_feature([y*h/nh for y in ymn]))
    example.features.feature['image/object/bbox/ymax'].CopyFrom(
        dataset_util.float_list_feature(y*h/nh for y in ymx))
    return example

def mix(inputs, converter):
    for zipped in itertools.zip_longest(*(tf.python_io.tf_record_iterator(f) for f in inputs)):
        for record in zipped:
            if record is None:
                continue
            yield process(record, converter)

def main(_):
    if FLAGS.input_records == '':
        raise ValueError("No input records")
    inputs = [x.strip() for x in FLAGS.input_records.split(',')]
    bad_inputs = [f for f in inputs if not os.path.exists(f)]
    if len(bad_inputs) != 0:
        raise ValueError("Records: " + ", ".join(bad_inputs) + " not found")

    if FLAGS.output_record == '':
        raise ValueError("No output record")

    if FLAGS.output_shape is None:
        tf.logging.info("Keep original shape")
        converter = passthrough
    else:
        shape = tuple([int(x) for x in FLAGS.output_shape.split(',')])
        if len(shape) != 2:
            raise ValueError("Invalid shape " + FLAGS.output_shape)
        tf.logging.info("Desired shape: %s", str(shape))
        converter = lambda w, h, bytes: reshape(w, h, bytes, shape)

    with tf.python_io.TFRecordWriter(FLAGS.output_record) as writer:
        for e in mix(inputs, converter):
            writer.write(e.SerializeToString())

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
