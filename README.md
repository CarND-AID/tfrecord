[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bd7e2a1edc424df891900a1e22b2532a)](https://www.codacy.com/app/CarND-AID/tfrecord?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=CarND-AID/tfrecord&amp;utm_campaign=Badge_Grade)

# Tensorflow Record API Helpers

## Content
* `create_bosch_tf_record.py` - convert [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) to Tensorflow Record format
* `create_lisa_tf_record.py` - convert [LISA Traffic Light Dataset](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset) to Tensorflow Record format
* `mix_tf_record.py` - interleave (and crop to the same size) one or more Tensorflow Record files
* `dump_tf_record.py` - dump sample of records from Tensorflow Record file

## Usage
```
usage: create_bosch_tf_record.py [-h] [--yaml YAML]
                                 [--output_path OUTPUT_PATH]
                                 [--label_map_path LABEL_MAP_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --yaml YAML           Root directory to Bosch dataset.
  --output_path OUTPUT_PATH
                        Path to output TFRecord
  --label_map_path LABEL_MAP_PATH
                        Path to label map proto
```

```
usage: create_lisa_tf_record.py [-h] [--data_dir DATA_DIR]
                                [--output_path OUTPUT_PATH]
                                [--label_map_path LABEL_MAP_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Root directory to LISA dataset.
  --output_path OUTPUT_PATH
                        Path to output TFRecord
  --label_map_path LABEL_MAP_PATH
                        Path to label map proto
```

```
usage: mix_tf_record.py [-h] [--input_records INPUT_RECORDS]
                        [--output_record OUTPUT_RECORD]
                        [--output_shape OUTPUT_SHAPE]

optional arguments:
  -h, --help            show this help message and exit
  --input_records INPUT_RECORDS
                        Comma separated list of input recordsx
  --output_record OUTPUT_RECORD
                        Path to output TFRecord
  --output_shape OUTPUT_SHAPE
                        Output record image shape. Comma separated integers
                        w,h
```

```
usage: dump_tf_record.py [-h] [--input_record INPUT_RECORD]
                         [--output_path OUTPUT_PATH]
                         [--num_samples NUM_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  --input_record INPUT_RECORD
                        TFRecord to process
  --output_path OUTPUT_PATH
                        Path to output images
  --num_samples NUM_SAMPLES
                        Number of samples

```
