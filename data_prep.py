import os
import numpy as np

from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)

image_dir = ""

def _image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )

def _float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_files(label_file):
    """Loads image filenames, classes, and bounding boxes"""
    files, classes, bboxes = [], [], []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            fname, cls, xmin, ymin, xmax, ymax = line.strip('\n').split()
            files.append(os.path.join(image_dir, fname))
            classes.append(int(cls))
            bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    return files, classes, bboxes

# Parameters

height, width = 300,300

label_file = os.getcwd() + "/data/validation.txt"
data_set_type = "validation"

tf_record_location = "gs://vertex-central-1f/covid_proj_tfrecords/"

# Parses a text file line by line, extracting 3 strings and creating 3 arrays
files, classes, bboxes = get_files(label_file)

dataset_size = len(files)
files_per_record = 1500

print("Number of files:", dataset_size)
print("Number of records =", dataset_size//files_per_record)

def create_records():
    for i in tqdm(range(dataset_size//files_per_record)):
        with tf.io.TFRecordWriter(tf_record_location + data_set_type + '/TFRECORD_%i'% i) as file_writer:
            for f in range(files_per_record):
                file_index = (i * files_per_record) + f
                if file_index > dataset_size - 1:
                    break
                    
                im = Image.open(files[file_index])
                x1,y1,x2,y2 = bboxes[file_index]
                
                w,h = im.size
                im.close()
                
                x1 = np.float32(x1/w)
                x2 = np.float32(x2/w)
                y1 = np.float32(y1/h)
                y2 = np.float32(y2/h)
                
                image_bytes = tf.io.decode_png(tf.io.read_file(files[file_index]), channels=3)
                
                # print(image_bytes)
                image = tf.image.resize(image_bytes,(height,width), method='nearest')
                # target_class = tf.cast(target_class, tf.int32)
                
                # plt.figure(figsize=(7, 7))
                # plt.imshow(image.numpy())
                # plt.show()
                
                bounding_box = [x1,y1,x2,y2]
                target_class = classes[file_index]
                
                # print(files[file_index])
                # print(image)
                # print(image_feature(image_bytes))
                
                record_bytes = tf.train.Example(features=tf.train.Features(feature={
                        "image": _image_feature(tf.io.encode_png(image).numpy()),
                        "bounding_box": _float_feature_list(bounding_box),
                        "target_class": _int64_feature(target_class),
                    })).SerializeToString()
                
                file_writer.write(record_bytes)
                
                # return
            # return

create_records()