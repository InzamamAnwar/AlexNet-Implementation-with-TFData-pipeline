import tensorflow as tf
import random
import cv2
import sys
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _load_images(address):
    img = cv2.cvtColor(cv2.imread(address), cv2.COLOR_RGB2GRAY)
    return img


def _create_record(filenames, labels, tfrecord_filename):
    assert len(filenames) == len(labels)
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    for i in range(len(filenames)):
        feature = {
            'label': _int64_feature(labels[i]),
            'data': _bytes_feature(tf.compat.as_bytes(_load_images(filenames[i]).tostring()))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file, seed, t2tRatio):
    """Build a list of all images files and labels in the data set.
    Args:
      data_dir: string, path to the root directory of images.
        Assumes that the image data set resides in JPEG files located in
        the following directory structure.
          data_dir/dog/another-image.JPEG
          data_dir/dog/my-image.jpg
        where 'dog' is the label associated with these images.
      labels_file: string, path to the labels file.
        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          dog
          cat
          flower
        where each line corresponds to a label. We map each label contained in
        the file to an integer starting with the integer 0 corresponding to the
        label contained in the first line.
      seed: integer used as seed for randomizing images
      test2train: float value less than 1 used to divide train & test set
    Returns:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []

    label_index = 0

    # Construct the list of JPEG files and labels.
    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        print('Finished finding files in %d of %d classes.' % (
                label_index, len(unique_labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(seed)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    train_filenames = filenames[0:int(t2tRatio * len(filenames))]
    train_labels = labels[0:int(t2tRatio * len(labels))]

    print('***TRAIN*** Found %d JPEG files across %d labels inside %s.' %
          (len(train_filenames), len(unique_labels), data_dir))
    _create_record(train_filenames, train_labels, 'train.tfrecords')

    test_filenames = filenames[int(t2tRatio * len(filenames)): ]
    test_labels = labels[int(t2tRatio * len(labels)): ]

    print('***TEST*** Found %d JPEG files across %d labels inside %s.' %
          (len(test_filenames), len(unique_labels), data_dir))
    _create_record(test_filenames, test_labels, 'test.tfrecords')

    return filenames, texts, labels


"""
    Args:
        data_dir: Path to the data(images). In directory
                  the images should be in seperate folder
                  & folder names should represent label.
        labels_file: text file containing all the labels 
                     For example, if there are two labels
                     cat, and dog then text file will have 
                     cat and dog in two lines
        seed: seed used to shuffle the data. Use same seed for repeatability
        t2tRation: (Train to test ratio) used to divide data into train and test set
    Return:
        filenames: list of strings; each string is a path to an image file.
        texts: list of strings; each string is the class, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth.        
"""

filenames, text, labels = _find_image_files(data_dir='Images', labels_file='labelfile.txt',
                                            seed=12345, t2tRatio=0.8)