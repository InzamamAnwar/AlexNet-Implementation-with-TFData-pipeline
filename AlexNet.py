"""
    This code is based on paper written by Alex Krizhevsky et al. Architecture of the network is exactly based on the
    paper. All the hyper parameters are defined at the start of the code which can be adjusted according to user needs.
    TF Data pipeline is used to provide data to the network. First compile the "tfrecords" by running "make_TFRecords.py".
    Data should be saved in the Images folder with folder names representing labels. In "labelfile.txt" labels should be
    place on separate lines.

"""


import tensorflow as tf
import numpy as np
import time

num_labels = 2
num_train_images = 720
num_test_images = 120
image_shape = [224, 224, 1]
num_epochs = 200
train_batch_size = 30
test_batch_size = 30
learning_rate = 1e-3
num_steps = int(num_train_images / train_batch_size)


def reformat_dom(label):
    return (np.arange(num_labels) == label[:, None]).astype(np.float32)


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def decode(serialized_example, img_shape=image_shape):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }
    )
    image = tf.decode_raw(features['data'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, img_shape)
    return image, label


def normalize(image, label):
    image = tf.cast(image, dtype=tf.float32) * (1. / 255)
    return image, label


def reformat(image, label):
    label = tf.one_hot(indices=label, depth=num_labels)
    return image, label


def data_input(batch_size, tf_filename):
    with tf.name_scope('data_input'):
        dataset = tf.data.TFRecordDataset(tf_filename)
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.map(reformat)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


graph = tf.Graph()
with graph.as_default():
    tf_data = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1],
                                                      image_shape[2]], name='InputData')
    tf_label = tf.placeholder(dtype=tf.float32, shape=[None, num_labels])

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    with tf.variable_scope('Conv_1') as Conv_1:
        W = tf.Variable(tf.random_normal(shape=[11, 11, 1, 96], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.zeros(shape=[96]), dtype=tf.float32, name='B')
        L1 = tf.nn.conv2d(input=tf_data, filter=W, strides=[1, 4, 4, 1], padding='VALID')
        L1_b = tf.nn.bias_add(value=L1, bias=B)
        L1_act = tf.nn.relu(L1_b)
        L1_p = tf.nn.max_pool(value=L1_act, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('Conv_2') as Conv_2:
        W = tf.Variable(tf.random_normal(shape=[5, 5, 96, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[256]), dtype=tf.float32)
        L2 = tf.nn.conv2d(input=L1_p, filter=W, strides=[1, 1, 1, 1], padding='SAME')
        L2_b = tf.nn.bias_add(value=L2, bias=B)
        L2_act = tf.nn.relu(L2_b)
        L2_p = tf.nn.max_pool(value=L2_act, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('Conv_3') as Conv_3:
        W = tf.Variable(tf.random_normal(shape=[3, 3, 256, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.zeros(shape=[384]), dtype=tf.float32)
        L3 = tf.nn.conv2d(input=L2_p, filter=W, strides=[1, 1, 1, 1], padding='SAME')
        L3_b = tf.nn.bias_add(value=L3, bias=B)
        L3_act = tf.nn.relu(L3_b)

    with tf.variable_scope('Conv_4') as Conv_4:
        W = tf.Variable(tf.random_normal(shape=[3, 3, 384, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[384]), dtype=tf.float32)
        L4 = tf.nn.conv2d(input=L3_act, filter=W, strides=[1, 1, 1, 1], padding='SAME')
        L4_b = tf.nn.bias_add(value=L3, bias=B)
        L4_act = tf.nn.relu(L4_b)

    with tf.variable_scope('Conv_5') as Conv_5:
        W = tf.Variable(tf.random_normal(shape=[3, 3, 384, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[256]), dtype=tf.float32)
        L5 = tf.nn.conv2d(input=L4_act, filter=W, strides=[1, 1, 1, 1], padding='SAME')
        L5_b = tf.nn.bias_add(value=L5, bias=B)
        L5_act = tf.nn.relu(L5_b)
        L5_p = tf.nn.max_pool(value=L5_act, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('Conv_6') as Conv_6:
        W = tf.Variable(tf.random_normal(shape=[3, 3, 256, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[256]), dtype=tf.float32)
        L9 = tf.nn.conv2d(input=L5_p, filter=W, strides=[1, 1, 1, 1], padding='SAME')
        L9_b = tf.nn.bias_add(value=L9, bias=B)
        L9_act = tf.nn.relu(L9_b)
        L9_p = tf.nn.max_pool(value=L9_act, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('Conv_7') as Conv_7:
        W = tf.Variable(tf.random_normal(shape=[3, 3, 256, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[256]), dtype=tf.float32)
        L10 = tf.nn.conv2d(input=L9_p, filter=W, strides=[1, 1, 1, 1], padding='SAME')
        L10_b = tf.nn.bias_add(value=L10, bias=B)
        L10_act = tf.nn.relu(L10_b)
        L10_p = tf.nn.max_pool(value=L10_act, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('Dense_1') as Dense_1:
        L5_f = tf.layers.flatten(inputs=L10_p)
        L5_d = tf.nn.dropout(x=L5_f, keep_prob=keep_prob)
        flat_shape = L5_f.get_shape().as_list()
        W = tf.Variable(tf.random_normal(shape=[flat_shape[1], 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[4096], dtype=tf.float32), name='B')
        L6 = tf.matmul(L5_d, W) + B
        L6_act = tf.nn.relu(L6)

    with tf.variable_scope('Dense_2') as Dense_2:
        L6_d = tf.nn.dropout(x=L6_act, keep_prob=keep_prob)
        W = tf.Variable(tf.random_normal(shape=[4096, 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[4096], dtype=tf.float32), name='B')
        L7 = tf.matmul(L6_d, W) + B
        L7_act = tf.nn.relu(L7)

    with tf.variable_scope('Dense_3') as Dense_3:
        W = tf.Variable(tf.random_normal(shape=[4096, num_labels], mean=0.0, stddev=0.01, dtype=tf.float32), name='W')
        B = tf.Variable(tf.ones(shape=[num_labels], dtype=tf.float32), name='B')
        L8 = tf.matmul(L7_act, W) + B

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L8, labels=tf_label))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    train_predictions = tf.nn.softmax(logits=L8)
    test_predictions = tf.nn.softmax(logits=L8)


with tf.Session(graph=graph) as sess:

    train_filename = 'train.tfrecords'

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    print('Initialized')

    print('********************** VARIABLES **************************')
    total_parameter = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        name = var.name
        print(name, '=', shape)
        total_parameter = total_parameter + (np.prod(var.get_shape().as_list()))
    print('Total number of trainable parameters in network = ', total_parameter)
    print('********************** VARIABLES **************************')

    getImage = data_input(train_batch_size, train_filename)

    # Train function
    start_time = time.clock()
    for epoch in range(num_epochs):
        print('****************** Epoch %d ******************' % (epoch))
        for step in range(num_steps):
            batch_data, batch_label = sess.run(getImage)

            feed_dict = {tf_data: batch_data, tf_label: batch_label, keep_prob: 0.5}

            _, l, train_outcome = sess.run(fetches=[optimizer, cross_entropy_loss, train_predictions], feed_dict=feed_dict)

            if (step % 100 == 0):
                print('Minibatch loss at Step = %d: %f' % (step, l))
                print('Minibatch accuracy = %f' % (accuracy(train_outcome, batch_label)))

    train_time = time.clock()
    print('*************** Training Time = %f ***************' % (train_time - start_time))

    # Test function
    train_time = time.clock()
    test_filename = 'test.tfrecords'
    total_accuracy = 0.0
    getImage = data_input(test_batch_size, test_filename)

    for step in range(int (num_test_images / test_batch_size)):
        batch_data, batch_label = sess.run(getImage)

        feed_dict = {tf_data: batch_data, keep_prob: 1.0}
        test_outcome = sess.run(fetches=test_predictions, feed_dict=feed_dict)
        total_accuracy = total_accuracy + accuracy(predictions=test_outcome, labels=batch_label)
    print('Test accuracy is = %f' % (total_accuracy / int(num_test_images / test_batch_size)))

    test_time = time.clock()
    print('*************** Test Time = %f ***************' % (test_time - train_time))
    print('*************** Total Time = %f ***************' % (time.clock() - start_time))
