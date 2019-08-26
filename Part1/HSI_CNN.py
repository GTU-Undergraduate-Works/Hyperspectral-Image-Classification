import tensorflow as tf
import numpy as np
import parameter
import HoustonDataset
import time
from datetime import timedelta
import math
from sklearn import metrics
import os

dataset = HoustonDataset.Houston()

PATCH_SIZE = parameter.hsi_patch_size
KERNEL_SIZE = parameter.kernel_size
CONV1 = parameter.conv1
CONV2 = parameter.conv2
FC1 = parameter.fc1
FC2 = parameter.fc2
LEARNING_RATE = parameter.learning_rate

train_img = dataset.get_patches(PATCH_SIZE, Train=True, PCA=True, LiDAR=False, n_components=3)
test_img = dataset.get_patches(PATCH_SIZE, Train=False, PCA=True, LiDAR=False, n_components=3)

NUM_BANDS = train_img.shape[1]

train_cls = dataset.get_train_labels()

train_cls = train_cls - 1

test_cls = dataset.get_test_labels()
train_labels = dataset.get_train_as_one_hot()
test_labels = dataset.get_test_as_one_hot()
train_img = np.array(train_img.transpose(0, 3, 1, 2).reshape(train_img.shape[0], PATCH_SIZE, PATCH_SIZE, NUM_BANDS))
test_img = np.array(test_img.transpose(0, 3, 1, 2).reshape(test_img.shape[0], PATCH_SIZE, PATCH_SIZE, NUM_BANDS))



NUM_CLS = test_labels.shape[1]




x = tf.placeholder(tf.float32, [None, PATCH_SIZE, PATCH_SIZE, NUM_BANDS])
y_true = tf.placeholder(tf.float32, [None, NUM_CLS])
pkeep = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool)


def batch_normalization(input, phase, scope):
    return tf.cond(phase,
                   lambda: tf.contrib.layers.batch_norm(input, decay=0.99, is_training=True,
                                                        updates_collections=None, center=True, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input, decay=0.99, is_training=False,
                                                        updates_collections=None, center=True, scope=scope, reuse=True))

def conv_layer(input, size_in, size_out, scope, use_pooling=True):
    w = tf.Variable(tf.truncated_normal([KERNEL_SIZE, KERNEL_SIZE, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME') + b
    conv_bn = batch_normalization(conv, phase, scope)
    y = tf.nn.relu(conv_bn)

    if use_pooling:
        y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return y

def fc_layer(input, size_in, size_out, scope, relu=True, dropout=True, batch_norm= False):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    logits = tf.matmul(input, w) + b

    if batch_norm:
        logits = batch_normalization(logits, phase, scope)

    if relu:
        y = tf.nn.relu(logits)
        if dropout:
            y = tf.nn.dropout(y, pkeep)
        return y

    return logits
        

conv1 = conv_layer(x, NUM_BANDS, CONV1, scope='conv1', use_pooling=True)
conv2 = conv_layer(conv1, CONV1, CONV2, scope='conv2', use_pooling=True)
flattened = tf.reshape(conv2, [-1, 7*7*CONV2])
fc1 = fc_layer(flattened, 7*7*CONV2, FC1, scope='fc1', relu=True, dropout=True, batch_norm=True)
fc2 = fc_layer(fc1, FC1, FC2, scope='fc2', relu=True, dropout=True, batch_norm=True)
logits = fc_layer(fc2, FC2, NUM_CLS, scope='fc_out', relu=False, dropout=False, batch_norm=False)
y = tf.nn.softmax(logits)



y_pred_cls = tf.argmax(y, 1)

xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

global_step = tf.Variable(0, trainable=False)
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
save_dir = 'WeightsHSI/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'hsi_cnn_model')


loss_graph = []
batch_size = parameter.batch_size

def training_step(iterations):
    num_images = train_labels.shape[0]
    num_batch = math.ceil(num_images / batch_size)
    start_time = time.time()
    for i in range(iterations):
        j = (i % num_batch) * batch_size
        k = min(num_images, j + batch_size)
        feed_dict_train = {x: train_img[j:k, :], y_true: train_labels[j:k, :], pkeep: 0.5, phase: True}
        [_, train_loss] = sess.run([optimizer, loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)
        if i % 3500 == 0:
            global LEARNING_RATE 
            LEARNING_RATE /= 2
        
        
        if i % 100 == 0:
            pred = sess.run(y_pred_cls, feed_dict=feed_dict_train)
            kappa = metrics.cohen_kappa_score(train_cls[j:k], pred)
            print('Iteration:', i, 'Training accuracy:', kappa, 'Training loss:', train_loss)

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage:", timedelta(seconds=int(round(time_dif))))
    
    


batch_size_test = 500
def test_accuracy():
    num_images = test_labels.shape[0]
    sum = 0.0
    i = 0
    while i < num_images:
        j = min(i+batch_size_test, num_images)
        feed_dict_test = {x: test_img[i:j, :], y_true: test_labels[i:j, :], pkeep: 1, phase: False}
        acc = sess.run(accuracy, feed_dict=feed_dict_test)
        sum += acc*(j-i)
        i = j
    print('Testing accuracy:', float(sum/num_images))
    


training_step(28000)
saver.save(sess, save_path=save_path, global_step=global_step)
test_accuracy()
test_accuracy()
        


