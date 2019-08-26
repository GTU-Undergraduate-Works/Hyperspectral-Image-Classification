# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import os
import CNN
import parameter
import HoustonDataset

PATCH_SIZE = parameter.patch_size



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10001, 'Number of steps to run trainer.')
flags.DEFINE_integer('conv1', 500, 'Number of filters in convolutional layer 1.')
flags.DEFINE_integer('conv2', 100, 'Number of filters in convolutional layer 2.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 84, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
# flags.DEFINE_string('train_dir', '1.mat', 'Directory to put the training data.')

learning_rate = 0.1
num_epochs = 20
max_steps = 100001
IMAGE_SIZE = parameter.patch_size
conv1 = parameter.conv1
conv2 = parameter.conv2
fc1 = parameter.fc1,
fc2 = parameter.fc2
batch_size = parameter.batch_size




def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, CNN.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


"""
Bu fonksiyona parametre olarak dataset yerine görüntü ve labellar verilecek
"""
def fill_feed_dict(images, labels, images_pl, labels_pl):


    feed_dict = {
      images_pl: images,
      labels_pl: labels,
    }
    return feed_dict

"""
Bu fonksiyona parametre olarak dataset değil de görüntü ve labellar verilecek
Bu fonksiyonda şimdilik hata var düzeltilecek
"""
def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,iamges, labels):

    true_count = 0  # Counts the number of correct predictions.

    feed_dict = fill_feed_dict(iamges, labels,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / 100
    print('  Total number of samples: %d  Correct classification: %d  Correct rate : %0.04f' %(100, true_count, precision))




def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on IndianPines.

    dataset = HoustonDataset.Houston()
    train_img = dataset.get_patches(PATCH_SIZE, Train=True, PCA=False, LiDAR=False)
    test_img = dataset.get_patches(PATCH_SIZE, Train=False, PCA=False, LiDAR=False)

    NUM_BANDS = train_img.shape[1]

    train_cls = dataset.get_train_labels()
    test_cls = dataset.get_test_labels()
    train_labels = dataset.get_train_as_one_hot()
    test_labels = dataset.get_test_as_one_hot()
    train_img = np.array(train_img.transpose(0, 3, 1, 2).reshape(train_img.shape[0], PATCH_SIZE, PATCH_SIZE, NUM_BANDS))
    test_img = np.array(test_img.transpose(0, 3, 1, 2).reshape(test_img.shape[0], PATCH_SIZE, PATCH_SIZE, NUM_BANDS))




    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = CNN.inference(images_placeholder,FLAGS.conv1,FLAGS.conv2,FLAGS.hidden1,FLAGS.hidden2)
        # Add to the Graph the Ops for loss calculation.
        loss = CNN.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = CNN.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = CNN.evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        #    summary_op = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        #sess = tf.Session()
        sess = tf.Session()
        # Instantiate a SummaryWriter to output summaries and the Graph.
        #    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(train_img, train_labels,
                                       images_placeholder,
                                       labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)


            if (step+1)%10000==0 and step<=90000:
                FLAGS.learning_rate=FLAGS.learning_rate-0.0001
            # Write the summaries and print an overview fairly often.
            if step % 50 == 0:
                duration = time.time() - start_time
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                #             summary_str = sess.run(summary_op, feed_dict=feed_dict)
                #             summary_writer.add_summary(summary_str, step)
                #             summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step) % 500 == 0 or (step + 1) == FLAGS.max_steps:
                filename = 'model_spatial_CNN_' + str(IMAGE_SIZE) + 'X' + str(IMAGE_SIZE) + '.ckpt'
                chickpoint_dir = "Weights/"
                saver.save(sess, chickpoint_dir,global_step=step)

                # Evaluate against the training set.
                print('Training set test result :')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        train_img, train_labels)
                print('Test set test result:')

                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        test_img, test_labels)


                # Evaluate against the validation set.
                #             print('Validation Data Eval:')
                #             do_eval(sess,
                #                     eval_correct,
                #                     im   data_sets.validation)
                #             # Evaluate against the test set.
                #             print('Test Data Eval:')
                #             do_eval(sess,
                #                     eval_correct,
                #                     images_placeholder,
                #                     labels_placeholder,ages_placeholder,
                #                     labels_placeholder,
                #
                #                     data_sets.test)

run_training()