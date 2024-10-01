#!/usr/bin/env python3
"""
Trains a loaded NN model using mini-batch gradient descent
"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5,
        load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Trains a loaded NN model using mini-batch gradient descent

    X_train: numpy.ndarray of shape (m, 784) containing the training data
        m: number of data points
        784: number of input features
    Y_train: one-hot numpy.ndarray of shape (m, 10) containing the training
    labels
        10: number of output classes
    X_valid: numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid: one-hot numpy.ndarray of shape (m, 10) containing the validation
    labels
    batch_size: number of data points in a batch
    epochs: number of times the training should pass through the whole dataset
    load_path: path from which to load the model
    save_path: path to where the model should be saved after training

    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]
        if m % batch_size == 0:
            num_batch = m // batch_size
        else:
            num_batch = m // batch_size + 1

        for epoch in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(
                loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for i in range(num_batch):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    if end > m:
                        end = m
                    sess.run(train_op, feed_dict={
                        x: X_shuffled[start:end], y: Y_shuffled[start:end]})

        save_path = saver.save(sess, save_path)
    return save_path
