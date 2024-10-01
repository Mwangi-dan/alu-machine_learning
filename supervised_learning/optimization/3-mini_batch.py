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

    After every 100 steps gradient descent within an epoch,
    the following should be printed:

    The training function should allow for a smaller final batch
    1. meta graph and restore session
    2. Get the following tensors and ops from the collection restored
        x is a placeholder for the input data
        y is a placeholder for the labels
        accuracy is an op to calculate the accuracy of the model
        loss is an op to calculate the cost of the model
        train_op is an op to perform one pass of gradient descent on the model
    3. loop over epochs:
        shuffle data
        loop over the batches:
            get X_batch and Y_batch from data
            train your model
    4. Save session


    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            loss_t, acc_t = sess.run([loss, accuracy],
                                     feed_dict={x: X_train, y: Y_train})
            loss_v, acc_v = sess.run([loss, accuracy],
                                     feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(loss_v))
            print("\tValidation Accuracy: {}".format(acc_v))

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                if i % 100 == 0 and i != 0:
                    loss_t, acc_t = sess.run([loss, accuracy],
                                             feed_dict={
                                                 x: X_train, y: Y_train})
                    loss_v, acc_v = sess.run([loss, accuracy],
                                             feed_dict={
                                                 x: X_valid, y: Y_valid})
                    print("\tStep {}:".format(i))
                    print("\t\tCost: {}".format(loss_t))
                    print("\t\tAccuracy: {}".format(acc_t))
                    print("\t\tValidation Cost: {}".format(loss_v))
                    print("\t\tValidation Accuracy: {}".format(acc_v))

        save_path = saver.save(sess, save_path)
    return save_path
