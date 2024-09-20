#!/usr/bin/env python3
"""
Module that builds, trains, and saves a neural network classifier
"""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier
    After every 100 iterations, the 0th iteration
    and `iterations` iteration:
    - it prints the training, training accuracy,
    validation cost and validation accuracy

    X_train: numpy.ndarray containing the training input data
    Y_train: numpy.ndarray containing the training labels
    X_valid: numpy.ndarray containing the validation input data
    Y_valid: numpy.ndarray containing the validation labels
    layer_sizes: list containing the number of
    nodes in each layer of the network
    activations: list containing the activation
    functions for each layer of the network
    alpha: learning rate
    iterations: number of iterations to train over
    save_path: designates where to save the model

    Returns: the path where the model was saved
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations + 1):
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0 or i == iterations:
                train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: X_train, y: Y_train})
                valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
                valid_accuracy = sess.run(accuracy,
                                          feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))
        save_path = saver.save(sess, save_path)
    return save_path
