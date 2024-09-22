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
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            t_loss, t_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_train, y: Y_train})
            v_loss, v_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_loss))
                print("\tTraining Accuracy: {}".format(t_accuracy))
                print("\tValidation Cost: {}".format(v_loss))
                print("\tValidation Accuracy: {}".format(v_accuracy))
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        i += 1
        loss_train, accuracy_train = sess.run(
            [loss, accuracy], feed_dict={x: X_train, y: Y_train})
        loss_valid, accuracy_valid = sess.run(
            [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
        accuracy_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        print("After {} iterations:".format(i))
        print("\tTraining Cost: {}".format(loss_train))
        print("\tTraining Accuracy: {}".format(accuracy_train))
        print("\tValidation Cost: {}".format(loss_valid))
        print("\tValidation Accuracy: {}".format(accuracy_valid))
        return saver.save(sess, save_path)
