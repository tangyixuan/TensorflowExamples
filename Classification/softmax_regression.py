'''
This code only use one layer y = Wx + b, followed by the softmax layer
Also, it use a 1D array to represent each image instead of 2D
Can reach accuracy: 91.8%
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
# 55k train, 10k test, 5k validation
print mnist.train.images.shape, mnist.train.labels.shape
print mnist.test.images.shape, mnist.test.labels.shape
print mnist.validation.images.shape, mnist.validation.labels.shape

import tensorflow as tf

# paras
num_features = 784
num_class = 10
num_epochs = 1000

# placeholder
x = tf.placeholder(tf.float32, [None, num_features])
y_ = tf.placeholder(tf.float32, [None, num_class]) # label

# model
W = tf.Variable(tf.zeros([num_features, num_class]))
b = tf.Variable(tf.zeros([num_class]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

# loss and train algorithm
cross_entopy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entopy)

# train
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(num_epochs):
    batch_xs, batch_ys =  mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_:batch_ys})

# test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print accuracy.eval({x: mnist.test.images, y_:mnist.test.labels})
