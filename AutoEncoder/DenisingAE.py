import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in+fan_out))
    high = constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform([fan_in, fan_out], minval = low, maxval = high, dtype = tf.float32)

class AddictiveGaussianNoiseAutoEncoder(object):
    def _initialize_weights(self):
        all_weights = dict()

        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]),dtype=tf.float32)
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input]), dtype=tf.float32)
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input]),dtype = tf.float32)

        return all_weights

    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        # para
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # placeholder
        self.x = tf.placeholder(tf.float32, [None, self.n_input])

        # model
        # h = softplus(W(x+noise)+b)
        # y = Wh + b
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),self.weights['b2'])

        # loss = square error / 2
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        # train
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x:X, self.scale: self.training_scale})
        return cost

    # no training involved
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale: self.training_scale})

    # encode using learned weights
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x:X, self.scale:self.training_scale})

    # decode using learned weights
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden:hidden})

    # encode + decode
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBias(self):
        return self.sess.run(self.weights['b1'])

# run on MNIST data
mnist = input_data.read_data_sets('../data/MNIST/', one_hot=True)

agn_ae = AddictiveGaussianNoiseAutoEncoder(mnist.train.images.shape[1], 64)

for i in range(10000):
    batch_sx, batch_sy = mnist.train.next_batch(100)
    cost = agn_ae.partial_fit(batch_sx)
    if i%50==0:
        print "epoch", i, "loss", cost
    
