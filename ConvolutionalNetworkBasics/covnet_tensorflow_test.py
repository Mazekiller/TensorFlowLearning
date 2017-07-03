from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data(".", one_hot=True, reshape=False)

import tensorflow as tf

#Hyperparameters.
learn_rate = 0.000001
epochs = 10
batch_size = 128

test_validation_size = 256 #size depends on memory.
n_labels = 10
dropout = 0.75

#weights and biases
weights = {
	'covlayer1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	'covlayer2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	'fullconnect': tf.Variable(tf.random_normal([7*7*64, 1024])),
	'output': tf.Variable(tf.random_normal([1024, n_labels]))
}

biases = {
	'covlayer1': tf.Variable(tf.zeros([32])),
	'covlayer2': tf.Variable(tf.zeros([64])),
	'fullconnect': tf.Variable(tf.zeros([1024])),
	'output': tf.Variable(tf.zeros([n_labels]))
}
