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
#the dimensions in the cov nets have to do with the size of the image and filter sizes.
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

#Function to calculate convolutions.
def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

#To help with max pooling.
def maxpool2d(x, k=2):
	return tf.nn.max_pool(
		x,
		ksize=[1, k, k, 1],
		strides=[1, k, k, 1],
		padding='SAME'
	)

def conv_net(x, weights, biases, dropout):
	#Layer 1 28*28*1 to 14*14*32
	conv1 = conv2d(x, weights['covlayer1'], biases['covlayer1'])
	conv1 = maxpool2d(conv1, k=2)
	
	#Layer 2 14*14*32 to 7*7*64
	conv2 = conv2d(conv1, weights['covlayer2'], biases['covlayer2'])
	conv2 = maxpool2d(conv2, k=2)
	
	#Fully connected layer. 7*7*64 to 1024
	fc1 = tf.reshape(conv2, [-1, weights['fullconnect'].get_shape().as_list()[0])
	fc1 = tf.add(tf.matmul(fc1, weights['fullconnect']), biases['fullconnect'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)
	
	#Output layer, 1024 to 10
	out = tf.add(tf.matmul(fc1, weights['output']), biases['output'])
	return output
	
#Session code.

#Prepare variables.
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_labels])
keep_prob = tf.placeholder(tf.float32)

logits = conv_net(x, weights, biases, keep_prob)

#Define cost function and optimization.
cost = tf.reduce_mean(\
		tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

#Gradient descent optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)\
		.minimize(cost)
		
#Calculate accuracy.
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
#Initialize all globals in session.
init = tf.global_variables_initializer()

#Start session.
with tf.Session() as sess:
	sess.run(init)
	
	#Start the epochs loop.
	for epoch in range(epochs):
		#Batch loop.
		for batch in range(mnist.train.num_examples//batch_size):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={ x: batch_x, y: batch_y, keep_prob: dropout })
	
	
	
	
	
	
	
	
	
	
	
	
