#Bring the data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

#Hyper parameters
learning_rate = 0.001
epochs = 20
batch_size = 128 #The batches on how the data is being processed.
display_step = 1 #don't know what this is.

n_input = 784 #Data input for MNIST
n_labels = 10 #Number of possible labels.

#hidden layer number of nodes
n_hidden_layer = 256

#weights and biases
weights = {
	'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer]), tf.float32),
	'output': tf.Variable(tf.random_normal([n_hidden_layer, n_labels]), tf.float32)
}

biases = {
	'hidden_layer': tf.Variable(tf.zeros([n_hidden_layer]), tf.float32),
	'output': tf.Variable(tf.zeros([n_labels]), tf.float32)
}

#set up the input values.
'''This one works.'''
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_labels])

x_flat = tf.reshape(x, [-1, n_input])

print(x_flat.shape)
print(x.shape)

'''
This one fails.

x_unflat = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_labels])

x = tf.reshape(x_unflat, [-1, n_input])
x_flat = tf.reshape(x_unflat, [-1, n_input])
print(x_unflat.shape)
print(x_flat.shape)
print(x.shape)'''

#setup layer operations
hidden_layer = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
hidden_layer = tf.nn.relu(hidden_layer)

output_logits = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])

#set up cost and optimizer.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
			.minimize(cost)

#set up tensorflow session.

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	#run the epochs.
	for epoch in range(epochs):
		#batch the data for faster training.
		total_batch = int(mnist.train.num_examples/batch_size)
		
		#iterate through the data
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			
			#Train the network.
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			
		print('Epoch ' + str(epoch + 1) + ' finished.')














