
import numpy as np
import tensorflow as tf
import math
import scipy


""" Get character data from text file """

data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

ix=[]

for ch in data:
    ix.append(char_to_ix[ch])

ixx=ix[:data_size-1]						# input is from beginning to just befor end of file
ixy=ix[1:]									# output is (x+1)th character

X_train= np.eye(vocab_size)[ixx]
Y_train= np.eye(vocab_size)[ixy]
X_train= X_train.reshape(vocab_size,1,data_size)
Y_train= Y_train.reshape(vocab_size,1,data_size)



def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- Input dimensions
	n_y -- Class dimensions
        
    Returns:
    Placeholders for input and output respectively
    """

    X = tf.placeholder(tf.float32, [n_x, 1, None])
    Y = tf.placeholder(tf.float32, [n_y, 1, None])

    
    return X, Y


def initialize_parameters(n_x, n_h, n_y):
    """
    Initializes all weight parameters with given dimensions.
	
	Arguements
    parameters -- a dictionary of tensors containing weights
    """
    

    Wxh = tf.get_variable("Wxh", [n_h, n_x], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Whh = tf.get_variable("Whh", [n_h, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Wy = tf.get_variable("Wy", [n_y, n_h], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    h0 = tf.get_variable("h0", [n_h,1], initializer= tf.contrib.layers.xavier_initializer(seed=0))    #Initial state


    parameters = {"Wxh": Wxh,
                  "Whh": Whh,
                 "Wy": Wy,
                 "h0": h0}
    
    return parameters


def forward_propagation_step(X, parameters, prev_state):
    
    """
    Forward propagation defined as in a single hidden unit IndRNN. The equations are as follows:
    
            h(t)= tanh(Wxh.X + Whh*h(t-1))      #where * is hadamard product
            O = softmax(Wyh.h(t))
            
    Returns: state and output at time 't'
    """
    
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    Wy = parameters['Wy']
    
    Z1= tf.matmul(Wxh, X)
    Z2= tf.multiply(Whh, prev_state)    #hadamard product
    curr_state = tf.tanh(tf.add(Z1, Z2))
    O = tf.nn.softmax(tf.reshape(tf.matmul(Wy, curr_state), [-1]))

    return O, curr_state


def compute_cost(output, Y):
    """
    Computes the cost
    
    Arguments:
    output -- output of forward propagation
    Y -- "true" labels
    
    Returns:
    cost - softmax loss
    """
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    
    return cost


def model(n_x, n_h, n_y, X_train, Y_train, learning_rate=0.015,
          num_epochs=1000, minibatch_size=64, print_cost=True):
    """
    Implements entire model with IndRNN
    
    Arguments:
    X_train -- training set input
    Y_train -- training set output labels
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    
    Returns:
	predictions -- Model's predicted output
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    tf.reset_default_graph()		# to be able to rerun the model without overwriting tf variables
    (_,_ n_len) = X_train.shape		# get length of sequence
    costs = []						# To keep track of the cost
    
    X, Y = create_placeholders(n_x, n_y)
 
    parameters = initialize_parameters(n_x, n_h, n_y)
	
	""" Start computation graph """

    state=[]						# keep track of states
    state.append(parameters['h0'])
	output=[]					# store model's output for entire sequence

    for t in range(n_len):
        model_out, next_state= forward_propagation_step(X[:,:,t],parameters, state[t])
        state.append(next_state)
        output.append(model_out)	
		
    output = tf.convert_to_tensor(output, dtype=tf.float32)								# convert list of outputs to tensor
    output=tf.reshape(output,[n_x,1,n_len])

    cost = compute_cost(prediction_series, Y)											# compute cost

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)		# Optimizer for backprop
	
	
	""" End computation graph """
	
    init = tf.global_variables_initializer()					# initialize variables
     

    with tf.Session() as sess:

        sess.run(init)
        
        # Training loop
		
        for epoch in range(num_epochs):
            
            _ , temp_cost, prediction_series  = sess.run([optimizer, cost, output], feed_dict={X:X_train, Y:Y_train})
            
            if  epoch % 50 == 0:
                print ("Cost after epoch %i: %f" % (epoch, temp_cost))
        
                
        return prediction_series, parameters


prediction_series, trained_params = model(26, 557, 200, 26, X_train, Y_train)
