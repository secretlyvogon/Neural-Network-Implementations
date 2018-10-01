# IndRNNTF
Implementation of IndRNN in Tensorflow as describes in the following paper: https://arxiv.org/pdf/1803.04831v3.pdf

So far it only includes a single layer IndRNN. 

The idea of an IndRNN is based on using a hadamard product in the recurrent cell which makes every neuron independent with a layer. Since the cell weights are no a single vector instead of a matrix, there are fewer parameters. 
