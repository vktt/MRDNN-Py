# MRDNN
Manifold Regularized Deep Neural Networks written in Python using numpy and gnumpy to run on GPUs.


### Background 
The base of this code (pretrianed deep neural networks) is taken from gdbn code written by George Dahl. http://www.cs.toronto.edu/~gdahl/. I have modified the DNN training by adding manifold regularization to his code. For a detailed overview of Manifold Regularized Neural Networks (MRDNN), please see my paper from Interspeech 2014 here: http://www.ece.mcgill.ca/~vtomar/Publications/Interspeech2014-MRDNN.pdf

There is no separate documentation as of yet. But the code should be easy enough to understand with the inline comments.

### Training data -- DNN
* trainInps is an N x D-dimensional array. N feature vectors each having D dimensions
* labels is a vector of dimensionality N containing a label for each feature vector
* trainTargs converts the labels vector into numberic classes and  1-hot target vectors for the DNN or MRDNN training

### Training data -- Manifold based graphs
- The network reads in pre-computed affinity graphs for the manifold data. Each graph W is a N x N sparse matrix represented by two matrices:


### Dependencies
gnumpy: http://www.cs.toronto.edu/~tijmen/gnumpy.html

cudamat: http://code.google.com/p/cudamat/ (for running on GPUs), or

npmat: http://www.cs.toronto.edu/~ilya/npmat.py) (a non-gpu cudamat simulation)

### Running the Example

Download the gzipped data from http://www.cs.toronto.edu/~gdahl/mnist.npz.gz and unzip it into the same folder as all of the code (or change the line 
f = num.load("mnist.npz")
in mnistExample.py. Then you should be able to run the example with
$ python mnistExample.py
assuming you have obtained all the dependencies for 