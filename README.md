# MRDNN
Manifold Regularized Deep Neural Networks written in Python using numpy and gnumpy to run on GPUs.


### Background 
The base of this code (pretrianed deep neural networks) is taken from gdbn code written by George Dahl. http://www.cs.toronto.edu/~gdahl/. I have modified the DNN training by adding manifold regularization to his code. For a detailed overview of Manifold Regularized Neural Networks (MRDNN), please see my paper from Interspeech 2014 here: http://www.ece.mcgill.ca/~vtomar/Publications/Interspeech2014-MRDNN.pdf . Please cite the paper if you use this code. I have also uploaded a pdf for the system architecture: MRDNN_architecture.pdf in the repository. 

There is no separate documentation as of yet. But the code should be easy enough to understand with the inline comments.

### Training data -- DNN
* trainInps is an N x D-dimensional array. N feature vectors each having D dimensions
* labels is a vector of dimensionality N containing a label for each feature vector
* trainTargs converts the labels vector into numberic classes and  1-hot target vectors for the DNN or MRDNN training

### Training data -- Manifold based graphs
Manifold based relationships are build in the file mrdnn/manifold.py and mrdnn/manifold_noPen.py
- The network reads in pre-computed affinity graphs for the manifold data. Each graph W is a N x N sparse matrix represented by two matrices:
* Dist: This is a N x k matrix (k = number of nearest neighbors of each vector). Each j\th row of Dist contains distance of vector x_j to k of it's nearest neighbors.
* Ind: This is also a N x k matrix. Each row contains the indices of the nearest neighbors saved in Dist.
* If both the intrinsic and penalty graphs are used, Dist and Ind for these graphs are differenciated by further suffices: Disti and Indi for the intrinsic graph and Distp and Indp for the penalty graph.

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