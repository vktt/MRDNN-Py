# MRDNN
Manifold Regularized Deep Neural Networks written in Python using numpy and gnumpy to run on GPUs.


### Background 
The base of this code (pretrianed deep neural networks) is taken from gdbn code written by George Dahl. http://www.cs.toronto.edu/~gdahl/. I have modified the DNN training by adding manifold regularization to his code. For a detailed overview of Manifold Regularized Neural Networks (MRDNN), please see my paper from Interspeech 2014 here: http://www.ece.mcgill.ca/~vtomar/Publications/Interspeech2014-MRDNN.pdf

### Dependencies
gnumpy: http://www.cs.toronto.edu/~tijmen/gnumpy.html
and cudamat: http://code.google.com/p/cudamat/ (for running on GPUs), or
npmat: http://www.cs.toronto.edu/~ilya/npmat.py) (a non-gpu cudamat simulation)

### Running the Example

Download the gzipped data from http://www.cs.toronto.edu/~gdahl/mnist.npz.gz and unzip it into the same folder as all of the code (or change the line 
f = num.load("mnist.npz")
in mnistExample.py. Then you should be able to run the example with
$ python mnistExample.py
assuming you have obtained all the dependencies for 