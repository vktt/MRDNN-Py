

__author__ = "Vikrant Singh Tomar \nMcGill University\n2013"
__email__ = "vikrant.tomar@mail.mcgill.ca"
__license__="GPL v3"

import numpy as np


def pca(X, no_dims=39):
	""" Trains and returns a PCA projection matrix on 
	a set of given features. X is a ndarray from numpy. 
	The rows of the matrix 	represent the feature vectors,
	and the columns represent the dimensions each vector has. 

	no_dims refers to the number of eigenvectors to keep,
			or the projection dimension
	 """
	# Make Zero mean -- this uses broacasting
	X = X - np.mean(X,0)

	# Computer covariance matrix
	if X.shape[1] < X.shape[0]:
		C = np.cov(X,rowvar=0)
	else:
		C = (1/X.shape[0]) * np.dot(X,X.T) # if N>D, we better use this matrix for the eigendecomposition


	# Perform eigendecomposition of C
	(lmbda, M) = np.linalg.eig(C)
	M = -M #to make most eigenvectors positive

	if no_dims > M.shape[1]:
		no_dims = M.shape[1]
		print 'Target dimensionality reduced to ', no_dims, '.'

	# Sort eigenvectors in descending order
	ind = np.argsort(lmbda) #ascending is default.. we would traverse ind in reverse order ind[::-1]
	ind = ind[::-1][0:no_dims]

	lmbda = lmbda[ind]
	M = M[:,ind]

	if not (X.shape[1] < X.shape[0]):
		M = np.dot(X.T,M) * (1 / np.sqrt(X.shape[0]*lmbda))
	
	return M








