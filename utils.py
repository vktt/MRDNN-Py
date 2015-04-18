import numpy as np
import gnumpy as gpu

def mult3d(x,y): 
	"""
	% 3d matrix multiplication of 2d parts. Assumes that only the 2d
	% counterpars of the 3d matrices are to be multipled. The third dimension
	% (i.e. pages) remains independent.
	% By - Vikrant Singh Tomar
	% http://www.ece.mcgill.ca/~vtomar
	"""

	x = x if isinstance(x, gpu.garray) else gpu.garray(x)
	y = y if isinstance(y, gpu.garray) else gpu.garray(y)

	gpu.free_reuse_cache()
	ans3d=gpu.zeros((x.shape[0],x.shape[1],y.shape[2]))
	for i in xrange(ans3d.shape[0]):
		ans3d[i]= gpu.dot(x[i],y[i])

	return ans3d


def trans3d(x):
	"""
	% 3d matrix transpose of 2d parts. Assumes that only the 2d
	% counterpars of the 3d matrices are to be transposed. The third dimension
	% (i.e. pages) remains independent.
	% Works for both numpy and gnumpy arrays
	% By - Vikrant Singh Tomar
	% http://www.ece.mcgill.ca/~vtomar
	"""
	return x.transpose((0,2,1))