
__author__ = "Vikrant Singh Tomar \nMcGill University\n2013"
__email__ = "vikrant.tomar@mail.mcgill.ca"
__license__="GPL v3"

""" This module reads .mat output files of matlab (--v7.3 or greater) and writes those 
	as .npz files using numpy"""

import h5py #for reading matlab files
import numpy as np
import time

def read_matlab_data(matfile):

	# mat_variables is a tuple/list of strings 
	# example = ('Disti', 'Distp', 'Indi', 'Indp')

	f = h5py.File(matfile,'r')
	# print f.keys()
	mat_variables=[varname.encode('ascii','ignore') for varname in f.keys()] #This converts the var names from unicode to ascii

	for varname in mat_variables:
		exec(varname + '= np.array(f.get(varname)).T' )
    
	f.close()
	
	t=time.time()
	np.savez_compressed(matfile[:-4]+'.npz',Disti=Disti, Distp=Distp, Indi=Indi, Indp=Indp)
	print 'Time taken', t - time.time()


	t=time.time()
	np.savez(matfile[:-4]+'_uncomp.npz',Disti=Disti, Distp=Distp, Indi=Indi, Indp=Indp)
	print 'Time in uncompress', t - time.time()


if __name__ == '__main__':

	matfile='input_file.mat'
	read_matlab_data(matfile)

	print 'All done!!!'
