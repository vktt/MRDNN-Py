"""
 Copyright (c) 2013,2014, 2015 Vikrant Tomar
 
 License: GPL v3. See attached LICENSE file

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.  THE
 SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,  EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES  OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT  HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR  OTHER DEALINGS IN THE
 SOFTWARE.
"""

import numpy as np
import gnumpy as gpu
import itertools
import time
import cPickle as pickle
import sys, os

# For plotting the error curves
import matplotlib
matplotlib.use('pdf') #so that it save the pdf of curve and does not try to use $DISPLAY
import matplotlib.pyplot as plt

try:
	from mrdnn_expEta_sfast import *
	from manifold_noPen import * 
except:
	sys.path.append("./mrdnn/")
	sys.path.append("./helpers/")
	from mrdnn_expEta_sfast import *
	from manifold_noPen import * 

_featType =  'mfcc_117z'  # some features tag .. just nomenclature
_pretrain = 0

if _featType is 'mfcc_117z':
	import xfNN_HTK_mfcc as xf
	start_layer = 117 #dimension of the input features = dimension of the input layer of DNN
	input_data = 'training_data.npz'
	val_data = 'devset.npz'
	feat_dir = '/path/to/input/features/' # for transforming the outputs
	out_layer = 3320 # output feature dimensions
elif _featType is 'zlpda_39z':
	import xfNN_HTK_zlpda as xf
	start_layer = 117 #dimension of the input features = dimension of the input layer of DNN
	input_data = 'training_data_zlpda.npz'
	val_data = 'devset_zlpda.npz'
	feat_dir = '/path/to/input/features/' # for transforming the outputs
	out_layer = 3320 # output feature dimensions
#######################################################################################

def numMistakes(targetsMB, outputs):
	if not isinstance(outputs, np.ndarray):
		outputs = outputs.as_numpy_array()
	if not isinstance(targetsMB, np.ndarray):
		targetsMB = targetsMB.as_numpy_array()
	return np.sum(outputs.argmax(1) != targetsMB.argmax(1))

def sampleMinibatch(mbsz, inps, targs, mlgrphz):
	idx = np.random.randint(inps.shape[0], size=(mbsz,))
	tmpgraph = mlgraph()
	tmpgraph.vals = mlgrphz.vals[idx]
	tmpgraph.indx = mlgrphz.indx[idx]
	return inps[idx], targs[idx], tmpgraph

def main():
	mbsz = 256
	layerSizes = [start_layer, 1024, 1024, 1024, 1024, 40, out_layer]
	scales = [0.05 for i in xrange(len(layerSizes)-1)] #weightScales
	fanOuts = [None for i in xrange(len(layerSizes)-1)]
	learnRate = 0.001 #Gradient learning rate eta
	learnRatesMultiplier = 0.95 # Exponentially decaying rate
	epochs = 100

	### Graph-embedding related parameters
	t1 = 1000 #Gaussian kernel scale factor for intrinsic graph
	t2 = 3000 #Gaussian kernel scale factor for penalty graph
	nbrhoodsz = 10
	mlgamma = 0.001 # Manifold regularization parameter

	######################################## LOAD DATA ############################################
	### Training data
	## trainInps is an N x D-dimensional array. N feature vectors each having D dimensions
	## labels is a vector of dimensionality N containing a label for each feature vector
	# trainTargs converts the labels vector into numberic classes and  1-hot target vectors for the DNN or MRDNN training


	print 'Loading data ...'
	
	root_dir='/path/to/trainng/data/'
	trainInps = pickle.load(open(root_dir+input_data))
	trainInps=np.asarray(trainInps)
	labels = pickle.load(open('/path/to/labels/for/training/data/.pkl'))

	#
	C = np.unique(labels)
	trainTargs = (np.tile(labels,(len(C),1)).T==C).astype(np.int32)
	del C
	
	## Manifold based graph's data
	print 'Load graph-embedding based neighborhs'
	root_dir += 'path/to/manifold/graph/data/'
	mlgrph = mlgraph()
	mlgrph.populate(root_dir, t1, t2, nbrhoodsz)

	print 'Data load done.'

	### Validation data

	f = np.load(val_data)
	validSet={}
	validSet['trainInps'] = f['trainInps']
	val_labels = f['trainTargs']
	
	C = np.unique(val_labels)
	val_labels = (np.tile(val_labels,(len(C),1)).T==C).astype(np.int32)
	validSet['trainTargs']=val_labels
	del C, val_labels, f

	print 'Data load done.'

	####################################################################################

	mbPerEpoch = int(np.ceil(float(trainInps.shape[0])/mbsz)) # Number of minibatches in each epoch
	mbStream = (sampleMinibatch(mbsz, trainInps, trainTargs, mlgrph) for unused in itertools.repeat(None))
	
	print 'Initializing the network with architecture:', layerSizes
	net = buildDBN(layerSizes, scales, fanOuts, Softmax(), realValuedVis = True, useReLU = True)
	net.learnRates = [learnRate for x in net.learnRates]
	net.learnRatesMultiplier = learnRatesMultiplier # for exponentially decaying learning rate
	net.nestCompare = True #this flag existing is a design flaw that I might address later, for now always set it to True
	net.mlgamma = mlgamma if mlgamma else 1.0 # regularization parameter for the manifold learning constraints in the obj. fn.

	if sum(net.dropouts):
		useDropout = True
	else:
		useDropout = False

	####################################################################################
	## Uncomment if you want to do pretraining
	#Pretraining
	# t = [time.time()]

	# if _pretrain:
	#     net.pretrain_learnRates = [preTlearnRate for x in net.pretrain_learnRates]
	#     for i in xrange(len(layerSizes)-2):
	#         print 'Pretraining layer', i+1, 'WeightMatrix Size:', net.weights[i].shape
	#         # for ep, recErr in enumerate(net.preTrainIth(i, mbStream, pretrain_epochs, int(float(mbPerEpoch)/2) )):
	#         for ep, recErr in enumerate(net.preTrainIth(i, mbStream, pretrain_epochs, mbPerEpoch )):
	#             t.append(time.time())
	#             print recErr
	#             # print 'Epoch:', ep, 'Reconstruction Error:', recErr
	#             # print 'Time taken in this epoch:', t[-1] - t[-2] , 'Total time taken:', t[-1] - t[0]
	#             print

	####################################################################################
	##### FineTuning
	
	# Where to save the trained networks?
	netname='DNNML'+str(len(layerSizes)-2)+'_BN_' + _featType + '_1024hu_'+str(mbsz)+'b_eta'+str(learnRate)+'_nbrs'+str(nbrhoodsz)+'_gm'+str(mlgamma)+'_ReLU'+'_expEta'+str(net.learnRatesMultiplier)+'_L2_'+str(net.L2Costs[0])+'_noPen_sfast'
	netDir = 'networks/'+netname
	if not os.path.isdir(netDir):
		os.makedirs(netDir)


	print 'Ready for Fine-tunining'
	t = [time.time()]
	valErr = []
	trErr=[]
	for ep, (trCE, valEr) in enumerate(net.fineTune(mbStream, trainInps, epochs, mbPerEpoch, numMistakes, validSet, progressBar = True, useDropout=useDropout)):
		t.append(time.time())
		valErr.append(valEr)
		trErr.append(trCE)
		print 'Epoch:', ep, 'Train Error:', trCE, 'Val Error:', valEr
		print 'Time taken in this epoch:', t[-1] - t[-2] , 'Total time taken:', t[-1] - t[0]
		print
		del net.stateML, net.actsML, net.pivt, net.actsMLpvt, net.acts, net.state
		# Save network after every epoch
		pickle.dump(net, open( netDir+'/DNN_epc_'+str(ep)+'.pkl', 'wb' ))
		####################################################################################
		# Transforming features every 10 epochs to check recognition performance
		if ep > 0 and ep%10 == 0:
			print 'Now transforming features...'
			outLayer = 5 #  Which layer to take output from
			nCxt = 0
			xf.main(netname+'_epc'+str(ep), net, nCxt, outLayer, feat_dir)
	

	####### Save the error curve plots

	plt.plot(valErr, label='devset')
	plt.hold(True)
	plt.plot(trErr, label='trainset')
	plt.xlabel('Number of Epochs')
	plt.ylabel('Cross Entropy Error')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, fancybox=True, shadow=True)
	plt.savefig('figures/a4/'+netname+'_epc'+str(epochs)+'.pdf')

	print 'Network training done.'

	
#######################################################################################

if __name__ == "__main__":
	main()
	
	print 'All done!!!'
	