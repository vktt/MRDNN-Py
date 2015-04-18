
import numpy as np
import gnumpy as gpu
import itertools
import time
import cPickle as pickle
import sys, os

import matplotlib
matplotlib.use('pdf') #so that it does not try to use $DISPLAY
import matplotlib.pyplot as plt

try:
	from dbn3_expEta_superfast import *
	from manifold_noPen import * 
except:
	sys.path.append("/home/vikrant/customToolboxes/python/dnn_mlregularized")
	from dbn3_expEta_superfast import *
	from manifold_noPen import * 

_featType =  'mfcc_117z'  # a2_mfcc117z, 'mfcc_117z' # or 'lpda_39z' or 'zlpda_39z'
_pretrain = 0

if _featType is 'mfcc_117z':
	import xfNN_HTK_mfcc as xf
	start_layer = 117
	input_data = 'data_noisytr_16k_z.npz'
	val_data = 'devset_16k_z.npz'
	feat_dir = '/home/vikrant/data/Aurora4a_mfcc_0_z/'
	out_layer = 121
elif _featType is 'lpda_39z':
	import xfNN_HTK_lpda as xf
	start_layer = 39
	input_data = 'data_noisytr_16k_lpda39_Z_origZmlf.npz'
	val_data = 'devset_16k_lpda39_Z_origZmlf.npz'
	feat_dir = '/home/vikrant/data/Aurora4a_mfcc_0/'
	out_layer = 121
elif _featType is 'zlpda_39z':
	import xfNN_HTK_zlpda as xf
	start_layer = 39
	input_data = 'data_noisytr_16k_zlpda39_Z_origZmlf.npz'
	val_data = 'devset_16k_zlpda39_Z_origZmlf.npz'
	feat_dir = '/home/vikrant/data/Aurora4a_mfcc_0_z/'
	out_layer = 121
elif _featType is 'a2_mfcc117z':
	import xfNN_HTK_mfcc_A2 as xf
	start_layer = 117
	input_data = 'data_multitr_Z.npz'
	feat_dir = '/home/vikrant/data/Aurora2_mfcc_0_z/'
	out_layer = 180
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
	scales = [0.05 for i in xrange(len(layerSizes)-1)] #weightScales 0.05
	fanOuts = [None for i in xrange(len(layerSizes)-1)]
	# preTlearnRate = 0.005
	# pretrain_epochs = 7
	learnRate = 0.00001 #Gradient learning rate eta
	learnRatesMultiplier = 0.95 # Exponentially decaying rate
	epochs = 100

	######################################## LOAD DATA ############################################
	### Training data

	print 'Loading data ...'
		
	# root_dir = '/home/vikrant/data/Aurora2matdata/'
	# f = np.load( root_dir + input_data)
	# trainInps = f['trainInps']
	# labels = f['trainTargs'].T
	# del f
	
	root_dir='/home/vikrant/data/Aurora4matdata/'
	trainInps = pickle.load(open('/home/vikrant/data/Aurora4matdata/data_noisytr_16k_z_X.pkl'))
	trainInps=np.asarray(trainInps)
	labels = pickle.load(open('/home/vikrant/data/Aurora4matdata/data_noisytr_16k_z_labels.pkl'))

	C = np.unique(labels)
	trainTargs = (np.tile(labels,(len(C),1)).T==C).astype(np.int32)
	del C

	### Graph-embedding related parameters
	t1 = 1000
	t2 = 3000
	nbrhoodsz = 10
	mlgamma = 0.02 # Manifold regularization parameter
	
	print 'Load graph-embedding based neighborhs'
	root_dir += 'out_data_noisytr_16k_euclsq_z/'
	# root_dir += 'out_data_multitr_euclsq_z/'
	mlgrph = mlgraph()
	mlgrph.populate(root_dir, t1, t2, nbrhoodsz)

	print 'Data load done.'

	### Validation data

	f = np.load('/home/vikrant/data/Aurora4matdata/' + val_data)
	validSet={}
	validSet['trainInps'] = f['trainInps']
	val_labels = f['trainTargs']
	
	C = np.unique(val_labels)
	val_labels = (np.tile(val_labels,(len(C),1)).T==C).astype(np.int32)
	validSet['trainTargs']=val_labels
	del C, val_labels, f

	print 'Data load done.'
	print trainInps.shape, trainTargs.shape, validSet['trainTargs'].shape, validSet['trainInps'].shape

	####################################################################################

	mbPerEpoch = int(np.ceil(float(trainInps.shape[0])/mbsz)) # Number of minibatches in each epoch
	mbStream = (sampleMinibatch(mbsz, trainInps, trainTargs, mlgrph) for unused in itertools.repeat(None))
	
	print 'Initializing the network with architecture:', layerSizes
	net = buildDBN(layerSizes, scales, fanOuts, Softmax(), realValuedVis = True, useReLU = True)
	net.learnRates = [learnRate for x in net.learnRates]
	net.learnRatesMultiplier = learnRatesMultiplier
	# net.L2Costs = [0 for x in net.L2Costs]
	net.nestCompare = True #this flag existing is a design flaw that I might address later, for now always set it to True
	net.mlgamma = mlgamma if mlgamma else 1.0

	#net.dropouts = [0.4 for i in xrange(len(layerSizes) -1 )]
	# net.momentum = 0.5
	# net.nesterov = True

	## Load old network 
	# netname='DNN4_BN_lpda_39z_1024hu_20epc_500batch_eta0.3_scale0.05'
	# mlp_dir = '/home/vikrant/data/tandem_py/noisytr/networks/' + netname
	# mlp_path= mlp_dir + '.pkl'
	# net = pickle.load(open(mlp_path,'rb'))
	# epochs = 10

	if sum(net.dropouts):
		useDropout = True
	else:
		useDropout = False

	####################################################################################
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
	# del net
	# net = pickle.load(open('networks/DNNMLreg5_BN_mfcc_117z_1024hu_256batch_eta0.1_scale0.05_nbrhoodsz30_newMLErrnoErrDiv/DNN_epc_4.pkl','rb'))
	
	netname='DNNML'+str(len(layerSizes)-2)+'_BN_' + _featType + '_1024hu_'+str(mbsz)+'b_eta'+str(learnRate)+'_nbrs'+str(nbrhoodsz)+'_gm'+str(mlgamma)+'_ReLU'+'_expEta'+str(net.learnRatesMultiplier)+'_L2_'+str(net.L2Costs[0])+'_noPen_sfast'
	netDir = 'networks/'+netname
	if not os.path.isdir(netDir):
		os.makedirs(netDir)

	# del net
	# print netDir+'/DNN_epc_5.pkl'
	# net = pickle.load(open( netDir+'/DNN_epc_4.pkl', 'rb' ))

	# net.learnRates = [0.0025026 for x in net.learnRates]

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
		pickle.dump(net, open( netDir+'/DNN_epc_'+str(ep)+'.pkl', 'wb' ))
		####################################################################################
		# Transforming features
		if ep > 0 and ep%10 == 0:
			print 'Now transforming features...'
			outLayer = 5 #  Which layer to take output from
			nCxt = 0
			xf.main(netname+'_epc'+str(ep), net, nCxt, outLayer, feat_dir)
	

	####### Plot the error curves

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
	