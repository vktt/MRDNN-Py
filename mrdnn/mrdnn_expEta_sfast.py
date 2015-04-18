"""
 Copyright (c) 2013,2014, 2015 Vikrant Tomar
 Based on the code by George Dahl

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

import numpy as num
# import fastnumpy as num
import gnumpy as gnp
import itertools
from utils import *
from activationFunctions import *
from pretrain import CD1
from pretrain import Binary as RBMBinary
from pretrain import Gaussian as RBMGaussian
from pretrain import ReLU as RBMReLU
from counter import Progress
# import time

class DummyProgBar(object):
	def __init__(self, *args): pass
	def tick(self): pass
	def done(self): pass

def initWeightMatrix(shape, scale, maxNonZeroPerColumn = None, uniform = False):
	#number of nonzero incoming connections to a hidden unit
	fanIn = shape[0] if maxNonZeroPerColumn==None else min(maxNonZeroPerColumn, shape[0])
	if uniform:
		W = scale*(2*num.random.rand(*shape)-1)
	else:
		W = scale*num.random.randn(*shape)
	for j in range(shape[1]):
		perm = num.random.permutation(shape[0])
		W[perm[fanIn:],j] *= 0
	return W

def validShapes(weights, biases):
	if len(weights) + 1 == len(biases):
		t1 = all(b.shape[0] == 1 for b in biases)
		t2 = all(wA.shape[1] == wB.shape[0] for wA, wB in zip(weights[:-1], weights[1:]))
		t3 = all(w.shape[1] == hb.shape[1] for w, hb in zip(weights, biases[1:]))
		t4 = all(w.shape[0] == vb.shape[1] for w, vb in zip(weights, biases[:-1]))
		return t1 and t2 and t3 and t4
	return False

def garrayify(arrays):
	return [ar if isinstance(ar, gnp.garray) else gnp.garray(ar) for ar in arrays]

def numpyify(arrays):
	return [ar if isinstance(ar, num.ndarray) else ar.as_numpy_array(dtype=num.float32) for ar in arrays]

def loadDBN(path, outputActFunct, realValuedVis = False, useReLU = False):
	fd = open(path, 'rb')
	d = num.load(fd)
	weights = garrayify(d['weights'].flatten())
	biases = garrayify(d['biases'].flatten())
	genBiases = []
	if 'genBiases' in d:
		genBiases = garrayify(d['genBiases'].flatten())
	fd.close()
	return DBN(weights, biases, genBiases, outputActFunct, realValuedVis, useReLU)

def buildDBN(layerSizes, scales, fanOuts, outputActFunct, realValuedVis, useReLU = False, uniforms = None):
	shapes = [(layerSizes[i-1],layerSizes[i]) for i in range(1, len(layerSizes))]
	assert(len(scales) == len(shapes) == len(fanOuts))
	if uniforms == None:
		uniforms = [False for s in shapes]
	assert(len(scales) == len(uniforms))
	
	initialBiases = [gnp.garray(0*num.random.rand(1, layerSizes[i])) for i in range(1, len(layerSizes))]
	initialGenBiases = [gnp.garray(0*num.random.rand(1, layerSizes[i])) for i in range(len(layerSizes) - 1)]
	initialWeights = [gnp.garray(initWeightMatrix(shapes[i], scales[i], fanOuts[i], uniforms[i])) \
					  for i in range(len(shapes))]
	
	net = DBN(initialWeights, initialBiases, initialGenBiases, outputActFunct, realValuedVis, useReLU)
	return net

def columnRMS(W):
	return gnp.sqrt(gnp.mean(W*W,axis=0))

def limitColumnRMS(W, rmsLim):
	"""
	All columns of W with rms entry above the limit are scaled to equal the limit.
	The limit can either be a row vector or a scalar.
	"""
	rmsScale = rmsLim/columnRMS(W)
	return W*(1 + (rmsScale < 1)*(rmsScale-1))
	
class DBN(object):
	def __init__(self, initialWeights, initialBiases, initialGenBiases, outputActFunct, realValuedVis = False, useReLU = False):
		self.realValuedVis = realValuedVis
		self.learnRates = [0.05 for i in xrange(len(initialWeights))]
		self.learnRatesMultiplier = 1 #For exponentially decaying weights
		self.momentum = 0.9
		self.L2Costs = [0.0001 for i in xrange(len(initialWeights))]
		self.dropouts = [0 for i in xrange(len(initialWeights))]
		self.nesterov = False
		self.nestCompare = False
		self.rmsLims = [None for i in xrange(len(initialWeights))]
		
		if self.realValuedVis:
			self.learnRates[0] = 0.005
		
		self.weights = initialWeights
		self.biases = initialBiases
		self.genBiases = initialGenBiases
		
		if useReLU:
			self.RBMHidUnitType = RBMReLU()
			self.hidActFuncts = [ReLU() for i in range(len(self.weights) - 1)]
		else:
			self.RBMHidUnitType = RBMBinary()
			self.hidActFuncts = [Sigmoid() for i in range(len(self.weights) - 1)]
		self.outputActFunct = outputActFunct
		
		#state variables modified in bprop
		self.WGrads = [gnp.zeros(self.weights[i].shape) for i in range(len(self.weights))]
		self.biasGrads = [gnp.zeros(self.biases[i].shape) for i in range(len(self.biases))]
	
	def weightsDict(self):
		d = {}
		if len(self.weights) == 1:
			d['weights'] = num.empty((1,), dtype=num.object)
			d['weights'][0] = numpyify(self.weights)[0]
			d['biases'] = num.empty((1,), dtype=num.object)
			d['biases'][0] = numpyify(self.biases)[0]
		else:
			d['weights'] = num.array(numpyify(self.weights)).flatten()
			d['biases'] = num.array(numpyify(self.biases)).flatten()
			if len(self.genBiases) == 1:
				d['genBiases'] = num.empty((1,), dtype=num.object)
				d['genBiases'][0] = numpyify(self.genBiases)[0]
			else:
				d['genBiases'] = num.array(numpyify(self.genBiases)).flatten()
		return d
	
	def scaleDerivs(self, scale):
		for i in range(len(self.weights)):
			self.WGrads[i] *= scale
			self.biasGrads[i] *= scale
	
	def loadWeights(self, path, layersToLoad = None):
		fd = open(path, 'rb')
		d = num.load(fd)
		if layersToLoad != None:
			self.weights[:layersToLoad] = garrayify(d['weights'].flatten())[:layersToLoad]
			self.biases[:layersToLoad] = garrayify(d['biases'].flatten())[:layersToLoad]
			self.genBiases[:layersToLoad] = garrayify(d['genBiases'].flatten())[:layersToLoad] #this might not be quite right
		else:
			self.weights = garrayify(d['weights'].flatten())
			self.biases = garrayify(d['biases'].flatten())
			if 'genBiases' in d:
				self.genBiases = garrayify(d['genBiases'].flatten())
			else:
				self.genBiases = []
		fd.close()
	
	def saveWeights(self, path):
		num.savez(path, **self.weightsDict())
		
	def preTrainIth(self, i, minibatchStream, epochs, mbPerEpoch):
		#initialize CD gradient variables
		self.dW = gnp.zeros(self.weights[i].shape)
		self.dvb = gnp.zeros(self.genBiases[i].shape)
		self.dhb = gnp.zeros(self.biases[i].shape)
		
		for ep in xrange(epochs):
			recErr = 0
			totalCases = 0
			for j in range(mbPerEpoch):
				inpMB = minibatchStream.next()
				curRecErr = self.CDStep(inpMB, i, self.learnRates[i], self.momentum, self.L2Costs[i])
				recErr += curRecErr
				totalCases += inpMB.shape[0]
			yield recErr/float(totalCases)
	
	def fineTune(self, minibatchStream, trainInps, epochs, mbPerEpoch, loss = None, validSet = False, progressBar = True, useDropout = False):
		for ep in xrange(epochs):
			print
			print 'learnRates:', self.learnRates
			totalCases = 0
			sumErr = 0
			sumLoss = 0
			if self.nesterov:
				step = self.stepNesterov
			else:
				step = self.step
			prog = Progress(mbPerEpoch) if progressBar else DummyProgBar()
			for i in range(mbPerEpoch):
				# print 'Epoch:', ep, 'minibatch', i
				
				(inpMB, targMB, mbgraph) = minibatchStream.next()
				if len(targMB.shape) != 3: # Convert to a cubic matrix (3d matrix)
					targMB = targMB.reshape(-1,1,targMB.shape[1])
				
				# Each dimensions of inpMB (3d), refers to a pivot vector. Now, we want to select
				# training samples that falls in the neighborhood of this guy, and store in the 
				# corresponding dimension of xsl (x_selected).


				xsl = np.zeros((mbgraph.indx.shape[0],mbgraph.indx.shape[1],trainInps.shape[1]))
				for j in xrange(mbgraph.indx.shape[0]): 
					xsl[j] = trainInps[mbgraph.indx[j]-1]
					# -1 because I need to covert the indices from matlab format to python
				
				#distribute graph.vals to 3d
				vals_select = mbgraph.vals #It has been converted to 3d inside manifold.py
				del mbgraph

				err = step(xsl, vals_select, inpMB, targMB, self.learnRates, self.momentum, self.L2Costs, useDropout)
				# gnp.free_reuse_cache()

				sumErr += err
				# print err, sumErr
				totalCases += inpMB.shape[0]
				prog.tick()
			prog.done()
			self.learnRates = [y*self.learnRatesMultiplier for y in self.learnRates]
			# If validation set is given
			if validSet:
				val_outputActs = self.fprop_xf(validSet['trainInps'])
				val_error = self.outputActFunct.error(gnp.garray(validSet['trainTargs']), self.state[-1], val_outputActs)                
				yield sumErr/float(totalCases), val_error/validSet['trainInps'].shape[0]/vals_select.shape[1]
			else:
				yield sumErr/float(totalCases)
	
	def totalLoss(self, minibatchStream, lossFuncts):
		totalCases = 0
		sumLosses = num.zeros((1+len(lossFuncts),))
		for inpMB, targMB in minibatchStream:
			inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
			targetBatch = targMB if isinstance(targMB, gnp.garray) else gnp.garray(targMB)

			outputActs = self.fprop(inputBatch)
			sumLosses[0] += self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
			for j,f in enumerate(lossFuncts):
				sumLosses[j+1] += f(targetBatch, outputActs)
			totalCases += inpMB.shape[0]
		return sumLosses / float(totalCases)

	def predictions(self, minibatchStream, asNumpy = False):
		for inpMB in minibatchStream:
			inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
			outputActs = self.fprop(inputBatch)
			yield outputActs.as_numpy_array() if asNumpy else outputActs

	def CDStep(self, inputBatch, layer, learnRate, momentum, L2Cost = 0):
		"""
		layer=0 will train the first RBM directly on the input
		"""
		inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
		mbsz = inputBatch.shape[0]
		vis = self.fprop(inputBatch, layer)
		GRBMFlag = layer==0 and self.realValuedVis
		visType = RBMGaussian() if GRBMFlag else self.RBMHidUnitType
		visHidStats, hidBiasStats, visBiasStats, negVis = \
					 CD1(vis, self.weights[layer], self.genBiases[layer], self.biases[layer], visType, self.RBMHidUnitType)
		factor = 1-momentum if not self.nestCompare else 1
		self.dW = momentum*self.dW + factor*visHidStats
		self.dvb = momentum*self.dvb + factor*visBiasStats
		self.dhb = momentum*self.dhb + factor*hidBiasStats

		if L2Cost > 0:
			self.weights[layer] *= 1-L2Cost*learnRate*factor 
		
		self.weights[layer] += (learnRate/mbsz) * self.dW
		self.genBiases[layer] += (learnRate/mbsz) * self.dvb
		self.biases[layer] += (learnRate/mbsz) * self.dhb

		#we compute squared error even for binary visible unit RBMs because who cares
		return gnp.sum((vis-negVis)**2)

	def fpropBprop(self, xsl, vals, inputBatch, targetBatch, useDropout):
		targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)

		if useDropout:
			self.fpropDropout(inputBatch)
		else:
			self.fprop(xsl, inputBatch)

		ps = self.pivt[-1].shape
		outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, None, self.actsMLpvt)
		error_grad = self.outputActFunct.error3d(targetBatch, self.pivt[-1])

		MLerr = 2.0*(self.actsMLpvt - self.actsML)
		error = error_grad/float(MLerr.shape[1]) + self.mlgamma/(2.0 * self.stateML[-1].shape[1]) * (MLerr**2 * vals).sum()
		ml_sense, pivt_sense = self.bprop(outputErrSignal, self.mlgamma*MLerr*vals)

		return error, ml_sense, pivt_sense
	
	def constrainWeights(self):
		for i in range(len(self.rmsLims)):
			if self.rmsLims[i] != None:
				self.weights[i] = limitColumnRMS(self.weights[i], self.rmsLims[i])
	
	def step(self, xsl, vals, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False):
		mbsz = inputBatch.shape[0]
		inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)

		xsl = xsl if isinstance(xsl, gnp.garray) else gnp.garray(xsl)
		vals = vals if isinstance(vals, gnp.garray) else gnp.garray(vals)

		# Get error
		error, ml_sense, pivt_sense = self.fpropBprop(xsl, vals, inputBatch, targetBatch, useDropout)

		# free some memory
		# del xsl, vals
		# gnp.free_reuse_cache()

		# Update the gradients using momentum
		factor = 1-momentum if not self.nestCompare else 1.0
		self.scaleDerivs(momentum)

		# For Manifold learning
		for i, (WGrad, biasGrad) in enumerate(self.gradientsML( ml_sense, pivt_sense )):
			self.WGrads[i] += learnRates[i]*factor*(WGrad/mbsz - L2Costs[i]*self.weights[i])
			self.biasGrads[i] += (learnRates[i]*factor/mbsz)*biasGrad       

		# Update the weigths
		self.applyUpdates(self.weights, self.biases, self.weights, self.biases, self.WGrads, self.biasGrads)
		self.constrainWeights()
		return error
	
	def applyUpdates(self, destWeights, destBiases, curWeights, curBiases, WGrads, biasGrads):
		for i in range(len(destWeights)):
			destWeights[i] = curWeights[i] + WGrads[i]
			destBiases[i] = curBiases[i] + biasGrads[i]
	
	def stepNesterov(self, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False):
		mbsz = inputBatch.shape[0]
		inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
		targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)
		
		curWeights = [w.copy() for w in self.weights]
		curBiases = [b.copy() for b in self.biases]
		self.scaleDerivs(momentum)
		self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)
		
		nodeSensitivity, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout)
		
		#self.scaleDerivs(momentum)
		for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, nodeSensitivity)):
			self.WGrads[i] += learnRates[i]*(WGrad/mbsz - L2Costs[i]*self.weights[i])
			self.biasGrads[i] += (learnRates[i]/mbsz)*biasGrad

		self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)
		self.constrainWeights()
		return error, outputActs

	def gradDebug(self, inputBatch, targetBatch):
		inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
		targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)
		

		mbsz = inputBatch.shape[0]
		outputActs = self.fprop(inputBatch)
		outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, self.state[-1], outputActs)
		#error = self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
		nodeSensitivity = self.bprop(outputErrSignal)
		for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, nodeSensitivity)):
			#update the weight increments
			self.WGrads[i] = WGrad
			self.biasGrads[i] = biasGrad
		allWeightGrads = itertools.chain(self.WGrads, self.biasGrads)
		return gnp.as_numpy_array(gnp.concatenate([dw.ravel() for dw in allWeightGrads])) 
	
	def fprop_xf(self, inputBatch, weightsToStopBefore = None ):
		"""
		Only used during feature dumping after the network has been trained.

		Perform a (possibly partial) forward pass through the
		network. Updates self.state which, on a full forward pass,
		holds the input followed by each hidden layer's activation and
		finally the net input incident on the output layer. Note that state does NOT constrain
		the activation of the output layer. For a full
		forward pass, we return the actual output unit activations. In
		a partial forward pass we return None.
		"""
		if weightsToStopBefore == None:
			weightsToStopBefore = len(self.weights)
		#self.state holds everything before the output nonlinearity, including the net input to the output units
		self.state = [inputBatch]
		for i in range(min(len(self.weights) - 1, weightsToStopBefore)):
			curActs = self.hidActFuncts[i].activation(gnp.dot(self.state[-1], self.weights[i]) + self.biases[i])
			self.state.append(curActs)

		if weightsToStopBefore >= len(self.weights):
			self.state.append(gnp.dot(self.state[-1], self.weights[-1]) + self.biases[-1])
			self.acts = self.outputActFunct.activation(self.state[-1])
			return self.acts
		#we didn't reach the output units
		# To return the first set of hidden activations, we would set
		# weightsToStopBefore to 1.
		return self.state[weightsToStopBefore]

	
	def fprop(self, xsl, inputBatch, weightsToStopBefore = None ):
		"""
		Perform a (possibly partial) forward pass through the
		network. Updates self.state which, on a full forward pass,
		holds the input followed by each hidden layer's activation and
		finally the net input incident on the output layer. For a full
		forward pass, we return the actual output unit activations. In
		a partial forward pass we return None.
		"""

		# Create a 3d copy of the inputBatch, and replicate the pivot vectors to the 
		# size of xsl

		sx, sx3d = xsl.shape[1], xsl.shape[0]
		self.stateML = [ xsl ]

		inp3d = inputBatch.reshape(inputBatch.shape[0],1,inputBatch.shape[1])
		self.pivt = [ inp3d ]

		if weightsToStopBefore == None:
			weightsToStopBefore = len(self.weights)
		#self.state holds everything before the output nonlinearity, including the net input to the output units
		# self.state = [inputBatch]

		for i in xrange(min(len(self.weights) - 1, weightsToStopBefore)):

			curActsML = self.hidActFuncts[i].activation( \
				gnp.dot( self.stateML[-1], self.weights[i]) + self.biases[i] )

			curpivtML =  self.hidActFuncts[i].activation( \
				gnp.dot( self.pivt[-1] , self.weights[i]) + self.biases[i] )

			self.pivt.append(curpivtML)
			self.stateML.append(curActsML)

		if weightsToStopBefore >= len(self.weights):
			
			self.stateML.append(gnp.dot(self.stateML[-1], self.weights[-1]) + self.biases[-1])
			self.actsML = self.outputActFunct.activationML3d(self.stateML[-1])

			self.pivt.append(gnp.dot(self.pivt[-1], self.weights[-1]) + self.biases[-1])
			self.actsMLpvt = self.outputActFunct.activationML3d(self.pivt[-1])

			# return self.actsML

		# Otherwise we didn't reach the output units
		# To return the first set of hidden activations, we would set
		# weightsToStopBefore to 1.
		# return self.stateML[weightsToStopBefore]
	
	def fpropDropout(self, inputBatch, weightsToStopBefore = None ):
		"""
		Perform a (possibly partial) forward pass through the
		network. Updates self.state which, on a full forward pass,
		holds the input followed by each hidden layer's activation and
		finally the net input incident on the output layer. For a full
		forward pass, we return the actual output unit activations. In
		a partial forward pass we return None.
		"""
		if weightsToStopBefore == None:
			weightsToStopBefore = len(self.weights)
		#self.state holds everything before the output nonlinearity, including the net input to the output units
		self.state = [inputBatch * (gnp.rand(*inputBatch.shape) > self.dropouts[0])]
		for i in range(min(len(self.weights) - 1, weightsToStopBefore)):
			dropoutMultiplier = 1.0/(1.0-self.dropouts[i])
			curActs = self.hidActFuncts[i].activation(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[i]) + self.biases[i])
			self.state.append(curActs * (gnp.rand(*curActs.shape) > self.dropouts[i+1]) )
		if weightsToStopBefore >= len(self.weights):
			dropoutMultiplier = 1.0/(1.0-self.dropouts[-1])
			self.state.append(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[-1]) + self.biases[-1])
			self.acts = self.outputActFunct.activation(self.state[-1])
			# return self.acts
		#we didn't reach the output units
		# To return the first set of hidden activations, we would set
		# weightsToStopBefore to 1.
		# return self.state[weightsToStopBefore]

	# def bprop(self, outputErrSignal, MLerr, fpropState = None):
	def bprop(self, outputErrSignal, MLerr, fpropState = None):
		"""
		Perform a backward pass through the network. fpropState
		defaults to self.state (set during fprop) and outputErrSignal
		should be self.outputActFunct.dErrordNetInput(...).
		"""

		# Manifold learning 

		ml_sense = [None for i in range(len(self.weights))]
		pivt_sense = [None for i in range(len(self.weights))]

		ml_sense[-1] = MLerr * self.actsML * (1 - self.actsML)
		pivt_sense[-1] = outputErrSignal - MLerr * self.actsMLpvt * (1 - self.actsMLpvt)

		for i in reversed(range(len(self.weights) - 1)):

			ml_sense[i] = gnp.dot( ml_sense[i+1], self.weights[i+1].T )* self.hidActFuncts[i].dEdNetInput(self.stateML[i+1])
			pivt_sense[i] = gnp.dot( pivt_sense[i+1], self.weights[i+1].T )* self.hidActFuncts[i].dEdNetInput(self.pivt[i+1])

		return ml_sense, pivt_sense

	def gradients(self, fpropState, nodeSensitivity):
		"""
		Lazily generate (negative) gradients for the weights and biases given
		the result of fprop (fpropState) and the result of bprop
		(nodeSensitivity).
		"""
		assert(len(fpropState) == len(self.weights)+1)
		assert(len(nodeSensitivity) == len(self.weights) == len(self.biases))
		for i in range(len(self.weights)):
			yield gnp.dot(fpropState[i].T, nodeSensitivity[i]), nodeSensitivity[i].sum(axis=0)
	
	def gradientsML(self,  ml_sense, pivt_sense):
		"""
		Lazily generate (negative) gradients for the weights and biases given
		the result of fprop (fpropState) and the result of bprop
		(nodeSensitivity).
		"""
		assert(len(ml_sense) == len(pivt_sense) == len(self.weights) == len(self.biases))

		divv = self.stateML[-1].shape[1] # Number of neighbors
		for i in range(len(self.weights)):
		# Here we devide both the pivt and the mlreg related gradients by the total number of neighbors. Theorritically we only need to 
		# devide the mlreg by the number of neighbors to calculate the mean gradient. So one would think that we need to multiply the pivt
		# part by the neighbors before addition. But note that here we are taking advantage of the broadcasting or populating the pivt matrix in 3d 
		# with same values K times, where K is number of neighbors (divv in this section of the code). This serves are multiplying pivt by divv.

			tempWgrad = ( gnp.tensordot( self.pivt[i].tile((1,pivt_sense[i].shape[1],1)), pivt_sense[i], axes = ([0,1],[0,1]) )\
			+ gnp.tensordot( self.stateML[i], ml_sense[i], axes = ([0,1],[0,1]) ) )/divv
			
			tempBgrad = (pivt_sense[i] + ml_sense[i]).sum(axis=0).sum(axis=0)/divv

			# gnp.free_reuse_cache()
			yield tempWgrad, tempBgrad
	
	