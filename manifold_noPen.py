"""Constitues a manifold graph from given parameters"""

import numpy as np
import gnumpy as gpu
import itertools



class mlgraph(object):
    def __init__(self):
        self.vals = None
        self.idx = None

    def populate(self, root_dir, t1, t2 = None, nbrhoodsz = None):
        self.t1 = t1
        self.t2 = t2 if t2 else t1
        
        # Read neighborhood defnitions to build graph
        Disti = np.load(root_dir + 'Disti.npy')
        # Distp = np.load(root_dir + 'Distp.npy')
        Indi  = (np.load(root_dir + 'Indi.npy')).astype(np.int32)
        # Indp  = (np.load(root_dir + 'Indp.npy')).astype(np.int32)
        
        if nbrhoodsz:
            self.nbrhoodsz = nbrhoodsz
            Disti = Disti[:,:nbrhoodsz]
            # Distp = Distp[:,:nbrhoodsz]
            Indi  = Indi[:,:nbrhoodsz]
            # Indp  = Indp[:,:nbrhoodsz]
        else:
            self.nbrhoodsz = Disti.shape[1]

        Disti = np.exp(-Disti/t1)
        # Distp = -np.exp(-Distp/t1)
        Disti[np.isnan(Disti)] = 0.0
        # Distp[np.isnan(Distp)] = 0.0

        self.vals = Disti.reshape(Disti.shape[0], Disti.shape[1],1)
        self.indx = Indi 
        
        print 'Calculated Wi and Wp as per current configuration'




