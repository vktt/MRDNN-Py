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



class mlgraph(object):
    def __init__(self):
        self.vals = None
        self.idx = None

    def populate(self, root_dir, t1, t2 = None, nbrhoodsz = None):
        """Constitues a manifold graph from given parameters"""
        self.t1 = t1
        self.t2 = t2 if t2 else t1
        
        # Read neighborhood defnitions to build graph
        Disti = np.load(root_dir + 'Disti.npy')
        Distp = np.load(root_dir + 'Distp.npy')
        Indi  = (np.load(root_dir + 'Indi.npy')).astype(np.int32)
        Indp  = (np.load(root_dir + 'Indp.npy')).astype(np.int32)
        
        if nbrhoodsz:
            self.nbrhoodsz = nbrhoodsz
            Disti = Disti[:,:nbrhoodsz]
            Distp = Distp[:,:nbrhoodsz]
            Indi  = Indi[:,:nbrhoodsz]
            Indp  = Indp[:,:nbrhoodsz]
        else:
            self.nbrhoodsz = Disti.shape[1]

        Disti = np.exp(-Disti/t1)
        Distp = -np.exp(-Distp/t1)
        Disti[np.isnan(Disti)] = 0.0
        Distp[np.isnan(Distp)] = 0.0

        self.vals = np.concatenate((Disti.reshape(Disti.shape[0], Disti.shape[1],1), Distp.reshape(Distp.shape[0], Distp.shape[1],1)),axis=1)
        print self.vals.shape
        self.indx = np.concatenate((Indi, Indp),axis=1)

        print 'Calculated Wi and Wp as per current configuration'




