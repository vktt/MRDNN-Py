############################
# Author: Vikrant Tomar
# McGill University
# Date: 21_oct_2013
############################
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


"""
This module reads the .txt features, transforms those using the DNN or MRDNN
 and writes the acoustic feature files used by HTK
"""
__author__="Vikrant Singh Tomar <vikrant.tomar@mail.mcgill.ca>"


import sys,os 
import numpy as np
import cPickle as pickle
import gnumpy as gpu
import time

try:
    from mrdnn_expEta_sfast import *
    import htkmfc
    from pca import pca
except:
    sys.path.append("../mrdnn/")
    from mrdnn_expEta_sfast import *
    import htkmfc
    from pca import pca

def mkdir_p(path):
    #function for making directories.
    # This is equivalent to mkdir -p
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
#end

def apply_nn_train_prePCA(net, nCxt, outLayer, feat_dir, FeatList, outFeatDir, Nframes, useDropout):
    """Sends the training features for feedforward and collects the output in a matrix X for performing PCA"""

    fdir = '';
    dim = net.weights[-2].shape[1]
    X = np.zeros((Nframes,dim))

    inFeatList = open(feat_dir + FeatList).readlines()
    
    fro = 0;
    to = 0;

    for fname in inFeatList:
        if fname.rstrip()[-1] == ':':
            fdir = fname.rstrip()[:-1]+'/'
            continue
        elif fname.rstrip()[-3:]=='txt':
            utt = np.loadtxt(feat_dir + fdir + fname.rstrip())
            # if not useDropout:
            outputs = gpu.as_numpy_array(net.fprop_xf(utt, outLayer))
            # else:
            #     outputs = gpu.as_numpy_array(net.fpropDropout(utt, outLayer))
            assert(outputs.shape[1] == 40)
            fro = to
            to = fro + outputs.shape[0]
            # if X == None:
            # 	X = outputs
            # else:
            X[fro:to] = outputs
            # X = np.concatenate((X,outputs))
            # if i/1*1 == i:
            #   gpu.free_reuse_cache()
            # np.savetxt(feat_dir + outFeatDir + 'train_16k_prePCA/' + fname, gpu.as_numpy_array(outputs))
            np.save(feat_dir + outFeatDir + 'train_prePCA/' + fname[:-5], outputs)
        del outputs
        gpu.free_reuse_cache()

    #End of for
    return X


def apply_nn_train_PCA(P, feat_dir, FeatList, outFeatDir):
    """Apply PCA to the training files"""

    fdir = '';  
    inFeatList = open(feat_dir + FeatList).readlines()
    
    for fname in inFeatList:
        if fname.rstrip()[-1] == ':':
            fdir = fname.rstrip()[:-1]+'/'
            continue
        elif fname.rstrip()[-3:]=='txt':
            x = np.load(feat_dir + outFeatDir + fdir + fname[:-5]+'.npy')
            mapped_x = np.dot(x,P)

        outfile=htkmfc.HTKFeat_write(feat_dir + outFeatDir + 'train/' + fname[:-5], mapped_x.shape[1], htkmfc.USER)
        outfile.writeall(mapped_x)
        del outfile

def apply_nn_test(P, net, nCxt, outLayer, feat_dir, FeatList, outFeatDir, useDropout):
    "Sends the test features for feedforward, and applies the PCA calculated from training files"

    fdir = '';      
    inFeatList = open(feat_dir + FeatList).readlines()

    for fname in inFeatList:
        if fname == '\n':
            continue
        elif fname.rstrip()[-1] == ':':
            fdir = fname.rstrip()[:-1]+'/'
            print fdir
            continue
        elif fname.rstrip()[-3:]=='txt':
            utt = np.loadtxt(feat_dir + fdir + fname[:-1])
            # if not useDropout:
            outputs = gpu.as_numpy_array(net.fprop_xf(utt, outLayer))
            # else:
            # outputs = gpu.as_numpy_array(net.fpropDropout(utt, outLayer))
                
            assert(outputs.shape[1] == 40)
            outputs = np.dot(outputs, P)
            # if i/1*1 == i:
                # gpu.free_reuse_cache()

        outfile=htkmfc.HTKFeat_write(feat_dir + outFeatDir + 'test_feat/' + fdir[-9:] + fname[:-5], outputs.shape[1], htkmfc.USER)
        outfile.writeall(outputs)
        del outfile
        del outputs
        gpu.free_reuse_cache()


#######################################################################################

def main(mlp_experiment, net, nCxt, outLayer, feat_dir):

    ### Use dropout???
    if sum(net.dropouts):
        useDropout = True
    else:
        useDropout = False
    print 'useDropout', useDropout


    ### Create output directory
    outFeatDir = 'some path to output features/'
    
    # feat template: if your output feat directories obey a particular structure, have empty directories at this path
    feat_template='path for template directory structure needed, if needed'
    os.system('cp -R '+ feat_template + ' ' + feat_dir + outFeatDir)
    assert( os.path.isdir(feat_dir + outFeatDir + 'test_feat_16k/test_07') )
    print 'Output feat_dir:', feat_dir + outFeatDir

    ### Training files
    featList = 'files_train_noisy_16k.txt' #List of training data files
    Nframes=5438715; #Total number of vectors in the dataset

    print 'Transforming training features pre-PCA... '
    t1 = time.time()
    X = apply_nn_train_prePCA(net, nCxt, outLayer, feat_dir, featList, outFeatDir, Nframes, useDropout)
    t2 = time.time()
    print 'Total time taken for xfing training prePCA: ', (t2 - t1)/60, 'minutes'
    np.save('X_prePCA_scale0.05', X)
    # X = np.load('train_prePCA_X_likeMatlab.npy')

    print
    print 'Performing PCA...'
    P = pca(X,39)

    print
    print 'Transforming Training features post-PCA...'
    t1 = time.time()
    featList = 'files_train_noisy_16k_prePCA.txt'
    apply_nn_train_PCA(P, feat_dir, featList, outFeatDir)
    t2 = time.time()
    print 'Total time taken for applying PCA to training data:', (t2 - t1)/60, 'minutes'

    ### Testing data
    print
    print 'Transforming Test data...'
    featList = 'files_test_16k.txt'
    t1 = time.time()
    apply_nn_test(P, net, nCxt, outLayer, feat_dir, featList, outFeatDir, useDropout)
    t2 = time.time()
    print 'Total time taken for xfing testing features: ', (t2 - t1)/60, 'minutes'

#######################################################################################

if __name__ == '__main__':


    mlp_experiment = 'what_is_this_network_called'

    mlp_dir = '/some/path/' + mlp_experiment #where is this network stored
    net = pickle.load(open(mlp_dir + '.pkl','rb'))

    outLayer = 5 #  Which layer to take output from
    nCxt = 0

    feat_dir = '/some/path/' #where are your input features.. reads .txt format. Each row is a feature vector.

    main(mlp_experiment, net, nCxt, outLayer, feat_dir)
    print 'All done!!!'




    






