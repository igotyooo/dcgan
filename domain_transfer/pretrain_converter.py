import sys
import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
sys.path.append( '..' )
from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch
from load import faces
def transform( X, npx ):
    assert X[ 0 ].shape == ( npx, npx, 3 ) or X[ 0 ].shape == ( 3, npx, npx )
    if X[ 0 ].shape == ( npx, npx, 3 ):
        X = X.transpose( 0, 3, 1, 2 )
    return floatX( X / 127.5 - 1. )
def inverse_transform( X, npx ):
    X = ( X.reshape( -1, nc, npx, npx ).transpose( 0, 2, 3, 1 ) + 1. ) / 2.
    return X

# SET PARAMETERS.
k = 1             # # of discrim updates for each gen update.
l2 = 1e-5         # l2 weight decay.
nvis = 196        # # of samples to visualize during training.
b1 = 0.5          # momentum term of adam.
nz = 100 	  # # dim of central activation of converter.
nc = 3            # # of channels in image.
nbatch = 128      # # of examples in batch.
npx_in = 128      # # of pixels width/height of output images.
npx_out = 64      # # of pixels width/height of input images.
ncf = 128         # # of converter filters in last conv layer.
niter = 25        # # of iter at starting learning rate.
niter_decay = 0   # # of iter to linearly decay learning rate to zero.
lr = 0.0002       # initial learning rate for adam.
ntrain = 182236   # # of examples to train on.

# FILE I/O.
desc = 'pretrain_converter'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists( 'logs/' ):
    os.makedirs( 'logs/' )
if not os.path.exists( model_dir ):
    os.makedirs( model_dir )
if not os.path.exists( samples_dir ):
    os.makedirs( samples_dir )
f_log = open( 'logs/%s.ndjson'%desc, 'wb' )
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'cost',]
tr_data, te_data, tr_stream, val_stream, te_stream = faces( ntrain = ntrain )
tr_handle = tr_data.open(  )

# DEFINE NETWORKS.
relu = activations.Rectify(  )
sigmoid = activations.Sigmoid(  )
lrelu = activations.LeakyRectify(  )
tanh = activations.Tanh(  )
bce = T.nnet.binary_crossentropy
tifn = inits.Normal( scale = 0.02 )
gain_ifn = inits.Normal( loc = 1., scale = 0.02 )
bias_ifn = inits.Constant( c = 0. )
cw0 = tifn( ( ncf / 2, nc, 5, 5 ), 'cw0' )
cw1 = tifn( ( ncf, ncf / 2, 5, 5 ), 'cw1' )
cg1 = gain_ifn( ncf, 'cg1' )
cb1 = bias_ifn( ncf, 'cb1' )
cw2 = tifn( ( ncf * 2, ncf, 5, 5 ), 'cw2' )
cg2 = gain_ifn( ( ncf * 2 ), 'cg2' )
cb2 = bias_ifn( ( ncf * 2 ), 'cb2' )
cw3 = tifn( ( ncf * 4, ncf * 2, 5, 5 ), 'cw3' )
cg3 = gain_ifn( ( ncf * 4 ), 'cg3' )
cb3 = bias_ifn( ( ncf * 4 ), 'cb3' )
cw4 = tifn( ( ncf * 8, ncf * 4, 5, 5 ), 'cw4' )
cg4 = gain_ifn( ( ncf * 8 ), 'cg4' )
cb4 = bias_ifn( ( ncf * 8 ), 'cb4' )
cw5 = tifn( ( nz, ncf * 8, 4, 4 ), 'cw5' )
cg5 = gain_ifn( nz, 'cg5' )
cb5 = bias_ifn( nz, 'cb5' )
cw5d = tifn( ( ncf * 8 * 4 * 4, nz, 1, 1 ), 'cw5d' )
cg5d = gain_ifn( ncf * 8 * 4 * 4, 'cg5d' )
cb5d = bias_ifn( ncf * 8 * 4 * 4, 'cb5d' )
cw4d = tifn( ( ncf * 8, ncf * 4, 5, 5 ), 'cw4d' )
cg4d = gain_ifn( ( ncf * 4 ), 'cg4d' )
cb4d = bias_ifn( ( ncf * 4 ), 'cb4d' )
cw3d = tifn( ( ncf * 4, ncf * 2, 5, 5 ), 'cw3d' )
cg3d = gain_ifn( ( ncf * 2 ), 'cg3d' )
cb3d = bias_ifn( ( ncf * 2 ), 'cb3d' )
cw2d = tifn( ( ncf * 2, ncf, 5, 5 ), 'cw2d' )
cg2d = gain_ifn( ( ncf ), 'cg2d' )
cb2d = bias_ifn( ( ncf ), 'cb2d' )
cw1d = tifn( ( ncf, nc, 5, 5 ), 'cw1d' )
convert_params = [ cw0, cw1, cg1, cb1, cw2, cg2, cb2, cw3, cg3, cb3, cw4, cg4, cb4, cw5, cg5, cb5,
        cw5d, cg5d, cb5d, cw4d, cg4d, cb4d, cw3d, cg3d, cb3d, cw2d, cg2d, cb2d, cw1d ]
def converter( IS, w0, w1, g1, b1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5,
        w5d, g5d, b5d, w4d, g4d, b4d, w3d, g3d, b3d, w2d, g2d, b2d, w1d ):
    h0 = lrelu( dnn_conv( IS, w0, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h1 = lrelu( batchnorm( dnn_conv( h0, w1, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g1, b = b1 ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    h5 = lrelu( batchnorm( dnn_conv( h4, w5, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5, b = b5 ) )
    h5d = relu( batchnorm( dnn_conv( h5, w5d, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5d, b = b5d ) )
    h5d = h5d.reshape( ( h5d.shape[0], ncf * 8, 4, 4 ) )
    h4d = relu( batchnorm( deconv( h5d, w4d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4d, b = b4d ) )
    h3d = relu( batchnorm( deconv( h4d, w3d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3d, b = b3d ) )
    h2d = relu( batchnorm( deconv( h3d, w2d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2d, b = b2d ) )
    h1d = tanh( deconv( h2d, w1d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ) )
    y = h1d
    return y

# DEFINE TRAINING FUNCTION: FORWARD(COSTS) AND BACKWARD(UPDATE).
IS = T.tensor4(  )
IT = T.tensor4(  )
IT_hat = converter( IS, *convert_params )
cost = T.mean( ( IT - IT_hat ) ** 2 )
lrt = sharedX( lr )
c_updater = updates.Adam( lr = lrt, b1 = b1, regularizer=updates.Regularizer( l2 = l2 ) )
c_updates = c_updater( convert_params, cost )
updates = c_updates

# COMPILING TRAIN/TEST FUNCTIONS.
print 'COMPILING'
t = time(  )
_train_c = theano.function( [ IS, IT ], cost, updates = c_updates )
_convert = theano.function( [ IS ], IT_hat )
print '%.2f seconds to compile theano functions'%( time(  ) - t )

# DO THE JOB.
print desc.upper(  )
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time(  )
for epoch in range( niter ):
    for ISb, in tqdm( tr_stream.get_epoch_iterator(  ), total = ntrain / nbatch ):
        ISb = transform( ISb, npx_in )
        ITb = ISb # Should be fixed.
        cost = _train_c( ISb, ITb )
        n_updates += 1
        n_examples += len( ISb )
    c_cost = float( cost )
    log = [ n_epochs, n_updates, n_examples, time(  ) - t, c_cost ]
    print '%.0f %.4f'%( epoch, c_cost )
    f_log.write( json.dumps( dict( zip( log_fields, log ) ) ) + '\n' )
    f_log.flush(  )
    samples = np.asarray( _convert( imb, npx_out ) )
    color_grid_vis( inverse_transform( samples, npx_out ), ( 14, 14 ),
            'samples/%s/%d.png'%( desc, n_epochs ) )
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value( floatX( lrt.get_value(  ) - lr / niter_decay ) )
    if n_epochs in [ 1, 2, 3, 4, 5, 10, 15, 20, 25 ]:
        joblib.dump( [ p.get_value(  ) for p in convert_params ],
                'models/%s/%d_convert_params.jl'%( desc, n_epochs ) )
