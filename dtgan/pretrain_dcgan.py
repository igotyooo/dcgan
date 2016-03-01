import sys
import os
import json
from time import time
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
sys.path.append( '..' )
from lib import activations, updates, inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX
from Datain import Gan
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
nvis = 14         # # of samples to visualize during training.
b1 = 0.5          # momentum term of adam.
nz = 100 	  # # dim of central activation of converter.
nc = 3            # # of channels in image.
batch_size = 128  # # of examples in batch.
npx_t = 64        # # of pixels width/height of target images.
nf = 128          # Primary # of filters.
niter = 100       # # of iter at starting learning rate.
niter_decay = 0   # # of iter to linearly decay learning rate to zero.
lr = 0.0002       # Initial learning rate for adam.
shuffle = True    # Suffling training sample sequance.

# DEFINE NETWORKS.
relu = activations.Rectify(  )
sigmoid = activations.Sigmoid(  )
lrelu = activations.LeakyRectify(  )
tanh = activations.Tanh(  )
bce = T.nnet.binary_crossentropy
filt_ifn = inits.Normal( scale = 0.02 )
gain_ifn = inits.Normal( loc = 1., scale = 0.02 )
bias_ifn = inits.Constant( c = 0. )
cw5d = filt_ifn( ( nf * 8 * 4 * 4, nz, 1, 1 ), 'cw5d' )
cg5d = gain_ifn( nf * 8 * 4 * 4, 'cg5d' )
cb5d = bias_ifn( nf * 8 * 4 * 4, 'cb5d' )
cw4d = filt_ifn( ( nf * 8, nf * 4, 5, 5 ), 'cw4d' )
cg4d = gain_ifn( ( nf * 4 ), 'cg4d' )
cb4d = bias_ifn( ( nf * 4 ), 'cb4d' )
cw3d = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'cw3d' )
cg3d = gain_ifn( ( nf * 2 ), 'cg3d' )
cb3d = bias_ifn( ( nf * 2 ), 'cb3d' )
cw2d = filt_ifn( ( nf * 2, nf, 5, 5 ), 'cw2d' )
cg2d = gain_ifn( ( nf ), 'cg2d' )
cb2d = bias_ifn( ( nf ), 'cb2d' )
cw1d = filt_ifn( ( nf, nc, 5, 5 ), 'cw1d' )
dw1 = filt_ifn( ( nf, nc, 5, 5 ), 'dw1' )
dw2 = filt_ifn( ( nf * 2, nf, 5, 5 ), 'dw2' )
dg2 = gain_ifn( ( nf * 2 ), 'dg2' )
db2 = bias_ifn( ( nf * 2 ), 'db2' )
dw3 = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'dw3' )
dg3 = gain_ifn( ( nf * 4 ), 'dg3' )
db3 = bias_ifn( ( nf * 4 ), 'db3' )
dw4 = filt_ifn( ( nf * 8, nf * 4, 5, 5), 'dw4' )
dg4 = gain_ifn( ( nf * 8 ), 'dg4' )
db4 = bias_ifn( ( nf * 8 ), 'db4' )
dwy = filt_ifn( ( nf * 8 * 4 * 4, 1), 'dwy' )
converter_params = [ cw5d, cg5d, cb5d, cw4d, cg4d, cb4d, cw3d, cg3d, cb3d, cw2d, cg2d, cb2d, cw1d ]
discrim_params = [ dw1, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy ]
def converter( Z, w5d, g5d, b5d, w4d, g4d, b4d, w3d, g3d, b3d, w2d, g2d, b2d, w1d ):
    h5d = relu( batchnorm( dnn_conv( Z, w5d, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5d, b = b5d ) )
    h5d = h5d.reshape( ( h5d.shape[ 0 ], nf * 8, 4, 4 ) )
    h4d = relu( batchnorm( deconv( h5d, w4d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4d, b = b4d ) )
    h3d = relu( batchnorm( deconv( h4d, w3d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3d, b = b3d ) )
    h2d = relu( batchnorm( deconv( h3d, w2d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2d, b = b2d ) )
    h1d = tanh( deconv( h2d, w1d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ) )
    y = h1d
    return y
def discrim( X, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy ):
    h1 = lrelu( dnn_conv( X, w1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    h4 = T.flatten( h4, 2 )
    y = sigmoid( T.dot( h4, wy ) )
    return y

# DEFINE TRAINING FUNCTION: FORWARD(COSTS) AND BACKWARD(UPDATE).
Z = T.tensor4(  )
IT = T.tensor4(  )
IT_hat = converter( Z, *converter_params )
lrt = sharedX( lr )
p_real = discrim( IT, *discrim_params )
p_gen = discrim( IT_hat, *discrim_params )
d_cost_from_real = bce( p_real, T.ones( p_real.shape ) ).mean(  )
d_cost_from_gen = bce( p_gen, T.zeros( p_gen.shape ) ).mean(  )
d_cost = d_cost_from_real + d_cost_from_gen
c_cost_to_d = bce( p_gen, T.ones( p_gen.shape ) ).mean(  )
c_cost = c_cost_to_d
cost = [ c_cost, d_cost, d_cost_from_real, d_cost_from_gen ]
d_updater = updates.Adam( lr = lrt, b1 = b1, regularizer = updates.Regularizer( l2 = l2 ) )
c_updater = updates.Adam( lr = lrt, b1 = b1, regularizer = updates.Regularizer( l2 = l2 ) )
d_updates = d_updater( discrim_params, d_cost )
c_updates = c_updater( converter_params, c_cost )

# COMPILING TRAIN/TEST FUNCTIONS.
print 'COMPILING'
t = time(  )
_convert = theano.function( [ Z ], IT_hat )
_train_c = theano.function( [ Z, IT ], cost, updates = c_updates )
_train_d = theano.function( [ Z, IT ], cost, updates = d_updates )
print '%.2f seconds to compile theano functions.'%( time(  ) - t )

# PREPARE FOR DATAIN AND DEFINE SOURCE/TARGET.
di = Gan(  )
di.set_PRODUCTS(  ) # di.set_LookBook(  )
if shuffle:
    di.shuffle(  )
ims = di.load( npx_t, True )

# PREPARE FOR DATAOUT.
dataout = os.path.join( './dataout/', di.name.upper(  ) )
desc = 'pretrain_dcgan'.upper(  )
model_dir = os.path.join( dataout, desc, 'models'.upper(  ) )
sample_dir = os.path.join( dataout, desc, 'samples'.upper(  ) )
log_dir = os.path.join( dataout, desc )
if not os.path.exists( log_dir ):
    os.makedirs( log_dir )
if not os.path.exists( model_dir ):
    os.makedirs( model_dir )
if not os.path.exists( sample_dir ):
    os.makedirs( sample_dir )
f_log = open( os.path.join( log_dir, '%s.ndjson' % desc ), 'wb' )
log_fields = [
    'num_epoch', 
    'num_update', 
    'num_example', 
    't_spent',
    'c_cost',
    'd_cost',]

# DO THE JOB.
print desc.upper(  )
num_update = 0
num_epoch = 0
num_update = 0
num_example = 0
Zb_vis = floatX( np_rng.uniform( -1., 1., size = ( nvis ** 2, nz, 1, 1 ) ) )
t = time(  )
for epoch in range( niter ):
    # Load pre-trained param if exists.
    num_epoch += 1
    mpath_c = os.path.join( model_dir, 'C%03d.npy' % num_epoch )
    mpath_d = os.path.join( model_dir, 'D%03d.npy' % num_epoch )
    if os.path.exists( mpath_c ) and os.path.exists( mpath_d ):
        print( 'Epoch %02d: Load.' % num_epoch )
        data_c = np.load( mpath_c )
        for pi in range( len( converter_params ) ):
            converter_params[ pi ].set_value( data_c[ pi ] )
        data_d = np.load( mpath_d )
        for pi in range( len( discrim_params ) ):
            discrim_params[ pi ].set_value( data_d[ pi ] )
        continue
    # Training.
    num_batches = int( np.ceil( ims.shape[ 0 ] / float( batch_size ) ) )
    for idx in range( num_batches ):
        idxs = idx * batch_size
        idxe = min( idx * batch_size + batch_size, ims.shape[ 0 ] )
        ITb = transform( ims.take( np.arange( idxs, idxe ), axis = 0 ), npx_t )
        Zb = floatX( np_rng.uniform( -1., 1., size = ( len( ITb ), nz, 1, 1 ) ) )
        if num_update % 2 == 0:
            cost = _train_c( Zb, ITb )
        else:
            cost = _train_d( Zb, ITb )
        num_update += 1
        num_example += len( Zb )
        c_cost = float( cost[ 0 ] )
        d_cost = float( cost[ 1 ] )
        if np.mod( idx, num_batches / 20 ) == 0:
            prog = np.round( idx * 100. / num_batches )
            print( 'Epoch %02d: %03d%% (batch %06d / %06d), c_cost = %.4f, d_cost = %.4f' 
                    % ( num_epoch, prog, idx + 1, num_batches, c_cost, d_cost ) )
    # Leave logs.
    c_cost = float( cost[ 0 ] )
    d_cost = float( cost[ 1 ] )
    log = [ num_epoch, num_update, num_example, time(  ) - t, c_cost, d_cost ]
    f_log.write( json.dumps( dict( zip( log_fields, log ) ) ) + '\n' )
    f_log.flush(  )
    # Sample visualization.
    IT_hat_vis = np.asarray( _convert( Zb_vis ) )
    color_grid_vis( inverse_transform( IT_hat_vis, npx_t ), ( nvis, nvis ),
            os.path.join( sample_dir, '%03d.png' % num_epoch ) )
    # Save network.
    print( 'Epoch %02d: Save.' % num_epoch )
    np.save( mpath_c, [ p.get_value(  ) for p in converter_params ] )
    np.save( mpath_d, [ p.get_value(  ) for p in discrim_params ] )
