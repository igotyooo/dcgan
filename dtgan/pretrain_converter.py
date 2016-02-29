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
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX
from Datain import Datain
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
npx_s = 128       # # of pixels width/height of source images.
npx_t = 64        # # of pixels width/height of target images.
nf = 128          # Primary # of filters.
niter = 25        # # of iter at starting learning rate.
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
cw0 = filt_ifn( ( nf / 2, nc, 5, 5 ), 'cw0' )
cw1 = filt_ifn( ( nf, nf / 2, 5, 5 ), 'cw1' )
cg1 = gain_ifn( nf, 'cg1' )
cb1 = bias_ifn( nf, 'cb1' )
cw2 = filt_ifn( ( nf * 2, nf, 5, 5 ), 'cw2' )
cg2 = gain_ifn( ( nf * 2 ), 'cg2' )
cb2 = bias_ifn( ( nf * 2 ), 'cb2' )
cw3 = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'cw3' )
cg3 = gain_ifn( ( nf * 4 ), 'cg3' )
cb3 = bias_ifn( ( nf * 4 ), 'cb3' )
cw4 = filt_ifn( ( nf * 8, nf * 4, 5, 5 ), 'cw4' )
cg4 = gain_ifn( ( nf * 8 ), 'cg4' )
cb4 = bias_ifn( ( nf * 8 ), 'cb4' )
cw5 = filt_ifn( ( nz, nf * 8, 4, 4 ), 'cw5' )
cg5 = gain_ifn( nz, 'cg5' )
cb5 = bias_ifn( nz, 'cb5' )
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
converter_params = [ cw0, cw1, cg1, cb1, cw2, cg2, cb2, cw3, cg3, cb3, cw4, cg4, cb4, cw5, cg5, cb5,
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
    h5d = h5d.reshape( ( h5d.shape[0], nf * 8, 4, 4 ) )
    h4d = relu( batchnorm( deconv( h5d, w4d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4d, b = b4d ) )
    h3d = relu( batchnorm( deconv( h4d, w3d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3d, b = b3d ) )
    h2d = relu( batchnorm( deconv( h3d, w2d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2d, b = b2d ) )
    h1d = tanh( deconv( h2d, w1d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ) )
    y = h1d
    return y

# DEFINE TRAINING FUNCTION: FORWARD(COSTS) AND BACKWARD(UPDATE).
IS = T.tensor4(  )
IT = T.tensor4(  )
IT_hat = converter( IS, *converter_params )
cost = T.mean( ( IT - IT_hat ) ** 2 )
lrt = sharedX( lr )
c_updater = updates.Adam( lr = lrt, b1 = b1, regularizer=updates.Regularizer( l2 = l2 ) )
c_updates = c_updater( converter_params, cost )

# COMPILING TRAIN/TEST FUNCTIONS.
print 'COMPILING'
t = time(  )
_train_c = theano.function( [ IS, IT ], cost, updates = c_updates )
_convert = theano.function( [ IS ], IT_hat )
print '%.2f seconds to compile theano functions.'%( time(  ) - t )

# PREPARE FOR DATAIN AND DEFINE SOURCE/TARGET.
di = Datain(  )
di.set_LookBook(  )
ims_ssize = di.load( npx_s, True )
ims_tsize = di.load( npx_t, True )
if shuffle:
    di.shuffle(  )
sset_tr = di.d1set_tr
tset_tr = di.d2set_tr
sset_val = di.d1set_val
tset_val = di.d2set_val

# PREPARE FOR DATAOUT.
dataout = os.path.join( './dataout/', di.name.upper(  ) )
desc = 'pretrain_converter'.upper(  )
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
    'cost',]

# PLOT SOURCE/TARGET SAMPLE IMAGES.
np.random.seed( 0 )
vis_tr = np.random.permutation( len( sset_tr ) )
vis_tr = vis_tr[ 0 : nvis ** 2 ]
vis_ims_tr_s = ims_tsize.take( sset_tr.take( vis_tr ), axis = 0 )
vis_ims_tr_t = ims_tsize.take( tset_tr.take( vis_tr ), axis = 0 )
color_grid_vis( vis_ims_tr_s, ( nvis, nvis ),
            os.path.join( sample_dir, 'TR000S.png' ) )
color_grid_vis( vis_ims_tr_t, ( nvis, nvis ),
            os.path.join( sample_dir, 'TR000T.png' ) )
vis_ims_tr_s = transform( ims_ssize.take( sset_tr.take( vis_tr ), axis = 0 ), npx_s )
vis_val = np.random.permutation( len( sset_val ) )
vis_val = vis_val[ 0 : nvis ** 2 ]
vis_ims_val_s = ims_tsize.take( sset_val.take( vis_val ), axis = 0 )
vis_ims_val_t = ims_tsize.take( tset_val.take( vis_val ), axis = 0 )
color_grid_vis( vis_ims_val_s, ( nvis, nvis ),
            os.path.join( sample_dir, 'VAL000S.png' ) )
color_grid_vis( vis_ims_val_t, ( nvis, nvis ),
            os.path.join( sample_dir, 'VAL000T.png' ) )
vis_ims_val_s = transform( ims_ssize.take( sset_val.take( vis_val ), axis = 0 ), npx_s )

# DO THE JOB.
print desc.upper(  )
num_update = 0
num_epoch = 0
num_update = 0
num_example = 0
t = time(  )
for epoch in range( niter ):
    # Load pre-trained param if exists.
    num_epoch += 1
    mpath = os.path.join( model_dir, '%03d.npy' % num_epoch )
    if os.path.exists( mpath ):
        print( 'Epoch %02d: Load.' % num_epoch )
        data = np.load( mpath )
        for pi in range( len( converter_params ) ):
            converter_params[ pi ].set_value( data[ pi ] )
        continue
    # Training.
    num_batches = int( np.ceil( sset_tr.shape[ 0 ] / float( batch_size ) ) )
    for idx in range( num_batches ):
        idxs = idx * batch_size
        idxe = min( idx * batch_size + batch_size, sset_tr.shape[ 0 ] )
        ISb = transform( ims_ssize.take( sset_tr[ idxs : idxe ], axis = 0 ), npx_s )
        ITb = transform( ims_tsize.take( tset_tr[ idxs : idxe ], axis = 0 ), npx_t )
        cost = _train_c( ISb, ITb )
        num_update += 1
        num_example += len( ISb )
        if np.mod( idx, num_batches / 20 ) == 0:
            prog = np.round( idx * 100. / num_batches )
            print( 'Epoch %02d: %03d%% (batch %06d / %06d), cost = %.4f' 
                    % ( num_epoch, prog, idx + 1, num_batches, float( cost ) ) )
    # Leave logs.
    c_cost = float( cost )
    log = [ num_epoch, num_update, num_example, time(  ) - t, c_cost ]
    f_log.write( json.dumps( dict( zip( log_fields, log ) ) ) + '\n' )
    f_log.flush(  )
    # Sample visualization.
    samples = np.asarray( _convert( vis_ims_tr_s ) )
    color_grid_vis( inverse_transform( samples, npx_t ), ( nvis, nvis ),
            os.path.join( sample_dir, 'TR%03dT.png' % num_epoch ) )
    samples = np.asarray( _convert( vis_ims_val_s ) )
    color_grid_vis( inverse_transform( samples, npx_t ), ( nvis, nvis ),
            os.path.join( sample_dir, 'VAL%03dT.png' % num_epoch ) )
    # Save network.
    print( 'Epoch %02d: Save.' % num_epoch )
    np.save( mpath, [ p.get_value(  ) for p in converter_params ] )
