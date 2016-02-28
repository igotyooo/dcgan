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
shuffle = True    # Suffling training sample sequance.

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

# PREPARE FOR DATAIN.
di = Datain(  )
di.set_LookBook(  )
ims_s = di.load( npx_in, True )
ims_t = di.load( npx_out, True )
if shuffle:
    di.shuffle(  )

# PREPARE FOR DATAOUT.
dataout = os.path.join( './dataout/', di.name.upper(  ) )
desc = 'pretrain_converter'.upper(  )
model_dir = os.path.join( dataout, desc, 'models'.upper(  ) )
samples_dir = os.path.join( dataout, desc, 'samples'.upper(  ) )
logs_dir = os.path.join( dataout, desc )
if not os.path.exists( logs_dir ):
    os.makedirs( logs_dir )
if not os.path.exists( model_dir ):
    os.makedirs( model_dir )
if not os.path.exists( samples_dir ):
    os.makedirs( samples_dir )
f_log = open( os.path.join( logs_dir, '%s.ndjson' % desc ), 'wb' )
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'cost',]

# PLOT SOURCE/TARGET SAMPLE IMAGES.
np.random.seed( 0 )
vis_tr = np.random.permutation( len( di.sset_tr ) )
vis_tr = vis_tr[ 0 : nvis ]
vis_ims_tr_s = ims_t.take( di.sset_tr.take( vis_tr ), axis = 0 )
vis_ims_tr_t = ims_t.take( di.tset_tr.take( vis_tr ), axis = 0 )
color_grid_vis( vis_ims_tr_s, ( 14, 14 ),
            os.path.join( samples_dir, 'TR000S.png' ) )
color_grid_vis( vis_ims_tr_t, ( 14, 14 ),
            os.path.join( samples_dir, 'TR000T.png' ) )
vis_ims_tr_s = transform( ims_s.take( di.sset_tr.take( vis_tr ), axis = 0 ), npx_in )
vis_val = np.random.permutation( len( di.sset_val ) )
vis_val = vis_val[ 0 : nvis ]
vis_ims_val_s = ims_t.take( di.sset_val.take( vis_val ), axis = 0 )
vis_ims_val_t = ims_t.take( di.tset_val.take( vis_val ), axis = 0 )
color_grid_vis( vis_ims_val_s, ( 14, 14 ),
            os.path.join( samples_dir, 'VAL000S.png' ) )
color_grid_vis( vis_ims_val_t, ( 14, 14 ),
            os.path.join( samples_dir, 'VAL000T.png' ) )
vis_ims_val_s = transform( ims_s.take( di.sset_val.take( vis_val ), axis = 0 ), npx_in )

# DO THE JOB.
print desc.upper(  )
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time(  )
for epoch in range( niter ):
    # Load pre-trained param if exists.
    n_epochs += 1
    mpath = os.path.join( model_dir, '%03d.npy' % n_epochs )
    if os.path.exists( mpath ):
        print( 'Epoch %02d: Load.' % n_epochs )
        data = np.load( mpath )
        for pi in range( len( convert_params ) ):
            convert_params[ pi ].set_value( data[ pi ] )
        continue
    # Training.
    num_batches = int( np.ceil( di.sset_tr.shape[ 0 ] / float( nbatch ) ) )
    for idx in range( num_batches ):
        idxs = idx * nbatch
        idxe = min( idx * nbatch + nbatch, di.sset_tr.shape[ 0 ] )
        ISb = transform( ims_s.take( di.sset_tr[ idxs : idxe ], axis = 0 ), npx_in )
        ITb = transform( ims_t.take( di.tset_tr[ idxs : idxe ], axis = 0 ), npx_out )
        cost = _train_c( ISb, ITb )
        n_updates += 1
        n_examples += len( ISb )
        if np.mod( idx, num_batches / 20 ) == 0:
            prog = np.round( idx * 100. / num_batches )
            print( 'Epoch %02d: %03d%% (batch %06d / %06d), cost = %.4f' 
                    % ( n_epochs, prog, idx + 1, num_batches, float( cost ) ) )
    # Leave logs.
    c_cost = float( cost )
    log = [ n_epochs, n_updates, n_examples, time(  ) - t, c_cost ]
    f_log.write( json.dumps( dict( zip( log_fields, log ) ) ) + '\n' )
    f_log.flush(  )
    # Sample visualization.
    samples = np.asarray( _convert( vis_ims_tr_s ) )
    color_grid_vis( inverse_transform( samples, npx_out ), ( 14, 14 ),
            os.path.join( samples_dir, 'TR%03dT.png' % n_epochs ) )
    samples = np.asarray( _convert( vis_ims_val_s ) )
    color_grid_vis( inverse_transform( samples, npx_out ), ( 14, 14 ),
            os.path.join( samples_dir, 'VAL%03dT.png' % n_epochs ) )
    # Save network.
    print( 'Epoch %02d: Save.' % n_epochs )
    np.save( mpath, [ p.get_value(  ) for p in convert_params ] )
