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
from Datain import Pldt
def transform( X, npx ):
    assert X[ 0 ].shape == ( npx, npx, 3 ) or X[ 0 ].shape == ( 3, npx, npx )
    if X[ 0 ].shape == ( npx, npx, 3 ):
        X = X.transpose( 0, 3, 1, 2 )
    return floatX( X / 127.5 - 1. )
def inverse_transform( X, npx ):
    X = ( X.reshape( -1, nc, npx, npx ).transpose( 0, 2, 3, 1 ) + 1. ) / 2.
    return X

# SET PARAMETERS.
l2 = 1e-5           # l2 weight decay.
b1 = 0.5            # momentum term of adam.
nc = 3              # # of channels in image.
batch_size = 128    # # of examples in batch.
npx_in = 64         # # of pixels width/height of input images.
nf = 128            # Primary # of filters.
niter_lr0 = 15      # # of iter at starting learning rate.
niter = 20          # # of total iteration.
lr_decay = 10       # # of iter to linearly decay learning rate to zero.
lr = 0.0002         # Initial learning rate for adam.
shuffle = True      # Suffling training sample sequance.

# DEFINE NETWORKS.
relu = activations.Rectify(  )
sigmoid = activations.Sigmoid(  )
lrelu = activations.LeakyRectify(  )
tanh = activations.Tanh(  )
bce = T.nnet.binary_crossentropy
filt_ifn = inits.Normal( scale = 0.02 )
gain_ifn = inits.Normal( loc = 1., scale = 0.02 )
bias_ifn = inits.Constant( c = 0. )
dw1 = filt_ifn( ( nf, nc * 2, 5, 5 ), 'dw1' )
dw2 = filt_ifn( ( nf * 2, nf, 5, 5 ), 'dw2' )
dg2 = gain_ifn( ( nf * 2 ), 'dg2' )
db2 = bias_ifn( ( nf * 2 ), 'db2' )
dw3 = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'dw3' )
dg3 = gain_ifn( ( nf * 4 ), 'dg3' )
db3 = bias_ifn( ( nf * 4 ), 'db3' )
dw4 = filt_ifn( ( nf * 8, nf * 4, 5, 5), 'dw4' )
dg4 = gain_ifn( ( nf * 8 ), 'dg4' )
db4 = bias_ifn( ( nf * 8 ), 'db4' )
dw5 = filt_ifn( ( nf * 16, nf * 8, 4, 4 ), 'dw5' )
dg5 = gain_ifn( ( nf * 16 ), 'dg5' )
db5 = bias_ifn( ( nf * 16 ), 'db5' )
dwy = filt_ifn( ( nf * 16, 1 ), 'dwy' )
domain_discrim_params = [ dw1, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dw5, dg5, db5, dwy ]
def domain_discrim( X, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, wy ):
    h1 = lrelu( dnn_conv( X, w1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    h2 = lrelu( batchnorm( dnn_conv( h1, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    h3 = lrelu( batchnorm( dnn_conv( h2, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    h4 = lrelu( batchnorm( dnn_conv( h3, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    h5 = lrelu( batchnorm( dnn_conv( h4, w5, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5, b = b5 ) )
    y = sigmoid( T.dot( T.flatten( h5, 2 ), wy ) )
    return y

# DEFINE TRAINING FUNCTION: FORWARD(COSTS) AND BACKWARD(UPDATE).
I = T.tensor4(  )
Y = T.matrix(  )
Y_hat = domain_discrim( I, *domain_discrim_params )
lrt = sharedX( lr )
cost = bce( Y_hat, Y ).mean(  )
out = [ cost, Y_hat ]
d_updater = updates.Adam( lr = lrt, b1 = b1, regularizer = updates.Regularizer( l2 = l2 ) )
d_updates = d_updater( domain_discrim_params, cost )

# COMPILING TRAIN/TEST FUNCTIONS.
print 'COMPILING'
t = time(  )
_domain_discrim = theano.function( [ I, Y ], out )
_train_d = theano.function( [ I, Y ], out, updates = d_updates )
print '%.2f seconds to compile theano functions.'%( time(  ) - t )

# PREPARE FOR DATAIN AND DEFINE SOURCE/TARGET.
di = Pldt(  )
di.set_LOOKBOOK(  )
ims = di.load( npx_in, True )
if shuffle:
    di.shuffle(  )
sset_tr = di.d1set_tr
tset_tr = di.d2set_tr
pids_tr = di.pids_tr
sset_val = di.d1set_val
tset_val = di.d2set_val
pids_val = di.pids_val

# PREPARE FOR DATAOUT.
dataout = os.path.join( './dataout/', di.name.upper(  ) )
desc = 'train_domain_discrim'.upper(  )
model_dir = os.path.join( dataout, desc, 'models'.upper(  ) )
log_dir = os.path.join( dataout, desc )
if not os.path.exists( log_dir ):
    os.makedirs( log_dir )
if not os.path.exists( model_dir ):
    os.makedirs( model_dir )
f_log = open( os.path.join( log_dir, '%s.ndjson' % desc ), 'wb' )
log_fields = [ 'num_epoch', 'num_update', 'num_example', 't_spent', 'cost_tr', 'err_tr', 'cost_val', 'err_val' ]

# DO THE JOB.
print desc.upper(  )
num_update = 0
num_epoch = 0
num_update = 0
num_example = 0
t = time(  )
for epoch in range( niter ):
    # Decay learning rate if needed.
    num_epoch += 1
    if num_epoch > niter_lr0:
        print( 'Decaying learning rate.' )
        lrt.set_value( floatX( lrt.get_value(  ) - lr / lr_decay ) )
    # Load pre-trained param if exists.
    mpath_d = os.path.join( model_dir, 'D%03d.npy' % num_epoch )
    if os.path.exists( mpath_d ):
        print( 'Epoch %02d: Load.' % num_epoch )
        data_d = np.load( mpath_d )
        for pi in range( len( domain_discrim_params ) ):
            domain_discrim_params[ pi ].set_value( data_d[ pi ] )
        continue
    # Training.
    num_sample_tr = len( sset_tr )
    num_batch_tr = int( np.ceil( num_sample_tr / float( batch_size ) ) )
    cost_tr_cumm = 0.
    err_tr_cumm = 0.
    for bi_tr in range( num_batch_tr ):
        bis_tr = bi_tr * batch_size
        bie_tr = min( bi_tr * batch_size + batch_size, num_sample_tr )
        ISb_tr = ims.take( sset_tr[ bis_tr : bie_tr ], axis = 0 )
        ITb_tr = ims.take( tset_tr[ bis_tr : bie_tr ], axis = 0 )
        this_bsize = len( ISb_tr )
        Yb_tr = np.ones( ( this_bsize, 1 ), np.float32 )
        pb_tr = pids_tr[ bis_tr : bie_tr ]
        for i in range( this_bsize ):
            if np.random.uniform(  ) > 0.5:
                iid = tset_tr[ np.random.choice( ( pids_tr != pb_tr[ i ] ).nonzero(  )[ 0 ], 1 ) ]
                ITb_tr[ i, :, :, : ] = ims[ iid, :, :, : ]
                Yb_tr[ i, : ] = 0.
            if np.random.uniform(  ) > 0.5:
                ISb_tr[ i, :, :, : ] = np.fliplr( ISb_tr[ i, :, :, : ] )
            if np.random.uniform(  ) > 0.5:
                ITb_tr[ i, :, :, : ] = np.fliplr( ITb_tr[ i, :, :, : ] ) 
        ISb_tr = transform( ISb_tr, npx_in )
        ITb_tr = transform( ITb_tr, npx_in )
        Ib_tr = np.concatenate( ( ISb_tr, ITb_tr ), axis = 1 )
        out_tr = _train_d( Ib_tr, Yb_tr )
        num_update += 1
        num_example += this_bsize
        cost_tr = float( out_tr[ 0 ] )
        cost_tr_cumm += cost_tr * this_bsize
        err_tr = ( ( out_tr[ 1 ] > 0.5 ) - Yb_tr.astype( bool ) ).mean(  )
        err_tr_cumm += err_tr * this_bsize 
        if np.mod( bi_tr, num_batch_tr / 20 ) == 0:
            prog = np.round( bi_tr * 100. / num_batch_tr )
            print( 'Epoch %02d: %03d%% (batch %06d / %06d), cost_tr = %.4f, err_tr = %.4f' 
                    % ( num_epoch, prog, bi_tr + 1, num_batch_tr, cost_tr, err_tr ) )
    # Save network.
    cost_tr_cumm = cost_tr_cumm / num_sample_tr
    err_tr_cumm = err_tr_cumm / num_sample_tr
    print( 'Epoch %03d: cost_tr = %.4f, err_tr = %.4f' % ( num_epoch, cost_tr_cumm, err_tr_cumm ) )
    print( 'Epoch %02d: Save.' % num_epoch )
    np.save( mpath_d, [ p.get_value(  ) for p in domain_discrim_params ] )
    # Validation.
    num_sample_val = len( sset_val )
    num_batch_val = int( np.ceil( num_sample_val / float( batch_size ) ) )
    cost_val_cumm = 0.
    err_val_cumm = 0.
    for bi_val in range( num_batch_val ):
        bis_val = bi_val * batch_size
        bie_val = min( bi_val * batch_size + batch_size, num_sample_val )
        ISb_val = ims.take( sset_val[ bis_val : bie_val ], axis = 0 )
        ITb_val = ims.take( tset_val[ bis_val : bie_val ], axis = 0 )
        this_bsize = len( ISb_val )
        Yb_val = np.ones( ( this_bsize, 1 ), np.float32 )
        pb_val = pids_val[ bis_val : bie_val ]
        for i in range( this_bsize ):
            if np.random.uniform(  ) > 0.5:
                iid = tset_val[ np.random.choice( ( pids_val != pb_val[ i ] ).nonzero(  )[ 0 ], 1 ) ]
                ITb_val[ i, :, :, : ] = ims[ iid, :, :, : ]
                Yb_val[ i ] = 0.
        ISb_val = transform( ISb_val, npx_in )
        ITb_val = transform( ITb_val, npx_in )
        Ib_val = np.concatenate( ( ISb_val, ITb_val ), axis = 1 )
        out_val = _domain_discrim( Ib_val, Yb_val )
        cost_val_cumm += float( out_val[ 0 ] ) * this_bsize
        err_val_cumm += ( ( out_val[ 1 ] > 0.5 ) - Yb_val.astype( bool ) ).sum(  ) 
        if np.mod( bi_val, num_batch_val / 10 ) == 0:
            prog = np.round( bi_val * 100. / num_batch_val )
            print( 'Val) Epoch %02d: %03d%% (batch %06d / %06d)' 
                    % ( num_epoch, prog, bi_val + 1, num_batch_val ) )
    cost_val_cumm = cost_val_cumm / num_sample_val
    err_val_cumm = err_val_cumm / num_sample_val
    print( 'Val) Epoch %02d: cost_val = %.4f, err_val = %.4f' % ( num_epoch, cost_val_cumm, err_val_cumm ) )
    # Leave logs.
    log = [ num_epoch, num_update, num_example, time(  ) - t, cost_tr_cumm, err_tr_cumm, cost_val_cumm, err_val_cumm ]
    f_log.write( json.dumps( dict( zip( log_fields, log ) ) ) + '\n' )
    f_log.flush(  )
