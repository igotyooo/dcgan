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
nz = 100 	    # # dim of central activation of converter.
nc = 3              # # of channels in image.
batch_size = 128    # # of examples in batch.
npx_in = 64         # # of pixels width/height of input images.
nf = 128            # Primary # of filters.
nvis = 14           # # of samples to visualize during training.
niter_lr0 = 50      # # of iter at starting learning rate.
niter = 100          # # of total iteration.
lr_decay = 10       # # of iter to linearly decay learning rate to zero.
lr = 0.0002         # Initial learning rate for adam.
shuffle = True      # Suffling training sample sequance.
mpath_decoder = './dataout/PRODUCTS/PRETRAIN_DCGAN/MODELS/C034.npy'
mpath_domain_discrim = './dataout/LOOKBOOK/TRAIN_DOMAIN_DISCRIM/MODELS/D020.npy'

# LOAD CONVERTER-DECODER.
data_c_d = np.load( mpath_decoder )
cw5d = sharedX( data_c_d[ 0 ] )
cg5d = sharedX( data_c_d[ 1 ] )
cb5d = sharedX( data_c_d[ 2 ] )
cw4d = sharedX( data_c_d[ 3 ] )
cg4d = sharedX( data_c_d[ 4 ] )
cb4d = sharedX( data_c_d[ 5 ] )
cw3d = sharedX( data_c_d[ 6 ] )
cg3d = sharedX( data_c_d[ 7 ] )
cb3d = sharedX( data_c_d[ 8 ] )
cw2d = sharedX( data_c_d[ 9 ] )
cg2d = sharedX( data_c_d[ 10 ] )
cb2d = sharedX( data_c_d[ 11 ] )
cw1d = sharedX( data_c_d[ 12 ] )

# LOAD DOMAIN-DISCRIMINATOR.
data_d = np.load( mpath_domain_discrim )
dw1 = sharedX( data_d[ 0 ] )
dw2 = sharedX( data_d[ 1 ] )
dg2 = sharedX( data_d[ 2 ] )
db2 = sharedX( data_d[ 3 ] )
dw3 = sharedX( data_d[ 4 ] )
dg3 = sharedX( data_d[ 5 ] )
db3 = sharedX( data_d[ 6 ] )
dw4 = sharedX( data_d[ 7 ] )
dg4 = sharedX( data_d[ 8 ] )
db4 = sharedX( data_d[ 9 ] )
dw5 = sharedX( data_d[ 10 ] )
dg5 = sharedX( data_d[ 11 ] )
db5 = sharedX( data_d[ 12 ] )
dwy = sharedX( data_d[ 13 ] )

# INITIALIZE CONVERTER-ENCODER.
relu = activations.Rectify(  )
sigmoid = activations.Sigmoid(  )
lrelu = activations.LeakyRectify(  )
tanh = activations.Tanh(  )
bce = T.nnet.binary_crossentropy
filt_ifn = inits.Normal( scale = 0.02 )
gain_ifn = inits.Normal( loc = 1., scale = 0.02 )
bias_ifn = inits.Constant( c = 0. )
cw1e = filt_ifn( ( nf, nc, 5, 5 ), 'cw1e' )
cw2e = filt_ifn( ( nf * 2, nf, 5, 5 ), 'cw2e' )
cg2e = gain_ifn( ( nf * 2 ), 'cg2e' )
cb2e = bias_ifn( ( nf * 2 ), 'cb2e' )
cw3e = filt_ifn( ( nf * 4, nf * 2, 5, 5 ), 'cw3e' )
cg3e = gain_ifn( ( nf * 4 ), 'cg3e' )
cb3e = bias_ifn( ( nf * 4 ), 'cb3e' )
cw4e = filt_ifn( ( nf * 8, nf * 4, 5, 5 ), 'cw4e' )
cg4e = gain_ifn( ( nf * 8 ), 'cg4e' )
cb4e = bias_ifn( ( nf * 8 ), 'cb4e' )
cw5e = filt_ifn( ( nf * 16, nf * 8, 4, 4 ), 'cw5e' )
cg5e = gain_ifn( ( nf * 16 ), 'cg5e' )
cb5e = bias_ifn( ( nf * 16 ), 'cb5e' )
cw6e = filt_ifn( ( nz, nf * 16, 1, 1 ), 'cw6e' )


encoder_params = [ cw1e, cw2e, cg2e, cb2e, cw3e, cg3e, cb3e, cw4e, cg4e, cb4e, cw5e, cg5e, cb5e, cw6e ]
def converter( IS, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6 ):
    ch1e = lrelu( dnn_conv( IS, w1, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ) )
    ch2e = lrelu( batchnorm( dnn_conv( ch1e, w2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g2, b = b2 ) )
    ch3e = lrelu( batchnorm( dnn_conv( ch2e, w3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g3, b = b3 ) )
    ch4e = lrelu( batchnorm( dnn_conv( ch3e, w4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = g4, b = b4 ) )
    ch5e = lrelu( batchnorm( dnn_conv( ch4e, w5, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = g5, b = b5 ) )
    ch6e = tanh( dnn_conv( ch5e, w6, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ) )
    ch5d = relu( batchnorm( dnn_conv( ch6e, cw5d, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = cg5d, b = cb5d ) )
    ch5d = ch5d.reshape( ( ch5d.shape[ 0 ], nf * 8, 4, 4 ) )
    ch4d = relu( batchnorm( deconv( ch5d, cw4d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = cg4d, b = cb4d ) )
    ch3d = relu( batchnorm( deconv( ch4d, cw3d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = cg3d, b = cb3d ) )
    ch2d = relu( batchnorm( deconv( ch3d, cw2d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = cg2d, b = cb2d ) )
    ch1d = tanh( deconv( ch2d, cw1d, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ) )
    return ch1d
def domain_discrim( X ):
    dh1 = lrelu( dnn_conv( X, dw1, subsample=( 2, 2 ), border_mode = ( 2, 2 ) ) )
    dh2 = lrelu( batchnorm( dnn_conv( dh1, dw2, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = dg2, b = db2 ) )
    dh3 = lrelu( batchnorm( dnn_conv( dh2, dw3, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = dg3, b = db3 ) )
    dh4 = lrelu( batchnorm( dnn_conv( dh3, dw4, subsample = ( 2, 2 ), border_mode = ( 2, 2 ) ), g = dg4, b = db4 ) )
    dh5 = lrelu( batchnorm( dnn_conv( dh4, dw5, subsample = ( 1, 1 ), border_mode = ( 0, 0 ) ), g = dg5, b = db5 ) )
    dy = sigmoid( T.dot( T.flatten( dh5, 2 ), dwy ) )
    return dy

# DEFINE TRAINING FUNCTION: FORWARD(COSTS) AND BACKWARD(UPDATE).
IS = T.tensor4(  )
IT_hat = converter( IS, *encoder_params )
X = T.concatenate( [ IS, IT_hat ], axis = 1 )
Y_hat = domain_discrim( X )
lrt = sharedX( lr )
cost = bce( Y_hat, T.ones( Y_hat.shape ) ).mean(  )
out = [ cost, Y_hat ]
e_updater = updates.Adam( lr = lrt, b1 = b1, regularizer = updates.Regularizer( l2 = l2 ) )
e_updates = e_updater( encoder_params, cost )

# COMPILING TRAIN/TEST FUNCTIONS.
print 'COMPILING'
t = time(  )
_converter = theano.function( [ IS ], IT_hat )
_val_e = theano.function( [ IS ], out )
_train_e = theano.function( [ IS ], out, updates = e_updates )
print '%.2f seconds to compile theano functions.' % ( time(  ) - t )

# PREPARE FOR DATAIN AND DEFINE SOURCE/TARGET.
di = Pldt(  )
di.set_LOOKBOOK(  )
ims = di.load( npx_in, True )
if shuffle:
    di.shuffle(  )
sset_tr = di.d1set_tr
tset_tr = di.d2set_tr
sset_val = di.d1set_val
tset_val = di.d2set_val

# PREPARE FOR DATAOUT.
dataout = os.path.join( './dataout/', di.name.upper(  ) )
desc = 'train_encoder'.upper(  )
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
log_fields = [ 'num_epoch', 'num_update', 'num_example', 't_spent', 'cost_tr', 'err_tr', 'cost_val', 'err_val' ]

# PLOT SOURCE/TARGET SAMPLE IMAGES.
vis_tr = np.random.permutation( len( sset_tr ) )
vis_tr = vis_tr[ 0 : nvis ** 2 ]
vis_ims_tr_s = ims.take( sset_tr.take( vis_tr ), axis = 0 )
vis_ims_tr_t = ims.take( tset_tr.take( vis_tr ), axis = 0 )
color_grid_vis( vis_ims_tr_s, ( nvis, nvis ),
            os.path.join( sample_dir, 'TR000S.png' ) )
color_grid_vis( vis_ims_tr_t, ( nvis, nvis ),
            os.path.join( sample_dir, 'TR000T.png' ) )
vis_ims_tr_s = transform( ims.take( sset_tr.take( vis_tr ), axis = 0 ), npx_in )
vis_val = np.random.permutation( len( sset_val ) )
vis_val = vis_val[ 0 : nvis ** 2 ]
vis_ims_val_s = ims.take( sset_val.take( vis_val ), axis = 0 )
vis_ims_val_t = ims.take( tset_val.take( vis_val ), axis = 0 )
color_grid_vis( vis_ims_val_s, ( nvis, nvis ),
            os.path.join( sample_dir, 'VAL000S.png' ) )
color_grid_vis( vis_ims_val_t, ( nvis, nvis ),
            os.path.join( sample_dir, 'VAL000T.png' ) )
vis_ims_val_s = transform( ims.take( sset_val.take( vis_val ), axis = 0 ), npx_in )

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
    mpath_e = os.path.join( model_dir, 'CE%03d.npy' % num_epoch )
    if os.path.exists( mpath_e ):
        print( 'Epoch %02d: Load.' % num_epoch )
        data_d = np.load( mpath_e )
        for pi in range( len( encoder_params ) ):
            encoder_params[ pi ].set_value( data_d[ pi ] )
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
        this_bsize = len( ISb_tr )
        for i in range( this_bsize ):
            if np.random.uniform(  ) > 0.5:
                ISb_tr[ i, :, :, : ] = np.fliplr( ISb_tr[ i, :, :, : ] )
        ISb_tr = transform( ISb_tr, npx_in )
        out_tr = _train_e( ISb_tr )
        num_update += 1
        num_example += this_bsize
        cost_tr = float( out_tr[ 0 ] )
        cost_tr_cumm += cost_tr * this_bsize
        err_tr = ( out_tr[ 1 ] < 0.5 ).mean(  )
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
    np.save( mpath_e, [ p.get_value(  ) for p in encoder_params ] )
    # Validation.
    num_sample_val = len( sset_val )
    num_batch_val = int( np.ceil( num_sample_val / float( batch_size ) ) )
    cost_val_cumm = 0.
    err_val_cumm = 0.
    for bi_val in range( num_batch_val ):
        bis_val = bi_val * batch_size
        bie_val = min( bi_val * batch_size + batch_size, num_sample_val )
        ISb_val = ims.take( sset_val[ bis_val : bie_val ], axis = 0 )
        this_bsize = len( ISb_val )
        ISb_val = transform( ISb_val, npx_in )
        out_val = _val_e( ISb_val )
        cost_val_cumm += float( out_val[ 0 ] ) * this_bsize
        err_val_cumm += ( out_val[ 1 ] < 0.5 ).sum(  ) 
        if np.mod( bi_val, num_batch_val / 10 ) == 0:
            prog = np.round( bi_val * 100. / num_batch_val )
            print( 'Val) Epoch %02d: %03d%% (batch %06d / %06d)' 
                    % ( num_epoch, prog, bi_val + 1, num_batch_val ) )
    cost_val_cumm = cost_val_cumm / num_sample_val
    err_val_cumm = err_val_cumm / num_sample_val
    print( 'Val) Epoch %02d: cost_val = %.4f, err_val = %.4f' % ( num_epoch, cost_val_cumm, err_val_cumm ) )
    # Sample visualization.
    samples = np.asarray( _converter( vis_ims_tr_s ) )
    color_grid_vis( inverse_transform( samples, npx_in ), ( nvis, nvis ),
            os.path.join( sample_dir, 'TR%03dT.png' % num_epoch ) )
    samples = np.asarray( _converter( vis_ims_val_s ) )
    color_grid_vis( inverse_transform( samples, npx_in ), ( nvis, nvis ),
            os.path.join( sample_dir, 'VAL%03dT.png' % num_epoch ) )
    # Leave logs.
    log = [ num_epoch, num_update, num_example, time(  ) - t, cost_tr_cumm, err_tr_cumm, cost_val_cumm, err_val_cumm ]
    f_log.write( json.dumps( dict( zip( log_fields, log ) ) ) + '\n' )
    f_log.flush(  )
