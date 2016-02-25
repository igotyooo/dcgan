import sys
sys.path.append('..')
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
def transform(X):
    assert X[0].shape == (npx, npx, 3) or X[0].shape == (3, npx, npx)
    if X[0].shape == (npx, npx, 3):
        X = X.transpose(0, 3, 1, 2)
    return floatX(X / 127.5 - 1.)
def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
    return X

# SET PARAMETERS.
k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nz = 100 	  # # dim of central activation of transformer.
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
ntf = 128         # # of transformer filters in first conv layer
niter = 25        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 182236   # # of examples to train on

# FILE I/O.
desc = 'pretrain_transformer'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
f_log = open('logs/%s.ndjson'%desc, 'wb')
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'cost',
]
tr_data, te_data, tr_stream, val_stream, te_stream = faces(ntrain=ntrain)
tr_handle = tr_data.open()

# DEFINE NETWORKS.
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
tifn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)
tw1  = tifn((ntf, nc, 5, 5), 'tw1')
tw2 = tifn((ntf*2, ntf, 5, 5), 'tw2')
tg2 = gain_ifn((ntf*2), 'tg2')
tb2 = bias_ifn((ntf*2), 'tb2')
tw3 = tifn((ntf*4, ntf*2, 5, 5), 'tw3')
tg3 = gain_ifn((ntf*4), 'tg3')
tb3 = bias_ifn((ntf*4), 'tb3')
tw4 = tifn((ntf*8, ntf*4, 5, 5), 'tw4')
tg4 = gain_ifn((ntf*8), 'tg4')
tb4 = bias_ifn((ntf*8), 'tb4')
tw5 = tifn((nz, ntf*8, 4, 4), 'tw5')
tg5 = gain_ifn(nz, 'tg5')
tb5 = bias_ifn(nz, 'tb5')
tw5d = tifn((ntf*8*4*4, nz, 1, 1), 'tw5d')
tg5d = gain_ifn(ntf*8*4*4, 'tg5d')
tb5d = bias_ifn(ntf*8*4*4, 'tb5d')
tw4d = tifn((ntf*8, ntf*4, 5, 5), 'tw4d')
tg4d = gain_ifn((ntf*4), 'tg4d')
tb4d = bias_ifn((ntf*4), 'tb4d')
tw3d = tifn((ntf*4, ntf*2, 5, 5), 'tw3d')
tg3d = gain_ifn((ntf*2), 'tg3d')
tb3d = bias_ifn((ntf*2), 'tb3d')
tw2d = tifn((ntf*2, ntf, 5, 5), 'tw2d')
tg2d = gain_ifn((ntf), 'tg2d')
tb2d = bias_ifn((ntf), 'tb2d')
tw1d = tifn((ntf, nc, 5, 5), 'tw1d')
transform_params = [tw1, tw2, tg2, tb2, tw3, tg3, tb3, tw4, tg4, tb4, tw5, tg5, tb5,
        tw5d, tg5d, tb5d, tw4d, tg4d, tb4d, tw3d, tg3d, tb3d, tw2d, tg2d, tb2d, tw1d]
def transformer(NX, w1, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5,
        w5d, g5d, b5d, w4d, g4d, b4d, w3d, g3d, b3d, w2d, g2d, b2d, w1d):
    h1 = lrelu(dnn_conv(NX, w1, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h1, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    h5 = lrelu(batchnorm(dnn_conv(h4, w5, subsample=(1, 1), border_mode=(0, 0)), g=g5, b=b5))
    h5d = relu(batchnorm(dnn_conv(h5, w5d, subsample=(1, 1), border_mode=(0, 0)), g=g5d, b=b5d))
    h5d = h5d.reshape((h5d.shape[0], ntf*8, 4, 4))
    h4d = relu(batchnorm(deconv(h5d, w4d, subsample=(2, 2), border_mode=(2, 2)), g=g4d, b=b4d))
    h3d = relu(batchnorm(deconv(h4d, w3d, subsample=(2, 2), border_mode=(2, 2)), g=g3d, b=b3d))
    h2d = relu(batchnorm(deconv(h3d, w2d, subsample=(2, 2), border_mode=(2, 2)), g=g2d, b=b2d))
    h1d = tanh(deconv(h2d, w1d, subsample=(2, 2), border_mode=(2, 2)))
    y = h1d
    return y

# DEFINE TRAINING FUNCTION: FORWARD(COSTS) AND BACKWARD(UPDATE).
NX = T.tensor4()
gtX = T.tensor4()
tX = transformer(NX, *transform_params)
cost = T.mean((gtX - tX) ** 2)
lrt = sharedX(lr)
t_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
t_updates = t_updater(transform_params, cost)
updates = t_updates

# COMPILING TRAIN/TEST FUNCTIONS.
print 'COMPILING'
t = time()
_train_t = theano.function([NX, gtX], cost, updates=t_updates)
_transform = theano.function([NX], tX)
print '%.2f seconds to compile theano functions'%(time()-t)

# DO THE JOB.
print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(niter):
    for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain/nbatch):
        imb = transform(imb)
        gtb = imb # Should be fixed.
        cost = _train_t(imb, gtb)
        n_updates += 1
        n_examples += len(imb)
    t_cost = float(cost)
    log = [n_epochs, n_updates, n_examples, time()-t, t_cost]
    print '%.0f %.4f'%(epoch, t_cost)
    f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
    f_log.flush()
    samples = np.asarray(_transform(imb))
    color_grid_vis(inverse_transform(samples), (14, 14), 'samples/%s/%d.png'%(desc, n_epochs))
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [1, 2, 3, 4, 5, 10, 15, 20, 25]:
        joblib.dump([p.get_value() for p in transform_params], 'models/%s/%d_transform_params.jl'%(desc, n_epochs))
