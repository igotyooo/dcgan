import sys
import os
import re
import scipy.misc
import numpy as np
sys.path.append('..')

# Set paths.
data_dir_lookbook = '/nickel/datain/lookbook'
data_dir_tshirts40k = '/nickel/datain/tshirts40k'
# Data provider for pixel-level domain transfer
class Pldt:
    def __init__( self ):
        self.name = [  ]
        self.impaths = [  ]
        self.d1set_tr = [  ]
        self.d2set_tr = [  ]
        self.cls1_tr = [  ]
        self.cls2_tr = [  ]
        self.d1set_val = [  ]
        self.d2set_val = [  ]
        self.cls1_val = [  ]
        self.cls2_val = [  ]
        self.cls1_names = [  ]
        self.cls2_names = [  ]
    def set_LookBook( self ):
        # Set param.
        src_dir = data_dir_lookbook
        val_rate = 0.1
        # Make basic info.
        print( 'Set LookBook.' )
        iid2impath = [ os.path.join( src_dir, f ) for f in os.listdir( src_dir ) ]
        iid2impath.sort(  )
        iid2impath = np.array( iid2impath )
        np.random.seed( 0 )
        iid2cname = [  ]
        iid2clean = [  ]
        iid2pid = [  ]
        iid2colname = [  ]
        for iid in range( len( iid2impath ) ):
            iid2cname.append( re.findall( 'CLS_([\w+]+)___PID', iid2impath[ iid ] )[ 0 ] )
            iid2pid.append( int( re.findall( 'PID_([\d+]+)___COLOR', iid2impath[ iid ] )[ 0 ] ) )
            iid2colname.append( re.findall( 'COLOR_([\w+]+)___CLEAN', iid2impath[ iid ] )[ 0 ] )
            iid2clean.append( bool( int( re.findall( 'CLEAN_([\d+]+)___IID', iid2impath[ iid ] )[ 0 ] ) ) )
        cid2name, iid2cid = np.unique( iid2cname, return_inverse = True )
        colid2name, iid2colid = np.unique( iid2colname, return_inverse = True )
        iid2pid = np.array( iid2pid )
        iid2clean = np.array( iid2clean )
        # Pairing domain-1 and domain-2, and spliting train and val.
        d1set_tr = np.array( [  ], np.int )
        d2set_tr = np.array( [  ], np.int )
        d2set_val = np.array( [  ], np.int )
        d1set_val = np.array( [  ], np.int )
        pids = np.unique( iid2pid )
        for pid in pids:
            prodiid = np.logical_and( iid2pid == pid, iid2clean == True ).nonzero(  )[ 0 ]
            natiids = np.logical_and( iid2pid == pid, iid2clean == False ).nonzero(  )[ 0 ]
            if np.random.rand( 1 ) > val_rate:
                d1set_tr = np.hstack( ( d1set_tr, natiids ) )
                d2set_tr = np.hstack( ( d2set_tr, np.tile( prodiid, natiids.size ) ) )
            else:
                d1set_val = np.hstack( ( d1set_val, natiids ) )
                d2set_val = np.hstack( ( d2set_val, np.tile( prodiid, natiids.size ) ) )
        cls1_tr = iid2cid.take( d1set_tr )
        cls2_tr = iid2colid.take( d1set_tr )
        cls1_val = iid2cid.take( d1set_val )
        cls2_val = iid2colid.take( d1set_val )
        self.name = 'LookBook'
        self.impaths = iid2impath
        self.d1set_tr = d1set_tr
        self.d2set_tr = d2set_tr
        self.cls1_tr = cls1_tr
        self.cls2_tr = cls2_tr
        self.d1set_val = d1set_val
        self.d2set_val = d2set_val
        self.cls1_val = cls1_val
        self.cls2_val = cls2_val
        self.cls1_names = cid2name
        self.cls2_names = colid2name
        print( 'Done.' )
    def shuffle( self ):
        np.random.seed( 0 )
        rp = np.random.permutation( len( self.d1set_tr ) )
        self.d1set_tr = self.d1set_tr.take( rp )
        self.d2set_tr = self.d2set_tr.take( rp )
        self.cls1_tr = self.cls1_tr.take( rp )
        self.cls2_tr = self.cls2_tr.take( rp )
    def load( self, im_side, keep_aspect ):
        return load_ims( self.impaths, self.name, im_side, keep_aspect )

# Data provider for GAN.
class Gan:
    def __init__( self ):
        self.name = [  ]
        self.impaths = [  ]
    def set_LookBook( self ):
        src_dir = data_dir_lookbook
        print( 'Set LookBook.' )
        iid2impath = [ os.path.join( src_dir, f ) for f in os.listdir( src_dir ) ]
        iid2impath.sort(  )
        iid2impath = np.array( iid2impath )
        self.name = 'LookBook'
        self.impaths = iid2impath
        print( 'Done.' )
    def set_Tshirts40k( self ):
        src_dir = data_dir_tshirts40k
        print( 'Set Tshirts40k.' )
        iid2impath = [ os.path.join( src_dir, f ) for f in os.listdir( src_dir ) ]
        iid2impath.sort(  )
        iid2impath = np.array( iid2impath )
        self.name = 'Tshirts40k'
        self.impaths = iid2impath
        print( 'Done.' )
    def shuffle( self ):
        np.random.seed( 0 )
        rp = np.random.permutation( len( self.impaths ) )
        self.impaths = self.impaths.take( rp )
    def load( self, im_side, keep_aspect ):
        return load_ims( self.impaths, self.name, im_side, keep_aspect )

# Static function.
def load_ims( impaths, name, im_side, keep_aspect ):
    dname = os.path.join( './dataout/', name.upper(  ) )
    if not os.path.exists( dname ):
        os.makedirs( dname )
    fname = 'DI_%s_IS%d_KA%d.npy' % ( name.upper(  ), im_side, keep_aspect )
    fpath = os.path.join( dname, fname )
    try:
        print( 'Try to load datain: %s' % fname )
        ims = np.load( fpath )
    except:
        num_im = len( impaths )
        ims = np.zeros( ( num_im, im_side, im_side, 3 ), np.uint8 )
        print( 'Make datain. (%d, %d, %d, %d)' % ims.shape )
        for i in range( num_im ):
            if np.mod( i, num_im / 10 ) == 0:
                print( '%d%% (im %06d / %06d)' % ( np.round( i * 100. / num_im ), i, num_im ) )
            im = scipy.misc.imread( impaths[ i ] )
            if im.ndim < 3:
                im = np.asarray( np.dstack( ( im, im, im ) ), dtype = np.uint8 )
            if im.shape[ 2 ] > 3:
                im = im[ :, :, 0 : 3 ]
            if keep_aspect == True:
                nr, nc, _ = im.shape
                if nr > nc:
                    nc = int( np.floor( float( nc ) * float( im_side ) / float( nr ) ) )
                    nr = im_side
                    im_ = scipy.misc.imresize( im, [ nr, nc, 3 ] )
                    mleft = int( np.round( float( nr - nc ) / 2.0 ) )
                    mright = im_side - nc - mleft
                    mleft = 255 * np.ones( ( im_side, mleft, 3 ), np.uint8 )
                    mright = 255 * np.ones( ( im_side, mright, 3 ), np.uint8 )
                    im_ = np.concatenate( ( mleft, im_, mright ), axis = 1 )
                elif nr < nc:
                    nr = int( np.floor( float( nr ) * float( im_side ) / float( nc ) ) )
                    nc = im_side
                    im_ = scipy.misc.imresize( im, [ nr, nc, 3 ] )
                    mtop = int( np.round( float( nc - nr ) / 2.0 ) )
                    mbttm = im_side - nr - mtop
                    mtop = 255 * np.ones( ( mtop, im_side, 3 ), np.uint8 )
                    mbttm = 255 * np.ones( ( mbttm, im_side, 3 ), np.uint8 )
                    im_ = np.concatenate( ( mtop, im_, mbttm ), axis = 0 )
            else:
                im_ = scipy.misc.imresize( im, [ im_side, im_side, 3 ] )
            ims[ i, :, :, : ] = im_
        print( 'Save datain: %s' % fpath )
        np.save( fpath, ims )
    print( 'Done.' )
    return ims
