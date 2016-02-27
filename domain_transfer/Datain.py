import sys
import os
import re
import scipy.misc
import numpy as np
sys.path.append('..')
class Datain:
    def __init__( self ):
        self.name = [  ]
        self.impaths = [  ]
        self.sset_tr = [  ]
        self.tset_tr = [  ]
        self.cls1_tr = [  ]
        self.cls2_tr = [  ]
        self.sset_val = [  ]
        self.tset_val = [  ]
        self.cls1_val = [  ]
        self.cls2_val = [  ]
    def set_LookBook( self ):
        # Set param.
        src_dir = '/nickel/datain/lookbook'
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
        # Pairing source and target, and spliting train and val.
        sset_tr = np.array( [  ], np.int )
        tset_tr = np.array( [  ], np.int )
        tset_val = np.array( [  ], np.int )
        sset_val = np.array( [  ], np.int )
        pids = np.unique( iid2pid )
        for pid in pids:
            prodiid = np.logical_and( iid2pid == pid, iid2clean == True ).nonzero(  )[ 0 ]
            natiids = np.logical_and( iid2pid == pid, iid2clean == False ).nonzero(  )[ 0 ]
            if np.random.rand( 1 ) > val_rate:
                sset_tr = np.hstack( ( sset_tr, natiids ) )
                tset_tr = np.hstack( ( tset_tr, np.tile( prodiid, natiids.size ) ) )
            else:
                sset_val = np.hstack( ( sset_val, natiids ) )
                tset_val = np.hstack( ( tset_val, np.tile( prodiid, natiids.size ) ) )
        cls1_tr = iid2cid.take( sset_tr )
        cls2_tr = iid2colid.take( sset_tr )
        cls1_val = iid2cid.take( sset_val )
        cls2_val = iid2colid.take( sset_val )
        self.name = 'LookBook'
        self.impaths = iid2impath
        self.sset_tr = sset_tr
        self.tset_tr = tset_tr
        self.cls1_tr = cls1_tr
        self.cls2_tr = cls2_tr
        self.sset_val = sset_val
        self.tset_val = tset_val
        self.cls1_val = cls1_val
        self.cls2_val = cls2_val
        print( 'Done.' )
    def load( self, im_side, keep_aspect ):
        num_im = len( self.impaths )
        ims = np.zeros( ( num_im, im_side, im_side, 3 ), np.uint8 )
        print( 'Load data (%d, %d, %d, %d)' % ims.shape )
        for i in range( num_im ):
            if np.mod( i, num_im / 10 ) == 0:
                print( '%d%% (im %06d / %06d)' % ( np.round( i * 100. / num_im ), i, num_im ) )
            im = scipy.misc.imread( self.impaths[ i ] )
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
        return ims
