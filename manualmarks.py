# a python implementation of the the Raith manual markers feature

import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import io, img_as_float
from skimage import transform as tf
import os

# sources:
# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# http://stackoverflow.com/questions/28758079

def image_to_np(filename):
    return img_as_float(io.imread(filename))

class SelectPoints(object):
    def __init__(self, im):
        self.im = im
        self.points = []

    def getCoord(self):
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        plt.imshow(self.im, cmap = plt.cm.gray, aspect = 'auto', interpolation = 'none')
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()

    def __onclick__(self,click):
        self.points.append((click.xdata,click.ydata))

# def quadrant(pts):
#     """ return list of quadrants each point is in """
# 
#     q = np.zeros(4, dtype='int')
#     
#     # is it quadrant 1?
#     q += np.all(pts > 0, axis=1)
#     # is it in quadrant 2?
#     q += 2*((pts[:,1] > 0) & (pts[:,0] < 0))
#     # is it quadrant 3?
#     q += 3*np.all(pts < 0, axis=1)
#     # is it in quadrant 4?
#     q += 4*((pts[:,1] < 0) & (pts[:,0] > 0))
#     
#     return q

def order_points(pnts):
    """ order the points by angle phi measured from x-axis 
    
        returns: (points in counterclockwise order, type of square) """
    shifted = pnts - np.mean(pnts, axis=0) # center points around origin
    
    phi = np.arctan2(shifted[:,1],shifted[:,0])
    sumphi = abs(sum(phi)) #can be used to determine different square types
    
    pnts = pnts[np.argsort(phi)] # sort original counter clockwise
    
    if abs(sumphi-np.pi) < sumphi:
        # square vertices on axis
        if (pnts[-1,1]>pnts[0,1]) & (pnts[-1,0]<pnts[0,0]):
            return np.roll(pnts, 1, axis=0)
        else:  
            return pnts
    else:
        # square edges parallel to axis
        return pnts


def find_manual_marks(im):
    """ use the SelectPoints class and some truth statements to
        find exactly four points, then prompt the user to see if they are
        correct 
        
        im: image that will be displayed """
    
    sp = SelectPoints(im)
    while True:
        sp.getCoord()
        if len(sp.points) == 4:
            fig = plt.figure(figsize=(12,9))
            ax = fig.add_subplot(111)
            ax.imshow(im, cmap = plt.cm.gray, aspect = 'auto', interpolation = 'none')
            pnts = np.column_stack(zip(*sp.points))
            ax.plot(pnts[:,0], pnts[:,1], 'ro', ms = 10)
            ymax, xmax = im.shape
            ax.set_xlim(0,xmax)
            ax.set_ylim(ymax,0)
            plt.show()
            m = raw_input('try again (y/n)? ')
            if m.lower().startswith('y'):
                sp.points = []
                continue
            else: 
                break
        else:
            sp.points = []
            n = raw_input('incorrect number of points. try again (y/n)? ')
            if n.lower().startswith('y'):
                continue
            else: 
                break
    pnts = np.column_stack(zip(*sp.points))
    return order_points(pnts)
    #return pnts
    
def warp_transform(im, pnts, src, val=0.0):
    """ warps image according to pnts and src 
        
        im: image to warp
        pnts: location of points on im
        scr: locations pnts will be mapped to
        val: fill value for outside image edges """
        
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, pnts)
    return tf.warp(im, tform3, mode='constant', cval=val, output_shape=im.shape)

def estimate_square(pnts, shape):
    """ estimate what square to which the manual marks will be fit.
        
        works for vertices on the axes or edges parallel to 
        the axes. 
        
        pnts: ordered manual markers
        shape: image shape """
        
    # lots of this is redundant with the order_points function
    # but it is fast enough and you can see what's happening 

    d = np.linalg.norm(pnts-np.roll(pnts,1, axis=0), axis=1)
    a = np.mean(d)/2.0
    y0, x0 = (s/2.0 for s in shape)
    
    shifted = pnts - np.mean(pnts, axis=0) # center points around origin
    phi = np.arctan2(shifted[:,1],shifted[:,0])
    sumphi = abs(sum(phi)) #can be used to determine different square type
    
    if abs(sumphi-np.pi) < sumphi:
        # square vertices on axis
        return np.array(((x0-a*np.sqrt(2),y0),
                         (x0,y0-a*np.sqrt(2)),
                         (x0+a*np.sqrt(2),y0),
                         (x0,y0+a*np.sqrt(2))))
    else:
        # square edges parallel to axis
        return np.array(((x0-a,y0-a),
                         (x0+a,y0-a),
                         (x0+a,y0+a),
                         (x0-a,y0+a)))

def nanotube_markers(filename):
    """ these are currently my favorite settings for processing
        nanotube images using this set of functions. """

    # some stuff to handle files or filelists
    if type(filename)==type(''):
        filename = [filename]
    elif type(filename)==type([]):
        pass
    else:
        print "Enter an string or list of strings"

    # use the first entry to get the directory name
    path = os.path.dirname(filename[0])

    for f in filename:
        print 'working on file: {0}'.format(f)
        im = image_to_np(f)
        pnts = find_manual_marks(im)
        src = estimate_square(pnts,im.shape)
        warped = warp_transform(im,pnts,src,val=1.0)
        fnew = os.path.join(path,os.path.basename(f)[:-4]+'_warp.png')
        scipy.misc.toimage(warped).save(fnew)
        # scipy.misc.toimage(warped).save(f)