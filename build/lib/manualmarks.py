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

def order_points(pts):
	# 0 - top right
	# 1 - top left
	# 2 - bottom left
	# 3 - bottom right
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the bottom left point will have the smallest sum, whereas
	# the top right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[2] = pts[np.argmin(s)]
	rect[0] = pts[np.argmax(s)]
 
	# bottom right point will have the largest difference,
	# whereas the rop left will have the smallest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

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
            m = raw_input('try again (y)? ')
            if m.lower().startswith('y'):
                sp.points = []
                continue
            else: 
                break
        else:
            sp.points = []
            n = raw_input('incorrect number of points. try again (y)? ')
            if n.lower().startswith('y'):
                continue
            else: 
                break
    pts = np.column_stack(zip(*sp.points))
    return order_points(pts)
    
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
    d = np.linalg.norm(pnts-np.roll(pnts,1, axis=0), axis=1)
    a = np.mean(d)/2.0
    y0, x0 = (s/2.0 for s in shape)
    return np.array(((x0+a,y0+a),
                     (x0+a,y0-a),
                     (x0-a,y0-a),
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

    # use the first entry to create a new directory
    path = os.path.dirname(filename[0])

    for f in filename:
        im = image_to_np(f)
        pnts = find_manual_marks(im)
        src = estimate_square(pnts,im.shape)
        warped = warp_transform(im,pnts,src,val=1.0)
        fnew = os.path.join(path,os.path.basename(f)[:-4]+'_warp.png')
        scipy.misc.toimage(warped).save(fnew)