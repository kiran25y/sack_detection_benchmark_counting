"""Sample a nonnegative density on the native point chart using a fixed window."""
import numpy as np


def support_image(density,mask,xy,width,height):
    h,w=density.shape
    # A square of half-width eight on the published point metric's chart.
    radius=8*max(width,height)/1024
    x,y=xy.T+.5
    extent_x=w/width
    extent_y=h/height
    left=np.clip(np.ceil((x-radius)*extent_x-.5).astype(int),0,w)
    right=np.clip(np.floor((x+radius)*extent_x-.5).astype(int)+1,0,w)
    top=np.clip(np.ceil((y-radius)*extent_y-.5).astype(int),0,h)
    bottom=np.clip(np.floor((y+radius)*extent_y-.5).astype(int)+1,0,h)
    def integral(a):
        table=np.pad(a,((1,0),(1,0))).cumsum(axis=0).cumsum(axis=1)
        return table[bottom,right]-table[top,right]-table[bottom,left]+table[top,left]
    total=integral(density)
    pixels=integral(mask.astype(float))
    return np.maximum(total,0)*int(mask.sum())/np.maximum(pixels,1)


def rotated(xy,roi):
    # Reflect centers through the rectangle center in the pixel-edge chart.
    a=np.asarray(roi,dtype=float)
    return a[:2]+a[2:]-1-xy
