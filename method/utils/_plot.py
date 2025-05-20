import matplotlib.pyplot as plt
import numpy as np

# From https://github.com/williamgilpin/fnn

def lighter(clr, f=1/3):
    """
    An implementation of Mathematica's Lighter[] 
    function for RGB colors
    clr : 3-tuple or list, an RGB color
    f : float, the fraction by which to brighten
    """
    gaps = [f*(1 - val) for val in clr]
    new_clr = [val + gap for gap, val in zip(gaps, clr)]
    return new_clr

def darker(clr, f=1/3):
    """
    An implementation of Mathematica's Darker[] 
    function for RGB colors
    clr : 3-tuple or list, an RGB color
    f : float, the fraction by which to brighten
    """
    gaps = [f*val for val in clr]
    new_clr = [val - gap for gap, val in zip(gaps, clr)]
    return new_clr


def plot3dproj(x, y, z, *args, 
    ax=None,
    color=(0,0,0), 
    shadow_dist=1.0, 
    color_proj=None, 
    elev_azim=(39,-47), 
    show_labels=False, 
    aspect='auto',
    **kwargs):
    """
    Create a three dimensional plot, with projections onto the 2D coordinate
    planes
    
    Parameters
    ----------
    x, y, z : 1D arrays of coordinates to plot
    *args : arguments passed to the matplotlib plt.plot functions
    - color : length-3 tuple
        The RGB color (with each element in [0,1]) to use for the
        three dimensional line plot
    - color_proj : length-3 tuple
        The RGB color (with each element in [0,1]) to use for the
        two dimensional projection plots. Defaults to a lighter version of the 
        plotting color
    - shadow_dist : float
        The relative distance of axes to their shadow. If a single value, 
        then the same distance is used for all three axies. If a triple, then 
        different values are used for all axes
    - elev_azim : length-2 tuple
        The starting values of elevation and azimuth when viewing the figure
    - show_labels : bool
        Whether to show numerical labels on the axes
    - aspect_ratio : None or int
        The integer aspect ratio to impose on the axes. If not passed, the default
        aspect ratio is used
    """
    if not ax:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection= '3d')
    if not color_proj:
        color_proj = lighter(color, .6)

    if np.isscalar(shadow_dist) == 1:
        sdist_x = shadow_dist
        sdist_y = shadow_dist
        sdist_z = shadow_dist
    else:
        sdist_x, sdist_y, sdist_z = shadow_dist


    
    ax.plot(x, z, *args, zdir='y', zs=sdist_y*np.max(y), color=color_proj, **kwargs)
    ax.plot(y, z, *args, zdir='x', zs=sdist_x*np.min(x), color=color_proj, **kwargs)
    ax.plot(x, y, *args, zdir='z', zs=sdist_z*np.min(z), color=color_proj, **kwargs)
    ax.plot(x, y, z, *args, color=color, **kwargs)

    ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
    ax.set_aspect(aspect, adjustable='box') 
    
#     ratio = 1.0
#     xvals, yvals = ax.get_xlim(), ax.get_ylim()
#     xrange = xvals[1]-xvals[0]
#     yrange = yvals[1]-yvals[0]
#     ax.set_aspect(ratio*(xrange/yrange), adjustable='box')

    if not show_labels:
        ax.set_xticklabels([])                               
        ax.set_yticklabels([])                               
        ax.set_zticklabels([])
    #plt.show()

    return ax