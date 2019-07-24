import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from skimage import measure
from skimage import filters

def voronoi( input_data , plot_metrics=False , output_dir='./' , output_name='' , filter_tolerance=1 ):

    assert( input_data.ndim == 2 )
    
    xy0               = compute_voronoi_centers( input_data , filter_tolerance )
    vor               = Voronoi( xy0 )
    vertices_internal = compute_nvertices_for_each_center( xy0 , vor , [0,input_data.shape[0]] , [0,input_data.shape[1]] )

    return vertices_internal , vor
    
def compute_voronoi_centers( input_data , filter_tolerance=1 ):

    # Filter all data less/greater than chosen tolerance
    if ( filter_tolerance < 0 ):
        xy_filter = np.where( input_data < filter_tolerance )
    else:
        xy_filter = np.where( input_data > filter_tolerance )

    x     = np.arange( input_data.shape[0] )
    y     = np.arange( input_data.shape[1] )
    xx,yy = np.meshgrid(x,y)
    xx    = xx.T; yy = yy.T

    # Label connected components
    l            = np.minimum( len(x) , len(y) )
    n            = l / 10
    im           = filters.gaussian(input_data , sigma= l / (4. * n))
    blobs        = im < 0.7 * im.mean()
    all_labels   = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
    blobs_labels_unique = np.unique(blobs_labels.ravel())
    
    # Compute connected component centers
    # First one is generally the whole domain (ie background), so ignore it
    xy_centers = np.zeros([len(blobs_labels_unique) - 1 , 2])
    for i in range(1,len(blobs_labels_unique)):
        x_i = xx.ravel()[np.where(blobs_labels.ravel() == blobs_labels_unique[i])]
        y_i = yy.ravel()[np.where(blobs_labels.ravel() == blobs_labels_unique[i])]
        xy_centers[i-1,0] = np.mean(x_i)
        xy_centers[i-1,1] = np.mean(y_i)
    
    return xy_centers

def compute_nvertices_for_each_center( xy0 , vor , xminmax , yminmax):

    vertices_internal = []
    
    for i in range(len( vor.regions )):
        vi = np.array( [ vor.vertices[j] for j in vor.regions[i] ] )
        try:
            if ( np.any(vi[:,0] > xminmax[1]) | np.any(vi[:,0] < xminmax[0]) | np.any(vi[:,1] > yminmax[1]) | np.any(vi[:,1] < yminmax[0]) ):
                pass
            else:
                vertices_internal.append( vi )
        except:
            pass
    
    return vertices_internal


    
# This is just a Python closure to wrap the filter_tolerance argument
def make_voronoi( filter_tolerance ):
    def voronoi_wrapper( input_data, plot_metrics=True, output_dir='./', output_name='result',
                         range_rel=0.75, scale=[1,1], adjust=1.0, output_condition='' , filter_tolerance = filter_tolerance ):
        return voronoi( input_data , plot_metrics , output_dir , output_name ,
                        filter_tolerance )    
    return voronoi_wrapper
