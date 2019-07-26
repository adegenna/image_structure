import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi , Delaunay
from shapely.geometry.polygon import Polygon
from skimage import measure
from skimage import filters

def voronoi( input_data , plot_metrics=False , output_dir='./' , output_name='' , filter_tolerance=1 ):

    assert( input_data.ndim == 2 )
    
    xy0               = compute_voronoi_centers( input_data , filter_tolerance )
    vor               = Voronoi( xy0 )
    vertices_internal , centers_internal , vol_internal , skewness_internal = compute_nvertices_for_each_center(
        xy0 , vor , [0,input_data.shape[0]] , [0,input_data.shape[1]] )
    dist_centers_to_vertices = compute_distance_center_to_vertices( centers_internal , vertices_internal )

    return vertices_internal , centers_internal , vol_internal , skewness_internal , dist_centers_to_vertices , vor

def compute_cell_angles( vi ):
    
    p      = Polygon(vi)
    theta  = np.zeros(len(vi))
    coords = p.exterior.coords[0:-1]
    
    for i in range(len(vi)):
        v1        = np.array( coords[i] )
        v2        = np.array( coords[(i+1)%(len(coords))] )
        v3        = np.array( coords[(i+2)%(len(coords))] )
        a         = v1 - v2
        b         = v3 - v2
        theta[i]  = np.arccos( a.dot(b) / ( np.linalg.norm(a) * np.linalg.norm(b) ) )
    
    return theta

def compute_skewness( vi ):
    
    theta     = compute_cell_angles( vi )
    theta_0   = np.pi - 2*np.pi / len(vi)
    theta_max = np.max( theta )
    theta_min = np.min( theta )
    skewness  = np.maximum( (theta_max - theta_0) / (np.pi - theta_0) , (theta_0 - theta_min) / theta_0 )
    
    return skewness


def compute_distance_center_to_vertices( centers_internal , vertices_internal ):

    d_cv = np.zeros(len(centers_internal))
    for i in range(len(centers_internal)):
        d_cv[i] = np.sum( [np.linalg.norm(centers_internal[i] - vi) for vi in vertices_internal[i]] ) / len(vertices_internal[i])

    return d_cv
    
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
    centers_internal  = []
    vol_internal      = []
    skewness_internal = []
    
    for i in range(len( vor.regions )):
        vi  = np.array( [ vor.vertices[j] for j in vor.regions[i] ] )
        try:
            if ( (np.all(vi[:,0] <= xminmax[1]) & \
                  np.all(vi[:,0] >= xminmax[0]) & \
                  np.all(vi[:,1] <= yminmax[1]) & \
                  np.all(vi[:,1] >= yminmax[0]) ) & \
                 (np.all(np.array(vor.regions[i]) != -1)     )    ):
                vertices_internal.append( vi )
                centers_internal.append( vor.points[np.where(vor.point_region == i)[0][0]] )
                vol_internal.append( compute_volume(vi) )
                skewness_internal.append( compute_skewness( vi ) )
        except:
            pass
    
    return vertices_internal , centers_internal , vol_internal , skewness_internal


def compute_volume( vi ):
    
    tri          = Delaunay( vi )
    coord_groups = [tri.points[x] for x in tri.simplices]
    polygons     = [Polygon(x) for x in coord_groups]
    vol          = np.sum( [pi.area for pi in polygons] )
    
    return vol

# This is just a Python closure to wrap the filter_tolerance argument
def make_voronoi( filter_tolerance ):
    def voronoi_wrapper( input_data, plot_metrics=True, output_dir='./', output_name='result',
                         range_rel=0.75, scale=[1,1], adjust=1.0, output_condition='' , filter_tolerance = filter_tolerance ):
        return voronoi( input_data , plot_metrics , output_dir , output_name ,
                        filter_tolerance )    
    return voronoi_wrapper
