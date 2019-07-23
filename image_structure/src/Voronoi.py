import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def voronoi( input_data ,plot_metrics=False , output_dir='./' , output_name='' , interpolation_abscissa=None, filter_tolerance=1 ):

    xy0 = compute_voronoi_centers( input_data , filter_tolerance )
    vor = Voronoi( xy0 )
    n_vertices = compute_nvertices_for_each_center( xy0 , vor )

    return n_vertices

def compute_voronoi_centers( input_data , filter_tolerance=1 ):

    return xy_centers

def compute_nvertices_for_each_center( xy0 , vor ):

    return n_vertices


    
# This is just a Python closure to wrap the filter_tolerance argument
def make_voronoi( filter_tolerance ):
    def voronoi_wrapper( input_data, plot_metrics=True, output_dir='./', output_name='result',
                         range_rel=0.75, scale=[1,1], adjust=1.0, output_condition='' , filter_tolerance = filter_tolerance ):
        return voronoi( input_data, plot_metrics, output_dir, output_name,
                        range_rel, scale, adjust, output_condition, filter_tolerance )
    return voronoi_wrapper
