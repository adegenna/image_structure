import numpy as np
from image_structure.src.Fourier import *
from image_structure.src.YagerFourier import *
import sys

class ImageStructure:

    def __init__(self,inputs):
        """
        Light wrapper class for image structure functions
        """
        self.dimensions = int(inputs.dimensions)
        self.inputs     = inputs
        assert ((self.dimensions == 2) | (self.dimensions == 3))
        input_data = self.read_input_data(inputs.data_file,inputs.data_file_type)
        try:
            fields = list(vars(inputs))
        except:
            fields = inputs._fields
        shape_data = len(np.shape(input_data))
        if ( ('nx' in fields) & (shape_data == 1) ):
            input_data  = self.reshape_input_data( input_data , inputs )
        elif ( ('nx' not in fields) & (shape_data == 1) ):
            sys.exit('1d data detected; please specify nx,ny or nx,ny,nz')
        self.set_input_data( input_data )
        self.set_structure_function(inputs.structure_function)
        self.outdir    = '/'.join( inputs.output_file.split('/')[0:-1] ) + '/'

    def reshape_input_data(self,data,inputs):
        # Reshape if given 1D raveled data
        try:
            if (self.dimensions == 2):
                input_data = np.reshape( data , [inputs.nx , inputs.ny] )
            elif (self.dimensions == 3):
                input_data = np.reshape( data , [inputs.nx , inputs.ny , inputs.nz] )
            return input_data
        except:
            sys.exit('1d data detected; please specify nx,ny or nx,ny,nz')
        
    def set_input_data(self,data):
        self.input_data = data
        
    def read_input_data(self,data_file,data_file_type):
        """
        Method to read input file data
        """
        if (data_file_type == 'npy'):
            input_data = np.load(data_file)
        elif (data_file_type == 'csv'):
            try:
                input_data = np.loadtxt(data_file,delimiter=',')
            except:
                input_data = np.genfromtxt( data_file )
        else:
            sys.exit('Unsupported input data type. Supported types are: npy and csv.')

        return input_data

    def set_structure_function(self,structure_function_type):
        """
        Method to set the relevant structure function
        """
        if (structure_function_type == 'fourier'):
            self.structure_function = fit_gaussian_to_average_fourier_spectrum

        elif (structure_function_type == 'fourier_yager'):
            try:
                self.structure_function = make_structure_vector_yager_2d( interpolation_abscissa = self.inputs.interpolation_abscissa ) 
            except:
                sys.exit('Must specify interpolation_abscissa for this structure function choice')

        elif (structure_function_type == 'fourier_yager_full'):
            try:
                self.structure_function = make_fft_circavg_yager_2d( interpolation_abscissa = self.inputs.interpolation_abscissa ) 
            except:
                sys.exit('Must specify interpolation_abscissa for this structure function choice')

        elif (structure_function_type == 'voronoi'):
            try:
                self.structure_function = make_voronoi( filter_tolerance = self.inputs.filter_tolerance ) 
            except:
                sys.exit('Must specify filter_tolerance for this structure function choice')

            self.structure_function = voronoi

        else:
            sys.exit('Unsupported structure function type. Supported types are: fourier , fourier_yager , fourier_yager_full.')

    def compute_structure(self,plot_metrics=False,outdir=None,str_figure=None,interpolation_abscissa=None):
        """
        Main method to compute data structure
        """
        structure = self.structure_function(input_data   = self.input_data , \
                                            plot_metrics = plot_metrics , \
                                            output_dir   = outdir , \
                                            output_name  = str_figure  )
        return structure
    
