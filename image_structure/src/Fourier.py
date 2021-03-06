import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as opt

def fit_gaussian_to_average_fourier_spectrum(input_data,plot_metrics=False,output_dir=None,output_name=None,interpolation_abscissa=None):
    """
    Function to: 
    (1) compute the n-dimensional fourier transform of input_data
    (2) compute the 1-d average of this
    (3) fit gaussian to the peak
    return this gaussian fit
    """
    assert( (input_data.ndim == 2) | (input_data.ndim == 3) )
    data_hat    = np.fft.fftn(input_data)
    if (input_data.ndim == 2):
        nx,ny        = input_data.shape
        data_hat    *= (1./nx/ny)
        xhat         = np.fft.fftfreq(nx)*nx
        yhat         = np.fft.fftfreq(ny)*ny
        xxhat,yyhat  = np.meshgrid(xhat,yhat)
        freq_avg,data_hat_avg = compute_radial_average_2d(xxhat,yyhat,np.abs(data_hat))
    elif (input_data.ndim == 3):
        nx,ny,nz  = input_data.shape
        data_hat *= (1./nx/ny/nz)
        xhat      = np.fft.fftfreq(nx)*nx
        yhat      = np.fft.fftfreq(ny)*ny
        zhat      = np.fft.fftfreq(nz)*nz
        xxhat,yyhat,zzhat     = np.meshgrid(xhat,yhat,zhat)
        freq_avg,data_hat_avg = compute_radial_average_3d(xxhat,yyhat,zzhat,np.abs(data_hat))
    gauss_mean, gauss_sigma = fit_gaussian(freq_avg,data_hat_avg,plot_metrics,output_dir,output_name)
    return gauss_mean,gauss_sigma

def compute_yavg_over_unique_xvals(x,y):
    x_unique   = np.unique( x )
    y_avg      = np.zeros_like( x_unique )
    idx_start  = 0
    for i in range(len(x_unique)):
        count = 0
        while(True):
            count += 1
            if ((idx_start+count) == len(x)):
                idx_end = len(x)
                break
            else:
                if not np.isclose(x[idx_start+count],x_unique[i]):
                    idx_end = idx_start+count
                    break
        y_avg[i]    = np.mean( y[idx_start:idx_end] )
        idx_start   = idx_end
    return x_unique,y_avg

def compute_radial_average_2d(xx,yy,signal):
    assert( (xx.shape == yy.shape) & (xx.shape == signal.shape) )
    nx,ny               = xx.shape
    frequency_magnitude = np.sqrt( xx**2 + yy**2 ).ravel()
    idx_freq            = np.argsort( frequency_magnitude )
    frequency_magnitude = frequency_magnitude[idx_freq]
    signal              = (signal.ravel())[idx_freq]
    f_unique,signal_avg = compute_yavg_over_unique_xvals( frequency_magnitude, signal)    
    return f_unique , signal_avg

def compute_radial_average_3d(xx,yy,zz,signal):
    assert( (xx.shape == yy.shape) & (xx.shape == zz.shape) & (xx.shape == signal.shape) )
    nx,ny,nz            = xx.shape
    frequency_magnitude = np.sqrt( xx**2 + yy**2 + zz**2 ).ravel().astype(int)
    signal              = signal.ravel()
    signal_avg          = np.zeros((frequency_magnitude.max()-frequency_magnitude.min())+1)
    for i in range(len(frequency_magnitude)):
        signal_avg[frequency_magnitude[i]] += signal[i]
    return np.unique(frequency_magnitude) , signal_avg

def return_peak_centered_spectrum(x,y,idx_peak):
    # Reflect (x,y) about idx_peak
    x_reflect  = np.hstack( [-np.flipud(x[idx_peak+1:]) + 2*x[idx_peak] , x[idx_peak:]] )
    y_reflect  = np.hstack( [ np.flipud(y[idx_peak+1:])                 , y[idx_peak:]] )
    return x_reflect,y_reflect

def restrict_x_data(x,y,decay):
    # Chop-off everything that has decayed past decay*max(y)
    return x[y >= decay*np.max(y)] , y[y >= decay*np.max(y)]
    
def fit_gaussian(x,y,plot_fit=False,outdir=None,str_figure=None):
    d            = 0.2       # Decay parameter for gaussian fitting    
    x = x[1:]; y = y[1:];    # Throw away the zero-th order fourier wavenumber
        
    # Force mean of gaussian to be at peak and reflect x-axis about the peak for fitting
    idx_peak            = np.argmax( y )
    x_reflect,y_reflect = return_peak_centered_spectrum(x, y, idx_peak)
    x_fitting,y_fitting = restrict_x_data(x_reflect,y_reflect,d)
    guess               = [np.max(y_fitting), x[idx_peak] , (np.max(x_fitting)-np.min(x_fitting))/5.]
    try:
        params,uncert       = opt.curve_fit(gauss1d,x_fitting,y_fitting,p0=guess)
    except:
        x_fitting,y_fitting = restrict_x_data(x_reflect,y_reflect,d*0.5)
        guess               = [np.max(y_fitting), x[idx_peak] , (np.max(x_fitting)-np.min(x_fitting))/5.]
        params,uncert       = opt.curve_fit(gauss1d,x_fitting,y_fitting,p0=guess)
    params[1] = np.maximum( params[1] , 0 ) # Limiter on mean
    params[2] = np.maximum( params[2] , 0 ) # Limiter on std-dev
    
    if plot_fit:
        plt.figure()
        fit   = np.max(y)*np.exp(-0.5*(x-params[1])**2/params[2]**2)
        x_fit = x[ fit > np.max(y)*np.exp(-0.5*(20**2)) ]
        y_fit = y[ fit > np.max(y)*np.exp(-0.5*(20**2)) ]
        plt.plot(x_fit,y_fit,'b',linewidth=3)
        plt.plot(x_fit, np.max(y)*np.exp(-0.5*(x_fit-params[1])**2/params[2]**2),'r',linewidth=3)
        plt.xlabel(r'$|k|$',fontsize=20)
        plt.legend(['Avg Fourier magnitude','Gaussian fit'])
        if (outdir is not None):
            outfile = outdir + str_figure + ".png"
            plt.savefig(outfile, bbox_inches='tight')
    return params[1] , params[2]

def gauss1d(x,amp,mu,sigma):
    return amp*np.exp(-0.5*(x-mu)**2/sigma**2)
