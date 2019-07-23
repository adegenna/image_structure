#!/usr/bin/python3
import numpy as np
import lmfit
import pickle
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
import pylab as plt
    
def load_result_file(infile, w, h):

    result = np.load(infile)
    size, num_steps = result.shape
    
    frames = []
    for istep in range(num_steps):
        data = np.reshape(result[:,istep], (h,w))
        frames.append(data)
    
    return np.asarray(frames)
    
        
def plot2D(data, outfile='output.png', scale=[1,1], size=10.0, plot_buffers=[0.1,0.05,0.1,0.05]):
    
    print('  data ({} points) from {:.2g} to {:.2g} ({:.2g} ± {:.2g})'.format(data.shape, np.min(data), np.max(data), np.average(data), np.std(data)))
    
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15

    
    fig = plt.figure(figsize=(size,size), facecolor='white')
    left_buf, right_buf, bottom_buf, top_buf = plot_buffers
    fig_width = 1.0-right_buf-left_buf
    fig_height = 1.0-top_buf-bottom_buf
    ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )
    
    h, w = data.shape
    extent = [ 0, w*scale[0], 0, h*scale[1] ]
    
    im = plt.imshow(data, extent=extent, cmap='bone')
    ax.set_xlabel('$x \, (\mathrm{pixels})$', size=20)
    ax.set_ylabel('$y \, (\mathrm{pixels})$', size=20)
    
    
    plt.savefig(outfile, dpi=200)
    plt.close()
    
    
    
def plotFFT(data, outfile='output.png', scale=[1,1], size=10.0, ztrim=[0.01, 0.01], plot_buffers=[0.1,0.05,0.1,0.05]):
    
    print('  data ({} points) from {:.2g} to {:.2g} ({:.2g} ± {:.2g})'.format(data.shape, np.min(data), np.max(data), np.average(data), np.std(data)))
    
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15

    
    fig = plt.figure(figsize=(size,size), facecolor='white')
    left_buf, right_buf, bottom_buf, top_buf = plot_buffers
    fig_width = 1.0-right_buf-left_buf
    fig_height = 1.0-top_buf-bottom_buf
    ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )
    
    # Compute vscale
    values = np.sort(data.flatten())
    zmin = values[ +int( len(values)*ztrim[0] ) ]
    zmax = values[ -int( len(values)*ztrim[1] ) ]
    
    h, w = data.shape
    origin = [int(w/2), int(h/2)]
    extent = [ -(w/2)*scale[0], +(w/2)*scale[0], -(h/2)*scale[1], +(h/2)*scale[1] ]
    
    im = plt.imshow(data, extent=extent, cmap='jet', vmin=zmin, vmax=zmax)
    ax.set_xlabel('$q_x \, (\mathrm{pixels}^{-1})$', size=20)
    ax.set_ylabel('$q_y \, (\mathrm{pixels}^{-1})$', size=20)
    
    
    plt.savefig(outfile, dpi=200)
    plt.close()
    
    
def plot1DFFT(x, y, outfile='output.png', x_expected=0, range_rel=0, fit_line=None, fit_line_e=None, fit_result=None, size=10.0, plot_buffers=[0.1,0.05,0.1,0.05]):
    
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15

    
    fig = plt.figure(figsize=(size,size*3/4), facecolor='white')
    left_buf, right_buf, bottom_buf, top_buf = plot_buffers
    fig_width = 1.0-right_buf-left_buf
    fig_height = 1.0-top_buf-bottom_buf
    ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )
    
    ax.plot(x, y, '-', color='k', linewidth=2.0)
    ax.set_xlabel('$q \, (\mathrm{pixels}^{-1})$', size=20)
    ax.set_ylabel('$I_{\mathrm{FFT}}(q) \, (\mathrm{a.u.})$', size=20)
    xi, xf, yi, yf = ax.axis()
    xi, xf, yi, yf = 0, np.max(x), 0, yf
    
    
    ax.axvline(x_expected, color='b', linewidth=2.0, alpha=0.5, dashes=[5,5])
    ax.axvspan(x_expected*(1-range_rel), x_expected*(1+range_rel), color='b', alpha=0.05)
    s = '$q_{{0,\mathrm{{expected}} }} = {:.2g} \, \mathrm{{pix}}^{{-1}}$'.format(x_expected)
    ax.text(x_expected, yf, s, size=20, color='b', verticalalignment='top', horizontalalignment='left')
    
    if fit_line is not None:
        ax.plot(fit_line[0], fit_line[1], '-', color='purple', linewidth=2.0)
    if fit_line_e is not None:
        ax.plot(fit_line_e[0], fit_line_e[1], '-', color='purple', linewidth=0.5)
    if fit_result is not None:
        x = fit_result.params['x_center'].value
        s = '$q_{{0,\mathrm{{fit}} }} = {:.2g} \, \mathrm{{pix}}^{{-1}}$'.format(x)
        ax.text(x_expected, yi, s, size=20, color='purple', verticalalignment='bottom', horizontalalignment='left')
        ax.axvline(x, color='purple', linewidth=1.0, alpha=0.5)


        els = [
            '$p = {:.2g} \, \mathrm{{a.u.}}$'.format(fit_result.params['prefactor'].value) ,
            '$q_0 = {:.2g} \, \mathrm{{pix}}^{{-1}}$'.format(x) ,
            '$\sigma_0 = {:.2g} \, \mathrm{{pix}}^{{-1}}$'.format(fit_result.params['sigma'].value) ,
            '$m = {:.3g} \, \mathrm{{a.u./pix^{{-1}} }}$'.format(fit_result.params['m'].value) ,
            '$b = {:.3g} \, \mathrm{{a.u.}}$'.format(fit_result.params['b'].value) ,
            ]
        s = '\n'.join(els)
        ax.text(xf, yf, s, size=20, color='purple', verticalalignment='top', horizontalalignment='right')
        
    
    
    ax.axis([xi,xf,yi,yf])
    
    plt.savefig(outfile, dpi=200)
    plt.close()

    

def circular_average(data, scale=[1,1], origin=None, bins_relative=2.0):
    
    h, w = data.shape
    x_scale, y_scale = scale
    if origin is None:
        x0, y0 = int(w/2), int(h/2)
    else:
        x0, y0 = origin
        
    # Compute map of distances to the origin
    x = (np.arange(w) - x0)*x_scale
    y = (np.arange(h) - y0)*y_scale
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2 + Y**2)

    # Compute histogram
    data = data.ravel()
    R = R.ravel()
    
    scale = (x_scale + y_scale)/2.0
    r_range = [0, np.max(R)]
    num_bins = int( bins_relative * abs(r_range[1]-r_range[0])/scale )
    num_per_bin, rbins = np.histogram(R, bins=num_bins, range=r_range)
    idx = np.where(num_per_bin!=0) # Bins that actually have data
    
    r_vals, rbins = np.histogram( R, bins=num_bins, range=r_range, weights=R )
    r_vals = r_vals[idx]/num_per_bin[idx]
    I_vals, rbins = np.histogram( R, bins=num_bins, range=r_range, weights=data )
    I_vals = I_vals[idx]/num_per_bin[idx]

    return r_vals, I_vals
        
        
    
# Define the fit model
def model(v, x):
    '''Gaussian with linear background.'''
    m = v['prefactor']*np.exp( -np.square(x-v['x_center'])/(2*(v['sigma']**2)) ) + v['m']*x + v['b']
    return m

def func2minimize(params, x, data):
    v = params.valuesdict()
    m = model(v, x)
    
    return m - data    
    
def peak_fit(xs, ys, x_expected, range_rel=0.2, vary=True, eps=1e-10):

    # Trim the curve to extract just the part we want
    xi, xf = x_expected*(1-range_rel), x_expected*(1+range_rel)
    idx_start, idx_end = np.where( xs>xi )[0][0], np.where( xs>xf )[0][0]
    xc = xs[idx_start:idx_end]
    yc = ys[idx_start:idx_end]
    span = np.max(xc)-np.min(xc)
    
    # Estimate linear background using the two endpoints
    m = (yc[-1]-yc[0])/(xc[-1]-xc[0])
    b = yc[0] - m*xc[0]
    
    # Estimate prefactor
    idx = np.where( xc>=x_expected )[0][0]
    xpeak, ypeak = xc[idx], yc[idx]
    p = ypeak - (m*xpeak + b)
    
    # Estimate standard deviation
    yc_peakonly = yc - (m*xc + b)
    mean = np.average(xc, weights=np.clip(yc_peakonly, eps, ypeak))
    variance = np.average(np.square(xc - mean), weights=np.clip(yc_peakonly, eps, ypeak))
    std = np.sqrt(variance)

    # Start the fit model using our best estimates; restrict the parameter ranges to be 'reasonable'
    params = lmfit.Parameters()
    params.add('prefactor', value=p, min=0, max=np.max(yc)+eps, vary=vary)
    params.add('x_center', value=x_expected, min=np.min(xc), max=np.max(xc)+eps, vary=vary)
    params.add('sigma', value=std, min=span*0.00001, max=span*0.75, vary=vary)
    params.add('m', value=m, min=abs(m)*-5, max=abs(m)*+5+eps, vary=vary)
    params.add('b', value=b, min=min(0, b*5), max=max(np.max(ys)*2, abs(b)*5)+eps, vary=vary)
    
    lm_result = lmfit.minimize(func2minimize, params, args=(xc, yc))
    
    fit_x = np.linspace(np.min(xc), np.max(xc), num=max(500, len(xc)))
    fit_y = model(lm_result.params.valuesdict(), fit_x)
    fit_line = fit_x, fit_y
    
    xe = 0.5
    xi = np.min(xc)-xe*span
    xf = np.max(xc)+xe*span
    fit_x = np.linspace(xi, xf, num=2000)
    fit_y = model(lm_result.params.valuesdict(), fit_x)
    fit_line_extended = fit_x, fit_y
        
    return lm_result, fit_line, fit_line_extended
    
def fft_circavg_yager_2d(input_data, plot_metrics=True, output_dir='./', output_name='result',
                         range_rel=0.75, scale=[1,1], adjust=1.0, output_condition='' , interpolation_abscissa = None ):
    assert( input_data.ndim == 2 )
    
    x_scale, y_scale = scale
    
    if plot_metrics:
        plot2D(input_data, outfile='{}{}2D_{}.png'.format(output_dir, output_name, output_condition), scale=[x_scale, y_scale])

    # Compute FFT
    result_fft = np.fft.fft2(input_data)
    
    # Recenter FFT (by default the origin is in the 'corners' but we want the origin in the center of the image)
    hq, wq = result_fft.shape
    result_fft = np.concatenate( (result_fft[int(hq/2):,:], result_fft[0:int(hq/2),:]), axis=0 )
    result_fft = np.concatenate( (result_fft[:,int(wq/2):], result_fft[:,0:int(wq/2)]), axis=1 )
    origin = [int(wq/2), int(hq/2)]
    qx_scale, qy_scale = 2*np.pi/(x_scale*wq), 2*np.pi/(y_scale*hq)

    data = np.absolute(result_fft)

    if plot_metrics:
        plotFFT(data, outfile='{}{}FFT_{}.png'.format(output_dir, output_name, output_condition), scale=[qx_scale, qy_scale])

    # Compute 1D curve by doing a circular average (about the origin)
    qs, data1D = circular_average(data, scale=[qx_scale, qy_scale], origin=origin)
    
    # Optionally adjust the curve to improve data extraction
    if adjust is not None:
        data1D *= np.power(qs, adjust)

    # Optionally interpolate
    if interpolation_abscissa is not None:
        assert( (interpolation_abscissa[0] >= qs[0]) & (interpolation_abscissa[-1] <= qs[-1]) )
        data1D = np.interp( interpolation_abscissa , qs , data1D )
        qs     = interpolation_abscissa

    # Find the peak of the data
    idx = np.where( data1D==np.max(data1D) )[0][0]
    idx = max(idx, 3)
    xpeak, ypeak = qs[idx], data1D[idx]

    # Fit the 1D curve to a Gaussian
    lm_result, fit_line, fit_line_extended = peak_fit(qs, data1D, x_expected=xpeak, range_rel=range_rel)

    if plot_metrics:
        plot1DFFT(qs, data1D, outfile='{}{}1DFFT_{}.png'.format(output_dir, output_name, output_condition), x_expected=xpeak, range_rel=range_rel, fit_line=fit_line, fit_line_e=fit_line_extended, fit_result=lm_result)

    # Save structure vector to output_dir
    np.savetxt( output_dir + output_name + 'fft_circavg.out' , [qs , data1D] )
    
    return qs , data1D , lm_result
    
def structure_vector_yager_2d(input_data, plot_metrics=True, output_dir='./', output_name='result',
                              range_rel=0.75, scale=[1,1], adjust=1.0, output_condition='', interpolation_abscissa=None):

    assert( input_data.ndim == 2 )
    
    qs , data1D , lm_result = fft_circavg_yager_2d(input_data, plot_metrics, output_dir, output_name,
                                                   range_rel, scale, adjust, output_condition, interpolation_abscissa)
    
    p = lm_result.params['prefactor'].value # Peak height (prefactor)
    q = lm_result.params['x_center'].value # Peak position (center)
    sigma = lm_result.params['sigma'].value # Peak width (stdev)
    I = p*sigma*np.sqrt(2*np.pi) # Integrated peak area
    m = lm_result.params['m'].value # Baseline slope
    b = lm_result.params['b'].value # Baseline intercept

    # Save structure vector to output_dir
    np.savetxt( output_dir + output_name + 'structure_vector.out' , [p, q, sigma, I, m, b] )

    return p, q, sigma, I, m, b

# This is just a Python closure to wrap the interpolation_abscissa argument
def make_fft_circavg_yager_2d( interpolation_abscissa ):
    def fft_circavg_wrapper( input_data, plot_metrics=True, output_dir='./', output_name='result',
                             range_rel=0.75, scale=[1,1], adjust=1.0, output_condition='' , interpolation_abscissa = interpolation_abscissa):
        return fft_circavg_yager_2d( input_data, plot_metrics, output_dir, output_name,
                                     range_rel, scale, adjust, output_condition, interpolation_abscissa )
    return fft_circavg_wrapper

# This is just a Python closure to wrap the interpolation_abscissa argument
def make_structure_vector_yager_2d( interpolation_abscissa ):
    def structure_vector_wrapper( input_data, plot_metrics=True, output_dir='./', output_name='result',
                                  range_rel=0.75, scale=[1,1], adjust=1.0, output_condition='' , interpolation_abscissa = interpolation_abscissa):
        return structure_vector_yager_2d( input_data, plot_metrics, output_dir, output_name,
                                          range_rel, scale, adjust, output_condition, interpolation_abscissa )
    return structure_vector_wrapper


if __name__ == '__main__':

    # Define expectations for simulations results    
    w, h = 500, 500 # Simulation size
    x_scale, y_scale = 1, 1 # Conversion of simulation units into realspace units
    
    # Example: A single input file.
    ########################################
    
    # Load simulation output    
    m, sigma = 0.2, 0.1
    m, sigma = -0.5, 0.1
    #infile = 'result_m{:.1f}_sigma{:.1f}.npy'.format(m, sigma)
    #frames = load_result_file(infile, w, h)
    infile = '/home/adegennaro/Projects/exalearn/image_structure/data/2D_N100_T2000_sig0.5_m0.1_size8_gray.npy'
    frames = np.expand_dims( np.load( infile ) , 0 )
    print(frames.shape)
    
    for iframe, frame in enumerate(frames):
        
        print('frame {}'.format(iframe))
        
        output_condition = 'm{:.1f}_sigma{:.1f}_frame{:d}'.format(m, sigma, iframe)
        # Compute structure vector
        print( frame , type(frame) )
        print( output_condition , type(output_condition) )
        
        vector = structure_vector_yager_2d(frame, plot_metrics=True, scale=[x_scale, y_scale], output_condition=output_condition)    








