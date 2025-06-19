import numpy as np 
import matplotlib.pyplot as plt
import scipy as scp 
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import generic_filter
from scipy.ndimage import gaussian_filter1d
#############################################################################################################
########################## cCOMPLEX TRACE ATTRIUTES #########################################################
#############################################################################################################
def Hilbert_data(data):
    """
    Compute the Hilbert transform of the data.

    The attribute is computed over the last dimension. That is, time should
    be in the last dimension, so a 100 inline, 100 crossline seismic volume
    with 250 time slices would have shape (100, 100, 250).

    Args:
        data (ndarray): The data array to use for calculating energy.

    Returns:
        ndarray: An array the same dimensions as the input array.
    """
    return hilbert(data)
def instantaneous_amplitude(data):
    """
    Compute instantaneous amplitude, also known as the envelope or
    reflection strength.

    The attribute is computed over the last dimension. 
    That is, time should be in the last dimension, 
    so a 100 inline, 100 crossline seismic volume
    with 250 time slices would have shape (100, 100, 250).

    Args:
        traces (ndarray): The data array to use for calculating energy.
    Returns:
        ndarray: An array the same dimensions as the input array.
    
    """
    return np.abs(hilbert(data))

envelope = instantaneous_amplitude
reflection_strength = instantaneous_amplitude

def quadrature(traces):
    """
    Compute the quadrature trace.

    See https://wiki.seg.org/wiki/Instantaneous_attributes.

    Args:
        traces (ndarray): The data array to use for calculating energy.

    Returns:
        ndarray: An array the same dimensions as the input array.
    """
    h = hilbert(traces)
    hilb_trans=np.imag(h)
    #return np.abs(h) * np.sin(np.log(h).imag)
    return hilb_trans

def instantaneous_phase(traces):
    """
    Compute the instantaneous phase of the data.

    .. math::

        \\phi(t) = {\\rm Im}[\\ln h(t)]
        
    
    See https://wiki.seg.org/wiki/Instantaneous_attributes.

    Args:
        traces (ndarray): The data array to use for calculating energy.

    Returns:
        ndarray: An array the same dimensions as the input array.
    """
    return np.angle(hilbert(traces))
def _inst_freq_claerbout(traces, dt):
    """
    Compute the instantaneous frequency using Claerbout's (1985) approximation.
    This is also the formulation given in Yilmaz (2001).

    Formulation from Barnes, A, 2016, Handbook of Poststack Seismic Attributes,
    SEG Books.

    Args:
        traces (ndarray): The data array to use for calculating energy.
        dt (float): The sample interval in seconds, e.g. 0.004 for 4 ms sample
            interval (250 Hz sample frequency).
    Returns:
        ndarray: An array the same dimensions as the input array.
    """
    h = hilbert(traces)
    term = (h[1:] - h[:-1]) / (h[1:] + h[:-1])
    return (1 / (np.pi * dt)) * np.imag(term)

def _inst_freq_scheuer_oldenburg(traces, dt):
    """Instantaneous frequency after Scheuer & Oldenburg (1988)
    
    Scheuer, TE and DW Oldenburg (1988). Local phase velocity from complex seismic data.
    Geophysics 53 (12), p1503. DOI: http://dx.doi.org/10.1190/1.1442431.
    
    Formulation from Barnes, A, 2016, Handbook of Poststack Seismic Attributes,
    SEG Books:
    
    .. math::

        f_i(t) = \frac{1}{2\pi} \ \mathrm{Im} \left[\frac{h'(t)}{h(t)} \right]\approx \frac{1}{\pi T} 
        \ \mathrm{Im} \left[\frac{h(t+T) - h(t)}{h(t+T) + h(t)} \right] 

    Args:
        traces (ndarray): The data array to use for calculating energy.
        dt (float): The sample interval in seconds, e.g. 0.004 for 4 ms sample
            interval (250 Hz sample frequency).
    Returns:
        ndarray: An array the same dimensions as the input array.
    """
    y = quadrature(traces)
    expr = (traces[:-1] * y[1:] - traces[1:] * y[:-1]) / (traces[:-1] * traces[1:] + y[1:] * y[:-1])
    return (1 / (2 * np.pi * dt)) * np.arctan(expr)

def instantaneous_frequency(traces, dt, kind='scipy', percentile_clip=99):
    """
    Compute instantaneous frequency with a discrete approximation.

    The attribute is computed over the last dimension. That is, time should
    be in the last dimension, so a 100 inline, 100 crossline seismic volume
    with 250 time slices would have shape (100, 100, 250).

    These attributes can be noisy so a percentile clips is applied.

    Args:
        traces (ndarray): The data array to use for calculating energy.
        dt (float): The sample interval in seconds, e.g. 0.004 for 4 ms sample
            interval (250 Hz sample frequency).
        kind (str): "scipy", "claerbout" or "so" to denote a naive method from
            the SciPy docs, Claerbout's (1985) method or that of Scheuer & Oldenburg
            (1988). Claerbout's approximation is not stable above about half the
            Nyquist frequency (i.e. one quarter of the sampling frequency). The
            SciPy implementation is not recommended for seismic data.
        percentile_clip (float): Percentile at which to clip the data.
            Computed from the absolute values, clipped symmetrically
            at -p and +p, where p is the value at the 98th percentile.
    Returns:
        ndarray: An array the same dimensions as the input array.
    """
    methods = {'claerbout': _inst_freq_claerbout, 'so': _inst_freq_scheuer_oldenburg,
               'scipy': lambda traces, dt: np.diff(instantaneous_phase(traces)) / (2.0 * np.pi * dt)}
    func = methods.get(kind)
    if func is None:
        m = f'{kind} is not supported, use "so" (Scheuer-Oldenburg, recommended), "claerbout" or "scipy".'
        raise NotImplementedError(m)
    f = func(traces, dt)
    p = np.percentile(np.abs(f), percentile_clip)
    return np.clip(f, a_min=-p, a_max=p)

def RMS(data):
    """
    Root mean square.
    
    Example
    >>> rms([3, 4, 5])
    4.08248290463863
    """
    data = np.asanyarray(data)
    #np.sqrt(np.sum(data**2) / data.size)
    return np.sqrt(np.sum(data**2) / data.size)

def Rms_calc(data, tx, window):
    """
    compute the root mean square of the data.

    Args:
        data (_type_): seismic array

    Returns:
        seismic _rms: RMS of data
        
        
    generic_filter wants the data, the function, and the size of the sub-volume to pass in to the 
    callback function. This can be trace-by-trace (use tx=1 for the first two dimensions) 
    or multi-trace (e.g. use tx=3). A larger template will be slower to compute of course.
    """
    ty=tx
    if data.ndim == 3:
        seismic_rms = generic_filter(data, RMS, size=(tx, ty, window))
    elif data.ndim == 2:
        seismic_rms = generic_filter(data, RMS, size=(tx, window))
    else:
        raise NotImplementedError("Expected 2D or 3D seismic data.")
    return seismic_rms

def RMS_window(data, window_down,window_top, tx, window,dt):
    """"
    This function calculates the RMs amplitude of a subvolume of the seismic data.

    Args:
        data (_type_): seismic traces 2d or 3d
        window_top (_type_): window top analysis in time in ms
        window_base (_type_): window base analysis in time in ms
        tx (_type_): traces to compute the RMS amplitude in x and yin y for the 2d case
        window (_type_): samples used to compute the RMS amplitude as a moving window

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    samples=data.shape[-1]
    Time=dt*(samples-1)*1000#3 THIS GIVES THE TOTAL RECORDING TIME  in ms

    time_down=window_down # smaller time in ms
    time_up=window_top # larger time  in ms
    
    time_delta=time_up-time_down
    samples_window=(np.round(time_delta/(dt*1000)))+1
    ty=tx
    
    if time_down<0 or time_down>Time or time_down>time_up or time_up>Time or samples_window*2<=window:
        raise ValueError("The window time is out of range or smaller than the window size. Please check the input values.")
    else:
        sample_down=int((time_down/dt)+1)
        sample_up=int((time_up/dt)+1)
        
        if data.ndim == 3:
            subcube_data=data[:,:,sample_up:sample_down+1]
        
            seismic_rms = generic_filter(subcube_data, RMS, size=(tx, ty, window))
        elif data.ndim == 2:
            subcube_data=data[:,sample_up:sample_down+1]
        
            seismic_rms = generic_filter(subcube_data, RMS, size=(tx, window))
            
        else:
            raise NotImplementedError("Expected 2D or 3D seismic data.")

    return seismic_rms


#############################################################################################################
########################## DISCONTINUITY TRACE ATTRIBUTES #########################################################
#############################################################################################################

def bahorich_coherence(data, zwin):
    """Compute Bahorich & Farmer (1995) coherence.

    Args:
        data (_type_): data traces
        zwin (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    if data.ndim == 2:
        nj,nk=data.shape
        out = np.zeros_like(data)
        padded = np.pad(data, ((0, 1), (zwin//2, zwin//2)), mode='reflect')
        for j, k in np.ndindex(nj, nk):
            
            center_trace = data[j,:]
            
        
            y_trace = padded[j+1, k:k+zwin]
            ycor = np.correlate(center_trace, y_trace)
            center_std = center_trace.std()
            py = ycor.max() / (ycor.size * center_std * y_trace.std())
            
            out[j,k] = np.sqrt(py)

    elif data.ndim == 3:
        ni, nj, nk = data.shape
        out = np.zeros_like(data)
    
        # Pad the input to make indexing simpler. We're not concerned about memory usage.
        # We'll handle the boundaries by "reflecting" the data at the edge.
        padded = np.pad(data, ((0, 1), (0, 1), (zwin//2, zwin//2)), mode='reflect')

        for i, j, k in np.ndindex(ni, nj, nk):
            # Extract the "full" center trace
            center_trace = data[i,j,:]
            
            # Use a "moving window" portion of the adjacent traces
            x_trace = padded[i+1, j, k:k+zwin]
            y_trace = padded[i, j+1, k:k+zwin]

            # Cross correlate. `xcor` & `ycor` will be 1d arrays of length
            # `center_trace.size - x_trace.size + 1`
            xcor = np.correlate(center_trace, x_trace)
            ycor = np.correlate(center_trace, y_trace)
            
            # The result is the maximum normalized cross correlation value
            center_std = center_trace.std()
            px = xcor.max() / (xcor.size * center_std * x_trace.std())
            py = ycor.max() / (ycor.size * center_std * y_trace.std())
            out[i,j,k] = np.sqrt(px * py)
        
    else:
        raise NotImplementedError("Expected 2D or 3D seismic data.")
    
    return out



def moving_window(traces, func, window):
    """
    Helper function for multi-trace attribute generation.
    This function applies a 3D function func to process a
    region of shape `window` over a dataset `data`.
    """
    wrapped = lambda x: func(x.reshape(window))
    return generic_filter(traces, wrapped, window)



def marfurt(traces):
    """
    Marfurt, K., V. Sudhaker, A. Gersztenkorn, K. D. Crawford, and S. E. Nissen, 1999,
    Coherency calculations in the presence of structural dip: GEOPHYSICS, 64, 104-111.
    doi:10.1190/1.1444508
    """
    i, x, t = traces.shape
    traces = traces.reshape(-1, t)
    square_sums = np.sum(traces, axis=0)**2
    sum_squares = np.sum(traces**2, axis=0)
    c = square_sums.sum() / (sum_squares.sum() + 1e-12)
    return c / (i * x)



def gersztenkorn(traces):
    """
    Gersztenkorn, A., and K. J. Marfurt, 1999, Eigenstructure‐based coherence
    computations as an aid to 3-D structural and stratigraphic mapping:
    GEOPHYSICS, 64, 1468-1479. doi:10.1190/1.1444651
    """
    # Stack traces in 3D traces into 2D array.
    traces = traces.reshape(-1, traces.shape[-1])

    # Calculate eigenvalues of covariance matrix.
    cov = traces.dot(traces.T)
    vals = np.linalg.eigvalsh(cov)
    return vals.max() / (vals.sum() + 1e-12)



def gradients(traces, sigma):
    """Builds a 4-d array of the gaussian gradient of *seismic*."""
    grads = []
    for axis in range(3):
        grad = gaussian_filter1d(traces, sigma, axis=axis, order=1)
        grads.append(grad[..., np.newaxis])
    return np.concatenate(grads, axis=3)



def moving_window4d(grad, window, func):
    """Applies the given function *func* over a moving *window*, reducing 
    the input *grad* array from 4D to 3D."""
    # Pad in the spatial dimensions, but leave the gradient dimension unpadded.
    half_window = [(x // 2, x // 2) for x in window] + [(0, 0)]
    padded = np.pad(grad, half_window, mode='reflect')
    
    out = np.empty(grad.shape[:3], dtype=float)
    for i, j, k in np.ndindex(out.shape):
        region = padded[i:i+window[0], j:j+window[1], k:k+window[2], :]
        out[i,j,k] = func(region)
    return out



def gst_calc(region):
    """Calculate gradient structure tensor coherence on a local region.
    Intended to be applied with *moving_window4d*."""
    region = region.reshape(-1, 3)
    gst = region.T.dot(region)
    eigs = np.sort(np.linalg.eigvalsh(gst))[::-1]
    eigs= np.where(np.isnan(eigs), 0, eigs)
    
    return (eigs[0]-eigs[1]) / (eigs[0]+eigs[1])



def gst_discontinuity(seismic, window, sigma=1):
    """
    Gradient structure tensor discontinuity.

    Randen, T., E. Monsen, C. Singe, A. Abrahamsen, J. Hansen, T. Saeter, and J. Schlaf, 2000,
    Three-dimensional texture attributes for seismic data analysis, 70th Annual International Meeting,
    SEG, Expanded Abstracts, 668-671.
    """
    grad = gradients(seismic, sigma)
    return moving_window4d(grad, window, gst_calc)



def discontinuity(traces, duration, dt, step_out=1, kind='gst', sigma=1):
    """
    Compute discontinuity for a seismic section using one of various methods.

    Expects time or depth to be in the last axis of a 2D or 3D input.

    :param traces: A 2D or 3D NumPy array arranged as (cdp, twt) or
        (iline, xline, twt).
    :param duration: The length in seconds of the window trace kernel
        used to calculate the discontinuity.
    :keyword dt (default=1): The sample interval of the traces in sec.
        (eg. 0.001, 0.002, ...). Will default to one, allowing
        duration to be given in samples.
    :keyword step_out (default=1):
        The number of adjacent traces to the kernel to compute discontinuity over.
    :keyword kind (default='gst'):
        The method to use for the computation. Can be "marfurt", "gersztenkorn"
        or "gst" (gradient structure tensor).
    :keyword sigma (default=1):
        The width of the Gaussian function used to compute gradients.
    """
    if traces.ndim == 2:
        traces = traces[:, None, :]
        window = 2*step_out+1, 1, int(duration / dt)
    elif traces.ndim == 3:
        window = 2*step_out+1, int(duration / dt), 2*step_out+1
    else:
        raise NotImplementedError("Expected 2D or 3D seismic data.")

    methods = {
        "marfurt": moving_window(traces, marfurt, window),
        "gersztenkorn": moving_window(traces, gersztenkorn, window),
        "gst": gst_discontinuity(traces, window, sigma)
    }

    return np.squeeze(methods[kind])


similarity = discontinuity