import matplotlib.pyplot as plt
import numpy as np
from obspy.io.segy.segy import _read_segy

import numpy as np
from scipy.ndimage import gaussian_filter
from math import atan
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

from scipy.ndimage import gaussian_filter, convolve
import cupy as cp
import time
from scipy.optimize import minimize
import os
import numpy as np
import cv2
from scipy.ndimage import median_filter, gaussian_filter
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # for parallel processing
from lowRank_processing import *



def plot_it(ax, data, vmin, vmax, title, cmap, ticks, ticklabels, alpha=1.0, interpolation='bilinear'):
    """_summary_

    Args:
        ax (_type_): _description_
        data (_type_): aray to plot must be the trasponse 
        vmin (_type_): min of data range that colormap uses
        vmax (_type_): max of data range that colormap uses
        title (_type_): title of the plot
        cmap (_type_): colormap for the plot
        ticks (_type_): _description_
        ticklabels (_type_): _description_
        alpha (float, optional): _description_. Defaults to 1.0.
        interpolation (str, optional): _description_. Defaults to 'bilinear'.
    """
    #fig, ax = plt.subplots(figsize=(16,10))
    im = ax.imshow(data.T, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto', alpha=alpha, interpolation=interpolation)
    ax.set_title(title)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    cbar = fig.colorbar(im, 
                        ax=ax,
                        ticks=ticks,
                        orientation='horizontal',
                        fraction=0.03, pad=0.03)
    cbar.ax.set_xticklabels(ticklabels)
    plt.show()



# Nombre del archivo SEGY
filename = r'D:\SEGY\SEGY_Files\OPEX\SBP_SEGY\Envelope\ETKAL-202-SBP_L-08_P2-CH1.sgy'

# Leer el archivo SEGY
stream = _read_segy(filename, headonly=True)

# Extraer los datos de la traza
traces = stream.traces

# Crear un arreglo numpy para almacenar los datos
data = np.zeros((len(traces), len(traces[0].data)))

# Llenar el arreglo con los datos de cada traza
for i, trace in enumerate(traces):
    data[i, :] = trace.data

# Obtener información de la geometría de las trazas
first_trace = traces[0]
trace_header = first_trace.header

# Extraer la información necesaria de la cabecera (puedes ajustar según tus necesidades)
sample_interval = trace_header.sample_interval_in_ms_for_this_trace / 1000.0  # Intervalo de muestreo en segundos
n_samples = trace_header.number_of_samples_in_this_trace  # Número de muestras por traza

# Crear un arreglo de tiempo para el eje x del gráfico
time = np.arange(0, n_samples * sample_interval, sample_interval)


vmin, vmax = np.min(data), np.max(data)
vrng = max(abs(vmin), vmax)

fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,data, -vrng/5, vrng/5, 'Seismic', 'seismic', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'])

print('Plotear todas las trazas')
#plt.figure(figsize=(10, 6))
#for trace_data in data:
#    plt.plot(time, trace_data, color='black', alpha=0.5)

#plt.title('Datos SEGY')
#plt.xlabel('Tiempo (s)')
#plt.ylabel('Amplitud')
#plt.grid(True)
#plt.tight_layout()
#plt.show()
def main():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_directory)

    # Initialize parallel processing
    # Adjust 'n_jobs' as per your system capabilities
    parallel_pool = Parallel(n_jobs=6)

    # Load test image
    x = img_as_float(io.imread('simulated4_2_noise.bmp'))

    # Filter
    Id = 255 - x
    Id = median_filter(Id, size=(3, 3))
    Id = gaussian_filter(Id, sigma=2)

    sigmas = [7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 20, 30]
    Ivessel0, Direction, whatScale = FrangiFilter2D(Id, sigmas)
    Ivessel = Direction

    Id = 255 - Id
    Id = median_filter(Id, size=(3, 3))
    Id = gaussian_filter(Id, sigma=2)

    sigmas = [6.5, 10, 30, 40, 50]
    Ivessel1, Direction2, whatScale2 = FrangiFilter2D(Id, sigmas)
    Ivessel2 = Direction2
    Ivessel[Ivessel0 < Ivessel1] = Ivessel2[Ivessel0 < Ivessel1]

    # Parameters
    l = 29
    s = 0.8
    f = 3
    t = 36
    sigma = 25

    # Apply FNLM filter
    Output = FNLM(x, f, t, sigma, Ivessel, l, s)

    # Display the output
    plt.figure()
    plt.imshow(Output, cmap='gray')
    plt.title('Output')
    plt.show()

    plt.figure()
    plt.imshow(x, cmap='gray')
    plt.title('Input')
    plt.show()

    # Load clean image for comparison
    Clean = img_as_float(io.imread('simulated4_2_clean.bmp'))

    # Calculate PSNR and SSIM
    snr_value = psnr(Output, Clean)
    mssim, ssim_map = ssim(Output, Clean, full=True)
    Mssim = mssim

    print(f'PSNR: {snr_value}')
    print(f'SSIM: {Mssim}')

if __name__ == '__main__':
    main()