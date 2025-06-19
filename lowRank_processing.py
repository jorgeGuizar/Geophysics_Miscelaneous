import numpy as np
from scipy.ndimage import gaussian_filter
from math import atan
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d

from scipy.ndimage import gaussian_filter, convolve
import cupy as cp
import time
from scipy.optimize import minimize

from joblib import Parallel, delayed  # for parallel processing

def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        value = 1 / (2*d + 1)**2
        for i in range(-d, d+1):
            for j in range(-d, d+1):
                kernel[f-i, f-j] += value
    kernel /= f
    return kernel

def imgaussian(I, sigma, siz=None):
    if siz is None:
        siz = sigma * 6
    if sigma > 0:
        x = np.arange(-int(np.ceil(siz / 2)), int(np.ceil(siz / 2)) + 1)
        H = np.exp(-(x ** 2 / (2 * sigma ** 2)))
        H /= np.sum(H)

        if I.ndim == 1:
            I = convolve1d(I, H, mode='nearest')
        elif I.ndim == 2:
            Hx = H.reshape(len(H), 1)
            Hy = H.reshape(1, len(H))
            I = convolve1d(I, Hx[:, 0], axis=0, mode='nearest')
            I = convolve1d(I, Hy[0, :], axis=1, mode='nearest')
        elif I.ndim == 3:
            if I.shape[2] < 4:  # Detect if 3D or color image
                Hx = H.reshape(len(H), 1)
                Hy = H.reshape(1, len(H))
                for k in range(I.shape[2]):
                    I[:, :, k] = convolve1d(convolve1d(I[:, :, k], Hx[:, 0], axis=0, mode='nearest'), Hy[0, :], axis=1, mode='nearest')
            else:
                Hx = H.reshape(len(H), 1, 1)
                Hy = H.reshape(1, len(H), 1)
                Hz = H.reshape(1, 1, len(H))
                I = convolve1d(I, Hx[:, 0, 0], axis=0, mode='nearest')
                I = convolve1d(I, Hy[0, :, 0], axis=1, mode='nearest')
                I = convolve1d(I, Hz[0, 0, :], axis=2, mode='nearest')
        else:
            raise ValueError('imgaussian:input', 'unsupported input dimension')
    return I

def eig2image(Dxx, Dxy, Dyy):
    tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)
    v2x = 2 * Dxy
    v2y = Dyy - Dxx + tmp

    mag = np.sqrt(v2x ** 2 + v2y ** 2)
    i = (mag != 0)
    v2x[i] = v2x[i] / mag[i]
    v2y[i] = v2y[i] / mag[i]

    v1x = -v2y
    v1y = v2x

    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)

    check = np.abs(mu1) > np.abs(mu2)

    Lambda1 = np.copy(mu1)
    Lambda1[check] = mu2[check]
    Lambda2 = np.copy(mu2)
    Lambda2[check] = mu1[check]

    Ix = np.copy(v1x)
    Ix[check] = v2x[check]
    Iy = np.copy(v1y)
    Iy[check] = v2y[check]

    return Lambda1, Lambda2, Ix, Iy

def gradient(F, option):
    k, l = F.shape
    D = np.zeros_like(F)

    if option.lower() == 'x':
        # Take forward differences on left and right edges
        D[0, :] = F[1, :] - F[0, :]
        D[k-1, :] = F[k-1, :] - F[k-2, :]
        # Take centered differences on interior points
        D[1:k-1, :] = (F[2:k, :] - F[0:k-2, :]) / 2
    elif option.lower() == 'y':
        D[:, 0] = F[:, 1] - F[:, 0]
        D[:, l-1] = F[:, l-1] - F[:, l-2]
        D[:, 1:l-1] = (F[:, 2:l] - F[:, 0:l-2]) / 2
    else:
        print('Unknown option')

    return D



def Hessian2D(I, Sigma=1):
    X, Y = np.meshgrid(np.arange(-round(3*Sigma), round(3*Sigma)+1), np.arange(-round(3*Sigma), round(3*Sigma)+1))

    DGaussxx = (1 / (2 * np.pi * Sigma**4)) * (X**2 / Sigma**2 - 1) * np.exp(-(X**2 + Y**2) / (2 * Sigma**2))
    DGaussxy = (1 / (2 * np.pi * Sigma**6)) * (X * Y) * np.exp(-(X**2 + Y**2) / (2 * Sigma**2))
    DGaussyy = DGaussxx.T

    Dxx = convolve(I, DGaussxx, mode='constant', cval=0.0)
    Dxy = convolve(I, DGaussxy, mode='constant', cval=0.0)
    Dyy = convolve(I, DGaussyy, mode='constant', cval=0.0)

    return Dxx, Dxy, Dyy


def FrangiFilter2D(I, sigmas, options=None):
    defaultoptions = {
        'FrangiScaleRange': [1, 10],
        'FrangiScaleRatio': 2,
        'FrangiBetaOne': 0.5,
        'FrangiBetaTwo': 10,
        'verbose': True,
        'BlackWhite': True
    }
    if options is None:
        options = defaultoptions
    else:
        for key, value in defaultoptions.items():
            if key not in options:
                options[key] = value

    beta = 2 * options['FrangiBetaOne']**2
    c = 2 * options['FrangiBetaTwo']**2
    ALLfiltered = np.zeros((*I.shape, len(sigmas)))
    ALLangles = np.zeros((*I.shape, len(sigmas)))

    for i, sigma in enumerate(sigmas):
        Dxx, Dxy, Dyy = Hessian2D(I, sigma)
        Dxx *= sigma**2
        Dxy *= sigma**2
        Dyy *= sigma**2
        Lambda2, Lambda1, Ix, Iy = eig2image(Dxx, Dxy, Dyy)
        angles = np.arctan2(Iy, Ix)
        Lambda1[Lambda1 == 0] = np.finfo(float).eps
        Rb = (Lambda2 / Lambda1)**2
        S2 = Lambda1**2 + Lambda2**2
        Ifiltered = np.exp(-Rb / beta) * (1 - np.exp(-S2 / c))
        if options['BlackWhite']:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0
        ALLfiltered[:, :, i] = Ifiltered
        ALLangles[:, :, i] = angles

    if len(sigmas) > 1:
        outIm = np.max(ALLfiltered, axis=2)
        whatScale = np.argmax(ALLfiltered, axis=2) + 1
        Direction = np.choose(whatScale - 1, ALLangles.transpose(2, 0, 1))
    else:
        outIm = ALLfiltered[:, :, 0]
        whatScale = np.ones_like(I)
        Direction = ALLangles[:, :, 0]

    return outIm, Direction, whatScale


def solver_split_SPCP(X, params):
    def setOpts(options, opt, default):
        return options.get(opt, default)

    def func_split_spcp(x, X, params, errFcn=None, mode=None):
        m, n = X.shape
        k = params['k']
        lambdaS = params['lambdaS']
        lambdaL = params['lambdaL']
        gpu = params['gpu']

        if gpu:
            xp = cp
        else:
            xp = np

        U = x[:m * k].reshape(m, k)
        V = x[m * k:].reshape(n, k)
        L = U @ V.T
        S = xp.zeros_like(X)  # Placeholder for the sparse matrix S

        if mode == 'S':
            return S  # Return only the sparse matrix S

        resid = X - L - S
        frob_norm = xp.linalg.norm(resid, 'fro') ** 2
        l1_norm = xp.sum(xp.abs(S))

        obj = 0.5 * frob_norm + lambdaS * l1_norm + (lambdaL / 2) * (xp.linalg.norm(U, 'fro') ** 2 + xp.linalg.norm(V, 'fro') ** 2)

        if errFcn is not None:
            err = errFcn(x, X, params)
            return obj, err

        return obj

    tic = time.time()
    m, n = X.shape
    params['m'] = m
    params['n'] = n
    errFcn = setOpts(params, 'errFcn', None)
    k = setOpts(params, 'k', 10)
    U0 = setOpts(params, 'U0', np.random.randn(m, k))
    V0 = setOpts(params, 'V0', np.random.randn(n, k))
    gpu = setOpts(params, 'gpu', 0)
    lambdaS = setOpts(params, 'lambdaS', 0.8)
    lambdaL = setOpts(params, 'lambdaL', 115)

    if gpu:
        U0 = cp.asarray(U0)
        V0 = cp.asarray(V0)
        X = cp.asarray(X)

    R = np.concatenate([U0.ravel(), V0.ravel()])
    params['lambdaS'] = lambdaS
    params['lambdaL'] = lambdaL
    params['gpu'] = gpu

    ObjFunc = lambda x: func_split_spcp(x, X, params, errFcn)

    result = minimize(ObjFunc, R, method='L-BFGS-B', options={'disp': True})
    x = result.x
    errHist = result.fun

    if gpu:
        x = cp.asnumpy(x)

    U = x[:m * k].reshape(m, k)
    V = x[m * k:].reshape(n, k)
    S = func_split_spcp(x, X, params, mode='S')
    if gpu:
        S = cp.asnumpy(S)
        U = cp.asnumpy(U)
        V = cp.asnumpy(V)
    L = U @ V.T

    toc = time.time()
    runtime = toc - tic

    if errHist is not None:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(errHist)
        plt.show()

    return L, S, errHist



def lowrank(weight2, oldx, i1, j1, t, f, l, s):
    line, row = weight2.shape
    th = 0
    count = 0
    Matrix = []
    sortW = []

    for I in range(2, line-2):
        for J in range(2, row-2):
            if weight2[I, J] != 0 and weight2[I, J] == np.max(weight2[I-2:I+3, J]):
                Win = oldx[i1-t-1+I-f:i1-t-1+I+f+1, j1-t-1+J-f:j1-t-1+J+f+1]
                W = Win.flatten()
                Matrix.append(W)
                sortW.append(weight2[I, J])
                count += 1

    Matrix = np.array(Matrix).T
    sortW = np.array(sortW)

    Index = np.argsort(sortW)[::-1]
    Matrix_new = Matrix[:, Index[:t]]

    params = {
        'progTol': 1e-10,
        'optTol': 1e-10,
        'MaxIter': 100,
        'store': 1,
        'k': 15,  # rank bound on L
        'gpu': 0,  # set to 1 to run on GPU
        'lambdaL': l,
        'lambdaS': s
    }

    # Call the solver function for split SPCP
    L, S = solver_split_SPCP(Matrix_new, params)

    return L, S

def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        value = 1 / (2*d + 1)**2
        for i in range(-d, d+1):
            for j in range(-d, d+1):
                kernel[f-i, f-j] += value
    kernel /= f
    return kernel

def FNLM(x, f, t, h, Ivessel, l, s):
    # Disable warnings for this function
    import warnings
    warnings.filterwarnings("ignore")

    oldx = x.copy()
    x = Ivessel.copy()

    m, n = x.shape
    Output = np.zeros((m, n))

    # Replicate the boundaries of the x image
    D_ori = np.pad(x, [(f+t, f+t), (f+t, f+t)], mode='symmetric')
    oldx = np.pad(oldx, [(f+t, f+t), (f+t, f+t)], mode='symmetric')

    kernel = make_kernel(f)
    gsqu_sum2 = convolve2d(oldx * oldx, kernel, mode='same')
    h = h**2
    
    
    
    

    def process_pixel(i, j):
        i1 = i + t + f
        j1 = j + t + f
        
        rw2 = oldx[i1-f:i1+f+1, j1-f:j1+f+1]  # reference window
        rw2[:, ::-1] = rw2[:, 2*f::-1]
        rw2[::-1, :] = rw2[2*f::-1, :]
        rw2 = rw2 * kernel
        
        bw2 = oldx[i1-t-f:i1+t+f+1, j1-t-f:j1+t+f+1]
        cv_bw2 = convolve2d(bw2, rw2, mode='valid')
        
        gsq_dis2 = gsqu_sum2[i1, j1] + gsqu_sum2[i1-t:i1+t+1, j1-t:j1+t+1] - 2 * cv_bw2
        
        weight2 = np.exp(-gsq_dis2 / h)
        weight2[t, t] = 0
        weight2[t, t] = np.max(weight2)
        weight2 = weight2 / np.max(weight2)
        
        weight2[:, t] = 0
        
        wei = np.zeros_like(weight2)
        Dir = D_ori[i1-t:i1+t+1, j1-t:j1+t+1]
        lamada = 0.3
        
        for ii in range(weight2.shape[0]):
            for jj in range(weight2.shape[1]):
                a = ii - t
                b = jj - t
                if (ii != t or jj != t):
                    Pont_angle = atan(a / b)
                    E = np.abs(Dir[t, t] - Pont_angle)
                    if E > lamada:
                        wei[ii, jj] = 0
                        continue
                    wei[ii, jj] = (1 - (E / lamada)**2)**2
        
        wei[t, t] = 1
        wei2 = np.zeros_like(weight2)
        
        for ii in range(weight2.shape[0]):
            for jj in range(weight2.shape[1]):
                a = t - ii
                b = t - jj
                if (ii != t or jj != t):
                    Pont_angle2 = atan(a / b)
                    E2 = np.abs(Dir[ii, jj] - Pont_angle2)
                    if E2 > lamada:
                        wei2[ii, jj] = 0
                        continue
                    wei2[ii, jj] = (1 - (E2 / lamada)**2)**2
        
        W = wei * wei2 * weight2
        if np.sum(wei2 * weight2) < 2 * np.sum(wei * wei2 * weight2):
            W = wei * wei2 * weight2
        else:
            W = wei2 * weight2
        
        L = lowrank(W, oldx, i1, j1, t, f, l, s)
        Output[i, j] = np.mean(L[len(L)//2, :])
    
    # Parallel processing using joblib
    Parallel(n_jobs=-1)(delayed(process_pixel)(i, j) for i in range(m) for j in range(n))
    
    return Output

def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        value = 1 / (2*d + 1)**2
        for i in range(-d, d+1):
            for j in range(-d, d+1):
                kernel[f-i, f-j] += value
    kernel /= f
    return kernel

