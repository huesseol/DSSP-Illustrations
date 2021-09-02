import numpy as np


#f should be function handler, returns pair of list (first being sample times and second sampled value of fourier transform)
def fourier_transform(f, t0, fidelity=0.01):
    t = np.arange(t0, -t0, fidelity) #sample times in time domain
    f_s = np.array([f(s) for s in t])
    w = np.fft.fftfreq(f_s.size)*2*np.pi/fidelity #sample times in frequency domain
    
    #Compute Fourier transform by numpy's FFT function (approximation)
    g = fidelity*np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi)) * np.fft.fft(f_s)
    #frequency normalization factor is 2*np.pi/dt
    
    #sort by w
    combined = np.column_stack((w, g))
    combined = combined[combined[:, 0].argsort()]
    
    return (np.real(combined[:, 0]), combined[:, 1])
    
    
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
#spectrum should have shape as returned by fourier_transform(f, t0)
def evaluate_spectrum(spectrum, w):
    return spectrum[1][find_nearest_index(spectrum[0], w)]
    
def evaluate_time(Q, w):
    t_values = Q[0]
    f_values = Q[1]
    return f_values[find_nearest_index(t_values, w)]
    
    