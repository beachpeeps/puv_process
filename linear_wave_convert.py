#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:41:48 2019

@author: cassandra
"""

import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

g = 9.81
rho = 1025

def get_k(omega, h):
    k = omega/np.sqrt(g*h)
    f = g*k*np.tanh(k*h) - omega**2
    while abs(f) > 1e-10:
      dfdk = g*k*h*(1/np.cosh(k*h))**2 + g*np.tanh(k*h)
      k = k - f/dfdk
      f = g*k*np.tanh(k*h) - omega**2
    return k

def p_to_eta(pressure, burial_depth_start, burial_depth_end, 
             dt=0.5, pressure_units='depth'):
    """
    Convert bottom pressure to surface elevation 
    
    @returns
        eta, h
        
        eta - the anomaly
        h   - the total depth
    """
    
    # Step 1: Convert pressure units if necessary
    if pressure_units == 'depth':
        p = pressure*rho*g
    else:
        p = pressure
        
    # Step 2: Divide into 15 minute segments with 50% overlap, 
    #         get mean p (and h), detrend
    # Step 2.i:    Generate the segment indices
    seg_length = 2*15*60
    overlap = 0.5
    seg_step = int(seg_length*(1-overlap))
    seg_indices = list(range(0, len(p)-seg_length, seg_step))
    # Step 2.ii:   Add a segment index corresponding to the last segment, which 
    #              has more overlap in order to work with all the data
    seg_indices.append(len(p)-seg_length)
    # Step 2.iii:  Create mean height for each segment
    mean_h = [np.nanmean(p[i:i+seg_length])/(rho*g) for i in seg_indices]
    # Step 2.iv:   Create the segments themselves
    p_segs = [p[i:i+seg_length] for i in seg_indices]
    # Step 2.v:    Figure out the burial depth for each segment
    burial_depths = np.interp([i+seg_length/2 for i in seg_indices],
                              [0, len(p)], [burial_depth_start, burial_depth_end])
    
    # Step 3: For each segment
    eta_segs = []
    for i, p_seg, h, burial_depth in zip(range(len(p_segs)), p_segs, mean_h, burial_depths):
        # Step 3.0.   subtract mean water level
        p_seg = p_seg - np.mean(p_seg)
        # Step 3.i.   fft to get p_n
        p_ns = fft(p_seg) 
        # Step 3.ii.  generate frequency info to get omega_n
        freq_pos = np.linspace(0, (dt*2)**(-1), int(seg_length/2))
        freq_ns = np.append(freq_pos, np.flip(-1*(freq_pos+(dt*seg_length)**(-1)), 0))
        omega_ns = freq_ns*2*np.pi
        # Step 3.iii. Cutoff noise and ringing frequencies (more than 0.3)
        cutoff = np.abs(freq_ns) > 0.3
        p_ns[cutoff] = 0
        # Step 3.iv. Compute k_n
        k_ns = [get_k(omega_n, h) for omega_n in omega_ns]
        # Step 3.v.  Compute eta_n from p_n (set any nans to zero)
        # Doing a correction for linear waves (cosh) and also sand burial (exp)
        eta_ns = [p_n * np.exp(k_n * burial_depth) * np.cosh(k_n * h) /  (rho * g)
                  for p_n, k_n, omega_n in zip(p_ns, k_ns, omega_ns)]
        # Step 3.vi. Compute eta using an inverse fft
        eta_segs.append(np.real(ifft(eta_ns)))
        
        print(i, "/", len(p_segs))
    
    #Step 4: Combine etas for one time series
    #Step 4.i.   Create a normalized triangle weight for averaging segments
    weight = np.convolve(np.ones(seg_length), np.ones(seg_length), mode='same')
    weight = weight - np.min(weight) + 1
    weight = weight / np.trapz(weight)
    # Step 4.ii.  Project segments and weights onto the original time series,
    #             appended by zeros
    new_eta = [np.zeros(len(p)) for seg in eta_segs]
    new_weights = [np.zeros(len(p)) for seg in eta_segs]
    for seg, i, j in zip(eta_segs, seg_indices, range(len(eta_segs))):
        new_weights[j][i:i+seg_length] = weight
        new_eta[j][i:i+seg_length] = seg
    # Step 4.iii. Do a weighted average with segment
    eta_full = np.average(np.array(new_eta), weights=np.array(new_weights), axis=0)
    # Step 4.iv:  Generate a hydrostatic height anomaly, to compare to expected eta
    h_tidal = savgol_filter(p/(rho*g), seg_length+1, 1)
    # Step 4.v:   Return
    return eta_full, (eta_full+h_tidal)

def u_bot_to_top(u, dt=0.5):
    pass

def main():
    df = pd.read_csv('/data/ib_data/IB-S02.csv')
    
    # Have to remove some spurious data at the start
    df = df[767000:767000+150000]
    
    # Get pressure
    u = df['u'].values
    
    # Get time
    t = df['t'].values.astype('datetime64')
    
    # Get u top 
    ut = u_bot_to_top(u)

    # Plot result
    plt.figure()
    plt.plot(t, u, label="U bot")
    plt.plot(t, ut, label = "U top")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()