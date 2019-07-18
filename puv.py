#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:03:37 2019

@author: cassandra
"""

import pandas as pd
import numpy as np
import gc

import linear_wave_convert

g = 9.81
rho = 1025

def get_k(omega, h):
    """
    Linear wave dispersion relation solver
    Given to me in matlab by Falk Feddersen, implemented in Python by me
    """
    k = omega/np.sqrt(g*h)
    f = g*k*np.tanh(k*h) - omega**2
    while abs(f) > 1e-10:
      dfdk = g*k*h*(1/np.cosh(k*h))**2 + g*np.tanh(k*h)
      k = k - f/dfdk
      f = g*k*np.tanh(k*h) - omega**2
    return k

def NortekVectorConvert(dat, vhd, csv, sample_Hz=2):
    """
    Take a NortekVector output file set (.dat and .vhd files only) and convert
    them into csv files with times and measurements
    
    @inputs
        dat - path to .dat file output by Nortek, containing PUV data
        vhd - path to .vhd file output by Nortek, containing start time
        csv - path to .csv file to save to
    """
    dat_cols = ["Burst","Ensemble","u","v","w",
                "str1","str2","str3",
                "snr1","snr2","snr3",
                "corr1","corr2","corr3",
                "p","analog1","analog2","checksum(1=failed)"]	

    df = pd.read_table(dat,sep='\s+',names=dat_cols)
    df = df[["u","v","w","p"]]
    gc.collect()
    
    mo, da, ye, ho, mi, se = open(vhd).readline().split(' ')[:6]
    start_time = pd.Timestamp(month = int(mo), day= int(da), year=int(ye), 
                              hour=int(ho), minute=int(mi), second=int(se))
    
    t0 = start_time.to_datetime64()
    timestep = np.timedelta64(int(1000*(sample_Hz**-1)),'ms')
    t = np.arange(t0, t0+len(df)*timestep, timestep)
    
    df['t'] = t
    df.to_csv(csv, index=False)

class PUV(object):
    """
    PUV object contains a dataframe with PUV data (among other stuff)
    and is tied to a specific CSV file. 
    
    Has the capacity to save header information associated with the file
    """
    
    def __init__(self, csv, sample_Hz=2):
        """
        Initialize this PUVCSV object with a .csv file
        """
        self.df = pd.read_csv(csv)
        self.filename = csv
        self.header = {'sample_Hz':sample_Hz}
        
        self.start_time = np.datetime64(self.df['t'].values[0])
        self.end_time = np.datetime64(self.df['t'].values[-1])
    
    @classmethod    
    def fromCSV(cls, csv):
        """
        Use an existing CSV file to initialize a PUV object
        """
        return cls(csv)
    
    @classmethod  
    def fromNortekVector(cls, dat, vhd, csv, sample_Hz=2):
        """
        Use NortekVectorConvert to initialize a PUV object
        """
        NortekVectorConvert(dat, vhd, csv, sample_Hz=sample_Hz)
        return cls(csv)

    def bottomPressureToElevation(self, burial_depth_start, burial_depth_end,
                                  pressure_units='depth'):
        """
        Convert from bottom pressure into sea level elevation with linear wave 
        theory
        """
        eta, h = linear_wave_convert.p_to_eta(self.df['p'].values,
                                              burial_depth_start, 
                                              burial_depth_end,
                                              dt=self.header['sample_Hz']**-1,
                                              pressure_units=pressure_units)
        self.df['eta'] = eta
        self.df['h'] = h
    
    def bottomCurrentToCurrent(self):
        pass
        
    def addCDIPBuoy(self, buoy_dat):
        """
        Add CDIP buoy z displacement measurements to the csv file from the 
        provided buoy_dat file
        
        Projects NaNs to fill gaps
        Overwrites existing data
        """
    
    def add(self, PUV):
        """
        Add another PUV to this one, concatenating in time
        
        Projects NaNs to fill gaps
        Overwrites existing data
        """
        
    def segment(self, start, end, csv):
        """
        Create a new PUV object with the times trimmed from start to end
        
        @params
            start - numpy datetime64 start time
            end   - numpy datetime64 end time
        @returns
            PUV object trimmed to these times specifically
        """
        t = np.array(self.df['t'].values, dtype='datetime64')
        self.df.iloc[np.where(np.logical_and(t >= start, t <= end))].to_csv(csv, index=False)
        return PUV.fromCSV(csv)
        
    def save(self, header=True):
        """
        Save, with header
        """
        self.df.to_csv(self.filename, index=False)

desired_times = [['2019-01-18T12:51:00'    , '2019-01-18T13:24:00'],
                 ['2019-01-18T13:21:00'    , '2019-01-18T13:54:00'],
                 ['2019-01-18T13:51:00'    , '2019-01-18T14:24:00'],
                 ['2019-01-18T14:21:00'    , '2019-01-18T14:54:00'],
                 ['2019-01-18T15:01:00'    , '2019-01-18T15:34:00'],
                 ['2019-01-18T15:51:00'    , '2019-01-18T16:24:00'],
                 ['2019-01-18T16:21:00'    , '2019-01-18T16:54:00'],
                 ['2019-01-18T16:50:00'    , '2019-01-18T17:23:00'],
                 ['2019-01-18T17:27:00'    , '2019-01-18T18:00:00']]

puv_raws = [['/media/reefbreakcopy/zdata/group/NortekVector/20190422_IB_South/IB-S02.dat',
             '/media/reefbreakcopy/zdata/group/NortekVector/20190422_IB_South/IB-S02.vhd'],
            ['/media/reefbreakcopy/zdata/group/NortekVector/20190422_IB_South/20190301-20190423/IB-S02.dat',
             '/media/reefbreakcopy/zdata/group/NortekVector/20190422_IB_South/20190301-20190423/IB-S02.vhd']]

puv_csv_path   = '/media/reefbreakcopy/zdata/group/NortekVector/20190422_IB_South/csvs/'
puv_julia_path = '/media/reefbreakcopy/zdata/group/NortekVector/20190422_IB_South/julia/'

deploy_times = [np.datetime64('2018-11-27T12:00:00.00').astype(int),
                np.datetime64('2018-11-27T12:00:00.00').astype(int)]
deploy_burial_depths = [0.73, 0.56]

def main():
    """
    For each desired time, create a PUV with CDIP buoy information, shoal, and save
    """
    # Step 1: Convert all the NortekVector data into CSV files
    csv0 = puv_csv_path + '2018_11-2019_03.csv'
    csv1 = puv_csv_path + '2019_03-2019_04.csv'
    csv2 = puv_csv_path + 'julia.csv'
    
    NortekVectorConvert(puv_raws[0][0], puv_raws[0][1], csv0)
    NortekVectorConvert(puv_raws[1][0], puv_raws[1][1], csv1)
    
    # Step 2: Initialize PUV object and add water elevation, water height, and surface currents
    puv = PUV.fromCSV(csv0)
    
    # Optionally select right away a subset of data to work on
    # puv = puv.segment(np.datetime64(desired_times[0][0]),
    #                   np.datetime64(desired_times[-1][-1]),
    #                   csv2)
    
    start_depth, end_depth = np.interp([puv.start_time.astype(int),
                                        puv.end_time.astype(int)],
                                       deploy_times, deploy_burial_depths)
    
    puv.bottomPressureToElevation(start_depth, end_depth)
    puv.save()
    
    # Step 3: Add the buoy data and shoal/deshoal 
    # [not implemented]
    
    # Step 4: Select segments corresponding to the desired_times list and save
    for timerange in desired_times:
        ti, tf = np.datetime64(timerange[0]), np.datetime64(timerange[1])
        filename = puv_julia_path + timerange[0] + '.csv'
        puv.segment(ti, tf, filename)
 
if __name__ == '__main__':
    main()