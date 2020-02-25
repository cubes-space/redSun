## Import libraries
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cons
import pandas as pd
import pickle
from scipy.interpolate import interp1d
import netCDF4 as net
import subprocess as sp
import scipy
import scipy.integrate as integrate
import math
import traceback
import random
import string
import itertools
import multiprocessing as mp
import xarray as xr
from fmcd import solarzenithangle,sunmarsdistance

## Import additional redSun functions
from rs_pv_func import *
from rs_lrt_func import *

## Try importing the Mars Climate Database
try:
    from mcd3 import mcd
    print('MCD Imported')
except:
    print('MCD Not Imported')

## Define Class Objects

class Parameters():
    '''
        The Parameters class takes in a filename, directory, and parameter set identification to construct an object of
    initial parameters from which grids are constructed
    '''
    def __init__(self, filename='parameter_sets.xlsx', directory='parameters/', parameter_set='test_value', from_file=True):
        df = pd.read_excel(directory + filename)
        param_dict = dict(zip(df['parameter'], df[parameter_set]))
        self.__dict__.update((k, v) for k, v in param_dict.items())

class Enviornment():
    '''
    The Environment class takes in **kwards similar to the Parameters() class then calls Parameters() to construct the
    initial enviornment object into which all downstream objects are loaded. This is the base object for redSun.
    '''
    def __init__(self, filename='parameter_sets.xlsx', directory='parameters/', parameter_set='test_value', from_file=True):
        df = pd.read_excel(directory + filename)
        param_dict = dict(zip(df['parameter'], df[parameter_set]))
        grid =  xr.Dataset()
        param_dict.update((i,np.arange(param_dict[i+'_min'], param_dict[i+'_max'] + 1, param_dict[i+'_step'])) for i in ['lat','lon','hr','ls'])
        self.__dict__.update((k, v) for k, v in param_dict.items())

        for i in ['lat','lon','hr','ls']:
            grid.coords[i] = (i,param_dict[i])

        flux_dict = get_extraFlux()

        grid.coords['wl'] = ('wl',flux_dict['lambda'])
        grid.coords['wl'].attrs['unit'] = 'nm'

        grid.coords['level'] = ('level', np.arange(0,20))
        grid.coords['level'].attrs['unit'] = 'km'
        self.grid = grid



    # def initialize

    def initialize_grid(self, filename='mcd_grid_variables.xlsx', directory='parameters/'):
        df_data = pd.read_excel(directory+filename,sheet_name='data')

        unit_dict = dict(zip(df_data['var_name'],df_data['units']))
        coord_dict = dict(zip(df_data['var_name'],df_data['dims']))

        encoding_dict = {}
        encoding_val = {'zlib':True,'_FillValue':0.0}

        for vari in df_data['var_name']:
            coords = coord_dict[vari].split(',')
            zi = [len(self.grid.coords[i]) for i in coords]
            self.grid[vari] = (tuple(coords), np.zeros(zi))
            self.grid[vari].attrs['units'] = unit_dict[vari]
            encoding_dict[vari] = encoding_val

        df_coord = pd.read_excel(directory + filename, sheet_name='coords')
        unit_dict.update(dict(zip(df_coord['var_name'],df_coord['units'])))

        for vari in df_coord['var_name']:
            self.grid.coords[vari].attrs['units'] = unit_dict[vari]
            encoding_dict[vari] = encoding_val

        self.encoding_dict = encoding_dict

    #     n = 0
    #     for i in df_data['var_name']:
    #         dims = df_data['dims'][n].split(',')
    #         lens = np.zeros(len(dims))
    #         m = 0
    #         for j in dims:
    #             lens[m] = int(len(self.grid.coords[j]))
    #             m = m + 1
    #         print(tuple(dims))
    #         print(np.array(lens))
    #         # self.grid[i] = (tuple(dims),np.zeros(tuple(lens)))
    #         n = n + 1
    #         return lens

    def calc_mcd_grid(self, scenario=1):
        '''
        Calculate the grid values from MCD using loaded parameters and store in grid xarray of Enviornment object
        :return:
        :rtype:
        '''
        ## Print the parameter sweep
        parameter_total = np.prod([len(self.grid.coords[i]) for i in ['lat', 'lon', 'ls', 'hr']])
        print('Total Number of Parameters in Sweep:' + str(parameter_total))

        ## Initialize MCD
        req = init_mcd(scenario=scenario)
        self.grid.coords['level'] = req.getextvar(3)
        self.grid.coords['level'].attrs['unit'] = 'km'


        ## Get extraterrestrial flux
        flux_dict = get_extraFlux()
        self.grid.coords['wl'] = ('wl',flux_dict['lambda'])
        self.grid.coords['wl'].attrs['unit'] = 'nm'
        extra_flux = flux_dict['F152']

        ## Define vector for loop
        coordv = [(self.grid.coords[i].values) for i in ['lat', 'lon', 'ls', 'hr']]
        [latv, lonv, lsv, hrv] = coordv
        coordvi = [(np.arange(0,len(i))) for i in coordv]
        [latvi, lonvi, lsvi, hrvi] = coordvi


        ## Loop over the lat,lon to get albedo data
        surf = xr.open_dataset('extras/surface.nc')
        alb = surf['albedo']
        self.grid['albedo'] = alb.interp(latitude=latv, longitude=lonv)

        ## Loop over the lat,lon,hr,time to get single point data along level
        for lsi in lsvi:
            for lati in latvi:
                for hri in hrvi:
                    for loni in lonvi:
                        mcd_dict = call_mcd(req,latvi[lati],lonvi[loni],lsvi[lsi],hrvi[hri],scenario=scenario)
                        # return mcd_dict
                        for vari in mcd_dict.keys():
                            if vari in ['alt_datum']:
                                self.grid[vari][lati,loni,:] = mcd_dict[vari]
                            elif vari in ['flux_dw_sw','flux_dw_lw','flux_uw_sw','flux_uw_lw']:
                                self.grid[vari][lati, loni, lsi, hri] = mcd_dict[vari][0]
                            elif vari not in ['sza','albedo','solar_corr','irr_TOA']:
                                self.grid[vari][lati, loni, :, lsi, hri] = mcd_dict[vari]

        ## Loop over lat,ls,hr to calculate solar zenith angle, then calculate flux @TOA
        for lsi in lsvi:
            r = sunmarsdistance(lsvi[lsi])
            for lati in latvi:
                for hri in hrvi:
                    sza = solarzenithangle(latvi[lati],lsvi[lsi],hrvi[hri])
                    self.grid['sza'][lati, lsi, hri] = sza
                    solar_corr = np.clip(np.cos(np.deg2rad(sza)), a_min=0, a_max=None) * ((1.52368**2)/(r**2))
                    self.grid['irr_TOA'][lati, lsi, hri, :] = solar_corr * extra_flux








## Define MCD-related functions

def init_mcd(scenario=1):
    '''
    This function initializes MCD. No input is required.
    The output is the MCD object called 'req'
    :param scenario:
    :type scenario:
    :return:
    :rtype:
    '''
    req = mcd()
    req.dust = scenario
    req.loct = 12 # local time
    req.xz = 0 # vertical coordinate
    req.xdate = 0 # areocentric longitude
    req.ack = ''
    req.fixedlt = True
    req.profile()
    req.update()
    return req

def call_mcd(req,lat,lon,ls,hr,scenario=1):
    '''
    This function calls mcd for a given lat,lon,ls,hr. It defaults to scenario=1 which is the standard
    climate scenario.
    :param req:
    :type req:
    :param lat:
    :type lat:
    :param lon:
    :type lon:
    :param ls:
    :type ls:
    :param hr:
    :type hr:
    :param scenario:
    :type scenario:
    :return:
    :rtype:
    '''
    ## Get MCD specifics
    K = cons.Boltzmann
    Av = cons.Avogadro
    req.dust = scenario
    req.lat = lat
    req.lon = lon
    req.xdate = ls
    req.loct = hr
    req.fixedlt = True
    req.profile()
    req.update()

    ## Define data variables of interest by ID
    datavars = [91, 93, 63, 62, 57, 42, 45, 44, 39, 38, 32, 31, 30, 33, 92, 2]

    ## Get data, store in vals variable
    vals = [req.getextvar(var) for var in datavars]

    ## Reformat data
    alt = vals[15]*1e-3
    dens = vals[14]
    pres = vals[0] * 1e-2
    temp = vals[1]
    air = (vals[0]/(K*vals[1]))*1e-6
    O3_pp = vals[2] * air
    O2_pp = vals[3] * air
    CO2_pp = vals[4] * air
    H2O_pp = vals[5] * air
    NO2_pp = vals[4] * 0
    reff_ice = vals[6] * 1e6
    iwc = air * vals[7] * (18/Av) * 1e3
    reff_dust = vals[8] * 1e6
    dust_conc = vals[9] * dens * 1e3
    flux_dw_sw = vals[10]
    flux_dw_lw = vals[11]
    flux_uw_sw = vals[12]
    flux_uw_lw = vals[13]

    ## Format as dictionary and return values
    keys = ['alt_datum','pressure','temperature','air_density','pp_O3','pp_O2','pp_H2O','pp_CO2','pp_NO2','content_dust','reff_dust','content_ice','reff_ice','flux_dw_sw','flux_dw_lw','flux_uw_sw','flux_uw_lw']
    vals = [alt, pres, temp, air, O3_pp, O2_pp, H2O_pp, CO2_pp, NO2_pp, dust_conc, reff_dust, iwc, reff_ice, flux_dw_sw, flux_dw_lw, flux_uw_sw, flux_uw_lw]

    return dict(zip(keys,vals))

def get_extraFlux(d=1.52368, directory='extras/', filenameIn = 'E490_00.xlsx'):
    '''
    Get the extraterrestrial spectrum from 'E490_00.xls'.
    Fix the units to wavelength in [nm] and flux in [W/m^2 nm]
    Correct for the average Mars-Sun distance d=1.52 [AU]
    Convert and return numpy arrays of corrected wavelength lambdaa and flux at average distance F152
    '''

    # import E490 spectrum
    df = pd.read_excel(directory+filenameIn)

    # convert microns to nm
    df['Wavelength, microns'] = df['Wavelength, microns']*1e3
    df['E-490 W/m2/micron'] = df['E-490 W/m2/micron']*1e-3

    # change column names
    df.columns = ['wavelength[nm]', 'flux [W/m^2 nm]']

    # define and implement distance correction
    corr = 1/(d**2)

    # convert
    lambdaa = np.array(df['wavelength[nm]'])
    F152 = np.array(df['flux [W/m^2 nm]'])*corr

    keys = ['lambda', 'F152']
    vals = [lambdaa[181:1522],F152[181:1522]]

    # export file
    return dict(zip(keys,vals))