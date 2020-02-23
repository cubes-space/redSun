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

## Import additional redSun functions
from rs_pv_func import *
from rs_geometry_func import *
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
        print('test')
        df = pd.read_excel(directory + filename)
        param_dict = dict(zip(df['parameter'], df[parameter_set]))
        param_dict['lat'] = np.arange(param_dict['latitude_min'], param_dict['latitude_max'] + 1, param_dict['latitude_step'])
        param_dict['lon'] = np.arange(param_dict['longitude_min'], param_dict['longitude_max'] + 1,param_dict['longitude_step'])
        param_dict['hr'] = np.arange(param_dict['time_min'], param_dict['time_max'] + 1,param_dict['time_step'])
        param_dict['ls'] = np.arange(param_dict['ls_min'], param_dict['ls_max'] + 1, param_dict['ls_step'])
        self.__dict__.update((k, v) for k, v in param_dict.items())

class Enviornment():
    '''
    The Environment class takes in **kwards similar to the Parameters() class then calls Parameters() to construct the
    initial enviornment object into which all downstream objects are loaded. This is the base object for redSun.
    '''
    def __init__(self, filename='parameter_sets.xlsx', directory='parameters/', parameter_set='test_value', from_file=True):
        self.parameters = Parameters(filename=filename, directory=directory, parameter_set=parameter_set, from_file=from_file)

    def initialize_grid(self, filename='grid_variables.xlsx', directory='parameters/'):
        data_dict = {}
        coord_dict {}

        coord_dict.update()


## Define Functions

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
    keys = ['altitude','pressure','temperature','air dens','O3_pp','O2_pp','H2O_pp','CO2_pp','NO2_pp','dust_conc','reff_dust','iwc','reff_ice','flux_dw_sw','flux_dw_lw','flux_uw_sw','flux_uw_lw']
    vals = [alt, pres, temp, air, O3_pp, O2_pp, H2O_pp, CO2_pp, NO2_pp, dust_conc, reff_dust, iwc, reff_ice, flux_dw_sw, flux_dw_lw, flux_uw_sw, flux_uw_lw]

    return dict(zip(keys,vals))

