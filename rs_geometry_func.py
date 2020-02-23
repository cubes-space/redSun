# This package contains the geometry functions for use in redSun
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

def calc_r(d=1.52368, e=0.0934, L_s=np.deg2rad(0), L_sp=np.deg2rad(250.99)):
    '''
    This function takes in 4 arguments (in the following order):
    * d: mean distance from Sun to Mars [AU] (defaults to 1.52AU or 227388763464 m)
    * e: the eccentricity of the orbit of Mars [unitless] (defaults to 0.0934)
    * L_s: the areocentric longitude [rad] which is a measure of orbital position or season (default to 0)
    * L_sp:  the solar longitude at the perihelion [rad] (default to np.rad(251))

    These values are used to calculate the Sun-Mars distance [AU] (or radius r) at any point in the Martian orbit
    '''
    r = (d*(1-e**2))/(1+e*np.cos(L_s-L_sp))
    return r

def calc_mu_0(t, epsilon, L_s, phi, P):
    '''
    This function takes in X arguments (in the following order):
    * t: time from local noon [sec] (default to 0)
    * epsilon: the Martian obliquity [rad]
    * L_s: the areocentric longitude [rad]
    * phi: the latitude [deg] (default to 0)
    * P: period or length of a martian sol [sec]

    These values are used to calculate the cosine of the zenith angle mu_0 [unitless]
    '''
    mu_0 = np.sin(phi)*np.sin(epsilon)*np.sin(L_s) + (np.cos(phi)*np.cos((2*np.pi*t)/P)*
                                                      ((1-(np.sin(epsilon)**2)*(np.sin(L_s)**2)))**(0.5))

    return mu_0

def get_correctFlux_TOA(F152, d=1.52368, e=0.0934, L_s=0, L_sp=250.99, t=0, epsilon=25.1919, phi=0, P=88775.244147, timedep=False):
    if timedep == True:
        # convert from MCD true time at 0 in hrs to time from local noon in seconds
        t = np.absolute((t-12)*60*60)

        # convert angular parameters from [deg] to [rad]
        L_s = np.deg2rad(L_s)
        L_sp = np.deg2rad(L_sp)
        epsilon = np.deg2rad(epsilon)
        phi = np.deg2rad(phi)

        # calculate the martian radius
        r = calc_r(d, e, L_s, L_sp)
        # calculate the cosine of the zenith angle
        mu_0 = calc_mu_0(t, epsilon, L_s, phi, P)
        mu_0 = mu_0.clip(min=0)


        #correct flux for Mars's r and d
        F = F152* mu_0 * (d**2.0)/(r**2.0)
        gF = F152 * ( np.sin(phi) * np.sin(epsilon) * np.sin(L_s) + np.cos(phi)*np.cos((2*np.pi*t)/(P)) * (1-(np.sin(epsilon)**2)*(np.sin(L_s)**2))*((1+e*np.cos(L_s-L_sp))/(1-e**2)))**2

        keys = ['corrected_flux','mu_0']
        vals = [F, mu_0]


    return dict(zip(keys,vals))
