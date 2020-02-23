# This package contains the libRadtran functions for use in redSun\
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

def write_atmos(alt, pres, temp, air, O3_pp, O2_pp, H2O_pp, CO2_pp, NO2_pp, filename='', path=''):
    filenameFull = path+filename+'atmos.DAT'
    atmos_profile = pd.DataFrame(np.fliplr(np.array([alt, pres, temp, air, O3_pp, O2_pp, H2O_pp, CO2_pp, NO2_pp])))
    atmos_profile = atmos_profile.transpose()
    # atmos_profile = atmos_profile[atmos_profile[0] > 0]
    atmos_profile.to_csv(filenameFull, sep='\t', index=False, header=False)
    return filenameFull

def write_dust(alt, dust_conc, reff_dust, filename='', path=''):
    filenameFull = path+filename+'dust.DAT'
    dust_profile = pd.DataFrame(np.fliplr(np.array([alt, dust_conc, reff_dust])))
    dust_profile = dust_profile.transpose()
    dust_profile = dust_profile[dust_profile[2] > 0.003228]
    dust_profile = dust_profile[dust_profile[0] > 0]
    dust_profile.to_csv(filenameFull, sep='\t', index=False, header=False)
    return filenameFull

def write_cloud(alt, iwc, reff_ice, filename='', path=''):
    filenameFull = path+filename+'cloud.DAT'
    cloud_profile = pd.DataFrame(np.fliplr(np.array([alt, iwc, reff_ice])))
    cloud_profile = cloud_profile.transpose()
    cloud_profile = cloud_profile[cloud_profile[2] > 0.003228]
    cloud_profile = cloud_profile[cloud_profile[0] > 0]
    cloud_profile.to_csv(filenameFull, sep='\t', index=False, header=False)
    return filenameFull

def write_flux(lambdaa,flux_corrected,filename='', path=''):
    filenameFull = path+filename+'flux.DAT'
    flux_profile = pd.DataFrame(np.array([lambdaa,flux_corrected]))
    flux_profile = flux_profile.transpose()
    flux_profile.to_csv(filenameFull, sep='\t', index=False, header=False)
    return filenameFull

def write_input(albedo = .2, profilename = 'input.inp', id='', wl_min=300.5,wl_max=4000,atmos_filename='atmos.DAT', dust_filename='dust.DAT', cloud_filename='cloud.DAT', flux_filename='flux.DAT', datum=0.0):
    profile ='''# libRadtran Calc test
# choose wavelength range for computation
wavelength {wl_min} {wl_max}
# load atmosphere profile
atmosphere_file {atmos_filename}
# update null mixing ratios
mixing_ratio CH4 0.0
mixing_ratio N2O 0.0
mixing_ratio F11 0.0
mixing_ratio F12 0.0
mixing_ratio F22 0.0
altitude {datum}
# load solar profile
# corrected for Sun-Mars Distance
# corrected for geometry
source solar {flux_filename}
# setup cloud profile (assuming water clouds)
ic_file 1D {cloud_filename}
ic_properties MieCalc/cloud.mie.cdf interpolate
# setup dust profile (using aerosol type for dust)
profile_file dust 1D {dust_filename}
profile_properties dust MieCalc/dust.mie.cdf interpolate
# reset earth_radius to Martian radius in [km]
earth_radius 3389.5
# choose radiative transfer solver
rte_solver disort pseudospherical
pseudospherical
# choose number of streams
number_of_streams 6
# define output
output_user lambda edir eglo edn eup enet esum
# choose albedo
albedo {albedo}
#verbose
quiet
'''
    text_file = open(id+profilename, "w")
    text_file.write(profile.format(wl_min=wl_min,wl_max=wl_max,
                                   atmos_filename=id+atmos_filename, dust_filename=id+dust_filename,
                                   cloud_filename=id+cloud_filename, flux_filename=id+flux_filename,
                                   albedo = albedo, datum=datum))

def call_libRadtran(input_filename='input.inp', id='', output_filename='output.DAT'):
    args = 'uvspec < ' + id+input_filename + " > " + id+output_filename
    p = sp.Popen(args, shell=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    p.wait()
    preturn = p.returncode
    return(output_filename,preturn)

def read_libRadtran(libRadtran_filename='output.DAT', id=''):
    df = pd.read_csv(id+libRadtran_filename, sep='\s+', names=['lambda', 'edir', 'eglo', 'edn', 'eup', 'enet', 'esum'])
    lambdaa = df['lambda']
    edir = df['edir']
    eglo = df['eglo']
    edn = df['edn']
    eup = df['eup']
    enet = df['enet']
    esum = df['esum']

    vals = [lambdaa, edir, eglo, edn, eup, enet, esum]
    return vals
