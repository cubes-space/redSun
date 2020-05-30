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
from fmcd import solarzenithangle,sunmarsdistance,ls2sol


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
        grid =  xr.Dataset()
        df = pd.read_excel(directory + filename)
        param_dict = dict(zip(df['parameter'], df[parameter_set]))
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


    def initialize_grid(self, filename='mcd_grid_variables.xlsx', directory='parameters/'):
        '''
        This function initializes the grid based on variables in filename
        :param filename:
        :type filename:
        :param directory:
        :type directory:
        :return:
        :rtype:
        '''

        ## Get mcd related variables from filename
        df_data = pd.read_excel(directory+filename,sheet_name='data')

        ## Construct unit and coord dictionaries with units and dimensions
        unit_dict = dict(zip(df_data['var_name'],df_data['units']))
        coord_dict = dict(zip(df_data['var_name'],df_data['dims']))

        ## Initialize encoding dictionary
        encoding_dict = {}
        encoding_val = {'zlib':True,'_FillValue':0.0}


        ## Loop through variables to initialize dataset as zero arrays and with encoding dict
        for vari in df_data['var_name']:
            coords = coord_dict[vari].split(',')
            zi = [len(self.grid.coords[i]) for i in coords]
            self.grid[vari] = (tuple(coords), np.zeros(zi))
            self.grid[vari].attrs['units'] = unit_dict[vari]
            encoding_dict[vari] = encoding_val

        ## Repead process for coords
        df_coord = pd.read_excel(directory + filename, sheet_name='coords')
        unit_dict.update(dict(zip(df_coord['var_name'],df_coord['units'])))

        ## Loop through coords to initialize dataset as zero arrays and with encoding dict
        for vari in df_coord['var_name']:
            self.grid.coords[vari].attrs['units'] = unit_dict[vari]
            encoding_dict[vari] = encoding_val

        ## Store encoding dict in object
        self.encoding_dict = encoding_dict

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
                        mcd_dict = call_mcd(req,latv[lati],lonv[loni],lsv[lsi],hrv[hri],scenario=scenario)
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
            for lati in latvi:
                for hri in hrvi:
                    sol = ls2sol(lsv[lsi])
                    sol = sol + hrv[hri]/24
                    ls_corr = homemade_sol2ls(sol)
                    r = sunmarsdistance(ls_corr)
                    sza = solarzenithangle(latv[lati],ls_corr,hrv[hri])
                    self.grid['sza'][lati, lsi, hri] = sza
                    solar_corr = np.clip(np.cos(np.deg2rad(sza)), a_min=0, a_max=None) * ((1.52368**2)/(r**2))
                    self.grid['solar_corr'][lati, lsi, hri] = solar_corr
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
    pp_O3 = vals[2] * air
    pp_O2 = vals[3] * air
    Cpp_O2 = vals[4] * air
    pp_H2O = vals[5] * air
    Npp_O2 = vals[4] * 0
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
    vals = [alt_datum, pres, temp, air, pp_O3, pp_O2, pp_H2O, pp_O2, pp_O2, dust_conc, reff_dust, iwc, reff_ice, flux_dw_sw, flux_dw_lw, flux_uw_sw, flux_uw_lw]

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

def homemade_sol2ls(sol):
    '''
    This function takes in a variable sol and returns the aerocentric longitude ls
    :param sol:
    :type sol:
    :return:
    :rtype:
    '''
    t_peri = 485.35
    N_s = 668.6
    Ls_peri = 250.99
    D_s = sol
    e=0.0934
    M = 2 * np.pi * ((D_s - t_peri) / N_s)
    E = calc_E(M)
    nu = 2 * np.arctan((np.sqrt(((1 + e) / (1 - e))) * np.tan(E / 2)))
    ls = np.mod((nu*180/np.pi)+Ls_peri,360)
    return ls


def calc_E(M,e=0.0934,n=1000):
    '''
    This little function takes in some mean anomaly M and uses it as the starting point to calculate
    the ecentric anomaly E using fixed point iteration
    :param M:
    :type M:
    :param e:
    :type e:
    :param n:
    :type n:
    :return:
    :rtype:
    '''
    E = M
    for k in range(0,n):
        E = M + e*np.sin(E)
    return E

# def parse_netcdf_by_index(inds,filename_netCDF='Initial_Grid.nc'):
#
#     ## Specify indices
#     # ind_lat = inds[1]
#     # ind_lon = inds[2]
#     # ind_hr = inds[3]
#     # ind_ls = inds[0]
#     d = dict(ls=inds[0], lat=inds[1], lon=inds[2], hr=inds[3])
#     ## load data from netcdf with dask and xarray
#     ds = xr.open_dataset(filename_netCDF, decode_cf=False)
#     ds = ds[d]
#     l = list(ds.keys())
#     for i in range(0,len(l)):
#         exec('%s=ds["%s"].values' % (l[i], l[i]))
#     args = [pressure,temperature,air_density,pp_O3,pp_O2,pp_CO2,pp_H2O,alt_datum,reff_ice,content_ice,reff_dust,content_dust,flux_dw_sw,irr_TOA,lambdaa,albedo]
#     return args

def paralell_lrt(inds=[0,10,10,6], filename_netCDF='Initial_Grid.nc'):

    ## load data from netcdf with dask and xarray
    d = dict(ls=inds[0], lat=inds[1], lon=inds[2], hr=inds[3], latitude=inds[1], longitude=inds[2])
    ds = xr.open_dataset(filename_netCDF, decode_cf=False)
    ds = ds[d]
    # l = list(ds.keys())
    # for i in range(0,len(l)):
    #     print(l[i])
    #     eval('%s=ds["%s"].values' % (l[i], l[i]))

    # irr_TOA = irr_TOA*1e3
    vals = 0
    atp_set = 3
    try:
        id = '__'+id_generator(size=10)
        ind_lat = inds[1]
        ind_lon = inds[2]
        ind_hr = inds[3]
        ind_ls = inds[0]
        atp = 0
        while atp < atp_set:
            if ds['flux_dw_sw'].values > 0:
                if np.sum(ds['irr_TOA'].values*1e3) > 0:
                    write_atmos(ds['alt_datum'].values,ds['pressure'].values,ds['temperature'].values,ds['air_density'].values, ds['pp_O3'].values, ds['pp_O2'].values, ds['pp_H2O'].values, ds['pp_O2'].values, ds['pp_O2'].values*0.0, filename=id)
#                     print(flux_dw_sw)
                    write_flux(ds['wl'].values, ds['irr_TOA'].values*1e3, filename=id)
                    write_cloud(ds['alt_datum'].values, ds['content_ice'].values,ds['reff_ice'].values, filename=id)
                    write_dust(ds['alt_datum'].values, ds['content_dust'].values, ds['reff_dust'].values, filename=id)
                    write_input(albedo=ds['albedo'].values, datum=ds['alt_datum'].values[0]+.001, id=id)
                    libRadtran_return = call_libRadtran(id=id)
                    if libRadtran_return[1] == 0:
                        vals = read_libRadtran(id=id)
                        [lambdaa, edir, eglo, edn, eup, enet, esum] = vals
                        [edir, eglo, edn, eup, enet, esum] = [edir*1e6, eglo*1e6, edn*1e6, eup*1e6, enet*1e6, esum*1e6]
                        mcd_lrt_error = calc_mcd_lrt_error(lambdaa*1e-9,eglo,ds['flux_dw_sw'].values)
                        vals = SQlim_1bg(np.array(lambdaa)*1e-9, np.array(eglo), ds['temperature'].values[0])
                        [j1_bg_vec,j1_etaPV,j1_etaPEC_H2,j1_etaPEC_NH3,j1_etaPEC_AA, j1_max_etaPV,j1_max_etaPEC_H2,j1_max_etaPEC_NH3,j1_max_etaPEC_AA,j1_bg_PVmax,j1_bg_H2max,j1_bg_NH3max,j1_bg_AAmax] = vals
                        vals = SQlim_2bg(np.array(lambdaa)*1e-9, np.array(eglo), ds['temperature'].values[0])
                        [j2_bg1_vec,j2_bg2_vec,j2_etaPV_2bg,j2_etaPEC_H2_2bg,j2_etaPEC_NH3_2bg,j2_etaPEC_AA_2bg, j2_max_etaPV_2bg,j2_max_etaPEC_H2_2bg,j2_max_etaPEC_NH3_2bg,j2_max_etaPEC_AA_2bg, j2_bg1_PVmax,j2_bg2_PVmax,j2_bg1_H2max,j2_bg2_H2max,j2_bg1_NH3max,j2_bg2_NH3max,j2_bg1_AAmax,j2_bg2_AAmax] = vals
                        vals = SQlim_3bg(np.array(lambdaa)*1e-9, np.array(eglo), ds['temperature'].values[0])
                        [j3_bg1_vec,j3_bg2_vec,j3_bg3_vec,j3_etaPV_3bg,j3_max_etaPV_3bg,j3_bg1max,j3_bg2max,j3_bg3max,j3_J_bg1,j3_J_bg2,j3_J_bg3,j3_v] = vals
                        vals = 0
                        vals = [lambdaa, j1_bg_vec, j2_bg1_vec, j2_bg2_vec, j3_bg1_vec, j3_bg2_vec, j3_bg3_vec, id, edir, edn, eup, eglo, mcd_lrt_error, j1_etaPV,j1_etaPEC_H2,j1_etaPEC_NH3,j1_etaPEC_AA, j1_max_etaPV,j1_max_etaPEC_H2,j1_max_etaPEC_NH3,j1_max_etaPEC_AA,j1_bg_PVmax,j1_bg_H2max,j1_bg_NH3max,j1_bg_AAmax,j2_etaPV_2bg,j2_etaPEC_H2_2bg,j2_etaPEC_NH3_2bg,j2_etaPEC_AA_2bg, j2_max_etaPV_2bg,j2_max_etaPEC_H2_2bg,j2_max_etaPEC_NH3_2bg,j2_max_etaPEC_AA_2bg, j2_bg1_PVmax,j2_bg2_PVmax,j2_bg1_H2max,j2_bg2_H2max,j2_bg1_NH3max,j2_bg2_NH3max,j2_bg1_AAmax,j2_bg2_AAmax,j3_etaPV_3bg,j3_max_etaPV_3bg,j3_bg1max,j3_bg2max,j3_bg3max,j3_J_bg1,j3_J_bg2,j3_J_bg3]
                        cleaner(id=id)
#                         if atp > 0:
#                             print('Passed ' + str(inds)+ ' after ' + str(atp) + ' attemps')
                        return inds,vals
                    else:
                        atp = atp + 1
#                         print(str(atp) + ' LRT Error at ' +  str(inds))
                        time.sleep(3)
                else:
                    return None
            else:
                return None
    except Exception:
        traceback.print_exc()
        cleaner(id=id)
    print('total fail at ' + str(inds))
    return None

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    '''Function produces an ID string of a size [default=6]'''
    return ''.join(random.choice(chars) for _ in range(size))

def cleaner(id='XXXXXX', directory=''):
    dirList = os.listdir()   # Use os.listdir() if want current directory
    dirList.sort()
    for i in dirList:
        if id in str(i):
            os.remove(i)

def calc_mcd_lrt_error(lambdaa,spec_flux,surf_flux):
    int_spec_flux = np.trapz(spec_flux, x=lambdaa)
    # print('mcd flux: ' + str(surf_flux))
    # print('lrt flux: ' + str(int_spec_flux))
    error = np.abs(int_spec_flux - surf_flux)/surf_flux
    return error

def save_singlepoint_netcdf(inds,vals,directory='singlePoints/'):
    [wl, j1_bg, j2_bg1, j2_bg2, j3_bg1, j3_bg2, j3_bg3, id, edir, edn, eup, eglo, mcd_lrt_error, j1_etaPV,j1_etaPEC_H2,j1_etaPEC_NH3,j1_etaPEC_AA, j1_max_etaPV,j1_max_etaPEC_H2,j1_max_etaPEC_NH3,j1_max_etaPEC_AA,j1_bg_PVmax,j1_bg_H2max,j1_bg_NH3max,j1_bg_AAmax,j2_etaPV_2bg,j2_etaPEC_H2_2bg,j2_etaPEC_NH3_2bg,j2_etaPEC_AA_2bg, j2_max_etaPV_2bg,j2_max_etaPEC_H2_2bg,j2_max_etaPEC_NH3_2bg,j2_max_etaPEC_AA_2bg, j2_bg1_PVmax,j2_bg2_PVmax,j2_bg1_H2max,j2_bg2_H2max,j2_bg1_NH3max,j2_bg2_NH3max,j2_bg1_AAmax,j2_bg2_AAmax,j3_etaPV_3bg,j3_max_etaPV_3bg,j3_bg1max,j3_bg2max,j3_bg3max,j3_J_bg1,j3_J_bg2,j3_J_bg3] = vals

    encodingDict = {
        'edir': {'zlib': True,  '_FillValue':0.0},
        'edn': {'zlib': True,  '_FillValue':0.0},
        'eup': {'zlib': True,  '_FillValue':0.0},
        'eglo': {'zlib': True,  '_FillValue':0.0},
        'mcd_lrt_error': {'zlib': True,  '_FillValue':0.0},
        'j1_etaPV': {'zlib': True,  '_FillValue':0.0},
        'j1_etaPEC_H2': {'zlib': True,  '_FillValue':0.0},
        'j1_etaPEC_NH3': {'zlib': True,  '_FillValue':0.0},
        'j1_etaPEC_AA': {'zlib': True,  '_FillValue':0.0},
        'j1_max_etaPV': {'zlib': True,  '_FillValue':0.0},
        'j1_max_etaPEC_H2': {'zlib': True,  '_FillValue':0.0},
        'j1_max_etaPEC_NH3': {'zlib': True,  '_FillValue':0.0},
        'j1_max_etaPEC_AA': {'zlib': True,  '_FillValue':0.0},
        'j1_bg_PVmax': {'zlib': True,  '_FillValue':0.0},
        'j1_bg_H2max': {'zlib': True,  '_FillValue':0.0},
        'j1_bg_NH3max': {'zlib': True,  '_FillValue':0.0},
        'j1_bg_AAmax': {'zlib': True,  '_FillValue':0.0},
        'j2_etaPV_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_etaPEC_H2_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_etaPEC_NH3_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_etaPEC_AA_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_max_etaPV_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_max_etaPEC_H2_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_max_etaPEC_NH3_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_max_etaPEC_AA_2bg': {'zlib': True,  '_FillValue':0.0},
        'j2_bg1_PVmax': {'zlib': True,  '_FillValue':0.0},
        'j2_bg2_PVmax': {'zlib': True,  '_FillValue':0.0},
        'j2_bg1_H2max': {'zlib': True,  '_FillValue':0.0},
        'j2_bg2_H2max': {'zlib': True,  '_FillValue':0.0},
        'j2_bg1_NH3max': {'zlib': True,  '_FillValue':0.0},
        'j2_bg2_NH3max': {'zlib': True,  '_FillValue':0.0},
        'j2_bg1_AAmax': {'zlib': True,  '_FillValue':0.0},
        'j2_bg2_AAmax': {'zlib': True,  '_FillValue':0.0},
        'j3_etaPV_3bg': {'zlib': True,  '_FillValue':0.0},
        'j3_max_etaPV_3bg': {'zlib': True,  '_FillValue':0.0},
        'j3_bg1max': {'zlib': True,  '_FillValue':0.0},
        'j3_bg2max': {'zlib': True,  '_FillValue':0.0},
        'j3_bg3max': {'zlib': True,  '_FillValue':0.0},
        'j3_J_bg1': {'zlib': True,  '_FillValue':0.0},
        'j3_J_bg2': {'zlib': True,  '_FillValue':0.0},
        'j3_J_bg3': {'zlib': True,  '_FillValue':0.0}
    }
    d = {
        'wl': {'dims': ('wl'), 'data': wl},
        'j1-bg': {'dims': ('j1-bg'), 'data': j1_bg},
        'j2-bg1': {'dims': ('j2-bg1'), 'data': j2_bg1},
        'j2-bg2': {'dims': ('j2-bg2'), 'data': j2_bg2},
        'j3-bg1': {'dims': ('j3-bg1'), 'data': j3_bg1},
        'j3-bg2': {'dims': ('j3-bg2'), 'data': j3_bg2},
        'j3-bg3': {'dims': ('j3-bg3'), 'data': j3_bg3},
        'lat': {'dims': (), 'data': inds[1]},
        'lon': {'dims': (), 'data': inds[2]},
        'ls': {'dims': (), 'data': inds[0]},
        'hr': {'dims': (), 'data': inds[3]},
        'edir': {'dims': ('wl'), 'data': edir},
        'edn': {'dims': ('wl'), 'data': edn},
        'eup': {'dims': ('wl'), 'data': eup},
        'eglo': {'dims': ('wl'), 'data': eglo},
        'mcd_lrt_error': {'dims': (), 'data': mcd_lrt_error},
        'j1_etaPV': {'dims': ('j1-bg'), 'data': j1_etaPV},
        'j1_etaPEC_H2': {'dims': ('j1-bg'), 'data': j1_etaPEC_H2},
        'j1_etaPEC_NH3': {'dims': ('j1-bg'), 'data': j1_etaPEC_NH3},
        'j1_etaPEC_AA': {'dims': ('j1-bg'), 'data': j1_etaPEC_AA},
        'j1_max_etaPV': {'dims': (), 'data': j1_max_etaPV},
        'j1_max_etaPEC_H2': {'dims': (), 'data': j1_max_etaPEC_H2},
        'j1_max_etaPEC_NH3': {'dims': (), 'data': j1_max_etaPEC_NH3},
        'j1_max_etaPEC_AA': {'dims': (), 'data': j1_max_etaPEC_AA},
        'j1_bg_PVmax': {'dims': (), 'data': j1_bg_PVmax},
        'j1_bg_H2max': {'dims': (), 'data': j1_bg_H2max},
        'j1_bg_NH3max': {'dims': (), 'data': j1_bg_NH3max},
        'j1_bg_AAmax': {'dims': (), 'data': j1_bg_AAmax},
        'j2_etaPV_2bg': {'dims': ('j2-bg1','j2-bg2'), 'data': j2_etaPV_2bg},
        'j2_etaPEC_H2_2bg': {'dims': ('j2-bg1','j2-bg2'), 'data': j2_etaPEC_H2_2bg},
        'j2_etaPEC_NH3_2bg': {'dims': ('j2-bg1','j2-bg2'), 'data': j2_etaPEC_NH3_2bg},
        'j2_etaPEC_AA_2bg': {'dims': ('j2-bg1','j2-bg2'), 'data': j2_etaPEC_AA_2bg},
        'j2_max_etaPV_2bg': {'dims': (), 'data': j2_max_etaPV_2bg},
        'j2_max_etaPEC_H2_2bg': {'dims': (), 'data': j2_max_etaPEC_H2_2bg},
        'j2_max_etaPEC_NH3_2bg': {'dims': (), 'data': j2_max_etaPEC_NH3_2bg},
        'j2_max_etaPEC_AA_2bg': {'dims': (), 'data': j2_max_etaPEC_AA_2bg},
        'j2_bg1_PVmax': {'dims': (), 'data': j2_bg1_PVmax},
        'j2_bg2_PVmax': {'dims': (), 'data': j2_bg2_PVmax},
        'j2_bg1_H2max': {'dims': (), 'data': j2_bg1_H2max},
        'j2_bg2_H2max': {'dims': (), 'data': j2_bg2_H2max},
        'j2_bg1_NH3max': {'dims': (), 'data': j2_bg1_NH3max},
        'j2_bg2_NH3max': {'dims': (), 'data': j2_bg2_NH3max},
        'j2_bg1_AAmax': {'dims': (), 'data': j2_bg1_AAmax},
        'j2_bg2_AAmax': {'dims': (), 'data': j2_bg2_AAmax},
        'j3_etaPV_3bg': {'dims': ('j3-bg1','j3-bg2','j3-bg3'), 'data': j3_etaPV_3bg},
        'j3_max_etaPV_3bg': {'dims': (), 'data': j3_max_etaPV_3bg},
        'j3_bg1max': {'dims': (), 'data': j3_bg1max},
        'j3_bg2max': {'dims': (), 'data': j3_bg2max},
        'j3_bg3max': {'dims': (), 'data': j3_bg3max}}

    test = xr.Dataset.from_dict(d)
    test.to_netcdf(directory+id+'_singlePoint.nc')


x = paralell_lrt(inds=[0,10,10,6], filename_netCDF='Initial_Grid.nc')
save_singlepoint_netcdf(x[0],x[1],directory='')
