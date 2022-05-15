# This package contains the photovoltaic functions for use in redSun
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

# 1 layer solar cell and photoelectrochemical production calcs
def SQlim_1bg(Lambda,Flux,T):

    # some general vars
    h = 6.63e-34
    c = 3e8
    kB = 1.38e-23
    q = 1.6e-19
    g = np.pi*2/((c**2)*(h**3))

    # bg range for all band gaps
    bg_first = 0.4 # eV
    bg_last = 3.0 # eV
    bg_step = 0.01

    bg_vec = np.arange(bg_first,bg_last+bg_step,bg_step)

    # alternate units for band gap arrays
    bg_m = (h*c) / (q*bg_vec)
    bg_nm = 1e9*bg_m
    bg_J = q*bg_vec

    # voltages for PEC calcs
    EredoxH2 = 1.23
    EredoxNH3 = 1.17
    EredoxAA = 1.09

    # overvoltage
    Vo = 0.7

    # ----------------------assumption that Lambda and Flux arranged in order of increasing wavelength---------------
    # ---------------------------------also, Flux and Lambda should have same length---------------------------------

    # generate dense solar spectra
    pchip_obj = scipy.interpolate.PchipInterpolator(Lambda,Flux)

    lam_step = 1e-9
    lam_m = np.arange(Lambda[0],Lambda[-1]+lam_step,lam_step)

    sFlux = pchip_obj(lam_m)

    # Set flux units to W/m^2/m
    sFlux = 1e9 * sFlux

    # define wavelength in nm as well
    lam_nm = 1e9 * lam_m

    # calculate photon flux as a function of wavelength
    E_lam = h*c/lam_m
    phFlux = 1e-9*sFlux/E_lam

    # calc solar intensity and photon flux
    Psun = np.trapz(sFlux,lam_m)
    phCount_vec = integrate.cumtrapz(phFlux)

    # make the photon flux count the same length as wavelength array
    phCount_vec = np.append(0,phCount_vec)

    # express photon flux as current density f(lamda)
    Jg = q*phCount_vec

    # define voltage range
    # voltages 20x less sparse than bgs
    vSparsity = 20
    vDiff = (bg_vec[1]-bg_vec[0])/vSparsity
    v = np.linspace(0,bg_vec[-1],math.ceil(1/vDiff))

    # find PEC voltage indices
    v_index_H2 = np.argmax(v>(EredoxH2+Vo))
    v_index_NH3 = np.argmax(v>(EredoxNH3+Vo))
    v_index_AA = np.argmax(v>(EredoxAA+Vo))

    # --------- Recombination current master array---------------------
    # predefine recombination current density, Jr
    Jr_master = np.zeros((len(lam_m),len(v)))
    Jr = np.zeros((len(bg_vec),len(v)))

    # dummy vars for Jr integration
    dummy = np.zeros(len(lam_m))
    dummy0 = np.zeros((len(lam_m),len(v)))


    for j in range(len(v)):

        dummy0[:,j] = E_lam - q*v[j]
        dummy0_1 = np.argmax(dummy0[:,j]<0)
        if dummy0_1 > 0:
            dummy0[dummy0_1:-1,j] = dummy0[dummy0_1-1,j]
            dummy0[-1,j] = dummy0[dummy0_1-1,j]

        dummy[0:-1] = integrate.cumtrapz((E_lam**2)/(np.exp((dummy0[:,j])/(kB*T))-1),E_lam)
        Jr_master[:,j] = q*g*dummy

    # --------end Recombination current master array---------------------

    # predefine generation current density arrays for each layer
    Jg_bg = np.zeros(len(bg_vec))

    # predefine recombination current density arrays for each layer
    Jr_bg = np.zeros((len(bg_vec),len(v)))

    # predefine total current
    J_bg = np.zeros((len(bg_vec),len(v)))

    # predefine efficiency arrays
    etaPV = np.zeros(len(bg_vec))
    etaPEC_H2 = np.zeros(len(bg_vec))
    etaPEC_NH3 = np.zeros(len(bg_vec))
    etaPEC_AA = np.zeros(len(bg_vec))

    for i in range(len(bg_vec)):

        # generation current density for top layer
        index_bg = np.argmax(E_lam<bg_J[i])
        Jg_bg[i] = Jg[index_bg-1]

        # recombination current density for top layer
        Jr_bg[i,:] = Jr_master[-1,:] - Jr_master[index_bg-1,:]

        # total current for top layer
        J_bg[i,:] = Jg_bg[i] - Jr_bg[i,:]

        # restrict to only relevant values
        dummy_J_bg = np.argmax(J_bg[i,:]<0)
        J_bg[i,dummy_J_bg:-1] = 0
        J_bg[i,-1] = 0

        # layer 1 voltage at max layer 1 efficiency
        v_index_PV = np.argmax(J_bg[i,:] * v / Psun)


        # efficiencies
        etaPV[i] = J_bg[i,v_index_PV] * v[v_index_PV] / Psun * 100

        etaPEC_H2[i] = J_bg[i,v_index_H2] * EredoxH2 / Psun * 100
        etaPEC_NH3[i] = J_bg[i,v_index_NH3] * EredoxNH3 / Psun * 100
        etaPEC_AA[i] = J_bg[i,v_index_AA] * EredoxAA / Psun * 100

    # max values and associated band gaps
    max_etaPV = np.amax(etaPV)
    max_etaPEC_H2 = np.amax(etaPEC_H2)
    max_etaPEC_NH3 = np.amax(etaPEC_NH3)
    max_etaPEC_AA = np.amax(etaPEC_AA)

    bg_PVmax = bg_vec[np.argmax(etaPV)]
    bg_H2max = bg_vec[np.argmax(etaPEC_H2)]
    bg_NH3max = bg_vec[np.argmax(etaPEC_NH3)]
    bg_AAmax = bg_vec[np.argmax(etaPEC_AA)]

    vals = [bg_vec,etaPV,etaPEC_H2,etaPEC_NH3,etaPEC_AA, max_etaPV,max_etaPEC_H2,max_etaPEC_NH3,max_etaPEC_AA,bg_PVmax,bg_H2max,bg_NH3max,bg_AAmax]
    return vals

# 2 layer solar cell and photoelectrochemical production calcs
def SQlim_2bg(Lambda,Flux,T):

    # some general vars
    h = 6.63e-34
    c = 3e8
    kB = 1.38e-23
    q = 1.6e-19
    g = np.pi*2/((c**2)*(h**3))

    # bg range for all band gaps
    bg_first = 0.4 # eV
    bg_last = 3.0 # eV
    bg_step = 0.01

    bg_vec = np.arange(bg_first,bg_last+bg_step,bg_step)
    # define individual band gap split points
    bg1_split = 1.2 #eV

    bg1_vec = np.arange(bg1_split,bg_last+bg_step,bg_step)
    bg2_vec = np.arange(bg_first,bg1_split+bg_step,bg_step)

    # alternate units for band gap arrays
    bg1_m = (h*c) / (q*bg1_vec)
    bg1_nm = 1e9*bg1_m
    bg1_J = q*bg1_vec

    bg2_m = (h*c) / (q*bg2_vec)
    bg2_nm = 1e9*bg2_m
    bg2_J = q*bg2_vec

    # voltages for PEC calcs
    EredoxH2 = 1.23
    EredoxNH3 = 1.17
    EredoxAA = 1.09

    # overvoltage
    Vo = 0.7

    # ----------------------assumption that Lambda and Flux arranged in order of increasing wavelength---------------
    # ---------------------------------also, Flux and Lambda should have same length---------------------------------

    # generate dense solar spectra
    pchip_obj = scipy.interpolate.PchipInterpolator(Lambda,Flux)

    lam_step = 1e-9
    lam_m = np.arange(Lambda[0],Lambda[-1]+lam_step,lam_step)

    sFlux = pchip_obj(lam_m)

    # Set flux units to W/m^2/m
    sFlux = 1e9 * sFlux

    # define wavelength in nm as well
    lam_nm = 1e9 * lam_m

    # calculate photon flux as a function of wavelength
    E_lam = h*c/lam_m
    phFlux = 1e-9*sFlux/E_lam

    # calc solar intensity and photon flux
    Psun = np.trapz(sFlux,lam_m)
    phCount_vec = integrate.cumtrapz(phFlux)

    # make the photon flux count the same length as wavelength array
    phCount_vec = np.append(0,phCount_vec)

    # express photon flux as current density f(lamda)
    Jg = q*phCount_vec

    # define voltage range
    # voltages 30x less sparse than bgs
    vSparsity = 10
    vDiff = (bg1_vec[1]-bg1_vec[0])/vSparsity
    v = np.linspace(0,bg1_vec[-1],math.ceil(1/vDiff))

    # v for PEC calcs
    v_pec = np.zeros(len(v))

    # --------- Recombination current master array---------------------
    # predefine recombination current density, Jr
    Jr_master = np.zeros((len(lam_m),len(v)))
    Jr = np.zeros((len(bg_vec),len(v)))

    # dummy vars for Jr integration
    dummy = np.zeros(len(lam_m))
    dummy0 = np.zeros((len(lam_m),len(v)))


    for j in range(len(v)):

        dummy0[:,j] = E_lam - q*v[j]
        dummy0_1 = np.argmax(dummy0[:,j]<0)
        if dummy0_1 > 0:
            dummy0[dummy0_1:-1,j] = dummy0[dummy0_1-1,j]
            dummy0[-1,j] = dummy0[dummy0_1-1,j]

        dummy[0:-1] = integrate.cumtrapz((E_lam**2)/(np.exp((dummy0[:,j])/(kB*T))-1),E_lam)
        Jr_master[:,j] = q*g*dummy

    # --------end Recombination current master array---------------------

    # predefine generation current density arrays for each layer
    Jg_bg1 = np.zeros(len(bg1_vec))
    Jg_bg2 = np.zeros((len(bg1_vec),len(bg2_vec)))

    # predefine recombination current density arrays for each layer
    Jr_bg1 = np.zeros((len(bg1_vec),len(v)))
    Jr_bg2 = np.zeros((len(bg1_vec),len(bg2_vec),len(v)))

    # predefine total current
    J_bg1 = np.zeros((len(bg1_vec),len(v)))
    J_bg2 = np.zeros((len(bg1_vec),len(bg2_vec),len(v)))

    # voltage for current matching routine
    v_2j = np.zeros((len(bg1_vec),len(bg2_vec)))

    # predefine efficiency arrays
    etaPV_2bg = np.zeros((len(bg1_vec),len(bg2_vec)))
    etaPEC_H2_2bg = np.zeros((len(bg1_vec),len(bg2_vec)))
    etaPEC_NH3_2bg = np.zeros((len(bg1_vec),len(bg2_vec)))
    etaPEC_AA_2bg = np.zeros((len(bg1_vec),len(bg2_vec)))

    for i in range(len(bg1_vec)):

        # generation current density for top layer
        index_bg1 = np.argmax(E_lam<bg1_J[i])
        Jg_bg1[i] = Jg[index_bg1-1]

        # recombination current density for top layer
        Jr_bg1[i,:] = Jr_master[-1,:] - Jr_master[index_bg1-1,:]

        # total current for top layer
        J_bg1[i,:] = Jg_bg1[i] - Jr_bg1[i,:]

        # restrict to only relevant values
        dummy_J_bg1 = np.argmax(J_bg1[i,:]<0)
        J_bg1[i,dummy_J_bg1:-1] = 0
        J_bg1[i,-1] = 0

        # layer 1 voltage at max layer 1 efficiency
        v_index_bg1 = np.argmax(J_bg1[i,:] * v / Psun * 100)


        for j in range(len(bg2_vec)):

            # generation current density for 2nd layer
            index_bg2 = np.argmax(E_lam<bg2_J[j])
            Jg_bg2[i,j] = Jg[index_bg2-1] - Jg[index_bg1-1]

            # recombination current density for 2nd layer
            Jr_bg2[i,j,:] = Jr_master[index_bg1-1,:] - Jr_master[index_bg2-1,:]

            # total current for bottom layer
            J_bg2[i,j,:] = Jg_bg2[i,j] - Jr_bg2[i,j,:]

            # restrict to only relevant values
            dummy_J_bg2 = np.argmax(J_bg2[i,j,:]<0)
            J_bg2[i,j,dummy_J_bg2:-1] = 0
            J_bg2[i,j,-1] = 0

            # layer 2 voltage at max layer 2 efficiency
            v_index_bg2 = np.argmax(J_bg2[i,j,:]*v/Psun)

            # ---------------match the higher current to the lower current by altering voltage-----------------
            if (J_bg1[i,v_index_bg1] > J_bg2[i,j,v_index_bg2]):
                # match layer 1 current to layer 2 current

                # voltage that "matches" layer currents
                v_index_bg1_m = np.argmin(np.abs(J_bg2[i,j,v_index_bg2]-J_bg1[i,:]))

                # define 2-layer voltage
                v_2j[i,j] = v[v_index_bg1_m] + v[v_index_bg2]

                # define 2-layer efficiency
                etaPV_2bg[i,j] = J_bg2[i,j,v_index_bg2] * v_2j[i,j] / Psun * 100

                # ----------routine for PEC calcs-------------------

                # H2 efficiency
                if (v_2j[i,j] > (EredoxH2 + Vo)):
                    etaPEC_H2_2bg[i,j] = J_bg2[i,j,v_index_bg2] * EredoxH2 / Psun * 100

                elif ((v[dummy_J_bg1-1]+v[dummy_J_bg2-1]) > (EredoxH2 + Vo)):


                    # for each v between vg_index_2 and dummy_J_bg2-1, find vg_index_1_m that matches J_bg1 and J_bg2.
                    # then, find where voltages sum to > Eredox--+Vo
                    # use J_bg2 at the voltage that corresponds to that location in efficiency calc.

                    v_2j_alt = np.zeros(len(J_bg2[i,j,v_index_bg2:dummy_J_bg2]))

                    for k in range(len(v_2j_alt)):

                        v_index_bg1_mPEC = np.argmin(np.abs(J_bg2[i,j,v_index_bg2+k]-J_bg1[i,:]))

                        v_2j_alt[k] = v[v_index_bg2+k] + v[v_index_bg1_mPEC]

                    # catch index error behavior
                    if len(v_2j_alt>0):
                        dummyNameHere = np.argmax((EredoxH2+Vo)<v_2j_alt)
                        if (dummyNameHere>0):
                            etaPEC_H2_2bg[i,j] = J_bg2[i,j,v_index_bg2+dummyNameHere] * EredoxH2 / Psun * 100

                # NH3 efficiency
                if (v_2j[i,j] > (EredoxNH3 + Vo)):
                    etaPEC_NH3_2bg[i,j] = J_bg2[i,j,v_index_bg2] * EredoxNH3 / Psun * 100

                elif ((v[dummy_J_bg1-1]+v[dummy_J_bg2-1]) > (EredoxNH3 + Vo)):


                    # for each v between vg_index_2 and dummy_J_bg2-1, find vg_index_1_m that matches J_bg1 and J_bg2.
                    # then, find where voltages sum to > Eredox--+Vo
                    # use J_bg2 at the voltage that corresponds to that location in efficiency calc.

                    v_2j_alt = np.zeros(len(J_bg2[i,j,v_index_bg2:dummy_J_bg2]))

                    for k in range(len(v_2j_alt)):

                        v_index_bg1_mPEC = np.argmin(np.abs(J_bg2[i,j,v_index_bg2+k]-J_bg1[i,:]))

                        v_2j_alt[k] = v[v_index_bg2+k] + v[v_index_bg1_mPEC]

                    # catch index error behavior
                    if len(v_2j_alt>0):
                        dummyNameHere = np.argmax((EredoxNH3+Vo)<v_2j_alt)
                        if (dummyNameHere>0):
                            etaPEC_NH3_2bg[i,j] = J_bg2[i,j,v_index_bg2+dummyNameHere] * EredoxNH3 / Psun * 100


                # Acetate efficiency
                if (v_2j[i,j] > (EredoxAA + Vo)):
                    etaPEC_AA_2bg[i,j] = J_bg2[i,j,v_index_bg2] * EredoxAA / Psun * 100

                elif ((v[dummy_J_bg1-1]+v[dummy_J_bg2-1]) > (EredoxAA + Vo)):


                    # for each v between vg_index_2 and dummy_J_bg2-1, find vg_index_1_m that matches J_bg1 and J_bg2.
                    # then, find where voltages sum to > Eredox--+Vo
                    # use J_bg2 at the voltage that corresponds to that location in efficiency calc.

                    v_2j_alt = np.zeros(len(J_bg2[i,j,v_index_bg2:dummy_J_bg2]))

                    for k in range(len(v_2j_alt)):

                        v_index_bg1_mPEC = np.argmin(np.abs(J_bg2[i,j,v_index_bg2+k]-J_bg1[i,:]))

                        v_2j_alt[k] = v[v_index_bg2+k] + v[v_index_bg1_mPEC]

                    # catch index error behavior
                    if len(v_2j_alt>0):
                        dummyNameHere = np.argmax((EredoxAA+Vo)<v_2j_alt)
                        if (dummyNameHere>0):
                            etaPEC_AA_2bg[i,j] = J_bg2[i,j,v_index_bg2+dummyNameHere] * EredoxAA / Psun * 100

                #-------------end routine for PEC calcs-----------------------

            else:
                # match layer 2 current to layer 1 current
                v_index_bg2 = np.argmin(np.abs(J_bg2[i,j,:]-J_bg1[i,v_index_bg1]))

                # define 2-layer voltage
                v_2j[i,j] = v[v_index_bg1] + v[v_index_bg2]

                # define 2-layer efficiency
                etaPV_2bg[i,j] = J_bg1[i,v_index_bg1] * v_2j[i,j] / Psun * 100

                # ----------routine for PEC calcs-------------------


                # H2 efficiency
                if (v_2j[i,j] > (EredoxH2 + Vo)):
                    etaPEC_H2_2bg[i,j] = J_bg1[i,v_index_bg2] * EredoxH2 / Psun * 100

                elif ((v[dummy_J_bg1-1]+v[dummy_J_bg2-1]) > (EredoxH2 + Vo)):

                    # for each v between vg_index_2 and dummy_J_bg2-1, find vg_index_1_m that matches J_bg1 and J_bg2.
                    # then, find where voltages sum to > Eredox--+Vo
                    # use J_bg2 at the voltage that corresponds to that location in efficiency calc.

                    v_2j_alt = np.zeros(len(J_bg1[i,v_index_bg1:dummy_J_bg1]))

                    for k in range(len(v_2j_alt)):

                        v_index_bg2_mPEC = np.argmin(np.abs(J_bg2[i,j,:]-J_bg1[i,v_index_bg1+k]))

                        v_2j_alt[k] = v[v_index_bg2_mPEC] + v[v_index_bg1+k]


                    # catch index error behavior
                    if len(v_2j_alt>0):
                        dummyNameHere = np.argmax((EredoxH2+Vo)<v_2j_alt)
                        if (dummyNameHere>0):
                            etaPEC_H2_2bg[i,j] = J_bg1[i,v_index_bg1+dummyNameHere] * EredoxH2 / Psun * 100


                # NH3 efficiency
                if (v_2j[i,j] > (EredoxNH3 + Vo)):
                    etaPEC_NH3_2bg[i,j] = J_bg1[i,v_index_bg2] * EredoxNH3 / Psun * 100

                elif ((v[dummy_J_bg1-1]+v[dummy_J_bg2-1]) > (EredoxNH3 + Vo)):

                    # for each v between vg_index_2 and dummy_J_bg2-1, find vg_index_1_m that matches J_bg1 and J_bg2.
                    # then, find where voltages sum to > Eredox--+Vo
                    # use J_bg2 at the voltage that corresponds to that location in efficiency calc.

                    v_2j_alt = np.zeros(len(J_bg1[i,v_index_bg1:dummy_J_bg1]))

                    for k in range(len(v_2j_alt)):

                        v_index_bg2_mPEC = np.argmin(np.abs(J_bg2[i,j,:]-J_bg1[i,v_index_bg1+k]))

                        v_2j_alt[k] = v[v_index_bg2_mPEC] + v[v_index_bg1+k]

                    # catch index error behavior
                    if len(v_2j_alt>0):
                        dummyNameHere = np.argmax((EredoxNH3+Vo)<v_2j_alt)
                        if (dummyNameHere>0):
                            etaPEC_NH3_2bg[i,j] = J_bg1[i,v_index_bg1+dummyNameHere] * EredoxNH3 / Psun * 100


                # AA efficiency
                if (v_2j[i,j] > (EredoxAA + Vo)):
                    etaPEC_AA_2bg[i,j] = J_bg1[i,v_index_bg2] * EredoxAA / Psun * 100

                elif ((v[dummy_J_bg1-1]+v[dummy_J_bg2-1]) > (EredoxAA + Vo)):

                    # for each v between vg_index_2 and dummy_J_bg2-1, find vg_index_1_m that matches J_bg1 and J_bg2.
                    # then, find where voltages sum to > Eredox--+Vo
                    # use J_bg2 at the voltage that corresponds to that location in efficiency calc.

                    v_2j_alt = np.zeros(len(J_bg1[i,v_index_bg1:dummy_J_bg1]))

                    for k in range(len(v_2j_alt)):

                        v_index_bg2_mPEC = np.argmin(np.abs(J_bg2[i,j,:]-J_bg1[i,v_index_bg1+k]))

                        v_2j_alt[k] = v[v_index_bg2_mPEC] + v[v_index_bg1+k]

                    # catch index error behavior
                    if len(v_2j_alt>0):
                        dummyNameHere = np.argmax((EredoxAA+Vo)<v_2j_alt)
                        if (dummyNameHere>0):
                            etaPEC_AA_2bg[i,j] = J_bg1[i,v_index_bg1+dummyNameHere] * EredoxAA / Psun * 100

                #-------------end routine for PEC calcs-----------------------
            # ----------------end current matching routing----------------------
    # max efficiency and band gaps
    max_etaPV_2bg = np.amax(etaPV_2bg)
    [bg1_idxMax,bg2_idxMax] = np.unravel_index(np.argmax(etaPV_2bg),np.shape(etaPV_2bg))
    bg1_PVmax = bg1_vec[bg1_idxMax]
    bg2_PVmax = bg2_vec[bg2_idxMax]

    max_etaPEC_H2_2bg = np.amax(etaPEC_H2_2bg)
    [bg1_H2idxMax,bg2_H2idxMax] = np.unravel_index(np.argmax(etaPEC_H2_2bg),np.shape(etaPEC_H2_2bg))
    bg1_H2max = bg1_vec[bg1_H2idxMax]
    bg2_H2max = bg2_vec[bg2_H2idxMax]

    max_etaPEC_NH3_2bg = np.amax(etaPEC_NH3_2bg)
    [bg1_NH3idxMax,bg2_NH3idxMax] = np.unravel_index(np.argmax(etaPEC_NH3_2bg),np.shape(etaPEC_NH3_2bg))
    bg1_NH3max = bg1_vec[bg1_NH3idxMax]
    bg2_NH3max = bg2_vec[bg2_NH3idxMax]

    max_etaPEC_AA_2bg = np.amax(etaPEC_AA_2bg)
    [bg1_AAidxMax,bg2_AAidxMax] = np.unravel_index(np.argmax(etaPEC_AA_2bg),np.shape(etaPEC_AA_2bg))
    bg1_AAmax = bg1_vec[bg1_AAidxMax]
    bg2_AAmax = bg2_vec[bg2_AAidxMax]

    vals = [bg1_vec,bg2_vec,etaPV_2bg,etaPEC_H2_2bg,etaPEC_NH3_2bg,etaPEC_AA_2bg, max_etaPV_2bg,max_etaPEC_H2_2bg,max_etaPEC_NH3_2bg,max_etaPEC_AA_2bg, bg1_PVmax,bg2_PVmax,bg1_H2max,bg2_H2max,bg1_NH3max,bg2_NH3max,bg1_AAmax,bg2_AAmax]
    return vals

# 3 layer solar cell (PV only) calcs
def SQlim_3bg(Lambda,Flux,T):

    # some general vars
    h = 6.63e-34
    c = 3e8
    kB = 1.38e-23
    q = 1.6e-19
    g = np.pi*2/((c**2)*(h**3))

    # bg range for all band gaps
    bg_first = 0.4 # eV
    bg_last = 3.0 # eV
    bg_step = 0.01

    bg_vec = np.arange(bg_first,bg_last+bg_step,bg_step)

    # define individual band gap split points
    bg1_split = 1.5 # eV
    bg2_split = 1.0 # eV

    bg1_vec = np.arange(bg1_split,bg_last+bg_step,bg_step)
    bg2_vec = np.arange(bg2_split,bg1_split+bg_step,bg_step)
    bg3_vec = np.arange(bg_first,bg2_split+bg_step,bg_step)

    # alternate units for band gap arrays
    bg1_m = (h*c) / (q*bg1_vec)
    bg1_nm = 1e9*bg1_m
    bg1_J = q*bg1_vec

    bg2_m = (h*c) / (q*bg2_vec)
    bg2_nm = 1e9*bg2_m
    bg2_J = q*bg2_vec

    bg3_m = (h*c) / (q*bg3_vec)
    bg3_nm = 1e9*bg3_m
    bg3_J = q*bg3_vec



    # ----------------------assumption that Lambda and Flux arranged in order of increasing wavelength---------------
    # ---------------------------------also, Flux and Lambda should have same length---------------------------------
    # generate dense solar spectra
    pchip_obj = scipy.interpolate.PchipInterpolator(Lambda,Flux)

    lam_step = 1e-9
    lam_m = np.arange(Lambda[0],Lambda[-1]+lam_step,lam_step)

    sFlux = pchip_obj(lam_m)

    # Set flux units to W/m^2/m
    sFlux = 1e9 * sFlux

    # define wavelength in nm as well
    lam_nm = 1e9 * lam_m

    # calculate photon flux as a function of wavelength


    E_lam = h*c/lam_m
    phFlux = 1e-9*sFlux/E_lam

    # calc solar intensity and photon flux
    Psun = np.trapz(sFlux,lam_m)
    phCount_vec = integrate.cumtrapz(phFlux)

    # make the photon flux count the same length as wavelength array
    phCount_vec = np.append(0,phCount_vec)

    # express photon flux as current density f(lamda)
    Jg = q*phCount_vec

    # define voltages
    # voltages 5x less sparse than bgs
    vSparsity = 5
    vDiff = (bg_vec[1]-bg_vec[0])/vSparsity
    v = np.linspace(0,bg_vec[-1],math.ceil(1/vDiff))


    # --------- Recombination current master array-----------------
    # predefine recombination current density, Jr
    Jr_master = np.zeros((len(lam_m),len(v)))
    Jr = np.zeros((len(bg_vec),len(v)))

    # dummy vars for Jr integration
    dummy = np.zeros(len(lam_m))
    dummy0 = np.zeros((len(lam_m),len(v)))


    for j in range(len(v)):

        dummy0[:,j] = E_lam - q*v[j]
        dummy0_1 = np.argmax(dummy0[:,j]<0)
        if dummy0_1 > 0:
            dummy0[dummy0_1:-1,j] = dummy0[dummy0_1-1,j]
            dummy0[-1,j] = dummy0[dummy0_1-1,j]

        dummy[0:-1] = integrate.cumtrapz((E_lam**2)/(np.exp((dummy0[:,j])/(kB*T))-1),E_lam)
        Jr_master[:,j] = q*g*dummy

    # -------------------------------------------------------------

    # predefine generation current density arrays for each layer
    Jg_bg1 = np.zeros(len(bg1_vec))
    Jg_bg2 = np.zeros((len(bg1_vec),len(bg2_vec)))
    Jg_bg3 = np.zeros((len(bg1_vec),len(bg2_vec),len(bg3_vec)))

    # predefine recombination current density arrays for each layer
    Jr_bg1 = np.zeros((len(bg1_vec),len(v)))
    Jr_bg2 = np.zeros((len(bg1_vec),len(bg2_vec),len(v)))
    Jr_bg3 = np.zeros((len(bg2_vec),len(bg3_vec),len(v)))

    # predefine total current
    J_bg1 = np.zeros((len(bg1_vec),len(v)))
    J_bg2 = np.zeros((len(bg1_vec),len(bg2_vec),len(v)))
    J_bg3 = np.zeros((len(bg1_vec),len(bg2_vec),len(bg3_vec),len(v)))

    # --------------------variables for current matching routine---------------------
    v_3j = np.zeros((len(bg1_vec),len(bg2_vec),len(bg3_vec)))
    # -------------------------------------------------------------------------------

    # predefine efficiency array
    etaPV_3bg = np.zeros((len(bg1_vec),len(bg2_vec),len(bg3_vec)))

    for i in range(len(bg1_vec)):

        # generation current density for top layer
        index_bg1 = np.argmax(E_lam<bg1_J[i])
        Jg_bg1[i] = Jg[index_bg1-1]

        # recombination current density for top layer
        Jr_bg1[i,:] = Jr_master[-1,:] - Jr_master[index_bg1-1,:]

        # total current for top layer
        J_bg1[i,:] = Jg_bg1[i] - Jr_bg1[i,:]

        # restrict to only relevant values
        dummy_J_bg1 = np.argmax(J_bg1[i,:]<0)
        J_bg1[i,dummy_J_bg1:-1] = 0
        J_bg1[i,-1] = 0

        # layer 1 voltage at max layer 1 efficiency
        v_index_bg1 = np.argmax(J_bg1[i,:] * v / Psun)


        for j in range(len(bg2_vec)):

            # generation current density for 2nd layer
            index_bg2 = np.argmax(E_lam<bg2_J[j])
            Jg_bg2[i,j] = Jg[index_bg2-1] - Jg[index_bg1-1]

            # recombination current density for 2nd layer
            Jr_bg2[i,j,:] = Jr_master[index_bg1-1,:] - Jr_master[index_bg2-1,:]

            # total current for 2nd layer layer
            J_bg2[i,j,:] = Jg_bg2[i,j] - Jr_bg2[i,j,:]

            # restrict to only relevant values
            dummy_J_bg2 = np.argmax(J_bg2[i,j,:]<0)
            J_bg2[i,j,dummy_J_bg2:-1] = 0
            J_bg2[i,j,-1] = 0

            # layer 2 voltage at max layer 2 efficiency
            v_index_bg2 = np.argmax(J_bg2[i,j,:]*v/Psun)


            for k in range(len(bg3_vec)):

                # generation current density for 3rd layer
                index_bg3 = np.argmax(E_lam<bg3_J[k])
                Jg_bg3[i,j,k] = Jg[index_bg3-1] - Jg[index_bg2-1]

                # recombination current density for 2nd layer
                Jr_bg3[j,k,:] = Jr_master[index_bg2-1,:] - Jr_master[index_bg3-1,:]

                # total current for 3rd layer
                J_bg3[i,j,k,:] = Jg_bg3[i,j,k] - Jr_bg3[j,k,:]

                # restrict to only relevant values
                dummy_J_bg3 = np.argmax(J_bg3[i,j,k,:]<0)
                J_bg3[i,j,k,dummy_J_bg3:-1] = 0
                J_bg3[i,j,k,-1] = 0

                # layer 3 voltage at max layer 3 efficiency
                v_index_bg3 = np.argmax(J_bg3[i,j,k,:]*v/Psun)


                # match the higher currents to the lowest current by altering voltage
                if (J_bg1[i,v_index_bg1] == np.min([J_bg1[i,v_index_bg1],
                                                    J_bg2[i,j,v_index_bg2],J_bg3[i,j,k,v_index_bg3]])):
                    # match layers 2 and 3 current to layer 1 current

                    # 2nd layer voltage that "matches" layer 1 current
                    v_index_bg2_m = np.argmin(np.abs(J_bg2[i,j,:]-J_bg1[i,v_index_bg1]))

                    # 3rd layer voltage that "matches" layer 1 current
                    v_index_bg3_m = np.argmin(np.abs(J_bg3[i,j,k,:]-J_bg1[i,v_index_bg1]))

                    # define 3-layer voltage
                    v_3j[i,j,k] = v[v_index_bg1] + v[v_index_bg2_m] + v[v_index_bg3_m]

                    # define 3-layer efficiency
                    etaPV_3bg[i,j,k] = J_bg1[i,v_index_bg1] * v_3j[i,j,k] / Psun * 100


                elif (J_bg2[i,j,v_index_bg2] == np.min([J_bg1[i,v_index_bg1],
                                                        J_bg2[i,j,v_index_bg2],J_bg3[i,j,k,v_index_bg3]])):
                    # match layers 1 and 3 current to layer 2 current

                    # 1st layer voltage that "matches" layer 2 current
                    v_index_bg1_m = np.argmin(np.abs(J_bg2[i,j,v_index_bg2]-J_bg1[i,:]))

                    # 3rd layer voltage that "matches" layer 2 current
                    v_index_bg3_m = np.argmin(np.abs(J_bg3[i,j,k,:]-J_bg2[i,j,v_index_bg2]))

                    # define 3-layer voltage
                    v_3j[i,j,k] = v[v_index_bg1_m] + v[v_index_bg2] + v[v_index_bg3_m]

                    # define 3-layer efficiency
                    etaPV_3bg[i,j,k] = J_bg2[i,j,v_index_bg2] * v_3j[i,j,k] / Psun * 100


                else:
                    # match layers 1 and 2 current to layer 3 current

                    # 1st layer voltage that "matches" layer 3 current
                    v_index_bg1_m = np.argmin(np.abs(J_bg3[i,j,k,v_index_bg3]-J_bg1[i,:]))

                    # 2nd layer voltage that "matches" layer 3 current
                    v_index_bg2_m = np.argmin(np.abs(J_bg2[i,j,:]-J_bg3[i,j,k,v_index_bg1]))

                    # define 3-layer voltage
                    v_3j[i,j,k] = v[v_index_bg1_m] + v[v_index_bg2_m] + v[v_index_bg3]

                    # define 3-layer efficiency
                    etaPV_3bg[i,j,k] = J_bg3[i,j,k,v_index_bg3] * v_3j[i,j,k] / Psun * 100

    # max efficiency and band gaps
    max_etaPV_3bg = np.amax(etaPV_3bg)
    [bg1_idxMax,bg2_idxMax,bg3_idxMax] = np.unravel_index(np.argmax(etaPV_3bg),np.shape(etaPV_3bg))
    bg1max = bg1_vec[bg1_idxMax]
    bg2max = bg2_vec[bg2_idxMax]
    bg3max = bg3_vec[bg3_idxMax]

    vals = [bg1_vec,bg2_vec,bg3_vec,etaPV_3bg,max_etaPV_3bg,bg1max,bg2max,bg3max,J_bg1,J_bg2,J_bg3,v]
    return vals
