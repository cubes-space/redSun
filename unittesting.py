import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob
import h5py
import multiprocessing as mp

def ls2sol(ls):
    N_s = 668.6
    ls_peri = 250.99
    t_peri = 485.35
    a = 1.52368
    e = 0.09340
    epsilon = 25.1919
    if (ls == 0).any():
        ls = .01
    nu = np.radians(ls) + 1.90258
    E = np.arctan((np.tan(nu/2))/(np.sqrt((1+e)/(1-e))))*2
    M = E - e*np.sin(E)
    Ds = (M/(2*np.pi))*N_s + t_peri
    if (Ds < 0).any():
        Ds = Ds + N_s
    if (Ds > N_s).any():
        Ds = Ds - N_s
    return(Ds)

def point_loop(file):
    sg = xr.open_dataset('StupidGridFull.nc', group='flux')
    ds = xr.open_dataset(file)
    lat = ds['lat'][0]
    lon = ds['lon'][0]
    G = np.zeros(len(ds['lon']))
    for ri in range(0,len(ds['lon'])):
        ls = ds['ls'][ri]
        hr = ds['hr'][ri]
        G[ri] = sg['flux_dw_sw'][lat,lon,ls,hr]
    lss = np.unique(ds['ls'])
    sg = 0
    sols = np.zeros(len(lss))
    for i in range(0,len(lss)):
        sols[i] = ls2sol(lss[i]*15)
    try:
        P = G[:, np.newaxis, np.newaxis] * ds['j2_etaPV_2bg']
        zz = np.zeros((len(lss),len(ds['j2-bg1']),len(ds['j2-bg2'])))
        for i in range(0,len(lss)):
            hr_int = np.where(ds['ls']==lss[i])
            x = np.array(ds['hr'][hr_int] * 2 * 1.02569 * 60 * 60)
            x
            for j in range(0,len(ds['j2-bg1'])):
                for k in range(0,len(ds['j2-bg2'])):
                    y = P[:,j,k][hr_int]
                    z = np.trapz(y,x=x)
                    zz[i,j,k] = z
        z = np.zeros((len(ds['j2-bg1']),len(ds['j2-bg2'])))
        for j in range(0,len(ds['j2-bg1'])):
            for k in range(0,len(ds['j2-bg2'])):
                y = zz[:,j,k]
                z[j,k] = np.trapz(y,x=sols)
        j2pv = np.max(z)
        j2pvi = np.unravel_index(np.argmax(z),np.shape(z), order='C')
        pv = j2pv * (1/688) * (1/1000) * (1/3600) * (1/24)
        bg1 = ds['j2-bg1'][j2pvi[0]]
        bg2 = ds['j2-bg2'][j2pvi[1]]
        return([[lat,lon,0],[pv,bg1,bg2]])
    except:
        return([[lat,lon,1],[file]])


file = 'redsun_timeseries_17_00.nc'

r = point_loop(file)