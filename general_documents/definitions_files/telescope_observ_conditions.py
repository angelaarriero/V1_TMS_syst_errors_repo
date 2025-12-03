import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time, TimeDelta
from PyAstronomy import pyasl
import datetime

from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time
import astropy.units as u

from astropy.coordinates import SkyCoord

from astropy.coordinates import Angle
#import skyproj
from astropy.io import fits
#import healpy as hp
import jdcal as j


def coord_fijas(total_time,AZ_fix,EL_fix,lugar,dates_sec_2,nsamp):
    
    AZ=np.zeros(int(nsamp.value))+AZ_fix
    EL = np.zeros(int(nsamp.value))+EL_fix
    c = SkyCoord(AZ*u.deg, EL*u.deg, frame='altaz',location=lugar, obstime=dates_sec_2)
    icrs_to_galac=c.icrs.transform_to('galactic')
    l = icrs_to_galac.l.value
    b = icrs_to_galac.b.value
    return c.icrs.ra,c.icrs.dec,l,b,AZ,EL

def coord_GB(total_time,Vaz,EL_fix,lugar,dates_sec_2,nsamp):
    
    AZ=np.mod((total_time*360/Vaz),360)
    EL = np.zeros(int(nsamp.value))+EL_fix
    c = SkyCoord(AZ*u.deg, EL*u.deg, frame='altaz',location=lugar, obstime=dates_sec_2)
    icrs_to_galac=c.icrs.transform_to('galactic')
    l = icrs_to_galac.l.value
    b = icrs_to_galac.b.value
    return c.icrs.ra,c.icrs.dec,l,b,AZ,EL


def coord_TMS(Vaz,az_i,az_f,total_time,nsamp,EL_fix,lugar,dates_sec_2):
    AZF= []
    AZ= []
    res1= []
    w=0
    a=0
    test=0
    res=0
    time_count=int(np.size(total_time)/10)### /10
    print(time_count)
    for i in range(time_count):
        for j in range(np.size(total_time)):
            AZ.append(az_i+(total_time[j])*Vaz)
            if AZ[j] > az_f:
                a=j
                break
        for k in range(a,np.size(total_time)):
            cons=total_time[a]*Vaz
            w=az_f-(total_time[k]*Vaz)+cons
            if w <= az_i:
                break
            AZ.append(w)
            AZ= AZ[:(np.size(total_time))]
    EL = np.zeros(int(nsamp.value))+EL_fix
    c = SkyCoord(AZ*u.deg, EL*u.deg, frame='altaz',location=lugar, obstime=dates_sec_2)
    icrs_to_galac=c.icrs.transform_to('galactic')
    l = icrs_to_galac.l.value
    b = icrs_to_galac.b.value
    return c.icrs.ra,c.icrs.dec,l,b,AZ,EL
    #return AZ,test,res1
    
def coord_TMS_elev(Vel,elev_ini,elev_final,total_time,nsamp,az_fix,lugar,dates_sec_2):
    ELF= []
    EL= []
    res1= []
    w=0
    a=0
    test=0
    res=0
    time_count=int(np.size(total_time)/10)### /10
    print(time_count)
    
    for i in range(time_count):
        for j in range(np.size(total_time)):
            EL.append(elev_ini+(total_time[j])*Vel)
            if EL[j] > elev_final:
                a=j
                break
        for k in range(a,np.size(total_time)):
            cons=total_time[a]*Vel
            w=elev_final-(total_time[k]*Vel)+cons
            if w <= elev_ini:
                break
            EL.append(w)
            EL= EL[:(np.size(total_time))]
    AZ = np.zeros(int(nsamp.value))+az_fix
    c = SkyCoord(AZ*u.deg, EL*u.deg, frame='altaz',location=lugar, obstime=dates_sec_2)
    icrs_to_galac=c.icrs.transform_to('galactic')
    l = icrs_to_galac.l.value
    b = icrs_to_galac.b.value
    return c.icrs.ra,c.icrs.dec,l,b,AZ,EL
    #return AZ,test,res1

def coord_TMS_elevAZmov(Vel,elev_ini,elev_final,total_time,nsamp,Vaz,lugar,dates_sec_2):
    ELF= []
    EL= []
    res1= []
    w=0
    a=0
    test=0
    res=0
    time_count=int(np.size(total_time)/10)### /10
    print(time_count)
    
    for i in range(time_count):
        for j in range(np.size(total_time)):
            EL.append(elev_ini+(total_time[j])*Vel)
            if EL[j] > elev_final:
                a=j
                break
        for k in range(a,np.size(total_time)):
            cons=total_time[a]*Vel
            w=elev_final-(total_time[k]*Vel)+cons
            if w <= elev_ini:
                break
            EL.append(w)
            EL= EL[:(np.size(total_time))]
    AZ = np.mod((total_time*360/Vaz),360)
    c = SkyCoord(AZ*u.deg, EL*u.deg, frame='altaz',location=lugar, obstime=dates_sec_2)
    icrs_to_galac=c.icrs.transform_to('galactic')
    l = icrs_to_galac.l.value
    b = icrs_to_galac.b.value
    return c.icrs.ra,c.icrs.dec,l,b,AZ,EL
    #return AZ,test,res1


def timedef(times_sec,f_sampling):
    t = Time(times_sec, format='fits', scale='utc')
    dt = t[1] - t[0] #me da el valor del tiempo en dias
    T_obs=dt*86400 # para pasar el valor de dias a segundos 1 dia tiene 86400 s
    t_sampling = 1.0 / f_sampling # time between samples (s)
    nsamp = (T_obs /t_sampling) # total number of samples (samples)
    total_time=np.linspace(0, int(T_obs.value),num=int(nsamp.value))
    dates_sec=t[0]+ dt * np.linspace(0, 1.,int(nsamp.value) )
    return dates_sec,total_time,nsamp

def value_pixels(map1_up):
    mask_arcade=hp.mask_good(map1_up)
    h=0
    g=0
    pix_val= []
    for i in range(np.size(mask_arcade[0])):
        if mask_arcade[0][i]==True:
            h+=1
            pix_val.append(i)
        if mask_arcade[0][i]==False:
            g+=1
    return pix_val