#!/usr/bin/env python
_citation_text = """
## You are using the Column Radar Coincident Location Scripting, an open source
## an open source library for working with weather radar data.
##
## If you use this software to prepare a publication, please cite:
##
## Freya I. Addison, University of Leeds,  https://orcid.org/0000-0001-8439-290X"""
print(_citation_text)

import os
import numpy as np 
import netCDF4 as nc
from netCDF4 import Dataset
import time
import csv
import simplekml
from scipy.special import ellipkinc
from geographiclib.geomath import Math
from geographiclib.constants import Constants
from geographiclib.geodesiccapability import GeodesicCapability
from geographiclib.geodesic import Geodesic
from netCDF4 import num2date
import scipy.special

import itertools
import h5py
import os.path
import sys
import numpy.ma as ma
import pandas as pd
import xarray as xr
#from joblib import Parallel, delayed
import multiprocessing as mpl
from multiprocessing import Process, JoinableQueue

np.set_printoptions(threshold=np.inf)


def maths(filenama, dat, quick, longitude_r,latitude_r, lon_col,lat_col,alt_col,gat_lim_ext,cs,radar):
	print(time.time(), "start",filenama)
	starttim=time.time()
	try:
                #print(quick)
                mid_lat=quick["Radar Voxel Mid Latitude"][:]
                mid_lon=quick["Radar Voxel Mid Longitude"][:]
                mid_azi=quick["Radar Voxel Mid Azimuth"][:]
                mid_hi=quick["Radar Voxel Mid Altitude t4"][:]
                mid_s=quick["Radar Voxel Min Distance Along the Earth"][:]
                whisky_lat=quick["Radar Voxel Whisky Latitude"][:]
                whisky_lon=quick["Radar Voxel Whisky Longitude"][:]
                whisky_azi=quick["Radar Voxel Whisky Azimuth"][:]
                whisky_hi=quick["Radar Voxel Whisky Altitude"][:]
                whisky_s=quick["Radar Voxel Whisky Distance Along the Earth"][:]
                xray_lat=quick["Radar Voxel Xray Latitude"][:]
                xray_lon=quick["Radar Voxel Xray Longitude"][:]
                xray_azi=quick["Radar Voxel Xray Azimuth"][:]
                xray_hi=quick["Radar Voxel Xray Altitude"][:]
                xray_s=quick["Radar Voxel Xray Distance Along the Earth"][:]
                yankee_lat=quick["Radar Voxel Yankee Latitude"][:]
                yankee_lon=quick["Radar Voxel Yankee Longitude"][:]
                yankee_azi=quick["Radar Voxel Yankee Azimuth"][:]
                yankee_hi=quick["Radar Voxel Yankee Altitude"][:]
                yankee_s=quick["Radar Voxel Yankee Distance Along the Earth"][:]
                zulu_lat=quick["Radar Voxel Zulu Latitude"][:]
                zulu_lon=quick["Radar Voxel Zulu Longitude"][:]
                zulu_azi=quick["Radar Voxel Zulu Azimuth"][:]
                zulu_hi=quick["Radar Voxel Zulu Altitude"][:]
                zulu_s=quick["Radar Voxel Zulu Distance Along the Earth"][:]
                uniform_lat=quick["Radar Voxel Uniform Latitude"][:]
                uniform_lon=quick["Radar Voxel Uniform Longitude"][:]
                uniform_azi=quick["Radar Voxel Uniform Azimuth"][:]
                uniform_hi=quick["Radar Voxel Uniform Altitude"][:]
                uniform_s=quick["Radar Voxel Uniform Distance Along the Earth"][:]
                victor_lat=quick["Radar Voxel Victor Latitude"][:]
                victor_lon=quick["Radar Voxel Victor Longitude"][:]
                victor_azi=quick["Radar Voxel Victor Azimuth"][:]
                victor_hi=quick["Radar Voxel Victor Altitude"][:]
                victor_s=quick["Radar Voxel Victor Distance Along the Earth"][:]
                sierra_lat=quick["Radar Voxel Sierra Latitude"][:]
                sierra_lon=quick["Radar Voxel Sierra Longitude"][:]
                sierra_azi=quick["Radar Voxel Sierra Azimuth"][:]
                sierra_hi=quick["Radar Voxel Sierra Altitude"][:]
                sierra_s=quick["Radar Voxel Sierra Distance Along the Earth"][:]
                tango_lat=quick["Radar Voxel Tango Latitude"][:]
                tango_lon=quick["Radar Voxel Tango Longitude"][:]
                tango_azi=quick["Radar Voxel Tango Azimuth"][:]
                tango_hi=quick["Radar Voxel Tango Altitude"][:]
                tango_s=quick["Radar Voxel Tango Distance Along the Earth"][:]
                el_rad0=quick["Radar Elevation"][:]
                rng0=quick["Radar Range"][:]
                el_ind0=np.arange(len(el_rad0))
                rng_ind0=np.arange(len(rng0))#rng_ind0=np.arange(1,len(rng0)+1)
                el_rad=np.reshape(el_rad0,(len(el_ind0),1))
                el_rad=np.tile(el_rad,(1,len(rng0)))
                rng=np.tile(rng0,(len(el_rad0),1))
                #print("tango-zulu",tango_lat.shape,zulu_lat.shape,tango_lat.ravel().shape)#np.asarray(tango_lat[0:-2])-np.asarray(zulu_lat[1:-1]))#print("rng_ind0",len(rng_ind0),len(rng0),rng_ind0[-1],rng_ind0[0])
                el_ind=np.reshape(el_ind0,(len(el_ind0),1))
                el_ind=np.tile(el_ind,(1,len(rng0)))
                rng_ind=np.tile(rng_ind0,(len(el_rad0),1))
                mingatwty=invertgeoidmulti(whisky_lat,whisky_lon,yankee_lat,yankee_lon)	   
                maxgatxtz=invertgeoidmulti(xray_lat,xray_lon,zulu_lat,zulu_lon)
                #min and max gate distances and azimuths calculations between the corners of the voxels
                #these will be used in the voxel bounds	

                """
		mingatltw=invertgeoidhlf(latitude_r,longitude_r,whisky_lat,whisky_lon)
		maxgatltx=invertgeoidhlf(latitude_r,longitude_r,xray_lat,xray_lon)
		maxgatltz=invertgeoidhlf(latitude_r,longitude_r,zulu_lat,zulu_lon)
		mingatlty=invertgeoidhlf(latitude_r,longitude_r,yankee_lat,yankee_lon)
                """
                mingatltw=invertgeoidhlf(lat_col,lon_col,whisky_lat,whisky_lon)
                maxgatltx=invertgeoidhlf(lat_col,lon_col,xray_lat,xray_lon)
                maxgatltz=invertgeoidhlf(lat_col,lon_col,zulu_lat,zulu_lon)
                mingatlty=invertgeoidhlf(lat_col,lon_col,yankee_lat,yankee_lon)
                midgatltm=invertgeoidhlf(lat_col,lon_col,mid_lat,mid_lon)
                print(np.asarray(mingatltw[1]).shape)#1800 189 ##(3600,998)
                #raw_input("sstop")
                #print(mingatltw)
		#these are used for the voxel bounds, between the radar and the voxels
                print("inverts dune", time.time())
		#listin the azimuths and just making them between 0 and 2pi for easier limit statements
                migwy_az=np.deg2rad(mingatwty[0])
                migwy_az[np.deg2rad(mingatwty[0])<0]+=2*np.pi
		#azimuth between whisky & yankee
                magxz_az=np.deg2rad(maxgatxtz[0])
                magxz_az[np.deg2rad(maxgatxtz[0])<0]+=2*np.pi
		#azimuth between xray and zulu
                miltw_az=np.deg2rad(mingatltw[0])
                miltw_az[np.deg2rad(mingatltw[0])<0]+=2*np.pi
		#azimuth between radar and whisky voxel corner
                maltx_az=np.deg2rad(maxgatltx[0])
                maltx_az[np.deg2rad(maxgatltx[0])<0]+=2*np.pi
		#azimuth between radar and xray voxel corner
                miaz=migwy_az
		#minimum radar azimuth
                mxaz=magxz_az
		#maximum radar azimuth
                miaz[migwy_az>magxz_az]=mxaz[migwy_az>magxz_az]
                mxaz[migwy_az<=magxz_az]=migwy_az[migwy_az<=magxz_az]
		#These just makes sure that the min and max azimuths are the right way round
					    
                closest=[]
                outer=[]
                rad2air=Geodesic.WGS84.Inverse(latitude_r,longitude_r,lat_col,lon_col,outmask=1929).values()
		#S & az calculated between the radar and the aircraft using the GeographicLib module
		
                sar=list(rad2air)[5]
                print("sar",sar)
		#distance along the earth (s), between radar and aircraft
                rad2air_az=np.deg2rad(list(rad2air)[6])
		
                if rad2air_az<0:
                	rad2air_az=rad2air_az+(2*np.pi)
                else:None

                az_lim_ext=np.arctan(gat_lim_ext/sar)

                #bnd1=np.where(np.logical_and(sar<mid_s+gat_lim_ext,np.logical_and(sar>mid_s-gat_lim_ext,np.logical_and(rad2air_az>=mid_azi-az_lim_ext,np.logical_and(rad2air_az<=mid_azi+az_lim_ext,~np.isnan(mid_azi))))))

                bnd1=np.where(np.logical_and(mingatltw[1]<=gat_lim_ext,np.logical_and(mingatlty[1]<=gat_lim_ext,np.logical_and(maxgatltx[1]<=gat_lim_ext,np.logical_and(maxgatltz[1]<=gat_lim_ext,~np.isnan(mid_azi))))))
                #arr1,arr2=np.split(np.asarray(bnd1),2,axis=1)#bnd1=np.asarray(bnd1)#np.savetxt("bnd1.csv",bnd1,delimiter=",")#bnd1=np.asarray(bnd1)#bnd1=np.where(np.logical_and(mingatltw[1]<=gat_lim_ext,np.logical_and(mingatlty[1]<=gat_lim_ext,np.logical_and(maxgatltx[1]<=gat_lim_ext,np.logical_and(maxgatltz[1]<=gat_lim_ext,np.logical_and(sar<mid_s+gat_lim_ext,np.logical_and(sar>mid_s-gat_lim_ext,~np.isnan(mid_azi))))))))
                #bndx=np.where(np.logical_and(mingatltw[1]<=gat_lim_ext,np.logical_and(mingatlty[1]<=gat_lim_ext,np.logical_and(maxgatltx[1]<=gat_lim_ext,np.logical_and(maxgatltz[1]<=gat_lim_ext,~np.isnan(mid_azi))))))#bnd11=np.asarray(bnd1)+1#raw_input("paws")#bnd1=np.where(np.logical_and(mingatltw[1]<=gat_lim_ext,np.logical_and(mingatlty[1]<=gat_lim_ext,np.logical_and(maxgatltx[1]<=gat_lim_ext,np.logical_and(maxgatltz[1]<=gat_lim_ext,np.logical_and(sar<mid_s+gat_lim_ext,np.logical_and(sar>mid_s-gat_lim_ext,np.logical_and(rad2air_az>=mid_azi-az_lim_ext,np.logical_and(rad2air_az<=mid_azi+az_lim_ext,~np.isnan(mid_azi))))))))))
                #print(bnd1.shape,bnd1.ravel().shape)#bx1=np.asarray(bnd1)-1#print(len(mid_lat[bnd1]),len(bnd1))
                #print("test",bnd1)
                #print(gat_lim_ext, az_lim_ext, np.median(mingatltw))
                con=0
                for i in range(0,len(mingatltw[1])):
                    
                    for j in range(0,len(mingatltw[1][i])):
                        #print("check",mingatltw[1][i][j])
                        if midgatltm[1][i][j] <= 100000:
                            con=con+1
                            print("uhuh",gat_lim_ext,midgatltm[1][i][j])
                        if mingatltw[1][i][j]<=gat_lim_ext:
                            print("yay",gat_lim_ext,mingatltw[1][i][j])
                #print(np.where(mingatltw[1]<=gat_lim_ext), len(mingatlty[1]<=gat_lim_ext), len(maxgatltx[1]<=gat_lim_ext),len(maxgatltz[1]<=gat_lim_ext),len(~np.isnan(mid_azi)))#print("here",[len(ark) for ark in bnd1])#min(arr1),max(arr1),min(arr2),max(arr2),#el_rad0[arr2].shape,rng0[arr1].shape)#len(tango_lat[np.where(tango_lat[bnd1]-zulu_lat[bnd11]))#(tango_lat[bnd1].shape too many indices for array)
                #print(con)
                #raw_input("paws")
                
			
                papas1=invertgeoidhlf(lat_col,lon_col,mid_lat[bnd1],mid_lon[bnd1])

                bndy2=np.arange(0,len(mid_s),1)
		
                bndy2=np.reshape(bndy2,(len(mid_s),1))
		
                size=(len(el_rad0),len(rng0))
                size=np.asarray(size)
                array=np.zeros(size)

                bndy2=np.tile(array,(1))

                bndy3=np.arange(len(array.ravel()))


                ultimatecoords=xr.Dataset(data_vars={"Range Index Value of matches": (("bndy"),rng_ind[bnd1]),
			    "Elevation Index Value of matches": (("bndy"),el_ind[bnd1]),
			    "Radar Voxel Mid Latitude": (("bndy"),mid_lat[bnd1]),
                            "Radar Voxel Mid Longitude": (("bndy"),mid_lon[bnd1]),
                            "Radar Voxel Mid Altitude": (("bndy"),mid_hi[bnd1]),
                            "Radar Voxel Mid Distance Along the Earth": (("bndy"),mid_s[bnd1]),
			    "Radar Voxel Whisky Latitude": (("bndy"),whisky_lat[bnd1]),
                            "Radar Voxel Whisky Longitude": (("bndy"),whisky_lon[bnd1]),
                            "Radar Voxel Whisky Altitude": (("bndy"),whisky_hi[bnd1]),
                            "Radar Voxel Whisky Distance Along the Earth": (("bndy"),whisky_s[bnd1]),
			    "Radar Voxel Xray Latitude": (("bndy"),xray_lat[bnd1]),
                            "Radar Voxel Xray Longitude": (("bndy"),xray_lon[bnd1]),
                            "Radar Voxel Xray Altitude": (("bndy"),xray_hi[bnd1]),
                            "Radar Voxel Xray Distance Along the Earth": (("bndy"),xray_s[bnd1]),
			    "Radar Voxel Yankee Latitude": (("bndy"),yankee_lat[bnd1]),
                            "Radar Voxel Yankee Longitude": (("bndy"),yankee_lon[bnd1]), 
                            "Radar Voxel Yankee Altitude": (("bndy"),yankee_hi[bnd1]),
                            "Radar Voxel Yankee Distance Along the Earth": (("bndy"),yankee_s[bnd1]),
			    "Radar Voxel Zulu Latitude": (("bndy"),zulu_lat[bnd1]),
                            "Radar Voxel Zulu Longitude": (("bndy"),zulu_lon[bnd1]),
                            "Radar Voxel Zulu Altitude": (("bndy"),zulu_hi[bnd1]),				
                            "Radar Voxel Zulu Distance Along the Earth": (("bndy"),zulu_s[bnd1]),
			    "Radar Voxel Uniform Latitude": (("bndy"),uniform_lat[bnd1]),
                            "Radar Voxel Uniform Longitude": (("bndy"),uniform_lon[bnd1]),
                            "Radar Voxel Uniform Altitude": (("bndy"),uniform_hi[bnd1]),
                            "Radar Voxel Uniform Distance Along the Earth": (("bndy"),uniform_s[bnd1]),
			    "Radar Voxel Victor Latitude": (("bndy"),victor_lat[bnd1]),
                            "Radar Voxel Victor Longitude": (("bndy"),victor_lon[bnd1]),
                            "Radar Voxel Victor Altitude": (("bndy"),victor_hi[bnd1]),
                            "Radar Voxel Victor Distance Along the Earth": (("bndy"),victor_s[bnd1]),
			    "Radar Voxel Sierra Latitude": (("bndy"),sierra_lat[bnd1]),
                            "Radar Voxel Sierra Longitude": (("bndy"),sierra_lon[bnd1]),
                            "Radar Voxel Sierra Altitude": (("bndy"),sierra_hi[bnd1]),
                            "Radar Voxel Sierra Distance Along the Earth": (("bndy"),sierra_s[bnd1]),
			    "Radar Voxel Tango Latitude": (("bndy"),tango_lat[bnd1]),
                            "Radar Voxel Tango Longitude": (("bndy"),tango_lon[bnd1]),
                            "Radar Voxel Tango Altitude": (("bndy"),tango_hi[bnd1]),				
                            "Radar Voxel Tango Distance Along the Earth": (("bndy"),tango_s[bnd1]),
			    "Radar Elevation": (("bndy"),el_rad[bnd1]),
			    "Radar Range": (("bndy"),rng[bnd1]),
			    "Distance between CS centroid and radar": (("bndy"),papas1[1]),
                            "Azimuth between CS centroid  and radar": (("bndy"),papas1[0])},
                                          attrs={
        "Radar Longitude": float(longitude_r),
        "Radar Latitude": float(latitude_r),
        "CS Longitude": float(lon_col),
        "CS Latitude": float(lat_col)
    },
                                          coords={"bndy":bndy2[bnd1]}
		)
		
                #ultimatecoords.to_netcdf("20210709_Col2500_northoltWS.nc")
                #findexn=open("20210709_Col2500_RothGrid1.csv","w")
                ultimatecoords.to_netcdf(str(dat)+"_Col5000_"+str(radar)+"_CS"+str(cs)+".nc")
                #findexn=open("20210709_Col2500_RothGrid1.csv","w")
                #np.savetxt(findexn,[el_ind[bnd1],rng_ind[bnd1],el_ind[bnd1],el_ind[bnd1]],delimiter=",")

	except KeyError:
		print("KeyError, not using file",filenama)
	
	print(time.time(),"end",filenama)
	return
def invertgeoidmulti(lat1,lon1,lat2,lon2):
	
	
	lin1a=lat1.ravel()
	lin1o=lon1.ravel()
	lin2a=lat2.ravel()
	lin2o=lon2.ravel()
	#print(lin1a[0],lin1o[0],lin2a[0],lin2o[0])
	
	corner=[Geodesic.WGS84.Inverse(iw,ix,iy,iz,outmask=1929).values() for (iw,ix,iy,iz) in zip(lin1a,lin1o,lin2a,lin2o)]
	
	cn=[]	
	cn.append([list(corner[ji]) for ji in range(0,len(lin1a))])
	#print("cn",cn[0][0],corner[0])
	azzy=[cn[0][noppy][6] for noppy in range(0,len(cn[0]))]
	azzy=np.reshape(azzy,np.shape(lat1))
	ss=[cn[0][noppy][5] for noppy in range(0,len(cn[0]))]
	ss=np.reshape(ss,np.shape(lat1))
	
	return(azzy,ss)
def invertgeoidhlf(lat1,lon1,lat2,lon2):
	lin2a=lat2.ravel()
	lin2o=lon2.ravel()
	corner=[Geodesic.WGS84.Inverse(lat1,lon1,iy,iz,outmask=1929).values() for (iy,iz) in zip(lin2a,lin2o)]
	cn=[]
	
	cn.append([list(corner[ji]) for ji in range(0,len(lin2a))])
	#print(cn[0][0],corner[0])
	azzy=[cn[0][noppy][6] for noppy in range(0,len(cn[0]))]
	azzy=np.reshape(azzy,np.shape(lon2))
	ss=[cn[0][noppy][5] for noppy in range(0,len(cn[0]))]
	ss=np.reshape(ss,np.shape(lon2))
	
	return(azzy,ss)

def is_valid_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            # Check for expected group structure
            if not any(k.startswith("dataset") for k in f.keys()):
                raise ValueError("No dataset groups found")
        return True
    except Exception as e:
        print(f"Invalid HDF5: {file_path} ({e})")
        return False
fcs1=Dataset("FAD_data_origsurvey_all.nc")
from datetime import datetime, timedelta

# Base timestamp (you confirmed this is what 0 maps to)
base_time = datetime(2021, 6, 15, 10, 0)

# Load raw time values (assuming this is a numpy masked array)
raw_offsets = fcs1["Date Time of First Sighting of Flying Ants"][:]

# Convert to datetimes
converted_datetimes = [base_time + timedelta(minutes=int(val)) for val in raw_offsets]
###Selects by Date
target_date="2021-07-09"
target_date_0 = "2021-07-13"
target_date_3 = "2021-07-16"
target_date_4 = "2021-07-17"
target_date_2="2021-07-15"
target_date_1="2021-07-14"
target_date_5="2021-07-18"
target_date_6="2021-07-19"
target_date_7="2021-07-20"

fcs1["Date Time of First Sighting of Flying Ants"][:]
var = fcs1['Date Time of First Sighting of Flying Ants']

dates = nc.num2date(var[:], units=var.units, calendar=var.calendar)

from datetime import datetime
# Extract dates and times
bdates = [date.strftime('%Y-%m-%d') for date in dates]
btimes = [date.strftime('%H:%M:%S') for date in dates]
bhours = [date.hour for date in dates]
bminutes = [date.minute for date in dates]

# Convert target date string to datetime object

daymins=24*60
target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
seconds_past = (target_datetime-dates[0]).total_seconds()
target_datetime_2 = datetime.strptime(target_date_2, "%Y-%m-%d")
seconds_past_2 = (target_datetime_2-dates[0]).total_seconds()
target_datetime_3 = datetime.strptime(target_date_3, "%Y-%m-%d")
seconds_past_3 = (target_datetime_3-dates[0]).total_seconds()
target_datetime_4 = datetime.strptime(target_date_4, "%Y-%m-%d")
seconds_past_4 = (target_datetime_4-dates[0]).total_seconds()
target_datetime_1 = datetime.strptime(target_date_1, "%Y-%m-%d")
seconds_past_1 = (target_datetime_1-dates[0]).total_seconds()
target_datetime_0 = datetime.strptime(target_date_0, "%Y-%m-%d")
seconds_past_0 = (target_datetime_0-dates[0]).total_seconds()
target_datetime_5 = datetime.strptime(target_date_5, "%Y-%m-%d")
seconds_past_5 = (target_datetime_5-dates[0]).total_seconds()
target_datetime_6 = datetime.strptime(target_date_6, "%Y-%m-%d")
seconds_past_6 = (target_datetime_6-dates[0]).total_seconds()
target_datetime_7 = datetime.strptime(target_date_7, "%Y-%m-%d")
seconds_past_7 = (target_datetime_7-dates[0]).total_seconds()

where090721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past/60), np.asarray(var) <= int(seconds_past/60)+daymins-1))
where170721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_4/60), np.asarray(var) <= int(seconds_past_4/60)+daymins-1))
where160721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_3/60), np.asarray(var) <= int(seconds_past_3/60)+daymins-1))
where130721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_0/60), np.asarray(var) <= int(seconds_past_0/60)+daymins-1))
where150721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_2/60), np.asarray(var) <= int(seconds_past_2/60)+daymins-1))
where140721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_1/60), np.asarray(var) <= int(seconds_past_1/60)+daymins-1))
where180721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_5/60), np.asarray(var) <= int(seconds_past_5/60)+daymins-1))
where190721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_6/60), np.asarray(var) <= int(seconds_past_6/60)+daymins-1))
where200721 = np.where(np.logical_and(np.asarray(var) >=int(seconds_past_7/60), np.asarray(var) <= int(seconds_past_7/60)+daymins-1))
whered=[where090721]#where170721,where160721 ,where130721,where150721 ,where140721,where180721,where190721,where200721]
crd=np.asarray(fcs1["Closest Radar Distances"])
crd[crd=="nan"]="999.99"
crd=crd.astype(float)

#date_list = ["20210716","20210714","20210715","20210716","20210717","20210718","20210719","20210720" ] 
date_list = ["20210709"]


#def maths(filenama, dat,  quick, longitude_r,latitude_r,rng0,el0_rad0, lon_col,lat_col,alt_col,gat_lim_ext):
#radar=Dataset("/gws/nopw/j04/ncas_radar_vol2/data/xband/chilbolton/cfradial/calib_v2/sur/20180214/ncas-mobile-x-band-radar-1_chilbolton_20180214-105033_SUR_v1.nc")
date_list = ["20210709"]#["20210713","20210714","20210715","20210716","20210717","20210718","20210719","20210720" ] 
root_dir = "/gws/nopw/j04/ncas_radar_vol3/ukmo-nimrod/raw_h5_data/single-site/"  # Top-level folder
folder_names = [name for name in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, name))] # Your folder list
import h5py
fcs1=Dataset("FAD_data_origsurvey_all.nc")
#file_path = "/gws/nopw/j04/ncas_radar_vol3/ukmo-nimrod/raw_h5_data/single-site/castor-bay/2021/20210716_polar_pl_radar07_aggregate.h5"
radscs=[]
for d in range(len(date_list)):
    radar_vals = np.unique(fcs1["Closest Radar Numbers"][whered[d]][:, 0])
    
    for r in radar_vals:
        r_str = str(r)
        if r_str != 'Non':
            r_fmt = r_str.zfill(2)  # pad with leading 0 if needed
            if r_fmt not in radscs:
                radscs.append(r_fmt)

#root_dir = "/gws/nopw/j04/ncas_radar_vol3/ukmo-nimrod/raw_h5_data/single-site/"
fp2_base = "~/coloc_code/"
from pyhigh import get_elevation
cs=0



#radarlp=read_nimrod_aggregated_odim_h5("/home/users/ee16fil/freya_thesis_data/20210709_polar_pl_radar05_aggregate.h5", "sp", "1000")
#"/gws/nopw/j04/ncas_radar_vol2/data/xband/chilbolton/cfradial/calib_v2/sur/20170513/ncas-mobile-x-band-radar-1_chilbolton_20170513-100426_SUR_v1.nc"
#'/gws/nopw/j04/ncas_radar_vol2/data/xband/chilbolton/cfradial/calib_v2/sur/20180214/ncas-mobile-x-band-radar-1_chilbolton_20180214-105033_SUR_v1.nc')
radar=Dataset('/gws/nopw/j04/ncas_radar_vol2/data/xband/chilbolton/cfradial/calib_v2/sur/20180214/ncas-mobile-x-band-radar-1_chilbolton_20180214-105033_SUR_v1.nc')
qk=Dataset("20180214Coords4.nc")

#qk=Dataset("20210709CoordsMOrefracCheniessp.nc")

#flp="20210709CoordsMOrefracCheniessp.nc"
#qk=Dataset("20170513Coords.nc")
#bentley woods:-1.6401474,51.0903015
#porton down iii: 51.1443825, -1.6826019
#maths('ncas-mobile-x-band-radar-1_chilbolton_20170513-100426_SUR_v1.nc',20170513qk,radar.variables['latitude'][0],radar.variables['longitude'][0],-1.6826019,51.1443825,130,2500)
maths('ncas-mobile-x-band-radar-1_chilbolton_20180214-105033_SUR_v1.nc',20180214,radar.variables['time'],qk,radar.variables['latitude'][0],radar.variables['longitude'][0],-1.6826019,51.1443825,130,2500)
#maths("20210709_polar_pl_radar05_aggregate.h5",20210709,radarlp.time,qk,radarlp.longitude["data"],radarlp.latitude["data"],-0.937112785658465,51.4465794965277,130,2500)
#maths("20210709_polar_pl_radar05_aggregate.h5",20210709,radarlp.time,qk,radarlp.longitude["data"],radarlp.latitude["data"],-0.417,51.549 ,33,2500)

#vcoords('ncas-mobile-x-band-radar-1_chilbolton_20180214-105033_SUR_v1.nc',20180214,radar.variables['time'],radar.variables['latitude'][0],radar.variables['longitude'][0],radar.variables['altitude'][0],1,radar.variables['range'][:],radar.variables['range'].meters_between_gates/2,radar.variables['elevation'][:],radar.variables['radar_beam_width_v'][0],radar.variables['azimuth'][:],radar.variables['radar_beam_width_h'][0])
