#!/usr/bin/env python
_citation_text = """
## You are using the Aircraft/Radar Coincident Location Scripting, an open source
## an open source library for working with weather radar data.
##
## If you use this software to prepare a publication, please cite:
##
## Freya I. Addison, University of Leeds,  https://orcid.org/0000-0001-8439-290X"""
print(_citation_text)
#import matplotlib.pyplot as plt  #not used
#import pyart #not used
#from matplotlib import rcParams #notused
import os
import numpy as np 
import netCDF4
from netCDF4 import Dataset
import time
import csv
import simplekml
from scipy.special import ellipkinc
from geographiclib.geomath import Math
from geographiclib.constants import Constants
from geographiclib.geodesiccapability import GeodesicCapability
from geographiclib.geodesic import Geodesic
#%matplotlib inline
#%pylab inline
import scipy.special
#import collections
#from collections import Counter
import itertools
import h5py
import os.path
import sys
import numpy.ma as ma
import pandas as pd
import xarray as xr

np.set_printoptions(threshold=np.inf)

def maths(filenama,dat,latitude_r,longitude_r,lat,lon,quick,gat_lim_ext,az_lim_ext,rng_lim_cls,klost):
	
	print(time.time(), "start",filenama)
	starttim=time.time()
	try:
		atim=lat
		if np.any(lat !=np.nan): 
			#checks that there are matching times
			print("yay")
			   
		else:
			return None
			#stops the script
		results={}
		#dictionary to write out ###IS THIS STILL USED
		kermit=0 
		###rng0=radar.variables['range'][:]/1000
		#turns the radar range into km 
		mid_lat=quick["Radar Voxel Mid Latitude"][:]
		mid_lon=quick["Radar Voxel Mid Longitude"][:]
		mid_azi=quick["Radar Voxel Mid Azimuth"][:]
		mid_hi=quick["Radar Voxel Mid Altitude"][:]
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
        
		#air_alt_err=7 #7m is the uncertainty in faam aircraft
		#mindif=((whisky_hi-mid_hi))-air_alt_err 
		#minimum distance in height of the voxel with an error adjustment to maximus
		#maxdif=air_alt_err+(mid_hi-zulu_hi)#*1000

		#print("invertgeoidmulti", time.time())
		mingatwty=invertgeoidmulti(whisky_lat,whisky_lon,yankee_lat,yankee_lon)	   
		maxgatxtz=invertgeoidmulti(xray_lat,xray_lon,zulu_lat,zulu_lon)
		#min and max gate distances and azimuths calculations between the corners of the voxels
		#these will be used in the voxel bounds	

		#print("invertgeoidhlf", time.time())
		mingatltw=invertgeoidhlf(latitude_r,longitude_r,whisky_lat,whisky_lon)
		maxgatltx=invertgeoidhlf(latitude_r,longitude_r,xray_lat,xray_lon)
		maxgatltz=invertgeoidhlf(latitude_r,longitude_r,zulu_lat,zulu_lon)
		mingatlty=invertgeoidhlf(latitude_r,longitude_r,yankee_lat,yankee_lon)
		#these are used for the voxel bounds, between the radar and the voxels
		print("inverts done", time.time())
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
		#print("rad2air invertend", time.time())
		rad2air=invertgeoidhlf(latitude_r,longitude_r,lat,lon)
		#S & az calculated between the radar and the aircraft using the GeographicLib module
		#print("rad2air invertend", time.time())
		sar=rad2air[1]
		#distance along the earth (s), between radar and aircraft
		rad2air_az=np.deg2rad(rad2air[0])
		rad2air_az[np.deg2rad(rad2air[0])<0]+=2*np.pi
		#azimuth between radar and aircraft
		#make sure that are natural numbers and not +extra pis for miaz & mxaz
			    	
		##########################################################
		###Reshape to 3D for efficiency
		##########################################################

		mid_lat=np.dstack([mid_lat]*len(lat))
		mid_lon=np.dstack([mid_lon]*len(lat))
		mid_hi=np.dstack([mid_hi]*len(lat))
		lat=np.tile(lat,mid_lat.shape[0:2])
		lat=lat.reshape(mid_lat.shape)
		lon=np.tile(lon,mid_lat.shape[0:2])
		lon=lon.reshape(mid_lat.shape)	
		#print("alt",alt.shape) #alt (66,)
		#alt=np.tile(alt,mid_lat.shape[0:2])
		#print("alt",alt.shape) #alt (157, 175758)
		#alt=alt.reshape(mid_lat.shape) #alt (157, 2663, 66)	
		#print "alt",lat.shape,lon.shape,mid_lat.shape,mid_lon.shape
		shpsr=np.shape(sar)
		#print "shape fix", np.shape(sar),np.shape(whisky_s),mid_lat.shape[0:2]
		sar=np.tile(sar,mid_lat.shape[0:2])
		sar=sar.reshape(mid_lat.shape) 
		rad2air_az=np.tile(rad2air_az,mid_lat.shape[0:2])
		rad2air_az=rad2air_az.reshape(mid_lat.shape) 
		#print "shape fix", np.shape(sar),np.shape(whisky_s),mid_lat.shape[0:2]
		#whisky_s=np.tile(whisky_s,mid_lat.shape[0:2])
		whisky_s=np.tile(whisky_s,shpsr)
		whisky_s=whisky_s.reshape(mid_lat.shape)
		yankee_s=np.tile(yankee_s,shpsr)
		yankee_s=yankee_s.reshape(mid_lat.shape)
		yankee_hi=np.tile(yankee_hi,shpsr)
		yankee_hi=yankee_hi.reshape(mid_lat.shape)
		whisky_hi=np.tile(whisky_hi,shpsr)
		whisky_hi=yankee_hi.reshape(mid_lat.shape)
		mxaz=np.tile(mxaz,shpsr)
		mxaz=mxaz.reshape(mid_lat.shape)
		miaz=np.tile(miaz,shpsr)
		miaz=miaz.reshape(mid_lat.shape)
		#print(time.time())
		print("shape rng",rngind.shape,len(elind),elind.shape,len(atim),atim.shape,filenama)
		elindy=np.tile(elind,mid_lat.shape[1:])
		elindy=elindy.reshape(mid_lat.shape)
		rngindy=np.tile(rngind,(len(elind),len(atim)))
		rngindy=rngindy.reshape(mid_lat.shape)
		atmi=np.arange(0,len(atim),1)
		atimy=np.tile(atmi,mid_lat.shape[0:2])
		atimy=atimy.reshape(mid_lat.shape) 
		lemons=np.empty([len(el_rad0),len(rng),len(atim)])
		sugar=np.empty([len(el_rad0),len(rng),len(atim)])

		########################################################################################
		###Limits are put in place to reduce the amount of time the code takes, by removing 
		###some of the voxels which would not be the closest.
		########################################################################################
		
		#gat_lim_ext=gat*3
		#alt_lim_ext=500
		#az_lim_ext=np.pi/2
		#rng_lim_cls=10000
		
		bnd=np.where(np.logical_or( np.logical_and(sar>=rng_lim_cls,np.logical_and(sar*np.cos(abs(rad2air_az-(miaz+0.5*(mxaz-miaz))))<=yankee_s +gat_lim_ext, np.logical_and(rad2air_az<=mxaz+az_lim_ext,rad2air_az>=miaz-az_lim_ext))),np.logical_and(sar>0,sar<=rng_lim_cls,))) #((1284159,) ~300secs #this
		####Where s>10km s*cos(az-min+err)<s_y+err, alt>min_hi-err, alt<max_hi+err, if s under 10km alt

		#print sar.shape, rad2air_az.shape
		#raw_input("yo")
		#print("bnd", time.time())
		#print lat[bnd]
		#print(lat.shape, lat[bnd].shape) #((706, 2663, 301), (6733528,))
		#raw_input("stop")
		#np.savetxt("array.txt", lat[bnd], fmt="%s")		
		if any(map(len,bnd)):
			print("good")
		else: 
			print("Above the radar beam")
			bnd=np.where(np.logical_and(sar>=0,np.logical_and(sar*np.cos(abs(rad2air_az-(miaz+0.5*(mxaz-miaz))))<=yankee_s +gat_lim_ext, np.logical_and(rad2air_az<=mxaz+az_lim_ext,rad2air_az>=miaz-az_lim_ext))))
			#less stringent limits
			#bnd=np.where(np.logical_and(sar>=0,np.logical_and(rad2air_az<=mxaz+np.pi/2,rad2air_az>=miaz-np.pi/2 )))
			if any(map(len,bnd)): 
				print("good, take 2")
			else:
				butt=np.where(np.logical_and(sar>=0,sar*np.cos(abs(rad2air_az-(miaz+0.5*(mxaz-miaz))))<=yankee_s +gat_lim_ext)) 
				#mutt=np.where(alt>=yankee_hi -alt_lim_ext)
				putt=np.where(np.logical_and(rad2air_az<=mxaz+az_lim_ext,rad2air_az>=miaz-az_lim_ext))
				print(np.asarray(butt).shape,np.asarray(putt).shape)
				print("out of bounds")
				return
				raw_input("stop")

		
		bndma=np.zeros(sar.shape)
		bndma[bnd]=1
		bndmic=ma.masked_where(bndma==0,sar)
		
		####################################################################################
		##Collocation between aircraft and radar voxels
		####################################################################################
		papa=invertgeoidmulti(lat[bnd],lon[bnd],mid_lat[bnd],mid_lon[bnd])
		#calculates the difference between the limited voxels and 
		#print("papa invert", time.time())#, lat.shape,mid_lat.shape	    
		
		#papa=invertgeoidhlf(lat[o],lon[o], mid_lat,mid_lon)
		papas=papa[1]
			
		#print("dis start",time.time()) #np.array				    
		dis=(np.float64(papas)**2 + np.float64(mid_hi[bnd])**2)**0.5
		#cheat trignometry of direct distance between radar voxel and aircraft	
		#dis=(np.float64(papas)**2 + (np.float64(alt)-np.float64(mid_hi*1000))**2)**0.5
		#lemons[bnd]=dis
		print(dis.shape)#, lemons.shape) #(1295299,) (157, 2663, 66)
			
		#print("dis end", dis.shape, time.time()) #(1295299,)
		dish=abs(np.float64(np.float64(mid_hi[bnd])))
		#difference in altitude between aircraft and center of the voxel
		ei=np.unique(elindy[bnd]) #makes sure there is no repetition		
		ri=np.unique(rngindy[bnd])
		ti=np.unique(atimy[bnd])
		#indexes of elevation, range and time
		erts=np.empty((len(ei),len(ri),len(ti)))
		print("erts", erts.shape)
		print([bnd][0][0][0])#0
		print("plop")
		plop=vindex[ti]
		#the collocated aircraft indexes
		Radnam=[filenama]*len(vindex[ti])

		###
		klost=9
		#For efficiency, only the closest voxels are outputted
		#fcd=np.argpartition(dis,klost)
		fcd=[]
		fcs=[]
		fch=[]
		fcr2az=[]
		fcr2as=[]
		fcv2ada=[]		
		fcws=[]
		fcys=[]
		fcwh=[]
		fcyh=[]
		fcyaz=[]
		fcwaz=[]
		fcmla=[]
		fcmlo=[]
		fcv2az=[]
		fcri=[]
		fcei=[]
		fcdn=[]
		fcsn=[]
		fchn=[]
		fcr2azn=[]
		fcr2asn=[]
		fcv2adan=[]		
		fcwsn=[]
		fcysn=[]
		fcwhn=[]
		fcyhn=[]
		fcyazn=[]
		fcwazn=[]
		fcmlan=[]
		fcmlon=[]
		fcv2azn=[]
		fcrin=[]
		fcein=[]

		hcd=[]
		hcs=[]
		hch=[]
		hcr2az=[]		
		hcr2as=[]
		hcv2ada=[]		
		hcws=[]
		hcys=[]
		hcwh=[]
		hcyh=[]
		hcyaz=[]
		hcwaz=[]
		hcmla=[]
		hcmlo=[]
		hcv2az=[]
		hcri=[]
		hcei=[]
		hcdn=[]
		hcsn=[]
		hchn=[]
		hcr2azn=[]
		hcr2asn=[]
		hcv2adan=[]		
		hcwsn=[]
		hcysn=[]
		hcwhn=[]
		hcyhn=[]
		hcyazn=[]
		hcwazn=[]
		hcmlan=[]
		hcmlon=[]
		hcv2azn=[]
		hcrin=[]
		hcein=[]

		scd=[]
		scs=[]
		sch=[]
		scr2az=[]
		scr2as=[]
		scv2ada=[]		
		scws=[]
		scys=[]
		scwh=[]
		scyh=[]
		scyaz=[]
		scwaz=[]
		scmla=[]
		scmlo=[]
		scv2az=[]
		scri=[]
		scei=[]
		scdn=[]
		scsn=[]
		schn=[]
		scr2azn=[]
		scr2asn=[]
		scv2adan=[]		
		scwsn=[]
		scysn=[]
		scwhn=[]
		scyhn=[]
		scyazn=[]
		scwazn=[]
		scmlan=[]
		scmlon=[]
		scv2azn=[]
		scrin=[]
		scein=[]

		#print("time", time.time())
		print("le",np.isin(atimy[bnd],ti).shape) #('le', 6733528)
		print("hm")
		
		for ink in range(0,len(ti)):
		#loops through all times which are close
			#rnd=np.isin(atimy[bnd],ink) #where times match
			rnd=np.isin(atimy[bnd],ti[ink]) #where times match
			
			print(atimy[bnd][0:5])
			###Closest few by direct direction
			#f refers to the closest voxels by direct distance (DD)
			fcd.append(np.partition(dis[rnd],klost)[:klost]) #Nearest direct distance 
			print(fcd,len(dis[rnd]), len(atimy[bnd]),ink)
			fcs.append(papas[np.argpartition(dis[rnd],klost)[:klost]]) #voxel to aircraft s 
			fch.append(mid_hi.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #voxel alt  
			fcr2az.append(rad2air_az.ravel()[np.argpartition(dis[rnd],klost)[:klost]])#radar to  air azi
			fcv2az.append(papa[0][np.argpartition(dis[rnd],klost)[:klost]]) #voxel to aircraft azi 
			fcr2as.append(sar.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #radar to aircraft s
			fcv2ada.append(dish.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #difference in aircraft & voxel alt 
			fcws.append(whisky_s.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #radar to whisky(min range) s
			fcys.append(yankee_s.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #radar to yankee(max range) s
			fcwh.append(whisky_hi.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #max voxel altitude
			fcyh.append(yankee_hi.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #min voxel altitude
			fcyaz.append(mxaz.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #max voxel azimuth
			fcwaz.append(miaz.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #min voxel azimuth
			fcmla.append(mid_lat.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #voxel latitude
			fcmlo.append(mid_lon.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #voxel longitude
			fcri.append(rngindy.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #radar range index
			fcei.append(elindy.ravel()[np.argpartition(dis[rnd],klost)[:klost]]) #radar elevation
			###Minimum by direct distance
			fcdn.append(min(fcd[ink])) #Direct Distance
			cumin=np.where(fcd[ink]==min(dis[rnd])) #finds the minimum direct distance
			cumin=np.asarray(cumin[0]) #provides sorted index number for the minimum direct distance
			fcsn.append(fcs[ink][cumin[0]]) #s 
			fchn.append(fch[ink][cumin[0]]) #altitude
			fcr2azn.append(fcr2az[ink][cumin[0]]) #radar to aircraft azimuth
			fcr2asn.append(fcr2as[ink][cumin[0]]) #radar to aircraft s
			fcv2adan.append(fcv2ada[ink][cumin[0]]) #difference in aircraft & voxel alt	
			fcwsn.append(fcws[ink][cumin[0]]) #radar to min range gate s
			fcysn.append(fcys[ink][cumin[0]]) #radar to max range gate s
			fcwhn.append(fcwh[ink][cumin[0]]) #max voxel altitude
			fcyhn.append(fcyh[ink][cumin[0]]) #min voxel altitude
			fcyazn.append(fcyaz[ink][cumin[0]]) #max azimuth 
			fcwazn.append(fcwaz[ink][cumin[0]]) #min azimuth
			fcmlan.append(fcmla[ink][cumin[0]]) #voxel latitude
			fcmlon.append(fcmlo[ink][cumin[0]]) #voxel longitude
			fcv2azn.append(fcv2az[ink][cumin[0]]) #voxel to aircraft azimuth
			fcrin.append(fcri[ink][cumin[0]]) #radar range index
			fcein.append(fcei[ink][cumin[0]]) #radar elevation index

			###Closest few by nearest altitude
			hch.append(mid_hi.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #the nearest altitude
			hcd.append(dis[np.argpartition(dish[rnd],klost)[:klost]]) #voxel to aircraft direct distance
			hcs.append(papas[np.argpartition(dish[rnd],klost)[:klost]]) #voxel to aircraft s
			hcr2az.append(rad2air_az.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #radar to aircraft azimuth
			hcv2az.append(papa[0][np.argpartition(dish[rnd],klost)[:klost]]) #voxel to aircraft azimuth
			hcr2as.append(sar.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #radar to aircraft s
			hcv2ada.append(np.partition(dish[rnd],klost)[:klost]) #difference in aircraft & voxel alt
			hcws.append(whisky_s.ravel()[np.argpartition(dish[rnd],klost)[:klost]])  #radar to whisky(min range) s
			hcys.append(yankee_s.ravel()[np.argpartition(dish[rnd],klost)[:klost]])  #radar to yankee(max range) s
			hcwh.append(whisky_hi.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #max voxel altitude
			hcyh.append(yankee_hi.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #min voxel altitude
			hcyaz.append(mxaz.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #max voxel azimuth
			hcwaz.append(miaz.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #min voxel azimuth
			hcmla.append(mid_lat.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #voxel latitude
			hcmlo.append(mid_lon.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #voxel longitude
			hcri.append(rngindy.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #radar range index
			hcei.append(elindy.ravel()[np.argpartition(dish[rnd],klost)[:klost]]) #radar elevation index	
			###Minimum by nearest altitude
			hchn.append(min(hch[ink])) #minimum altitude
			clove=np.where(hch[ink]==min(hch[ink]))
			clove=np.asarray(clove[0]) #index for min altitude
			hcsn.append(hcs[ink][clove[0]]) #voxel to aircraft s
			hcdn.append(hcd[ink][clove[0]]) #voxel to aircraft direct distance
			hcr2azn.append(hcr2az[ink][clove[0]]) #radar to aircraft azimuth
			hcr2asn.append(hcr2as[ink][cumin[0]]) #radar to aircraft s
			hcv2adan.append(hcv2ada[ink][cumin[0]]) #difference in aircraft & voxel alt			
			hcwsn.append(hcws[ink][clove[0]]) #radar to whisky s
			hcysn.append(hcys[ink][clove[0]]) #radar to yankee s
			hcwhn.append(hcwh[ink][clove[0]]) #max voxel altitude
			hcyhn.append(hcyh[ink][clove[0]]) #min voxel altitude
			hcyazn.append(hcyaz[ink][clove[0]]) #max voxel azimuth
			hcwazn.append(hcwaz[ink][clove[0]]) #min voxel azimuth
			hcmlan.append(hcmla[ink][clove[0]]) #voxel latitude
			hcmlon.append(hcmlo[ink][clove[0]]) #voxel longitude
			hcv2azn.append(hcv2az[ink][clove[0]]) #voxel to aircraft azimuth
			hcrin.append(hcri[ink][clove[0]]) #radar range index
			hcein.append(hcei[ink][clove[0]]) #radar elevation index

			###Closest few by s 
			scs.append(np.partition(papas[rnd],klost)[:klost]) #the nearest by distance along the Earth
			scd.append(dis[np.argpartition(papas[rnd],klost)[:klost]]) #Voxel to Aircraft Direct Distance
			sch.append(mid_hi.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #Voxel Altitude
			scr2az.append(rad2air_az.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #Radar to Aircraft Azimuth
			scv2az.append(papa[0][np.argpartition(papas[rnd],klost)[:klost]]) #Voxel to Aircraft Azimuth
			scr2as.append(sar.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #radar to aircraft s
			scv2ada.append(dish.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #difference in aircraft & voxel alt
			scws.append(whisky_s.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #radar to whisky s
			scys.append(yankee_s.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #radar to yankee s
			scwh.append(whisky_hi.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #max voxel altitude
			scyh.append(yankee_hi.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #min voxel altitude
			scyaz.append(mxaz.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #min voxel azimuth
			scwaz.append(miaz.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #max voxel azimuth
			scmla.append(mid_lat.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #voxel latitude
			scmlo.append(mid_lon.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #voxel longitude
			scri.append(rngindy.ravel()[np.argpartition(papas[rnd],klost)[:klost]]) #radar range index
			scei.append(elindy.ravel()[np.argpartition(papas[rnd],klost)[:klost]])	#radar elevation index
			

			scsn.append(min(scs[ink])) #minimum s
			mace=np.where(scs[ink]==min(scs[ink])) 
			mace=np.asarray(mace[0]) #index for minimum s
			scdn.append(scd[ink][mace[0]]) #Voxel to Aircraft Direct Distance
			schn.append(sch[ink][mace[0]]) #Voxel Altitude
			scr2azn.append(scr2az[ink][mace[0]]) #Radar to Aircraft Azimuth	
			scr2asn.append(scr2as[ink][cumin[0]]) #radar to aircraft s
			scv2adan.append(scv2ada[ink][cumin[0]]) #difference in aircraft & voxel alt		
			scwsn.append(scws[ink][mace[0]]) #radar to whisky s
			scysn.append(scys[ink][mace[0]]) #radar to yankee s
			scwhn.append(scwh[ink][mace[0]]) #max voxel altitude
			scyhn.append(scyh[ink][mace[0]]) #min voxel altitude
			scyazn.append(scyaz[ink][mace[0]]) #min voxel azimuth
			scwazn.append(scwaz[ink][mace[0]]) #max voxel azimuth
			scmlan.append(scmla[ink][mace[0]]) #voxel latitude
			scmlon.append(scmlo[ink][mace[0]]) #voxel longitude
			scv2azn.append(scv2az[ink][mace[0]]) #voxel to aircraft azimuth
			scrin.append(scri[ink][mace[0]]) #radar range index
			scein.append(scei[ink][mace[0]]) #radar elevation index
			
		#print("inktime", time.time())
		
		####Makes sure they are all in the array format
		fcd=np.asarray(fcd)
		
		fcs=np.asarray(fcs)
		
		fch=np.asarray(fch)
		fcr2az=np.asarray(fcr2az)
		fcws=np.asarray(fcws)
		fcys=np.asarray(fcys)
		fcwh=np.asarray(fcwh)
		fcyh=np.asarray(fcyh)
		fcyaz=np.asarray(fcyaz)
		fcwaz=np.asarray(fcwaz)
		fcmla=np.asarray(fcmla)
		fcmlo=np.asarray(fcmlo)
		fcv2az=np.asarray(fcv2az)
		fcr2as=np.asarray(fcr2as) 
		fcv2ada=np.asarray(fcv2ada) 	
		fcri=np.asarray(fcri)
		fcei=np.asarray(fcei)

		fchn=np.asarray(fchn)
		fcr2azn=np.asarray(fcr2azn)
		fcwsn=np.asarray(fcwsn)
		fcysn=np.asarray(fcysn)
		fcwhn=np.asarray(fcwhn)
		fcyhn=np.asarray(fcyhn)
		fcyazn=np.asarray(fcyazn)
		fcwazn=np.asarray(fcwazn)
		fcmlan=np.asarray(fcmlan)
		fcmlon=np.asarray(fcmlon)
		fcv2azn=np.asarray(fcv2azn)
		fcr2asn=np.asarray(fcr2asn) 
		fcv2adan=np.asarray(fcv2adan) 	
		fcrin=np.asarray(fcrin)
		fcein=np.asarray(fcein)

		hcd=np.asarray(hcd)
		hcs=np.asarray(hcs)
		hch=np.asarray(hch)
		hcr2az=np.asarray(hcr2az)
		hcws=np.asarray(hcws)
		hcys=np.asarray(hcys)
		hcwh=np.asarray(hcwh)
		hcyh=np.asarray(hcyh)
		hcyaz=np.asarray(hcyaz)
		hcwaz=np.asarray(hcwaz)
		hcmla=np.asarray(hcmla)
		hcmlo=np.asarray(hcmlo)
		hcv2az=np.asarray(hcv2az)
		hcr2as=np.asarray(hcr2as) 
		hcv2ada=np.asarray(hcv2ada) 	
		hcri=np.asarray(hcri)
		hcei=np.asarray(hcei)

		hchn=np.asarray(hchn)
		hcr2azn=np.asarray(hcr2azn)
		hcwsn=np.asarray(hcwsn)
		hcysn=np.asarray(hcysn)
		hcwhn=np.asarray(hcwhn)
		hcyhn=np.asarray(hcyhn)
		hcyazn=np.asarray(hcyazn)
		hcwazn=np.asarray(hcwazn)
		hcmlan=np.asarray(hcmlan)
		hcmlon=np.asarray(hcmlon)
		hcv2azn=np.asarray(hcv2azn)
		hcr2asn=np.asarray(hcr2asn) 
		hcv2adan=np.asarray(hcv2adan)
		hcrin=np.asarray(hcrin)
		hcein=np.asarray(hcein)

		scd=np.asarray(scd)
		scs=np.asarray(scs)
		sch=np.asarray(sch)
		scr2az=np.asarray(scr2az)
		scr2as=np.asarray(scr2as) 
		scv2ada=np.asarray(scv2ada)
		scws=np.asarray(scws)
		scys=np.asarray(scys)
		scwh=np.asarray(scwh)
		scyh=np.asarray(scyh)
		scyaz=np.asarray(scyaz)
		scwaz=np.asarray(scwaz)
		scmla=np.asarray(scmla)
		scmlo=np.asarray(scmlo)
		scv2az=np.asarray(scv2az)
		scri=np.asarray(scri)
		scei=np.asarray(scei)

		schn=np.asarray(schn)
		scr2azn=np.asarray(scr2azn)
		scwsn=np.asarray(scwsn)
		scysn=np.asarray(scysn)
		scwhn=np.asarray(scwhn)
		scyhn=np.asarray(scyhn)
		scyazn=np.asarray(scyazn)
		scwazn=np.asarray(scwazn)
		scmlan=np.asarray(scmlan)
		scmlon=np.asarray(scmlon)
		scv2azn=np.asarray(scv2azn)
		scr2asn=np.asarray(scr2asn) 
		scv2adan=np.asarray(scv2adan)
		scrin=np.asarray(scrin)
		scein=np.asarray(scein)

		
		klst=np.arange(0,klost)
		#creates new index for the number of closest voxels
		
		combo=xr.Dataset(data_vars={"Citizen Science Index": (("tim"),plop),
                            "Citizen Science Latitude": (("tim"),lat),
                            "Citizen Science Longitude": (("tim"),lon),
			    "Radar Name":(("tim"),Radnam),
                            "Radar Voxel Mid Latitude (DD)": (("tim","cls"),fcmla),
                            "Radar Voxel Mid Longitude (DD)": (("tim","cls"),fcmlo),
                            "Radar Voxel Mid Altitude (DD)": (("tim","cls"),fch),
                            "Radar Voxel Min Distance Along the Earth (DD)": (("tim","cls"),fcws),
                            "Radar Voxel Max Altitude (DD)": (("tim","cls"),fcwh),
                            "Radar Voxel Max Distance Along the Earth (DD)": (("tim","cls"),fcys),
			    "Radar to Aircraft S (DD)": (("tim","cls"), fcr2as),
			    "Difference in Altitude between Voxel and Aircraft (DD)": (("tim","cls"), fcv2ada),
                            "Radar Voxel Min Altitude (DD)": (("tim","cls"),fcyh),
                            "Radar Voxel Range Index (DD)": (("tim","cls"),fcri),
                            "Radar Voxel Elevation Index (DD)": (("tim","cls"),fcei),
                            "Distance Along the Earth Voxel to Aircraft (DD)": (("tim","cls"),fcs),
                            "Azimuth from Radar to Aircraft (DD)": (("tim","cls"),fcr2az),
			    "Azimuth from Voxel to Aircraft (DD)": (("tim","cls"),fcv2az),
			    "Direct Distance Voxel to Aircraft (DD)": (("tim","cls"),fcd),

                            "Radar Voxel Mid Latitude (DDmin)": (("tim"),fcmlan),
			    "Radar Voxel Mid Longitude (DDmin)": (("tim"),fcmlon),
                            "Radar Voxel Mid Altitude (DDmin)": (("tim"),fchn),
                            "Radar Voxel Min Distance Along the Earth (DDmin)": (("tim"),fcwsn),
                            "Radar Voxel Max Altitude (DDmin)": (("tim"),fcwhn),
                            "Radar Voxel Max Distance Along the Earth (DDmin)": (("tim"),fcysn),
                            "Radar Voxel Min Altitude (DDmin)": (("tim"),fcyhn),
                            "Radar Voxel Range Index (DDmin)": (("tim"),fcrin),
                            "Radar Voxel Elevation Index (DDmin)": (("tim"),fcein),
                            "Distance Along the Earth Radar to Aircraft (DDmin)": (("tim"),fcsn),
                            "Azimuth from Radar to Aircraft (DDmin)": (("tim"),fcr2azn),
			    "Azimuth from Voxel to Aircraft (DDmin)": (("tim"),fcv2azn),
			    "Radar to Aircraft S (DDmin)": (("tim"), fcr2asn),
			    "Difference in Altitude between Voxel and Aircraft (DDmin)": (("tim"), fcv2adan),
			    "Direct Distance Voxel to Aircraft (DDmin)": (("tim"),fcdn),

                            "Radar Voxel Mid Latitude (ByAlt)": (("tim","cls"),hcmla),
                            "Radar Voxel Mid Longitude (ByAlt)": (("tim","cls"),hcmlo),
                            "Radar Voxel Mid Altitude (ByAlt)": (("tim","cls"),hch),
                            "Radar Voxel Min Distance Along the Earth (ByAlt)": (("tim","cls"),hcws),
                            "Radar Voxel Max Altitude (ByAlt)": (("tim","cls"),hcwh),
                            "Radar Voxel Max Distance Along the Earth (ByAlt)": (("tim","cls"),hcys),
			    "Radar to Aircraft S (ByAlt)": (("tim","cls"), hcr2as),
			    "Difference in Altitude between Voxel and Aircraft (ByAlt)": (("tim","cls"), hcv2ada),
                            "Radar Voxel Min Altitude (ByAlt)": (("tim","cls"),hcyh),
                            "Radar Voxel Range Index (ByAlt)": (("tim","cls"),hcri),
                            "Radar Voxel Elevation Index (ByAlt)": (("tim","cls"),hcei),
                            "Distance Along the Earth Radar to Aircraft (ByAlt)": (("tim","cls"),hcs),
                            "Azimuth from Radar to Aircraft (ByAlt)": (("tim","cls"),hcr2az),
			    "Azimuth from Voxel to Aircraft (ByAlt)": (("tim","cls"),hcv2az),
			    "Direct Distance Voxel to Aircraft (ByAlt)": (("tim","cls"),hcd),
			     
                            "Radar Voxel Mid Latitude (ByAltmin)": (("tim"),hcmlan),
                            "Radar Voxel Mid Longitude (ByAltmin)": (("tim"),hcmlon),
                            "Radar Voxel Mid Altitude (ByAltmin)": (("tim"),hchn),
                            "Radar Voxel Min Distance Along the Earth (ByAltmin)": (("tim"),hcwsn),
                            "Radar Voxel Max Altitude (ByAltmin)": (("tim"),hcwhn),
                            "Radar Voxel Max Distance Along the Earth (ByAltmin)": (("tim"),hcysn),
			    "Radar to Aircraft S (ByAltmin)": (("tim"), hcr2asn),
			    "Difference in Altitude between Voxel and Aircraft (ByAltmin)": (("tim"), hcv2adan),
                            "Radar Voxel Min Altitude (ByAltmin)": (("tim"),hcyhn),
                            "Radar Voxel Range Index (ByAltmin)": (("tim"),hcrin),
                            "Radar Voxel Elevation Index (ByAltmin)": (("tim"),hcein),
                            "Distance Along the Earth Radar to Aircraft (ByAltmin)": (("tim"),hcsn),
                            "Azimuth from Radar to Aircraft (ByAltmin)": (("tim"),hcr2azn),
			    "Azimuth from Voxel to Aircraft (ByAltmin)": (("tim"),hcv2azn),
			    "Direct Distance Voxel to Aircraft (ByAltmin)": (("tim"),hcdn),

                            "Radar Voxel Mid Latitude (ByS)": (("tim","cls"),scmla),
                            "Radar Voxel Mid Longitude (ByS)": (("tim","cls"),scmlo),
                            "Radar Voxel Mid Altitude (ByS)": (("tim","cls"),sch),
                            "Radar Voxel Min Distance Along the Earth (ByS)": (("tim","cls"),scws),
                            "Radar Voxel Max Altitude (ByS)": (("tim","cls"),scwh),
                            "Radar Voxel Max Distance Along the Earth (ByS)": (("tim","cls"),scys),
			    "Radar to Aircraft S (ByS)": (("tim","cls"), scr2as),
			    "Difference in Altitude between Voxel and Aircraft (ByS)": (("tim","cls"), scv2ada),
                            "Radar Voxel Min Altitude (ByS)": (("tim","cls"),scyh),
                            "Radar Voxel Range Index (ByS)": (("tim","cls"),scri),
                            "Radar Voxel Elevation Index (ByS)": (("tim","cls"),scei),
                            "Distance Along the Earth Radar to Aircraft (ByS)": (("tim","cls"),scs),
                            "Azimuth from Radar to Aircraft (ByS)": (("tim","cls"),scr2az),
			    "Azimuth from Voxel to Aircraft (ByS)": (("tim","cls"),scv2az),
			    "Direct Distance Voxel to Aircraft (ByS)": (("tim","cls"),scd),
			     
                            "Radar Voxel Mid Latitude (BySmin)": (("tim"),scmlan),
                            "Radar Voxel Mid Longitude (BySmin)": (("tim"),scmlon),
                            "Radar Voxel Mid Altitude (BySmin)": (("tim"),schn),
                            "Radar Voxel Min Distance Along the Earth (BySmin)": (("tim"),scwsn),
                            "Radar Voxel Max Altitude (BySmin)": (("tim",),scwhn),
                            "Radar Voxel Max Distance Along the Earth (BySmin)": (("tim"),scysn),
                            "Radar Voxel Min Altitude (BySmin)": (("tim"),scyhn),
			    "Radar to Aircraft S (BySmin)": (("tim"), scr2asn),
			    "Difference in Altitude between Voxel and Aircraft (BySmin)": (("tim"), scv2adan),
                            "Radar Voxel Range Index (BySmin)": (("tim"),scrin),
                            "Radar Voxel Elevation Index (BySmin)": (("tim"),scein),
                            "Distance Along the Earth Radar to Aircraft (BySmin)": (("tim"),scsn),
                            "Azimuth from Radar to Aircraft (BySmin)": (("tim"),scr2azn),
			    "Azimuth from Voxel to Aircraft (BySmin)": (("tim"),scv2azn),
			    "Direct Distance Voxel to Aircraft (BySmin)": (("tim"),scdn)},
                  coords={"cls":klst,
                      "tim":ti}#newtime1[plop]}#ti}
                 )
		
		print("about to file",filenama[0:-3],str(dat)) 
		#raw_input("blurgh")
		#combo.to_netcdf(filenama+"Closest.nc")
		combo.to_netcdf(str(dat)+filenama[0:-3]+"Closest.nc",unlimited_dims=["Aircraft Time"])
		#raw_input("RedoneBounds_STOP")		
		
		print("kermit",kermit, time.time())
	
	except KeyError:
		print("KeyError, not using file",filenama)
	#except IndexError:
	#	print("IndexError, not using file",filenama)
	
	print(time.time(),"end",filenama)
	return
#"""


def invertgeoidmulti(lat1,lon1,lat2,lon2):
	print("invertgeoid_begin", time.time())
	lin1a=lat1.ravel()#.flatten()
	lin1o=lon1.ravel()#.flatten()
	lin2a=lat2.ravel()#.flatten()
	lin2o=lon2.ravel()#.flatten()
	#print("ravels_done", time.time())
	corner=[Geodesic.WGS84.Inverse(iw,ix,iy,iz,outmask=1929).values() for (iw,ix,iy,iz) in zip(lin1a,lin1o,lin2a,lin2o)]
	#print("geodesic_inverse_done", time.time())
	cn=[]	
	cn.append([list(corner[ji]) for ji in range(0,len(lin1a))])
	#print("appends_done", time.time())
	print("Reality check")
	print(cn[0][0][0],cn[0][0][1],cn[0][0][2],cn[0][0][3],cn[0][0][4],cn[0][0][5],cn[0][0][6])
	#raw_input("paws")
	azzy=[cn[0][noppy][5] for noppy in range(0,len(cn[0]))]
	azzy=np.reshape(azzy,np.shape(lat1))
	ss=[cn[0][noppy][2] for noppy in range(0,len(cn[0]))]
	ss=np.reshape(ss,np.shape(lat1))
	#print("reshapes_done", time.time())
	#print("shapiness", np.shape(ss))#, ss
	return(azzy,ss)

def invertgeoidhlf(lat1,lon1,lat2,lon2):
	lin2a=lat2.ravel()#.flatten()
	lin2o=lon2.ravel()#.flatten()
	corner=[Geodesic.WGS84.Inverse(lat1,lon1,iy,iz,outmask=1929).values() for (iy,iz) in zip(lin2a,lin2o)]
	cn=[]
	
	cn.append([list(corner[ji]) for ji in range(0,len(lin2a))])
	
	azzy=[cn[0][noppy][5] for noppy in range(0,len(cn[0]))]
	azzy=np.reshape(azzy,np.shape(lon2))
	ss=[cn[0][noppy][2] for noppy in range(0,len(cn[0]))]
	ss=np.reshape(ss,np.shape(lon2))
	return(azzy,ss)

#maths(filenama,dat,lat,lon,quick,gat_lim_ext,az_lim_ext,rng_lim_cls,klost)
