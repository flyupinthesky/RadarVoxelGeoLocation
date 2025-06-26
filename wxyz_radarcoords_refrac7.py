#!/usr/bin/env python
_citation_text = """
## You are using the Radar Voxel Location Script,
## an open source library for working with weather radar data.
##
## If you use this software to prepare a publication, please cite:
##
## Freya I. Addison, University of Leeds,  https://orcid.org/0000-0001-8439-290X"""
print(_citation_text)

import os
import numpy as np
import netCDF4
from netCDF4 import Dataset
import time
import csv

# import simplekml
from scipy.special import ellipkinc
from geographiclib.geomath import Math
from geographiclib.constants import Constants
from geographiclib.geodesiccapability import GeodesicCapability
from geographiclib.geodesic import Geodesic

import scipy.special

import itertools
import h5py
import os.path
import sys
import numpy.ma as ma
import pandas as pd
import xarray as xr

# from joblib import Parallel, delayed
import multiprocessing as mpl
from multiprocessing import Process, JoinableQueue

np.set_printoptions(threshold=np.inf)


def vcoords(
    filenama,
    dat,
    radtimestamp,
    latitude_r,
    longitude_r,
    rad_height_above_msl,
    rad_ant_length,
    rng0,
    gat,
    el0_rad0,
    beamwidthv,
    rad_azi,
    beamwidthh,
    N_s,
):
    print(time.time(), "start", filenama)
    starttim = time.time()
    try:
        lon_ra = np.deg2rad(longitude_r)
        # longitude of radar in radians
        # latitude of radar in decimal degrees north
        lat_ra = np.deg2rad(latitude_r)

        a = 6378137.0000  # /1000
        # semi-major axis of WGS-84 ellipsoid in km from https://www.mathworks.com/help/map/ref/wgs84ellipsoid.html
        b = 6356752.31424518  # /1000
        # semi-minor axis of WGS-84 ellipsoid in km from https://www.mathworks.com/help/map/ref/wgs84ellipsoid.html

        f = (a - b) / a
        # flattening of WGS-84 geoid
        R = np.sqrt((a**2) / (1 + np.sin(lat_ra) * ((1 / (1 - f) ** 2) - 1) ** 2))
        # radius of WGS-8 ellipsoid at coordinates of radar

        nin_rad = 0.5 * np.pi
        # shorthand for half pi in radians

        bwv = np.deg2rad(beamwidthv)
        bwh = np.deg2rad(beamwidthh)

        ab = np.sqrt((a**2) - (b**2))
        # shorthand of pythagoras theorem of a & b
        e = ab / a
        # eccentricity of WGS-84 Earth, e**2 = (a**2 - b**2)/a**2 from
        N = a * (1 - e**2 * np.sin(lat_ra) ** 2) ** -0.5  # Ellipsoid Normal at radar
        # radar height above mean sea level
        el_rad0 = np.deg2rad(el0_rad0)
        altitude_r = (
            rad_ant_length * (np.sin(el_rad0))
        ) + rad_height_above_msl  # /1000
        # altitude of radar: length of antenna*sin(elevation angle+90) +the height above sea level offset at CAMRa's location in km
        #print(altitude_r.shape,rng0.shape22)
        altitude_r = np.reshape(altitude_r, (len(altitude_r), 1))
        altitude_r = np.tile(altitude_r, (1, len(rng0)))
        N_1 = N + altitude_r

        rho_1 = (a * (1 - e**2)) * (1 - e**2 * np.sin(lat_ra)) ** (
            -3 / 2
        )  # Instantaneous radius of curvature
        ####WE NEED A DIAGRAM TO ILLUSTRATE THESE TERMS IN THE DOCUMENTATION
        # rho_cc=206264.806247096355156
        # rho seconds per radian The 3D Global Spatial Data Model pg 162
        print(rad_azi.shape)
        maxazi = np.deg2rad(rad_azi) + (0.5 * bwh)
        # print("maz", maxazi.shape)

        # to get the full beam ray, the horizontal beam width is added onto azimuth
        maxazi = np.reshape(maxazi, (len(maxazi), 1))
        maxazi = np.tile(maxazi, (1, len(rng0)))
        # print("maz2",maxazi.shape)
        # raw_input("stop")
        # This is reshapes the azimuth to a 2d array (el,rng)
        minazi = np.deg2rad(rad_azi) - (0.5 * bwh)
        minazi = np.reshape(minazi, (len(minazi), 1))
        minazi = np.tile(minazi, (1, len(rng0)))
        mazi = np.deg2rad(rad_azi)
        mazi = np.reshape(mazi, (len(mazi), 1))
        mazi = np.tile(mazi, (1, len(rng0)))
        mazir = rad_azi
        mazir = np.reshape(mazir, (len(mazir), 1))
        mazir = np.tile(mazir, (1, len(rng0)))


        R_ra = R + altitude_r
        # Radius of the Earth at the radar location+the altitude of the radar

        el_ind0 = np.arange(len(el_rad0))
        # indexes of elevations
        rng_ind0 = np.arange(len(rng0))
        # indexes of ranges
        azi_ind0 = np.arange(len(rad_azi))
        el_rad = np.reshape(el_rad0, (len(el_rad0), 1))
        el_rad = np.tile(el_rad, (1, len(rng0)))

        el_ind = np.reshape(el_ind0, (len(el_ind0), 1))
        el_ind = np.tile(el_ind, (1, len(rng0)))

        rng = np.tile(rng0, (len(el_rad0), 1))
        
        rng_ind = np.tile(rng_ind0, (len(el_rad0), 1))
        azi_ind = np.tile(azi_ind0, (len(rng0), 1))
        azi_ind = zip(*azi_ind)
        azi_ind = np.asarray(azi_ind)
        rtime = np.tile(radtimestamp, (len(rng0), 1))
        rtime = zip(*rtime)
        rtime = np.asarray(rtime)

        h_s = altitude_r
        mid_y = np.sqrt(N_1**2 + rng**2 + 2 * N_1 * rng * np.cos(el_rad + nin_rad))
        mid_c = np.zeros((len(el_rad0), len(rng0)))
        
        midFE=flatearth(rng,el_rad,rad_height_above_msl)
        midFEs=midFE[0] #s
        midFSa=midFE[1] #alt

        mazie= np.where((np.logical_and(mazir.ravel() > 180, mazir.ravel() < 360)), mazir.ravel() - 360, mazir.ravel())
        FEkest = [Geodesic.WGS84.Direct(latitude_r, longitude_r, fai, fsi, outmask=1929).values() for (fai, fsi) in zip(mazie, midFEs)]
        FEkq = []
        FEkq.append([list(FEkest[ji]) for ji in range(0, len(midFEs))])
        FEphi3 = [FEkq[0][Fnop][5] for Fnop in range(0, len(FEkq[0]))]
        FElat = np.reshape(FEphi3, np.shape(rng))

        FElon = [FEkq[0][Fnop][6] for Fnop in range(0, len(FEkq[0]))]
        FElon = np.reshape(FElon, np.shape(rng))
        midFEs=np.reshape(midFEs,np.shape(rng))
        midFSa=np.reshape(midFSa,np.shape(rng))


        midSE=spherical(R_ra,rng,el_rad,rad_height_above_msl)
        midSEs=midSE[0] #s
        midSSa=midSE[1] #a
        
        SEkest = [Geodesic.WGS84.Direct(latitude_r, longitude_r, sai, ssi, outmask=1929).values() for (sai, ssi) in zip(mazie.ravel(), midSEs)]
        SEkq = []
        SEkq.append([list(SEkest[ji]) for ji in range(0, len(midSEs))])
        SEphi3 = [SEkq[0][Snop][5] for Snop in range(0, len(SEkq[0]))]
        SElat = np.reshape(SEphi3, np.shape(rng))
        SElon = [SEkq[0][Snop][6] for Snop in range(0, len(SEkq[0]))]
        SElon = np.reshape(SElon, np.shape(rng))
        midSEs=np.reshape(midSEs,np.shape(rng))
        midSSa=np.reshape(midSSa,np.shape(rng))
        
        midSER=sphericalR(R_ra,rng,el_rad,rad_height_above_msl,N_s)
        print(rad_height_above_msl,R_ra[0][0])
        
        midSEsR=midSER[0] #s
        midSSaR=midSER[1] #alt
        
        SERkest = [Geodesic.WGS84.Direct(latitude_r, longitude_r, sai, ssi, outmask=1929).values() for (sai, ssi) in zip(mazie.ravel(), midSEsR)]
        SERkq = []
        SERkq.append([list(SERkest[ji]) for ji in range(0, len(midSEsR))])
        SERphi3 = [SERkq[0][Snop][5] for Snop in range(0, len(SERkq[0]))]
        SERlat = np.reshape(SERphi3, np.shape(rng))
        SERlon = [SERkq[0][Snop][6] for Snop in range(0, len(SERkq[0]))]
        SERlon = np.reshape(SERlon, np.shape(rng))
        midSEsR=np.reshape(midSEsR,np.shape(rng))
        midSSaR=np.reshape(midSSaR,np.shape(rng))
        
        for fxt0 in range(0, len(el_rad0)):
            # print("range",ranges[j])
            for hl0 in range(0, len(rng0)):
                if el_rad[fxt0][hl0] == 0:
                    mid_c[fxt0][hl0] = np.arctan(rng[fxt0][hl0] / N_1[fxt0][hl0])

                else:
                    mid_c[fxt0][hl0] = np.arccos(
                        (
                            (rng[fxt0][hl0] / 1000) ** 2
                            + (N_1[fxt0][hl0] / 1000) ** 2
                            - (mid_y[fxt0][hl0] / 1000) ** 2
                        )
                        / (2 * (rng[fxt0][hl0] / 1000) * (N_1[fxt0][hl0] / 1000))
                    )

                if np.isnan(mid_c[fxt0][hl0]) == True:
                    # print("nan",mid_c)
                    mid_c[fxt0][hl0] = np.arcsin(
                        rng[fxt0][hl0]
                        * (np.sin(el_rad[fxt0][hl0] + nin_rad) / mid_y[fxt0][hl0])
                    )
                    # We do this for very small elevations where numpy can't compute the arccos, it is less accurate as we are more reliant on our derived value of mid_y

                else:
                    None
        print("ogc",len(mid_c[np.isnan(mid_c)]))
        #raw_input("stop")
        rngind = np.arange(0, len(rng0), 1)
        elind = np.arange(0, len(el_rad0), 1)

        print(mazi.shape,lat_ra.shape,rngind.shape,len(rng0))

        derek = Geoid(
            mazi,
            lat_ra,
            mid_c,
            a,
            e,
            latitude_r,
            longitude_r,
            rho_1,
            mid_y,
            altitude_r,
            rng,
            el_rad,
            R,
            N,
            f,
        )
        print("shapes", rng.shape, el_rad.shape)
        og_el_rad = el_rad
        derek_lat = derek[3]
        derek_lon = derek[4]
        derek_azi = derek[0]
        derek_s = derek[1]
        h_a = derek[2][:,:]  # altitude of voxel
 
        derekR = GeoidR(
            mazi,
            lat_ra,
            mid_c,
            a,
            e,
            latitude_r,
            longitude_r,
            rho_1,
            mid_y,
            altitude_r,
            rng,
            el_rad,
            R,
            N,
            f,
        )
        print("shapes", rng.shape, el_rad.shape)
        og_el_radR = el_rad
        derekR_lat = derekR[3]
        derekR_lon = derekR[4]
        derekR_azi = derekR[0]
        derekR_s = derekR[1]
        hR_a = derekR[2][:,:]  # altitude of voxel
        print("og_elradR",el_rad[-1][-1])


        Ellip=Ellipsoid(
            mazi,
            lat_ra,
            mid_c,
            a,
            e,
            latitude_r,
            longitude_r,
            rho_1,
            mid_y,
            altitude_r,
            rng,
            el_rad,
            R,
            N,
            f,
        )

        Ellip_lat = Ellip[3]
        Ellip_lon = Ellip[4]
        Ellip_azi = Ellip[0]
        Ellip_s = Ellip[1]
        Ellip_h_a = Ellip[2][:,:]
        
        
        Rv = np.sqrt((a**2) / (1 + np.sin(derek[3]) * ((1 / (1 - f) ** 2) - 1) ** 2))
        N_delt = -0.00732 * np.exp(0.005577 * N_s)
        N_b = np.zeros((len(el_rad0), len(rng0)))
        H_b = np.zeros((len(el_rad0), len(rng0)))
        N_b[np.where(h_a <= 6000)] = -0.00732 * np.exp(0.005577 * N_s)
        H_b[np.where(h_a <= 6000)] = 1000 / (np.log(N_s / (N_s + 1000 * N_delt)))
        N_b[np.where(np.logical_and(h_a <= 9500, h_a > 6000))] = 102.9
        H_b[np.where(np.logical_and(h_a <= 9500, h_a > 6000))] = (
            h_a[np.where(np.logical_and(h_a <= 9500, h_a > 6000))]
            - h_s[np.where(np.logical_and(h_a <= 9500, h_a > 6000))]
        ) / np.log(N_s / N_b[np.where(np.logical_and(h_a <= 9500, h_a > 6000))])
        N_b[np.where(np.logical_and(h_a < 100000, h_a > 9500))] = 66.65
        H_b[np.where(np.logical_and(h_a < 100000, h_a > 9500))] = (
            h_a[np.where(np.logical_and(h_a < 100000, h_a > 9500))]
            - h_s[np.where(np.logical_and(h_a < 100000, h_a > 9500))]
        ) / np.log(N_s / N_b[np.where(np.logical_and(h_a < 100000, h_a > 9500))])
        test1 = el_rad
        el_rad[np.where(H_b == 0)] = (
            el_rad[np.where(H_b == 0)]
            + np.deg2rad(
                (0.01721 * h_s[np.where(H_b == 0)] + 0.02374) * np.rad2deg(el_rad[np.where(H_b == 0)])**2
                + (
                    0.01601 * h_s[np.where(H_b == 0)] ** 2
                    + 0.02317 * h_s[np.where(H_b == 0)]
                    + 0.6126
                )
                * np.rad2deg(el_rad[np.where(H_b == 0)])
                + (0.2483 * h_s[np.where(H_b == 0)] + 1.738)
            )
            ** -1
        )  # https://www.hindawi.com/journals/ijap/2020/2438515/
        
        print("t1",test1[-1][-1])
        test2 = el_rad
        print("t2",test2[-1][-1])

        el_rad[np.where(H_b != 0)] = refrac(
            rng[np.where(H_b != 0)],
            R,
            h_s[np.where(H_b != 0)],
            N_s,
            Rv[np.where(H_b != 0)],
            h_a[np.where(H_b != 0)],
            H_b[np.where(H_b != 0)],
            el_rad[np.where(H_b != 0)],
            0.1,
        )  # https://www.researchgate.net/publication/277011644_Earth_Curvature_and_Atmospheric_Refraction_Effects_on_Radar_Signal_Propagation
        test3 = el_rad
        print("t3",test3[-1][-1])
        print("t4",refrac(
            rng[-1][-1],
            R,
            h_s[-1][-1],
            N_s,
            Rv[-1][-1],
            h_a[-1][-1],
            H_b[-1][-1],
            test1[-1][-1],
            0.1,
        ) )
        # reshapes to el,rng and their indexes to (el,rng)
        minel = el_rad - (0.5 * bwv)
        # minimum elevation, elevation + the vertical beamwidth
        maxel = el_rad + (0.5 * bwv)

        mingat = rng - gat
        # the beginning of the range gate
        maxgat = rng + gat

        # the end of the range gate
        mid_y = np.sqrt(N_1**2 + rng**2 + 2 * N_1 * rng * np.cos(el_rad + nin_rad))
        # mid for the center of the range gate, y for the

        # c is angle between the y, using the cosine rule

        min_y = np.sqrt(
            N_1**2 + mingat**2 + 2 * N_1 * mingat * np.cos(maxel + nin_rad)
        )

        miny = np.sqrt(
            N_1**2 + mingat**2 + 2 * N_1 * mingat * np.cos(minel + nin_rad)
        )

        maxy = np.sqrt(
            N_1**2 + maxgat**2 + 2 * N_1 * maxgat * np.cos(maxel + nin_rad)
        )

        max_y = np.sqrt(
            N_1**2 + maxgat**2 + 2 * N_1 * maxgat * np.cos(minel + nin_rad)
        )

        min_c = np.zeros((len(el_rad0), len(rng0)))
        max_c = np.zeros((len(el_rad0), len(rng0)))
        minc = np.zeros((len(el_rad0), len(rng0)))
        maxc = np.zeros((len(el_rad0), len(rng0)))

        print("start_it", time.time())
        for fxt in range(0, len(el_rad0)):
            # print("range",ranges[j])
            for hl in range(0, len(rng0)):
                if el_rad[fxt][hl] == 0:
                    mid_c[fxt][hl] = np.arctan(rng[fxt][hl] / N_1[fxt][hl])
                    min_c[fxt][hl] = np.arctan(mingat[fxt][hl] / N_1[fxt][hl])
                    max_c[fxt][hl] = np.arctan(maxgat[fxt][hl] / N_1[fxt][hl])
                    minc[fxt][hl] = np.arctan(mingat[fxt][hl] / N_1[fxt][hl])
                    maxc[fxt][hl] = np.arctan(maxgat[fxt][hl] / N_1[fxt][hl])

                else:
                    mid_c[fxt][hl] = np.arccos(
                        (
                            (rng[fxt][hl] / 1000) ** 2
                            + (N_1[fxt][hl] / 1000) ** 2
                            - (mid_y[fxt][hl] / 1000) ** 2
                        )
                        / (2 * (rng[fxt][hl] / 1000) * (N_1[fxt][hl] / 1000))
                    )

                    min_c[fxt][hl] = np.arccos(
                        (
                            (mingat[fxt][hl] / 1000) ** 2
                            + (N_1[fxt][hl] / 1000) ** 2
                            - (min_y[fxt][hl] / 1000) ** 2
                        )
                        / (2 * (mingat[fxt][hl] / 1000) * (N_1[fxt][hl] / 1000))
                    )

                    max_c[fxt][hl] = np.arccos(
                        (
                            (maxgat[fxt][hl] / 1000) ** 2
                            + (N_1[fxt][hl] / 1000) ** 2
                            - (max_y[fxt][hl] / 1000) ** 2
                        )
                        / (2 * (maxgat[fxt][hl] / 1000) * (N_1[fxt][hl] / 1000))
                    )

                    minc[fxt][hl] = np.arccos(
                        (
                            (mingat[fxt][hl] / 1000) ** 2
                            + (N_1[fxt][hl] / 1000) ** 2
                            - (miny[fxt][hl] / 1000) ** 2
                        )
                        / (2 * (mingat[fxt][hl] / 1000) * (N_1[fxt][hl] / 1000))
                    )

                    maxc[fxt][hl] = np.arccos(
                        (
                            (maxgat[fxt][hl] / 1000) ** 2
                            + (N_1[fxt][hl] / 1000) ** 2
                            - (maxy[fxt][hl] / 1000) ** 2
                        )
                        / (2 * (maxgat[fxt][hl] / 1000) * (N_1[fxt][hl] / 1000))
                    )

                if np.isnan(mid_c[fxt][hl]) == True:
                    # print("nan",mid_c)
                    mid_c[fxt][hl] = np.arcsin(
                        rng[fxt][hl]
                        * (np.sin(el_rad[fxt][hl] + nin_rad) / mid_y[fxt][hl])
                    )
                    # We do this for very small elevations where numpy can't compute the arccos, it is less accurate as we are more reliant on our derived value of mid_y
                if np.isnan(min_c[fxt][hl]) == True:
                    min_c[fxt][hl] = np.arcsin(
                        (mingat[fxt][hl] * np.sin(nin_rad + maxel[fxt][hl]))
                        / min_y[fxt][hl]
                    )
                if np.isnan(max_c[fxt][hl]) == True:
                    max_c[fxt][hl] = np.arcsin(
                        (maxgat[fxt][hl] * np.sin(nin_rad + minel[fxt][hl]))
                        / max_y[fxt][hl]
                    )
                    # print("sinerule",mid_c)
                if np.isnan(minc[fxt][hl]) == True:
                    minc[fxt][hl] = np.arcsin(
                        (mingat[fxt][hl] * np.sin(nin_rad + minel[fxt][hl]))
                        / miny[fxt][hl]
                    )
                if np.isnan(maxc[fxt][hl]) == True:
                    maxc[fxt][hl] = np.arcsin(
                        (maxgat[fxt][hl] * np.sin(nin_rad + maxel[fxt][hl]))
                        / maxy[fxt][hl]
                    )
                    # print("sinerule",mid_c)
                else:
                    None

        print("finish_it", time.time())
        # for fxt loop adds 5mins onto loop
        mid_e = mid_c / a

        #####################################################
        ###This Geoid Section uses the direct method from the GeographicLib
        ###mid represents the center of the voxel, and wxyz, 4 corners.
        ###The Geoid module iterates to estimate the distance along the Earth
        ###using corrections to adjust from ellipsoid to geoid.
        #####################################################

        print("midGeoid", time.time())
        print(altitude_r[0][0],R)
        print("el_radg",el_rad[-1][-1])
        #raw_input("stop")
        mid = Geoid(
            mazi,
            lat_ra,
            mid_c,
            a,
            e,
            latitude_r,
            longitude_r,
            rho_1,
            mid_y,
            altitude_r,
            rng,
            el_rad,
            R,
            N,
            f,
        )
        # (maz,s,hi,lat2,lon2,azi2)
        # Geographic lib package provides the direct solution from A-B
        # This module estimates the latitude and longitude position of the mid point of each voxel
        #   0   1,2,3,4,5
        mid_lat = mid[3]
        mid_lon = mid[4]
        mid_azi = mid[0]
        mid_s = mid[1]
        mid_hi = mid[2].copy()

  
        mid_hi_p1 = np.roll(mid_hi, 1, axis=1)
        mid_hi_m1 = np.roll(mid_hi, -1, axis=1)
        mid_hi_md = mid_hi_m1 + 0.5 * (mid_hi_p1 - mid_hi_m1)

        #mid_hi_t1=mid_hi.copy()
        #mid_hi_t2=mid_hi.copy()
        mid_hi_t3 = mid_hi.copy()

        mid_hi_t3[
            np.where(
                np.logical_or(h_a / mid_hi < 0.15, h_a / mid_hi > 1.125),
                    ),
        ] = h_a[
            np.where(
                np.logical_or(h_a / mid_hi < 0.15, h_a / mid_hi > 1.125),
                    ),
        ].copy()
        print("geoid",h_a[0][0],h_a[-1][-1])
        print("geoid refrac",hR_a[0][0],hR_a[-1][-1])
        print("elliptical",Ellip_h_a[0][0],Ellip_h_a[-1][-1])
        print("flat earth", midFSa[0][0],midFSa[-1][-1])
        print("spherical",midSSa[0][0],midSSa[-1][-1])

        print("spherical refrac",midSSaR[0][0],midSSaR[-1][-1])
        print("geoid N refrac",mid_hi_t3[0][0],mid_hi_t3[-1][-1])
        #raw_input("stop")
        rdrcoords = xr.Dataset(
            data_vars={
                "Radar Voxel GE Latitude": (("el", "rng"), derek_lat[:, :]),
                "Radar Voxel GE Longitude": (("el", "rng"), derek_lon[:, :]),
                "Radar Voxel GE Azimuth": (("el", "rng"), derek_azi[:, :]),
                "Radar Voxel GE Altitude": (("el", "rng"), h_a[:, :]),
                "Radar Voxel GE Min Distance Along the Earth": (("el", "rng"),derek_s[:, :],),
                "Radar Voxel EE Latitude": (("el", "rng"), Ellip_lat[:, :]),
                "Radar Voxel EE Longitude": (("el", "rng"), Ellip_lon[:, :]),
                "Radar Voxel EE Azimuth": (("el", "rng"), Ellip_azi[:, :]),
                "Radar Voxel EE Altitude": (("el", "rng"), Ellip_h_a[:, :]),
                "Radar Voxel EE Min Distance Along the Earth": (("el", "rng"),Ellip_s[:, :],),
                "Radar Voxel FE Latitude": (("el", "rng"), FElat[:, :]),
                "Radar Voxel FE Longitude": (("el", "rng"), FElon[:, :]),
                "Radar Voxel FE Azimuth": (("el"), rad_azi[:]),
                "Radar Voxel FE Altitude": (("el","rng"), midFSa[:, :]),
                "Radar Voxel FE Min Distance Along the Earth": (("el", "rng"),midFEs[:, :],),
                "Radar Voxel SE Latitude": (("el", "rng"), SElat[:, :]),
                "Radar Voxel SE Longitude": (("el", "rng"), SElon[:, :]),
                "Radar Voxel SE Azimuth": (("el"), rad_azi[:]),
                "Radar Voxel SE Altitude": (("el","rng"), midSSa[:, :]),
                "Radar Voxel SE Min Distance Along the Earth": (("el", "rng"),midSEs[:, :],),
                "Radar Voxel GE with Refrac Latitude": (("el", "rng"), mid_lat[:, :]),
                "Radar Voxel GE with Refrac Longitude": (("el", "rng"), mid_lon[:, :]),
                "Radar Voxel GE with Refrac Azimuth": (("el", "rng"), mid_azi[:, :]),
                "Radar Voxel GE with Refrac Altitude": (("el", "rng"), mid_hi_t3[:, :]),
                "Radar Voxel GE with Refrac Min Distance Along the Earth": (("el", "rng"),mid_s[:, :],),
                "Radar Voxel SE with 43Refrac Latitude": (("el", "rng"), SERlat[:, :]),
                "Radar Voxel SE with 43Refrac Longitude": (("el", "rng"), SERlon[:, :]),
                "Radar Voxel SE with 43Refrac Azimuth": (("el"), rad_azi[:]),
                "Radar Voxel SE with 43Refrac Altitude": (("el","rng"), midSSaR[:, :]),
                "Radar Voxel SE with 43Refrac Min Distance Along the Earth": (("el", "rng"),midSEsR[:, :],),
                
                "Radar Elevation": (("el"), np.rad2deg(el_rad0[:])),
                "Radar Range": (("rng"), rng0[:]),
            },
            coords={"rng": rngind[:], "el": elind[:]},
        )
        #rdrcoords.to_netcdf("20220315CoordsTestFSGPole.nc")
        #rdrcoords.to_netcdf("20220315CoordsTestFSGRU.nc")
        rdrcoords.to_netcdf("20220315CoordsTestFSGMO.nc")
    except KeyError:
        print("KeyError, not using file", filenama)

    print(time.time(), "end", filenama)
    return

def flatearth(rng,el,height):
        #lat, lon from range in azimuth (SOHCAHTOA: rng*cos(el)=s)
        #alt from height determined by el (alt=height of radar+rng*sin(el))
    rng1=rng.ravel()
    el1=el.ravel()
    s_fe=rng1*np.cos(el1)
    alt_fe=height+rng1*np.sin(el1)
    print(alt_fe[0],alt_fe[-1])
    return(s_fe, alt_fe)

def spherical(R_E,rng,el,height):
        #lat, lon from 
    
    R_E1=R_E.ravel()
    rng1=rng.ravel()
    el1=el.ravel()
    midy=((R_E1+height)**2 +(rng1)**2 -2*((R_E1+height)*rng1*np.cos(el1+(np.pi/2))))**0.5
    alt_se=midy -R_E1  
    midc=np.arccos(((midy)**2 + (R_E1+height)**2 - (rng1)**2)/(2*midy*(R_E1+height)))    
    s_se=R_E1*midc
    print("no refrac",alt_se[0],alt_se[-1])
    #raw_input("stop")
    return(s_se,alt_se)

def sphericalR(R_E,rng,el,height,N_s):
        #lat, lon from 
    k=4/3
    
    R_E1=k*R_E.ravel() 
    rng1=rng.ravel()
    el1=el.ravel()
    

    midy1=((R_E1+height)**2 +(rng1)**2 -2*((R_E1+height)*rng1*np.cos(el1+(np.pi/2))))**0.5
    alt_ser=midy1 -R_E1
    #el2=np.arccos((R_E1 +height)/(R_E1 +alt_ser)) #+np.pi/2
    #el2=np.arccos((k*R_E1 +height)/(k*R_E1 +alt_ser))
    #print(el2[0],el2[-1])
    #elly=((k-1)/(2*k -1))*np.cos(el1)*((np.sin(el1)**2 + ((4*k -2)/(k-1))*(N_s*10**-6))**0.5 -np.sin(el1))
    #midy2=((R_E1+height)**2 +(rng1)**2 -2*((R_E1+height)*rng1*np.cos(el2+(np.pi/2))))**0.5
    #midyelly=((R_E1+height)**2 +(rng1)**2 -2*((R_E1+height)*rng1*np.cos(elly+(np.pi/2))))**0.5
 

    #alt_se2=midy2 -R_E1
    #alt_seelly=midyelly -R_E1

    
    #print("ydiff", el1[-1],el2[-1],elly[-1])
    #print("ydiff", midy1[-1],midy2[-1],midyelly[-1])
    #print("altdiff",alt_ser[-1],alt_se2[-1],alt_seelly[-1])

    midc1=np.arccos(((midy1)**2 + (R_E1)**2 - (rng1)**2)/(2*midy1*(R_E1)))
    
    #midc2=np.arccos(((midy2)**2 + (R_E1)**2 - (rng1)**2)/(2*midy2*(R_E1)))
    #midcelly=np.arccos(((midyelly)**2 + (R_E1)**2 - (rng1)**2)/(2*midyelly*(R_E1)))
    s_ser1=R_E.ravel()*midc1
    #s_ser2=R_E.ravel()*midc2
    #s_serelly=R_E.ravel()*midcelly
    print("orig",s_ser1[-1],alt_ser[-1])
    
    #print("orig",s_ser2[-1],alt_se2[-1])
    #print("elly",s_serelly[-1],alt_seelly[-1])
    #raw_input("stop")
    return(s_ser1,alt_ser)

def refrac(rg, R, h_s, N_s, Rv0, h_a0, H_b0, el_g0, mug):
    l = 0
    #print("og el",el_g0[0],el_g0[-1])
    for l in range(0, 2):
        ghg = Gh(R, h_s, N_s, Rv0, h_a0, H_b0)
        cosphi0 = ghg * np.cos(el_g0)
        delg0 = el_g0 - np.arccos(cosphi0)
        cosphi1 = ghg * np.cos(el_g0 - delg0 / 2)
        delg1 = el_g0 - np.arccos(cosphi1)
        cosphi2 = ghg * np.cos(el_g0 - delg1 / 2)
        R0 = abs(1 - cosphi0**2) ** -0.5 * (h_a0 - h_s)
        R1 = abs(1 - cosphi1**2) ** -0.5 * (h_a0 - h_s)
        R2 = abs(1 - cosphi2**2) ** -0.5 * (h_a0 - h_s)
        m0 = (R2 - R1) / delg1
        eta0 = R0 - rg
        el_g0 = el_g0 - mug * (eta0 / m0)
        l = l + 1
        #print("refrac el",el_g0[0],el_g0[-1])
        #raw_input("stop")
    return el_g0


def Gh(R, h_s, N_s, RV, ha, HB):
    Ghg = ((R + h_s) / (RV + ha)) * (
        1 + 10**-6 * N_s * np.exp(-(ha - h_s) / HB) / (1 + 10**-6 * N_s)
    )
    return Ghg

def Ellipsoid(
    az,
    lat_ra,
    _c,
    a,
    e,
    latitude_r,
    longitude_r,
    rho_1,
    _y,
    altitude_r,
    rnge,
    ele,
    R,
    N_1,
    f,
):
    # def GeoidMatrix(consts,matrix):
    # constants : lat_ra,a,e,latitude_r,longitude_r,rho_1,altitude_r,R
    # matrix: az - [n x elev_n x rang_n],mid_c - [n x elev_n x rang_n],_y - [n x elev_n x rang_n],rnge - [n x elev_n x rang_n],ele - [n x elev_n]
    # output:maz -[n x elev_n x rang_n],s -[n x elev_n x rang_n],hi -[n x elev_n x rang_n],lat2 -[n x elev_n x rang_n],lon2 -[n x elev_n x rang_n],azi2 -[n x elev_n x rang_n]

    # print("lat",latitude_r)

    maz = np.copy(az)
    # copy allows the order to remain the same and let us use the azimuth independently
    phi2 = np.zeros(np.shape(az))
    lat2 = np.zeros(az.shape)
    lon2 = np.zeros(az.shape)
    azi2 = np.zeros(az.shape)
    # set up the outputs with the correct shape
    # 3 steps processing

    phi2 = lat_ra + _c
    # print("initphi2",np.count_nonzero(np.isnan(phi2)),np.count_nonzero(np.isnan(lat_ra)),np.count_nonzero(np.isnan(_c)))#estimate of latitude
    phi2[(np.pi < maz) & (maz < (1.5 * np.pi))] = (
            lat_ra - _c[(np.pi < maz) & (maz < (1.5 * np.pi))]
        )

    s = a * abs(ellipkinc(phi2, e**2) - ellipkinc(lat_ra, e**2))
    # estimation of s
    lins = s.ravel()

    linaz = maz.ravel()

    linaz = np.rad2deg(linaz)

    linaz = np.where((np.logical_and(linaz > 180, linaz < 360)), linaz - 360, linaz)

    kest = [
            Geodesic.WGS84.Direct(
                latitude_r, longitude_r, xi, si, outmask=1929
            ).values()
            for (xi, si) in zip(linaz, lins)
        ]

    kq = []
    kq.append([list(kest[ji]) for ji in range(0, len(lins))])

    phi3 = [kq[0][nop][5] for nop in range(0, len(kq[0]))]
    phi2 = np.reshape(phi3, np.shape(s))
    lat2 = phi2
    
    rho_2 = (a * (1 - e**2)) * (1 - e**2 * np.sin(phi2)) ** (-3 / 2)
    rho_m = 0.5 * (rho_1 + rho_2)
    gamma = e**2 * (s / rho_m) * np.cos(maz) * (np.cos(phi2)) ** 2
    _d = a * e**2 * (s / rho_m) * np.cos(maz) * (np.cos(phi2))
    y_i = np.arcsin((_y * np.sin(gamma)) / _d)
    _h = altitude_r + rnge * np.sin(ele)
    # first approximation to height

    _N = a / np.sqrt(1 - e**2 * np.sin(phi2) ** 2)
    _R = np.sqrt((a**2) / (1 + np.sin(phi2) * ((1 / (1 - f) ** 2) - 1) ** 2))
    _dr = _R - R  # difference in height of radius of the Earth
    hi = _h + _dr

    lon2 = [kq[0][nop][6] for nop in range(0, len(kq[0]))]
    lon2 = np.reshape(lon2, np.shape(s))
    # print("longitude",np.count_nonzero(np.isnan(lon2)))
    azi2 = maz
    print("ellipsoid",hi[0],hi[-1])
    return (maz, s, hi, lat2, lon2)

def Geoid(
    az,
    lat_ra,
    _c,
    a,
    e,
    latitude_r,
    longitude_r,
    rho_1,
    _y,
    altitude_r,
    rnge,
    ele,
    R,
    N_1,
    f,
):
    # def GeoidMatrix(consts,matrix):
    # constants : lat_ra,a,e,latitude_r,longitude_r,rho_1,altitude_r,R
    # matrix: az - [n x elev_n x rang_n],mid_c - [n x elev_n x rang_n],_y - [n x elev_n x rang_n],rnge - [n x elev_n x rang_n],ele - [n x elev_n]
    # output:maz -[n x elev_n x rang_n],s -[n x elev_n x rang_n],hi -[n x elev_n x rang_n],lat2 -[n x elev_n x rang_n],lon2 -[n x elev_n x rang_n],azi2 -[n x elev_n x rang_n]

    # print("lat",latitude_r)

    maz = np.copy(az)
    # copy allows the order to remain the same and let us use the azimuth independently
    phi2 = np.zeros(np.shape(az))
    lat2 = np.zeros(az.shape)
    lon2 = np.zeros(az.shape)
    azi2 = np.zeros(az.shape)
    # set up the outputs with the correct shape
    # 3 steps processing
    og_c=_c
    for corr in range(0, 3):
        # Iterating through to get the finished estimation
        # direction check
        phi2 = lat_ra + _c
        # print("initphi2",np.count_nonzero(np.isnan(phi2)),np.count_nonzero(np.isnan(lat_ra)),np.count_nonzero(np.isnan(_c)))#estimate of latitude
        phi2[(np.pi < maz) & (maz < (1.5 * np.pi))] = (
            lat_ra - _c[(np.pi < maz) & (maz < (1.5 * np.pi))]
        )

        s = a * abs(ellipkinc(phi2, e**2) - ellipkinc(lat_ra, e**2))
        # estimation of s
        lins = s.ravel()

        linaz = maz.ravel()

        linaz = np.rad2deg(linaz)

        linaz = np.where((np.logical_and(linaz > 180, linaz < 360)), linaz - 360, linaz)

        kest = [
            Geodesic.WGS84.Direct(
                latitude_r, longitude_r, xi, si, outmask=1929
            ).values()
            for (xi, si) in zip(linaz, lins)
        ]

        kq = []
        kq.append([list(kest[ji]) for ji in range(0, len(lins))])

        phi3 = [kq[0][nop][5] for nop in range(0, len(kq[0]))]
        phi2 = np.reshape(phi3, np.shape(s))

        rho_2 = (a * (1 - e**2)) * (1 - e**2 * np.sin(phi2)) ** (-3 / 2)
        rho_m = 0.5 * (rho_1 + rho_2)
        gamma = e**2 * (s / rho_m) * np.cos(maz) * (np.cos(phi2)) ** 2
        _d = a * e**2 * (s / rho_m) * np.cos(maz) * (np.cos(phi2))
        y_i = np.arcsin((_y * np.sin(gamma)) / _d)
        _h = altitude_r + rnge * np.sin(ele)
        # first approximation to height
        print("h",_h[0][0],_h[-1][-1])

        _N = a / np.sqrt(1 - e**2 * np.sin(phi2) ** 2)

        _y= np.sqrt(_N**2 + rnge**2 + 2 * _N * rnge * np.cos(ele + 0.5*np.pi))
        _R = np.sqrt((a**2) / (1 + np.sin(phi2) * ((1 / (1 - f) ** 2) - 1) ** 2))
        _dr = _R - R  # difference in height of radius of the Earth
        hi = _h + _dr
        print("dr",_dr[0][0],_dr[-1][-1])
        if corr == 0:
            # skew normal correction
            dazi = hi / rho_m * e**2 * np.sin(maz) * np.cos(maz) * np.cos(phi2) ** 2
            azic = maz + dazi
            maz = azic
        if corr == 1:
            deltag = hi / rho_m * e**2 * np.sin(maz) * np.cos(maz) * np.cos(phi2) ** 2
            # reduced latitude correction
            maz = maz + deltag
        #print("shapes",rnge.shape,_N.shape,ele.shape,_c.shape,_c[np.where(ele==0)].shape,rnge[np.where(ele==0)].shape,rnge[np.where(ele==0)].shape)

        _c=np.arccos(((rnge/1000)**2 + (_N/1000)**2 - (_y/1000)**2)/(2*(rnge/1000)*(_N/1000)))
        _c[np.where(ele==0)]=np.arctan((rnge[np.where(ele==0)]/1000)/(_N[np.where(ele==0)]/1000))

        _c[np.isnan(_c)]=np.arcsin((rnge[np.isnan(_c)]/1000)* (np.sin(ele[np.isnan(_c)]+ np.pi*0.5) / (_y[np.isnan(_c)]/1000)))
        _c[np.isnan(_c)]=og_c[np.isnan(_c)]



    lat2 = phi2

    lon2 = [kq[0][nop][6] for nop in range(0, len(kq[0]))]
    lon2 = np.reshape(lon2, np.shape(s))
    # print("longitude",np.count_nonzero(np.isnan(lon2)))
    azi2 = maz
    print("geoid",hi[0][0],hi[-1][-1])
    
    return (maz, s, hi, lat2, lon2)


def GeoidR(
    az,
    lat_ra,
    _c,
    a,
    e,
    latitude_r,
    longitude_r,
    rho_1,
    _y,
    altitude_r,
    rnge,
    ele,
    R,
    N_1,
    f,
):
    # def GeoidMatrix(consts,matrix):
    # constants : lat_ra,a,e,latitude_r,longitude_r,rho_1,altitude_r,R
    # matrix: az - [n x elev_n x rang_n],mid_c - [n x elev_n x rang_n],_y - [n x elev_n x rang_n],rnge - [n x elev_n x rang_n],ele - [n x elev_n]
    # output:maz -[n x elev_n x rang_n],s -[n x elev_n x rang_n],hi -[n x elev_n x rang_n],lat2 -[n x elev_n x rang_n],lon2 -[n x elev_n x rang_n],azi2 -[n x elev_n x rang_n]

    # print("lat",latitude_r)
    R=7/6 * R
    a= 7/6 *a
    maz = np.copy(az)
    # copy allows the order to remain the same and let us use the azimuth independently
    phi2 = np.zeros(np.shape(az))
    lat2 = np.zeros(az.shape)
    lon2 = np.zeros(az.shape)
    azi2 = np.zeros(az.shape)
    # set up the outputs with the correct shape
    # 3 steps processing
    for corr in range(0, 3):
        # Iterating through to get the finished estimation
        # direction check
        phi2 = lat_ra + _c
        # print("initphi2",np.count_nonzero(np.isnan(phi2)),np.count_nonzero(np.isnan(lat_ra)),np.count_nonzero(np.isnan(_c)))#estimate of latitude
        phi2[(np.pi < maz) & (maz < (1.5 * np.pi))] = (
            lat_ra - _c[(np.pi < maz) & (maz < (1.5 * np.pi))]
        )

        s = a * abs(ellipkinc(phi2, e**2) - ellipkinc(lat_ra, e**2))
        # estimation of s
        lins = s.ravel()

        linaz = maz.ravel()

        linaz = np.rad2deg(linaz)

        linaz = np.where((np.logical_and(linaz > 180, linaz < 360)), linaz - 360, linaz)

        kest = [
            Geodesic.WGS84.Direct(
                latitude_r, longitude_r, xi, si, outmask=1929
            ).values()
            for (xi, si) in zip(linaz, lins)
        ]

        kq = []
        kq.append([list(kest[ji]) for ji in range(0, len(lins))])

        phi3 = [kq[0][nop][5] for nop in range(0, len(kq[0]))]
        phi2 = np.reshape(phi3, np.shape(s))

        rho_2 = (a * (1 - e**2)) * (1 - e**2 * np.sin(phi2)) ** (-3 / 2)
        rho_m = 0.5 * (rho_1 + rho_2)
        gamma = e**2 * (s / rho_m) * np.cos(maz) * (np.cos(phi2)) ** 2
        _d = a * e**2 * (s / rho_m) * np.cos(maz) * (np.cos(phi2))
        y_i = np.arcsin((_y * np.sin(gamma)) / _d)
        _h = altitude_r + rnge * np.sin(ele)
        # first approximation to height
        print("h",_h[0][0],_h[-1][-1])

        _N = a / np.sqrt(1 - e**2 * np.sin(phi2) ** 2)

        _y= np.sqrt(_N**2 + rnge**2 + 2 * _N * rnge * np.cos(ele + 0.5*np.pi))
        _R = np.sqrt((a**2) / (1 + np.sin(phi2) * ((1 / (1 - f) ** 2) - 1) ** 2))
        _dr = _R - R  # difference in height of radius of the Earth
        hi = _h + _dr
        print("dr",_dr[0][0],_dr[-1][-1])
        if corr == 0:
            # skew normal correction
            dazi = hi / rho_m * e**2 * np.sin(maz) * np.cos(maz) * np.cos(phi2) ** 2
            azic = maz + dazi
            maz = azic
        if corr == 1:
            deltag = hi / rho_m * e**2 * np.sin(maz) * np.cos(maz) * np.cos(phi2) ** 2
            # reduced latitude correction
            maz = maz + deltag

        _c=np.arccos(((rnge/1000)**2 + (_N/1000)**2 - (_y/1000)**2)/(2*(rnge/1000)*(_N/1000)))
        _c[np.where(ele==0)]=np.arctan((rnge[np.where(ele==0)]/1000)/(_N[np.where(ele==0)]/1000))

        _c[np.isnan(_c)]=np.arcsin((rnge[np.isnan(_c)]/1000)* (np.sin(ele[np.isnan(_c)]+ np.pi*0.5) / (_y[np.isnan(_c)]/1000)))

    lat2 = phi2

    lon2 = [kq[0][nop][6] for nop in range(0, len(kq[0]))]
    lon2 = np.reshape(lon2, np.shape(s))
    # print("longitude",np.count_nonzero(np.isnan(lon2)))
    azi2 = maz
    print("geoid frac refrac",hi[0][0],hi[-1][-1])
    
    return (maz, s, hi, lat2, lon2)



#radar = Dataset(
#    "/gws/nopw/j04/ncas_radar_vol2/data/xband/chilbolton/cfradial/calib_v2/sur/20180214/ncas-mobile-x-band-radar-1_chilbolton_20180214-105033_SUR_v1.nc"
#)
rt=[10000,15000,20000,30000,40000,50000,75000,100000,150000,200000]
rt=np.array(rt)
et0=[0,0.5,1,2,5,7,10,12.5,15]
et0=np.array(et0)

at0=[0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]
at0=np.array(at0)
extr=len(et0)*len(at0)

et=np.repeat(et0,len(at0))
at=np.resize(at0,len(et))
#et=np.tile(et0,(len(at0),1))
#el_rad = np.reshape(el_rad0, (len(el_rad0), 1))
#el_rad = np.tile(el_rad, (1, len(rng0)))
#at=np.reshape(at0,(len(at0),1))
#at=np.tile(at0,(len(et0),1))

# MO   50.81833333, -2.55472222,
# RU -21.0803, 55.38930,
#Troll -72.0019400, 2.5338900,

vcoords(
    "TestMO",
    20230315,
    1000,
    50.81833333, 
    -2.55472222,
    265,
    1,
    rt,
    600/2,
    et,
    1.1,
    at,
    1.1,
    313,
)
# radar=Dataset('/gws/nopw/j04/ncas_radar_vol2/data/xband/chilbolton/cfradial/calib_v2/sur/20170513/ncas-mobile-x-band-radar-1_chilbolton_20170513-100426_SUR_v1.nc')
# vcoords('ncas-mobile-x-band-radar-1_chilbolton_20170513-100426_SUR_v1.nc',20170513,radar.variables['time'],radar.variables['latitude'][0],radar.variables['longitude'][0],radar.variables['altitude'][0],1,radar.variables['range'][:],radar.variables['range'].meters_between_gates/2,radar.variables['elevation'][:],radar.variables['radar_beam_width_v'][0],radar.variables['azimuth'][:],radar.variables['radar_beam_width_h'][0])

# vcoords(filenama, dat, radtimestamp, latitude_r, longitude_r, rad_height_above_msl, rad_ant_length, rng0, gat, el0_rad0, beamwidthv, rad_azi, beamwidthh):

###klost only included when not a column
###CHANGE TO HAVE ONE RAD HEIGHT ABOVE MSL.
