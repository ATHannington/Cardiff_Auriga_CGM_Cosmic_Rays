"""
Author: A. T. Hannington
Created: 31/03/2022
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
import OtherConstants as oc
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
import h5py
import json
import copy

def cr_cgm_analysis(
    snapNumber,
    CRPARAMS,
    DataSavepathBase,
    FullDataPathSuffix=".h5",
    lazyLoadBool = True,
):

    out = {}
    DataSavepath = DataSavepathBase + f"Data_CR_{CRPARAMS['sim']['resolution']}_{CRPARAMS['sim']['CR_indicator']}"


    print("")
    print(f"[@{CRPARAMS['sim']['resolution']}, @{CRPARAMS['sim']['CR_indicator']}, @{int(snapNumber)}]: Starting Snap {snapNumber}")

    loadpath = CRPARAMS['sim']['simfile']

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber, dir=loadpath)

    # load in the gas particles mass and position only for HaloID 0.
    #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    #       gas and stars (type 0 and 4) MUST be loaded first!!
    snapGas = gadget_readsnap(
        snapNumber,
        loadpath,
        hdf5=True,
        loadonlytype=[0, 1],
        lazy_load=lazyLoadBool,
        subfind=snap_subfind,
    )

    # Load Cell IDs - avoids having to turn lazy_load off...
    # But ensures 'id' is loaded into memory before halo_only_gas_select is called
    #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
    #   Be in memory so taking the subset would be skipped.

    tmp = snapGas.data["id"]
    tmp = snapGas.data["sfr"]
    tmp = snapGas.data["hrgm"]
    tmp = snapGas.data["mass"]
    tmp = snapGas.data["pos"]
    tmp = snapGas.data["vol"]
    del tmp

    print(
        f"[@{CRPARAMS['sim']['resolution']}, @{CRPARAMS['sim']['CR_indicator']}, @{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
    )


    snapGas.calc_sf_indizes(snap_subfind, halolist=[CRPARAMS['HaloID']])

    # Centre the simulation on HaloID 0
    snapGas = set_centre(
        snap=snapGas, snap_subfind=snap_subfind, HaloID=CRPARAMS['HaloID'], snapNumber=snapNumber
    )

    # snapGas.select_halo(snap_subfind, do_rotation=False)
    # --------------------------#
    ##    Units Conversion    ##
    # --------------------------#

    # Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos *= 1e3  # [kpc]
    snapGas.vol *= 1e9  # [kpc^3]
    snapGas.mass *= 1e10  # [Msol]
    snapGas.hrgm *= 1e10  # [Msol]

    # Calculate New Parameters and Load into memory others we want to track
    snapGas = calculate_tracked_parameters(snapGas,oc.elements,oc.elements_Z,oc.elements_mass,oc.elements_solar,oc.Zsolar,oc.omegabaryon0,snapNumber)

    whereDM = np.where(snapGas.type == 1)[0]
    whereGas = np.where(snapGas.type == 0)[0]

    NDM = len(whereDM)
    NGas = len(whereGas)
    deleteKeys = []
    for key, value in snapGas.data.items():
        if value is not None:
            # print("")
            # print(key)
            # print(np.shape(value))
            if np.shape(value)[0] == (NGas + NDM) :
                # print("Gas")
                snapGas.data[key] = value.copy()[whereGas]
            elif np.shape(value)[0] == (NDM):
                # print("DM")
                deleteKeys.append(key)
            else:
                # print("Gas or Stars")
                pass
            # print(np.shape(snapGas.data[key]))

    for key in deleteKeys:
        del snapGas.data[key]

    # select the CGM, acounting for variable disk extent
    whereDiskSFR = np.where((snapGas.data["sfr"] > 0.0) & (snapGas.data["halo"]== 0) & (snapGas.data["subhalo"]== 0)) [0]
    maxDiskRadius = np.nanpercentile(snapGas.data["R"][whereDiskSFR],97.72,axis=0)
    whereCGM = np.where((snapGas.data["sfr"]<= 0.0) & (snapGas.data["R"]>=maxDiskRadius)) [0]

    for key, value in snapGas.data.items():
        if value is not None:
            snapGas.data[key] = value.copy()[whereCGM]

    # Select only gas in High Res Zoom Region
    snapGas = high_res_only_gas_select(snapGas, snapNumber)

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # Get lookback time in Gyrs
    # [0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    snapGas.data["Lookback"] = np.array([lookback])
    snapGas.data["Snap"] = np.array([snapNumber])


    if (
        (CRPARAMS["QuadPlotBool"] is True)
        # & (targetT == int(CRPARAMS["targetTLst"][0]))
        # & (rin == CRPARAMS["Rinner"][0])
    ):
        plot_projections(
            snapGas,
            snapNumber=snapNumber,
            targetT=None,
            rin=None,
            rout=None,
            TRACERSPARAMS=CRPARAMS,
            DataSavepath=DataSavepath,
            FullDataPathSuffix=None,
            titleBool = False,
            Axes=CRPARAMS["Axes"],
            zAxis=CRPARAMS["zAxis"],
            boxsize=CRPARAMS["boxsize"],
            boxlos=CRPARAMS["boxlos"],
            pixres=CRPARAMS["pixres"],
            pixreslos=CRPARAMS["pixreslos"],
        )



    # Trim snapshot...
    keys = list(snapGas.data.keys())
    for key in keys:
        if key not in CRPARAMS['saveParams']+CRPARAMS['saveEssentials']:
            del snapGas.data[key]

    # Make normal dictionary form of snapGas
    inner = {}
    for key, value in snapGas.data.items():
        if key in CRPARAMS['saveParams']+CRPARAMS['saveEssentials']:
            inner.update({key : value})

    # Add to final output
    out.update({(f"{CRPARAMS['sim']['resolution']}",f"{CRPARAMS['sim']['CR_indicator']}",f"{int(snapNumber)}") : inner})

    print(f"[@{CRPARAMS['sim']['resolution']}, @{CRPARAMS['sim']['CR_indicator']}, @{int(snapNumber)}]: Finishing process...")
    return out

def load_cr_parameters(CRPARAMSPATH = "CRParams.json"):
    CRPARAMS = json.load(open(CRPARAMSPATH, 'r'))

    if (CRPARAMS['sim']['with_CRs'] is True):
        CRPARAMS['sim']['CR_indicator'] = "with_CRs"
    else:
        CRPARAMS['sim']['CR_indicator'] = "no_CRs"

    try:
        CRPARAMS['finalSnap'] = copy.copy(CRPARAMS['snapMax'])
    except:
        pass

    return CRPARAMS

def flatten_wrt_time(dataDict,CRPARAMS,snapRange):

    print("Flattening with respect to time...")
    flatData = {}
    tmp = {}
    newKey = (f"{CRPARAMS['sim']['resolution']}",f"{CRPARAMS['sim']['CR_indicator']}")
    selectKey0 = (f"{CRPARAMS['sim']['resolution']}",f"{CRPARAMS['sim']['CR_indicator']}",f"{int(snapRange[0])}")

    keys = copy.deepcopy(list(dataDict[selectKey0].keys()))

    for ii,subkey in enumerate(keys):
        print(f"{float(ii)/float(len(keys)):3.1%}")
        concatenateList = []
        for snapNumber in snapRange:
            selectKey = (f"{CRPARAMS['sim']['resolution']}",f"{CRPARAMS['sim']['CR_indicator']}",f"{int(snapNumber)}")
            concatenateList.append(dataDict[subkey].copy())

            del dataDict[subkey]
            # # Fix values to arrays to remove concat error of 0D arrays
            # for k, val in dataDict.items():
            #     dataDict[k] = np.array([val]).flatten()
        outvals = np.concatenate(
            (concatenateList), axis=0
        )
        tmp.update({subkey: outvals})
    flatData.update({newKey : tmp})
    print("...done!")



    # print("")
    # print("***DEBUG!***")
    # print("flatData.keys()")
    # print(flatData.keys())
    # print("flatData[newKey].keys()")
    # print(flatData[newKey].keys())



    return flatData

def cr_calculate_statistics(
    dataDict,
    CRPARAMS,
    xParam = "R",
    Nbins=150,
):

    if xParam in CRPARAMS['logParameters']:
        xBins = np.logspace(start = np.log10(np.nanmin(dataDict[xParam])), stop=np.log10(np.nanmax(dataDict[xParam])), num=Nbins, base=10.0)
    else:
        xBins = np.linspace(start=np.nanmin(dataDict[xParam]), stop=np.nanmax(dataDict[xParam]), num=Nbins)

    xData = []
    whereList = []
    for xmin,xmax in zip(xBins[:-1],xBins[1:]):
        xData.append((float(xmax)-float(xmin))/2.)
        whereList.append(np.where((dataDict[xParam]>= xmin)&(dataDict[xParam]< xmax)) [0])

    statsData = {}
    for whereData in whereList:
        binnedData = {}
        for param, values in dataDict.items():
            if param in CRPARAMS['saveParams']:
                binnedData.update({param: values[whereData]})

        dat = calculate_statistics(
            binnedData,
            TRACERSPARAMS=CRPARAMS,
            saveParams=CRPARAMS['saveParams']
        )

        # Fix values to arrays to remove concat error of 0D arrays
        for k, val in dat.items():
            dat[k] = np.array([val]).flatten()

        if selectKey in list(statsData.keys()):
            for subkey, vals in dat.items():
                if subkey in list(statsData.keys()):

                    statsData[subkey] = np.concatenate(
                        (statsData[subkey], dat[subkey]), axis=0
                    )
                else:
                    statsData.update({subkey: dat[subkey]})
        else:
            statsData.update({selectKey: dat})

    statsData.update({f"{xParam}": xData})
    return statsData
