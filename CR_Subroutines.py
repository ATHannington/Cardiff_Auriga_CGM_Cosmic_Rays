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
import os

def cr_cgm_analysis(
    snapNumber,
    CRPARAMS,
    DataSavepathBase,
    FullDataPathSuffix=".h5",
    lazyLoadBool = True,
):

    out = {}

    # Generate halo directory
    try:
        os.mkdir(DataSavepathBase)
    except:
        pass
    else:
        pass

    DataSavepath = DataSavepathBase + f"Data_CR_{CRPARAMS['resolution']}_{CRPARAMS['CR_indicator']}"


    print("")
    print(f"[@{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Starting Snap {snapNumber}")

    loadpath = CRPARAMS['simfile']

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
        f"[@{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
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

    if (
        (CRPARAMS["QuadPlotBool"] is True)
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
            numThreads = CRPARAMS["QuadPlotNumThreads"]
        )


    print(f"[@{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Delete Dark Matter...")

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

    print(f"[@{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Select the CGM...")

    # select the CGM, acounting for variable disk extent
    whereDiskSFR = np.where((snapGas.data["sfr"] > 0.0) & (snapGas.data["halo"]== 0) & (snapGas.data["subhalo"]== 0) & (snapGas.data["R"] <= CRPARAMS['Rinner'])) [0]
    maxDiskRadius = np.nanpercentile(snapGas.data["R"][whereDiskSFR],97.72,axis=0)
    whereCGM = np.where((snapGas.data["sfr"]<= 0.0) & (snapGas.data["R"]>=maxDiskRadius) & (snapGas.data["R"] <= CRPARAMS['Router'])) [0]

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
    snapGas.data['maxDiskRadius'] = np.array([maxDiskRadius])

    print(f"[@{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Trim SnapShot...")

    # Trim snapshot...
    keys = list(snapGas.data.keys())
    for key in keys:
        if key not in CRPARAMS['saveParams']+CRPARAMS['saveEssentials']:
            del snapGas.data[key]

    print(f"[@{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Convert from SnapShot to Dictionary...")
    # Make normal dictionary form of snapGas
    inner = {}
    for key, value in snapGas.data.items():
        if key in CRPARAMS['saveParams']+CRPARAMS['saveEssentials']:
            inner.update({key : copy.deepcopy(value)})

    del snapGas

    # Add to final output
    out.update({(f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}",f"{int(snapNumber)}") : inner})

    print(f"[@{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Finishing process...")
    return out

def cr_parameters(CRPARAMSMASTER, simDict):
    CRPARAMS = copy.deepcopy(CRPARAMSMASTER)
    CRPARAMS.update(simDict)

    if (CRPARAMS['with_CRs'] is True):
        CRPARAMS['CR_indicator'] = "with_CRs"
    else:
        CRPARAMS['CR_indicator'] = "no_CRs"

    return CRPARAMS

def flatten_wrt_time(dataDict,CRPARAMS,snapRange):

    print("Flattening with respect to time...")
    flatData = {}
    tmp = {}
    newKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}")
    selectKey0 = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}",f"{int(snapRange[0])}")

    keys = copy.deepcopy(list(dataDict[selectKey0].keys()))

    for ii,subkey in enumerate(keys):
        print(f"{float(ii)/float(len(keys)):3.1%}")
        concatenateList = []
        for snapNumber in snapRange:
            selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}",f"{int(snapNumber)}")
            concatenateList.append(dataDict[selectKey][subkey].copy())

            del dataDict[selectKey][subkey]
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
    Nbins = 150,
    xlimDict = {
        "R": {"xmin": 0.0, "xmax": 500.0},
        "mass": {"xmin": 5.0, "xmax": 9.0},
        "L": {"xmin": 3.0, "xmax": 4.5},
        "T": {"xmin": 3.75, "xmax": 6.5},
        "n_H": {"xmin": -5.0, "xmax": 0.0},
        "B": {"xmin": -2.0, "xmax": 1.0},
        "vrad": {"xmin": -150.0, "xmax": 150.0},
        "gz": {"xmin": -1.5, "xmax": 0.5},
        "P_thermal": {"xmin": 1.0, "xmax": 4.0},
        "P_magnetic": {"xmin": -1.5, "xmax": 5.0},
        "P_kinetic": {"xmin": -1.0, "xmax": 8.0},
        "P_tot": {"xmin": -1.0, "xmax": 7.0},
        "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 3.0},
        "tcool": {"xmin": -5.0, "xmax": 2.0},
        "theat": {"xmin": -4.0, "xmax": 4.0},
        "tff": {"xmin": -1.5, "xmax": 0.5},
        "tcool_tff": {"xmin": -4.0, "xmax": 2.0},
        "rho_rhomean": {"xmin": 0.0, "xmax": 8.0},
        "dens": {"xmin": -30.0, "xmax": -22.0},
        "ndens": {"xmin": -6.0, "xmax": 2.0}
    },
    printpercent = 5.0):

    exclusions = ["Lookback","Snap","maxDiskRadius"]

    print("[@cr_calculate_statistics]: Generate bins")
    if xParam in CRPARAMS['logParameters']:
        xBins = np.logspace(start=xlimDict[xParam]['xmin'], stop=xlimDict[xParam]['xmax'], num=Nbins, base=10.0)
    else:
        xBins = np.linspace(start=xlimDict[xParam]['xmin'], stop=xlimDict[xParam]['xmax'], num=Nbins)

    xData = []
    whereList = []
    printcount = 0.0
    # print(xBins)
    print("[@cr_calculate_statistics]: Generate where in bins")
    for (ii,(xmin,xmax)) in enumerate(zip(xBins[:-1],xBins[1:])):
        # print(xmin,xParam,xmax)
        percentage = (float(ii)/float(len(xBins[:-1])))*100.
        if percentage >= printcount:
            print(f"{percentage:0.02f}% bins assigned!")
            printcount += printpercent
        xData.append((float(xmax)+float(xmin))/2.)
        whereList.append(np.where((dataDict[xParam]>= xmin)&(dataDict[xParam]< xmax)) [0])

    print("[@cr_calculate_statistics]: Bin data and calculate statistics")
    statsData = {}
    printcount = 0.0
    for ii, whereData in enumerate(whereList):
        percentage = (float(ii)/float(len(whereList)))*100.
        if percentage >= printcount:
            print(f"{percentage:0.02f}% data processed!")
            printcount += printpercent
        binnedData = {}
        for param, values in dataDict.items():
            if (param in CRPARAMS['saveParams'] + CRPARAMS['saveEssentials'])&(param not in exclusions):
                binnedData.update({param: values[whereData]})

        dat = calculate_statistics(
            binnedData,
            TRACERSPARAMS=CRPARAMS,
            saveParams=CRPARAMS['saveParams'],
            weightedStatsBool = True
        )

        # Fix values to arrays to remove concat error of 0D arrays
        for k, val in dat.items():
            dat[k] = np.array([val]).flatten()

        for subkey, vals in dat.items():
            if subkey in list(statsData.keys()):

                statsData[subkey] = np.concatenate(
                    (statsData[subkey], dat[subkey]), axis=0
                )
            else:
                statsData.update({subkey: dat[subkey]})


    statsData.update({f"{xParam}": xData})
    return statsData
