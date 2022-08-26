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
from scipy.stats import binned_statistic_dd
import h5py
import json
import copy
import os

DEBUG = False

def cr_analysis_radial(
    snapNumber, CRPARAMS, DataSavepathBase, FullDataPathSuffix=".h5",
    rotation_matrix=None
):
    analysisType = CRPARAMS["analysisType"]

    KnownAnalysisType = ["cgm", "ism", "all"]

    if analysisType not in KnownAnalysisType:
        raise Exception(
            f"ERROR! CRITICAL! Unknown analysis type: {analysisType}!"
            + "\n"
            + f"Availble analysis types: {KnownAnalysisType}"
        )
    out = {}

    saveDir = ( DataSavepathBase+f"{CRPARAMS['halo']}"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}/type-{analysisType}/"
    )

    # Generate halo directory
    tmp = "/"
    for savePathChunk in saveDir.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    DataSavepath = (
        saveDir + "CR_"
    )


    print("")
    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Starting Snap {snapNumber}"
    )

    loadpath = CRPARAMS["simfile"]

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber, dir=loadpath)

    # load in the gas particles mass and position only for HaloID 0.
    #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    #       gas and stars (type 0 and 4) MUST be loaded first!!
    snapGas = gadget_readsnap(
        snapNumber,
        loadpath,
        hdf5=True,
        loadonlytype=[0, 1, 4],
        lazy_load=True,
        subfind=snap_subfind,
    )

    # # load in the subfind group files
    # snap_subfind = load_subfind(100, dir="/home/universe/spxfv/Auriga/level4_cgm/h12_standard_CRs/output/")
    #
    # # load in the gas particles mass and position only for HaloID 0.
    # #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    # #       gas and stars (type 0 and 4) MUST be loaded first!!
    # snapGas = gadget_readsnap(
    #     100,
    #     "/home/universe/spxfv/Auriga/level4_cgm/h12_standard_CRs/output/",
    #     hdf5=True,
    #     loadonlytype=[0, 1],
    #     lazy_load=True,
    #     subfind=snap_subfind,
    # )
    # snapStars = gadget_readsnap(
    #     100,
    #     "/home/universe/spxfv/Auriga/level4_cgm/h12_standard_CRs/output/",
    #     hdf5=True,
    #     loadonlytype=[4],
    #     lazy_load=True,
    #     subfind=snap_subfind,
    # )
    snapStars = gadget_readsnap(
        snapNumber,
        loadpath,
        hdf5=True,
        loadonlytype=[4],
        lazy_load=True,
        subfind=snap_subfind,
    )

    snapGas.calc_sf_indizes(snap_subfind, halolist=[int(CRPARAMS["HaloID"])])
    if rotation_matrix is None:
        rotation_matrix = snapGas.select_halo(snap_subfind, do_rotation=True)
    else:
        snapGas.select_halo(snap_subfind, do_rotation=False)
        snapGas.rotateto(
            rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
        )

    snapStars.calc_sf_indizes(snap_subfind, halolist=[int(CRPARAMS["HaloID"])])
    if rotation_matrix is None:
        rotation_matrix = snapStars.select_halo(snap_subfind, do_rotation=True)
    else:
        snapStars.select_halo(snap_subfind, do_rotation=False)
        snapStars.rotateto(
            rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
        )

    # Load Cell Other params - avoids having to turn lazy_load off...
    for snap in [snapGas,snapStars]:
        for param in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]:
            try:
                tmp = snap.data[param]
            except:
                pass
    # tmp = snapGas.data["id"]
    # tmp = snapGas.data["sfr"]
    # tmp = snapGas.data["hrgm"]
    # tmp = snapGas.data["mass"]
    # tmp = snapGas.data["pos"]
    # tmp = snapGas.data["vol"]
    #
    # tmp = snapStars.data["mass"]
    # tmp = snapStars.data["pos"]
    # tmp = snapStars.data["age"]
    # tmp = snapStars.data["gima"]
    # tmp = snapStars.data["gz"]

    del tmp

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
    )

    snapGas.calc_sf_indizes(snap_subfind, halolist=[CRPARAMS["HaloID"]])
    # snapGas.calc_sf_indizes(snap_subfind, halolist=[0])

    snapStars.calc_sf_indizes(snap_subfind, halolist=[CRPARAMS["HaloID"]])

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

    snapStars.pos *= 1e3  # [kpc]
    snapStars.mass *= 1e10  # [Msol]

    snapGas.data["R"] = np.linalg.norm(snapGas.data["pos"], axis=1)
    snapStars.data["R"] = np.linalg.norm(snapStars.data["pos"], axis=1)

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Select stars..."
    )

    gasTypes = np.unique(snapGas.type)
    starTypes = np.unique(snapStars.type)

    whereWind = snapStars.data["age"] < 0.0

    snapStars = remove_selection(
        snapStars,
        removalConditionMask = whereWind,
        gasTypes=starTypes,
        errorString = "Remove Wind from Stars",
        DEBUG = DEBUG,
        )



    if analysisType == "cgm":
        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Select the CGM..."
        )
        whereNotCGM = (snapGas.data["R"] > CRPARAMS["Router"])

        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereNotCGM,
            gasTypes=gasTypes,
            errorString = "Remove NOT CGM from Gas",
            DEBUG = DEBUG,
            )

        # select the CGM, acounting for variable ISM extent
        # whereISMSFR = np.where((snapGas.data["sfr"] > 0.0) & (snapGas.data["halo"]== 0) & (snapGas.data["subhalo"]== 0)) [0]
        # maxISMRadius = np.nanpercentile(snapGas.data["R"][whereISMSFR],97.72,axis=0)

        whereNotCGMstars = (snapStars.data['age'] >= 0.0) \
            & (snapStars.data["R"] > CRPARAMS["Router"])

        snapStars = remove_selection(
            snapStars,
            removalConditionMask = whereNotCGMstars,
            gasTypes=starTypes,
            errorString = "Remove NOT CGM from Stars",
            DEBUG = DEBUG,
            )

    elif analysisType == "ism":
        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Select the ISM..."
        )

        whereNotISM = (snapGas.data["sfr"] < 0.0) \
        & (snapGas.data["R"] > CRPARAMS["Rinner"])

        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereNotISM,
            gasTypes=gasTypes,
            errorString = "Remove NOT ISM from Gas",
            DEBUG = DEBUG,
            )


        # select the CGM, acounting for variable ISM extent
        # whereISMSFR = np.where((snapGas.data["sfr"] > 0.0) & (snapGas.data["halo"]== 0) & (snapGas.data["subhalo"]== 0)) [0]
        # maxISMRadius = np.nanpercentile(snapGas.data["R"][whereISMSFR],97.72,axis=0)

        whereNotISMstars = (snapStars.data["R"] > CRPARAMS["Rinner"])

        snapStars = remove_selection(
            snapStars,
            removalConditionMask = whereNotISMstars,
            gasTypes=starTypes,
            errorString = "Remove NOT ISM from Stars",
            DEBUG = DEBUG,
            )

    elif analysisType == "all":

        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Select all of the halo..."
        )

        whereOutsideSelection = (snapGas.data["R"] > CRPARAMS["Router"])

        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereOutsideSelection,
            gasTypes=gasTypes,
            errorString = "Remove ALL Outside Selection from Gas",
            DEBUG = DEBUG,
            )

        # select the CGM, acounting for variable ISM extent
        # whereISMSFR = np.where((snapGas.data["sfr"] > 0.0) & (snapGas.data["halo"]== 0) & (snapGas.data["subhalo"]== 0)) [0]
        # maxISMRadius = np.nanpercentile(snapGas.data["R"][whereISMSFR],97.72,axis=0)

        whereOutsideSelectionStars = (snapStars.data["R"] > CRPARAMS["Router"])

        snapStars = remove_selection(
            snapStars,
            removalConditionMask = whereOutsideSelectionStars,
            gasTypes=starTypes,
            errorString = "Remove ALL Outside Selection from Stars",
            DEBUG = DEBUG,
            )


    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Select within R_virial..."
    )

    Rvir = (snap_subfind.data["frc2"] * 1e3)[int(CRPARAMS["HaloID"])]

    whereOutsideVirialStars = snapStars.data["R"] > Rvir

    snapStars = remove_selection(
        snapStars,
        removalConditionMask = whereOutsideVirialStars,
        gasTypes=starTypes,
        errorString = "Remove Outside Virial from Stars",
        DEBUG = DEBUG,
        )

    whereOutsideVirial = snapGas.data["R"] > Rvir

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereOutsideVirial,
        gasTypes=gasTypes,
        errorString = "Remove Outside Virial from Gas",
        DEBUG = DEBUG,
        )

    rmax = np.max(CRPARAMS["Router"])
    boxmax = rmax
    box = [boxmax, boxmax, boxmax]

    # Calculate New Parameters and Load into memory others we want to track
    snapGas = calculate_tracked_parameters(
        snapGas,
        oc.elements,
        oc.elements_Z,
        oc.elements_mass,
        oc.elements_solar,
        oc.Zsolar,
        oc.omegabaryon0,
        snapNumber,
        paramsOfInterest=CRPARAMS["saveParams"],
        mappingBool=True,
        box=box,
        numthreads=CRPARAMS["numThreads"],
        DataSavepath = DataSavepath,
        verbose = DEBUG,
    )
    # snapGas = calculate_tracked_parameters(snapGas,oc.elements,oc.elements_Z,oc.elements_mass,oc.elements_solar,oc.Zsolar,oc.omegabaryon0,100)
    if CRPARAMS["QuadPlotBool"] is True:
        plot_projections(
            snapGas,
            snapNumber=snapNumber,
            targetT=None,
            rin=None,
            rout=None,
            TRACERSPARAMS=CRPARAMS,
            DataSavepath=DataSavepath,
            FullDataPathSuffix=None,
            titleBool=False,
            Axes=CRPARAMS["Axes"],
            zAxis=CRPARAMS["zAxis"],
            boxsize=CRPARAMS["boxsize"],
            boxlos=CRPARAMS["boxlos"],
            pixres=CRPARAMS["pixres"],
            pixreslos=CRPARAMS["pixreslos"],
            numThreads=CRPARAMS["numThreads"],
        )

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Delete Dark Matter..."
    )

    whereDM = snapGas.data["type"] == 1

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereDM,
        gasTypes=gasTypes,
        errorString = "Remove DM from Gas",
        DEBUG = DEBUG,
        )

    whereStars = snapGas.data["type"] == 4

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereStars,
        gasTypes=gasTypes,
        errorString = "Remove Stars from Gas",
        DEBUG = DEBUG,
        )

    # whereDM = np.where(snapGas.type == 1)[0]
    # whereGas = np.where(snapGas.type == 0)[0]
    # NDM = len(whereDM)
    # NGas = len(whereGas)
    # deleteKeys = []
    # for key, value in snapGas.data.items():
    #     if value is not None:
    #         # print("")
    #         # print(key)
    #         # print(np.shape(value))
    #         if np.shape(value)[0] == (NGas + NDM):
    #             # print("Gas")
    #             snapGas.data[key] = value.copy()[whereGas]
    #         elif np.shape(value)[0] == (NDM):
    #             # print("DM")
    #             deleteKeys.append(key)
    #         else:
    #             # print("Gas or Stars")
    #             pass
    #         # print(np.shape(snapGas.data[key]))
    #
    # for key in deleteKeys:
    #     del snapGas.data[key]

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # Get lookback time in Gyrs
    # [0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    snapGas.data["Redshift"] = np.array([redshift])
    snapGas.data["Lookback"] = np.array([lookback])
    snapGas.data["Snap"] = np.array([snapNumber])
    snapGas.data["Rvir"] = np.array([Rvir])

    snapStars.data["Redshift"] = np.array([redshift])
    snapStars.data["Lookback"] = np.array([lookback])
    snapStars.data["Snap"] = np.array([snapNumber])
    snapStars.data["Rvir"] = np.array([Rvir])

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
    )
    # Make normal dictionary form of snapGas
    inner = {}
    for key, value in snapGas.data.items():
        if key in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]:
            if value is not None:
                inner.update({key: copy.deepcopy(value)})

    del snapGas

    innerStars = {}
    for key, value in snapStars.data.items():
        if key in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]:
            if value is not None:
                innerStars.update({key: copy.deepcopy(value)})

    del snapStars
    # Add to final output
    out.update(
        {
            (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}",
                f"{int(snapNumber)}",
            ): inner
        }
    )

    out.update(
        {
            (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}",
                f"{int(snapNumber)}",
                "Stars",
            ): innerStars
        }
    )

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Finishing process..."
    )
    return out, rotation_matrix


def cr_parameters(CRPARAMSMASTER, simDict):
    CRPARAMS = copy.deepcopy(CRPARAMSMASTER)
    CRPARAMS.update(simDict)

    if CRPARAMS["with_CRs"] is True:
        CRPARAMS["CR_indicator"] = "with_CRs"
    else:
        CRPARAMS["CR_indicator"] = "no_CRs"

    return CRPARAMS


def flatten_wrt_time(dataDict, CRPARAMS, snapRange):

    print("Flattening with respect to time...")
    flatData = {}

    print("Gas...")
    tmp = {}
    newKey = (f"{CRPARAMS['resolution']}", f"{CRPARAMS['CR_indicator']}")
    selectKey0 = (
        f"{CRPARAMS['resolution']}",
        f"{CRPARAMS['CR_indicator']}",
        f"{int(snapRange[0])}",
    )

    keys = copy.deepcopy(list(dataDict[selectKey0].keys()))

    for ii, subkey in enumerate(keys):
        print(f"{float(ii)/float(len(keys)):3.1%}")
        concatenateList = []
        for snapNumber in snapRange:
            selectKey = (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}",
                f"{int(snapNumber)}",
            )
            concatenateList.append(dataDict[selectKey][subkey].copy())

            del dataDict[selectKey][subkey]
            # # Fix values to arrays to remove concat error of 0D arrays
            # for k, val in dataDict.items():
            #     dataDict[k] = np.array([val]).flatten()
        outvals = np.concatenate((concatenateList), axis=0)
        tmp.update({subkey: outvals})
    flatData.update({newKey: tmp})

    print("Stars...")
    tmp = {}
    newKey = (f"{CRPARAMS['resolution']}", f"{CRPARAMS['CR_indicator']}", "Stars")
    selectKey0 = (
        f"{CRPARAMS['resolution']}",
        f"{CRPARAMS['CR_indicator']}",
        f"{int(snapRange[0])}",
        "Stars",
    )

    keys = copy.deepcopy(list(dataDict[selectKey0].keys()))

    for ii, subkey in enumerate(keys):
        print(f"{float(ii)/float(len(keys)):3.1%}")
        concatenateList = []
        for snapNumber in snapRange:
            selectKey = (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}",
                f"{int(snapNumber)}",
                "Stars",
            )
            concatenateList.append(dataDict[selectKey][subkey].copy())

            del dataDict[selectKey][subkey]
            # # Fix values to arrays to remove concat error of 0D arrays
            # for k, val in dataDict.items():
            #     dataDict[k] = np.array([val]).flatten()
        outvals = np.concatenate((concatenateList), axis=0)
        tmp.update({subkey: outvals})
    flatData.update({newKey: tmp})

    print("...flattening done!")
    return flatData


def cr_calculate_statistics(
    dataDict,
    CRPARAMS,
    xParam="R",
    Nbins=150,
    xlimDict={
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
        "ndens": {"xmin": -6.0, "xmax": 2.0},
    },
    printpercent=5.0,
):

    exclusions = ["Redshift", "Lookback", "Snap", "Rvir"]

    print("[@cr_calculate_statistics]: Generate bins")
    if xParam in CRPARAMS["logParameters"]:
        xBins = np.logspace(
            start=xlimDict[xParam]["xmin"],
            stop=xlimDict[xParam]["xmax"],
            num=Nbins,
            base=10.0,
        )
    else:
        xBins = np.linspace(
            start=xlimDict[xParam]["xmin"], stop=xlimDict[xParam]["xmax"], num=Nbins
        )

    xData = []
    whereList = []
    printcount = 0.0
    # print(xBins)
    print("[@cr_calculate_statistics]: Generate where in bins")
    for (ii, (xmin, xmax)) in enumerate(zip(xBins[:-1], xBins[1:])):
        # print(xmin,xParam,xmax)
        percentage = (float(ii) / float(len(xBins[:-1]))) * 100.0
        if percentage >= printcount:
            print(f"{percentage:0.02f}% bins assigned!")
            printcount += printpercent
        xData.append((float(xmax) + float(xmin)) / 2.0)
        whereList.append(
            np.where((dataDict[xParam] >= xmin) & (dataDict[xParam] < xmax))[0]
        )

    print("[@cr_calculate_statistics]: Bin data and calculate statistics")
    statsData = {}
    printcount = 0.0
    for ii, whereData in enumerate(whereList):
        percentage = (float(ii) / float(len(whereList))) * 100.0
        if percentage >= printcount:
            print(f"{percentage:0.02f}% data processed!")
            printcount += printpercent
        binnedData = {}
        for param, values in dataDict.items():
            if (param in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]) & (
                param not in exclusions
            ):
                binnedData.update({param: values[whereData]})

        dat = calculate_statistics(
            binnedData,
            TRACERSPARAMS=CRPARAMS,
            saveParams=CRPARAMS["saveParams"],
            weightedStatsBool=True,
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

    for param in exclusions:
        statsData.update({param: copy.deepcopy(dataDict[param])})

    statsData.update({f"{xParam}": xData})
    return statsData

def remove_selection(
    snap,
    removalConditionMask,
    gasTypes,
    errorString = "NOT SET",
    DEBUG = False,
    ):

    if type(gasTypes) != np.ndarray: gasTypes = np.array(gasTypes)

    removedTruthy = np.full(gasTypes.shape,fill_value=True)

    if DEBUG is True: print(errorString)

    for ii,tp in enumerate(gasTypes):
        if DEBUG is True: print(f"Type {tp}")
        try:
            whereToRemove = np.where(removalConditionMask
                & (snap.data["type"] == tp)
            )[0]
        except Exception as e:
            if DEBUG:
                print(f"[@remove_selection]: DEBUG! Exception: {str(e)}")
            #If data of type tp does not meet broadcasting shape for`
            # removalConditionMask, skip this type and continue
            removedTruthy[ii] = False
            continue

        whereType = np.where(snap.data["type"] == tp)[0]
        if whereToRemove.shape[0] > 0:
            for key, value in snap.data.items():
                if value is not None:
                    if value.shape[0] >= whereType.shape[0]:
                        try:
                            newvalue = np.delete(value,whereToRemove,axis=0)
                            snap.data[key]= newvalue
                        except Exception as e:
                            if DEBUG:
                                print(f"[remove_selection]: WARNING! {str(e)}. Could not remove selection from {key} for particles of type {tp}.")
        # Update removalConditionMask as shape of snap changes after above completes
        removalConditionMask = np.delete(removalConditionMask, whereToRemove,axis=0)
    noneRemovedTruthy = np.all(~removedTruthy)

    if noneRemovedTruthy is True:
        print(f"[@remove_selection]: WARNING! Selection Criteria for error string = '{errorString}', has removed NO entries. Check logic! ")
    elif DEBUG is True:
        if np.any(~removedTruthy):
            print(f"[@remove_selection]: WARNING! DEBUG! Selection criteria for error string = '{errorString}' not applied to particles of type:")
            print(f"{gasTypes[np.where(removedTruthy==False)[0]]}")
        else:
            print(f"[@remove_selection]: DEBUG! Selection criteria for error string = '{errorString}' was ~successfully~ applied!")

    nData = np.shape(snap.data["type"])[0]
    assert nData > 0,f"[@remove_selection]: FAILURE! CRITICAL!"+"\n"+f"Error String: {errorString} returned an empty snapShot!"
    return snap

# def cr_histogram_dd_summarise():
#
#
#     ####
#     # Firstly, make (N,D) array of data
#     ####
#     try:
#         del dataArray
#     except:
#         pass
#
#     paramIndexDict = {}
#     for k, dataDict in out.items():
#         for ii,(key, value) in enumerate(dataDict.items()):
#             # Fiddle with value shapes to accomodate for some being 2D (e.g. pos has x,y,z)
#             if np.shape(np.shape(value))[0]==1:
#                 value = value.reshape(-1,1)
#             #
#             try:
#                 dataArray = np.concatenate((dataArray,value),axis=1)
#             except:
#                 dataArray = value
#
#             paramIndexDict.update({key : ii})
#
#     summarisedOut = {}
#
#     # Get base histogram, and reuse result (ret) to reuse bins through all stats
#     ret = binned_statistic_dd(dataDict[~paramIndexDict['mass']], values = dataDict[paramIndexDict['mass']], binned_statistic_result=ret , statistic = "count")
#
#
#     Hdd, edges, _ = ret
#     summarisedOut.update({("count", None) : {"Hdd" : Hdd, "edges" : edges}})
#
#     Hdd, edges, _ = binned_statistic_dd(dataDict[~paramIndexDict['mass']], values = dataDict[paramIndexDict['mass']], binned_statistic_result=ret , statistic = "sum")
#
#     summarisedOut.update({("sum", "mass") : {"Hdd" : Hdd, "edges" : edges}})
#
#     otherWeights = np.unique(np.array(list(CRPARAMS["nonMassWeightDict"].values())))
#
#     for weight in otherWeights:
#         Hdd, edges, _ = binned_statistic_dd(dataDict[~paramIndexDict[weight]], values = dataDict[paramIndexDict[weight]], binned_statistic_result=ret , statistic = "sum")
#
#         summarisedOut.update({("sum", weight) : {"Hdd" : Hdd, "edges" : edges}})
#
#     for percentile in TRACERSPARAMS["percentiles"]:
#         Hdd, edges, _ = binned_statistic_dd(dataDict[~paramIndexDict['mass']], values = dataDict[paramIndexDict['mass']], binned_statistic_result=ret , statistic = lambda xx: np.nanpercentile(xx, percentile, axis=0))
#
#         summarisedOut.update({(percentile, "mass") : {"Hdd" : Hdd, "edges" : edges}})
#
#     for weight in otherWeights:
#         for percentile in TRACERSPARAMS["percentiles"]:
#             Hdd, edges, _ = binned_statistic_dd(dataDict[~paramIndexDict[weight]], values = dataDict[paramIndexDict[weight]] , binned_statistic_result=ret, statistic = lambda xx: np.nanpercentile(xx, percentile, axis=0))
#
#             summarisedOut.update({(percentile, weight) : {"Hdd" : Hdd, "edges" : edges}})
#
#     return summarisedOut,paramIndexDict
