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
from CR_Plotting_Tools import cr_plot_projections
import h5py
import json
import copy
import os

DEBUG = False

def cr_analysis_radial(
    snapNumber,
    CRPARAMS,
    DataSavepathBase,
    FullDataPathSuffix=".h5",
    logParameters = [],
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

    saveDir = ( DataSavepathBase+f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}/"
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
        lazy_load=False,
        subfind=snap_subfind,
    )

    # # load in the subfind group files
    # snap_subfind = load_subfind(100, dir="/home/universe/spxfv/Auriga/level4_cgm/h12_1kpc_CRs/output/")
    #
    # # load in the gas particles mass and position only for HaloID 0.
    # #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    # #       gas and stars (type 0 and 4) MUST be loaded first!!
    # snapGas = gadget_readsnap(100,"/home/universe/spxfv/Auriga/level4_cgm/h12_1kpc_CRs/output/",hdf5=True,loadonlytype=[0, 1, 4],lazy_load=True,subfind=snap_subfind)
    # snapStars = gadget_readsnap(
    #     100,
    #     "/home/universe/spxfv/Auriga/level4_cgm/h12_1kpc_CRs/output/",
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
        lazy_load=False,
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
    # for snap in [snapGas,snapStars]:
    #     # for param in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]:
    #     for param in snap.data.keys():
    #         try:
    #             tmp = snap.data[param]
    #         except:
    #             pass
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

    # del tmp

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
    )

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
    snapStars.gima *= 1e10  # [Msol]

    snapGas.data["R"] = np.linalg.norm(snapGas.data["pos"], axis=1)
    snapStars.data["R"] = np.linalg.norm(snapStars.data["pos"], axis=1)

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Select stars..."
    )


    whereWind = snapGas.data["age"] < 0.0

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereWind,
        errorString = "Remove Wind from Gas",
        DEBUG = DEBUG,
        )

    whereWindStars = snapStars.data["age"] < 0.0

    snapStars = remove_selection(
        snapStars,
        removalConditionMask = whereWindStars,
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
        errorString = "Remove Outside Virial from Stars",
        DEBUG = DEBUG,
        )

    whereOutsideVirial = snapGas.data["R"] > Rvir

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereOutsideVirial,
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
        logParameters = logParameters,
        paramsOfInterest=CRPARAMS["saveParams"],
        mappingBool=True,
        box=box,
        numthreads=CRPARAMS["numThreads"],
        DataSavepath = DataSavepath,
        verbose = DEBUG,
    )
    # snapGas = calculate_tracked_parameters(snapGas,oc.elements,oc.elements_Z,oc.elements_mass,oc.elements_solar,oc.Zsolar,oc.omegabaryon0,100)
    quadPlotDict = cr_calculate_projections(
        snapGas,
        snapNumber,
        CRPARAMS,
        Axes=CRPARAMS["Axes"],
        zAxis=CRPARAMS["zAxis"],
        boxsize=CRPARAMS["boxsize"],
        boxlos=CRPARAMS["boxlos"],
        pixres=CRPARAMS["pixres"],
        pixreslos=CRPARAMS["pixreslos"],
        numThreads=CRPARAMS["numThreads"],
    )

    if CRPARAMS["QuadPlotBool"] is True:
        cr_plot_projections(
            quadPlotDict,
            CRPARAMS,
            Axes=CRPARAMS["Axes"],
            zAxis=CRPARAMS["zAxis"],
            boxsize=CRPARAMS["boxsize"],
            boxlos=CRPARAMS["boxlos"],
            pixres=CRPARAMS["pixres"],
            pixreslos=CRPARAMS["pixreslos"],
            fontsize = CRPARAMS["fontsize"],
            fontsizeTitle = CRPARAMS["fontsizeTitle"],
            DPI=CRPARAMS["DPI"],
            numThreads=CRPARAMS["numThreads"],
            savePathKeyword = f"{int(snapNumber)}",
        )

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Delete Dark Matter..."
    )

    whereDM = snapGas.data["type"] == 1

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereDM,
        errorString = "Remove DM from Gas",
        DEBUG = DEBUG,
        )

    whereStars = snapGas.data["type"] == 4
    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereStars,
        errorString = "Remove Stars from Gas",
        DEBUG = DEBUG
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

    quadPlotDictOut = { (
            f"{CRPARAMS['resolution']}",
            f"{CRPARAMS['CR_indicator']}",
            f"{int(snapNumber)}",
        ): quadPlotDict
    }

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}, @{int(snapNumber)}]: Finishing process..."
    )
    return out, rotation_matrix , quadPlotDictOut


def cr_parameters(CRPARAMSMASTER, simDict):
    CRPARAMS = copy.deepcopy(CRPARAMSMASTER)
    CRPARAMS.update(simDict)

    if CRPARAMS["with_CRs"] is True:
        CRPARAMS["CR_indicator"] = "with_CRs"
    else:
        CRPARAMS["CR_indicator"] = "no_CRs"

    return CRPARAMS


def cr_flatten_wrt_time(dataDict, CRPARAMS, snapRange):

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

def map_params_to_types(snap, degeneracyBool = False):
    from itertools import combinations#
    import copy
    import pandas as pd
    from collections import Counter

    ADJUSTMENTFACTOR= 15

    #Unsorted unique using pandas!!
    types = pd.unique(snap.data["type"])
    lenTypes = [np.shape(np.where(snap.data["type"]==tp)[0])[0] for tp in types]
    possibleTypesCombos = []
    for jj in range(1,len(types)+1):
        possibleTypesCombos += list(combinations(types,r=jj))
    possibleTypesCombos = np.array(possibleTypesCombos)

    possibleValueLengths = []
    possibleValueLengthsSumTot = []
    for jj in range(1,len(types)+1):
        val = np.array(list(combinations(lenTypes,r=jj)))
        possibleValueLengths += val.tolist()
        possibleValueLengthsSumTot += np.sum(val,axis=-1).tolist()

    possibleValueLengths = np.array(possibleValueLengths)
    possibleValueLengthsSumTot = np.array(possibleValueLengthsSumTot)

    paramToTypeMap = {}
    for key, value in snap.data.items():
        if value is not None:
            whereValueShape = np.where(possibleValueLengthsSumTot == value.shape[0])[0]
            paramToTypeMap.update({
                key: copy.deepcopy(possibleTypesCombos[whereValueShape][0]),
            })
        else:
            pass
            # raise Exception(f"[@map_params_to_types]: value of None found for key {key}! This is not allowed! Make sure this function is removed before removal of any data, check logic; re-run!")
    paramToTypeMap.update({"lty" : copy.deepcopy(lenTypes)})
    paramToTypeMap.update({"ptyc" : copy.deepcopy(possibleTypesCombos)})
    paramToTypeMap.update({"pvl" : copy.deepcopy(possibleValueLengths)})
    paramToTypeMap.update({"pvl_tot" : copy.deepcopy(possibleValueLengthsSumTot)})

    if len(types)>1:
        countDict = Counter(lenTypes)
        #If length of type is not unique...
        if np.any(np.array(list(countDict.values()))>1):
            print(f"[map_params_to_types]: WARNING! Type lengths are degenerate!")
            print(f"[map_params_to_types]:",countDict)
            print(f"[map_params_to_types]: We will pad with np.nan by small amounts to try and break this degeneracy")

            nUnique = 0
            while nUnique != len(types):
                typeAdjustments = np.random.randint(low=1,high=ADJUSTMENTFACTOR,size=len(types))
                nUnique = len(np.unique(typeAdjustments))

            for ii,tp in enumerate(types):
                whereType = np.where(snap.data["type"]==tp)[0]
                for jj,(key, value) in enumerate(snap.data.items()):
                    if tp in paramToTypeMap[key]:

                        locOffset= np.array([jj for jj,tt in enumerate(types[:ii]) if tt not in paramToTypeMap[key]])
                        if (len(locOffset) == 0):
                            # no types before this type so set offset to zero
                            offset = 0
                        else:
                            offset = np.sum(np.array(paramToTypeMap["lty"])[locOffset])

                        whereAdd = whereType[0]-offset

                        for kk in range(0,typeAdjustments[ii]):
                            if key!="type":
                                if np.issubdtype(value.dtype,np.integer):
                                    addVal = np.nanmax(value) + 1
                                else:
                                    addVal = np.nan
                                value = np.insert(value,whereAdd,addVal,axis=0)
                            else:
                                value = np.insert(value,whereAdd,tp,axis=0)
                        # print(np.shape(newValue))
                        snap.data[key] = value
            #Recursive call to rerun this mapping and degeneracy detection
            snap,paramToTypeMap,degeneracyBool = map_params_to_types(snap,degeneracyBool = True)

    return snap, paramToTypeMap, degeneracyBool

def remove_selection(
    snap,
    removalConditionMask,
    errorString = "NOT SET",
    DEBUG = False,
    ):

    import copy
    import pandas as pd
    types = pd.unique(snap.data["type"])

    snap, paramToTypeMap, degeneracyBool = map_params_to_types(snap)

    if degeneracyBool is True:
        raise Exception(f"[remove_selection]: FAILURE! CRITICAL! Snapshot type lengths have been detected as degenerate by map_params_to_types() call in remove_selection()."+"\n"+"map_params_to_types() must be called seperately, prior to the evaluation of removalConditionMask in this call to remove_selection()"+"\n"+f"This error came from errorString {errorString} call to remove_selection()!")

    removedTruthy = np.full(types.shape,fill_value=True)
    if DEBUG is True: print("DEBUG!",errorString)

    # Find possible value length total that matches removalConditionMask
    # shape. From this, infer which parameters are involved in this
    # removalConditionMask. Save this possible type combination
    # as typeCombosArray, and use that to adjust how much the keys
    # in snap.data.keys() need to be offset based on which types
    # used in mask aren't relevant to that snap.data key.

    whereShapeMatch = np.where(paramToTypeMap["pvl_tot"] == removalConditionMask.shape[0])[0]

    typeCombos = paramToTypeMap["ptyc"][whereShapeMatch]

    tmp = typeCombos.tolist()
    tmp2 = [list(xx) for xx in tmp]
    typeCombosArray = np.array(tmp2)[0]


    for ii,tp in enumerate(types):
        if DEBUG is True: print(f" DEBUG! Type {tp}")
        if DEBUG is True: print(f" DEBUG! START Shape of Type {np.shape(np.where(snap.data['type']==tp)[0])}")
        if(tp in typeCombosArray):
            whereType = np.where(snap.data["type"]==tp)[0]
            whereTypeInTypes = np.where(types==tp)[0][0]
            if DEBUG: print("whereTypeInTypes",whereTypeInTypes)
            locTypesOffset= np.array([jj for jj,tt in enumerate(types[:whereTypeInTypes])])# if tt not in typeCombosArray])

            if (len(locTypesOffset) == 0):
                # no types before this type so set offset to zero
                typesOffset = 0
            else:
                typesOffset = np.sum(np.array(paramToTypeMap["lty"])[locTypesOffset])
            if DEBUG: print("locTypesOffset",locTypesOffset)
            # Type specific removal which adjusts for any type in types that
            # aren't part of those included in removalConditionMask
            if DEBUG: print(np.where(snap.data["type"]==tp)[0])
            whereToRemove = np.where(removalConditionMask[np.where(snap.data["type"]==tp)[0]-typesOffset])[0] + typesOffset
            if DEBUG: print(typesOffset)
            if DEBUG: print(whereToRemove)
            for jj,(key, value) in enumerate(snap.data.items()):
                if tp in paramToTypeMap[key]:
                    if value is not None:
                        if DEBUG: print(f"{jj}, {key}")

                        # For the key in snapshot data, retrieve types that
                        # contain that key (i.e. the types that have values
                        # for snap.data[key]). Find types in
                        # removalConditionMask relevant types that aren't
                        # used for this key, and remove the indexing
                        # of this unused data type (offset) from whereToRemove

                        locRemovalOffset= np.array([jj for jj,tt in enumerate(types[:whereTypeInTypes]) if tt not in paramToTypeMap[key]])
                        if DEBUG: print("locRemovalOffset",locRemovalOffset)
                        if DEBUG:
                            print(tp)
                            if len(locTypesOffset)>0:
                                print(types[locTypesOffset])
                            else:
                                print("No locs type")

                            if len(locRemovalOffset)>0:
                                print(types[locRemovalOffset])
                            else:
                                print("No locs removal")


                        if (len(locRemovalOffset) == 0):
                            # no types before this type so set offset to zero
                            removalOffset = 0
                        else:
                            removalOffset = np.sum(np.array(paramToTypeMap["lty"])[locRemovalOffset])
                        if DEBUG:
                            print("val",np.shape(value))
                            print("offset",removalOffset)
                            print("where",whereToRemove)
                            print("where - offset", whereToRemove - removalOffset)

                        try:
                            # Adjust whereToRemove for key specific types. If a
                            # type in types is not used for this param key
                            # we need to offset the whereToRemove to eliminate
                            # the indices that would match that now not Present
                            # particle type.

                            whereToRemoveForKey = copy.copy(whereToRemove) - removalOffset

                            newvalue = np.delete(value,whereToRemoveForKey,axis=0)
                            if newvalue.shape[0]>0:
                                snap.data[key] = newvalue
                            else:
                                # If no more data, set to None for cleaning
                                snap.data[key] = None

                                # If no dara for this type and key, remove this
                                # type from relevant to key mapping dict
                                remainingTypes = [tt for tt in paramToTypeMap[f"{key}"] if tt!=tp]

                                paramToTypeMap[f"{key}"] = copy.deepcopy(remainingTypes)

                        except Exception as e:
                            if DEBUG:
                                print(f"[remove_selection]: DEBUG! Shape key: {np.shape(value)}")
                                print(f"[remove_selection]: DEBUG! WARNING! {str(e)}. Could not remove selection from {key} for particles of type {tp}")

            # Need to remove all entries (deleted (True) or kept (False))
            # of this type so that next type has
            # correct broadcast shape for removalConditionMask and whereType
            removalConditionMask = np.delete(removalConditionMask,(whereType-typesOffset),axis=0)
        else:
            removedTruthy[ii] = False
            # continue
        # typeCombosArray = np.delete(typeCombosArray,np.where(typeCombosArray==tp)[0])
        #update length of types
        paramToTypeMap["lty"][ii] = (np.where(snap.data["type"]==tp)[0]).shape[0]
        if DEBUG: print("paramToTypeMap['lty'][ii]",paramToTypeMap["lty"][ii])


    noneRemovedTruthy = np.all(~removedTruthy)

    if noneRemovedTruthy is True:
        print(f"[@remove_selection]: WARNING! Selection Criteria for error string = '{errorString}', has removed NO entries. Check logic! ")

    elif DEBUG is True:
        if np.any(~removedTruthy):
            print(f"[@remove_selection]: WARNING! DEBUG! Selection criteria for error string = '{errorString}' not applied to particles of type:")
            print(f"{types[np.where(removedTruthy==False)[0]]}")
        else:
            print(f"[@remove_selection]: DEBUG! Selection criteria for error string = '{errorString}' was ~successfully~ applied!")

    snap = clean_snap_nones(snap)

    nData = np.shape(snap.data["type"])[0]
    assert nData > 0,f"[@remove_selection]: FAILURE! CRITICAL!"+"\n"+f"Error String: {errorString} returned an empty snapShot!"

    return snap

def clean_snap_nones(snap):
    deleteKeys = []
    for key, value in snap.data.items():
        if value is None:
            deleteKeys.append(key)

    for key in deleteKeys:
        del snap.data[key]

    return snap

def cr_calculate_projections(
    snapGas,
    snapNumber,
    CRPARAMS,
    Axes=[0, 1],
    zAxis=[2],
    boxsize=400.0,
    boxlos=20.0,
    pixres=0.2,
    pixreslos=0.2,
    numThreads=8,
):

    for param in ["Tdens", "rho_rhomean", "n_H", "B", "gz"]:
        try:
            tmp = snapGas.data[param]
        except:
            snapGas = calculate_tracked_parameters(
                snapGas,
                oc.elements,
                oc.elements_Z,
                oc.elements_mass,
                oc.elements_solar,
                oc.Zsolar,
                oc.omegabaryon0,
                snapNumber,
                paramsOfInterest=[param],
                mappingBool=True,
                numthreads=CRPARAMS["numThreads"],
                verbose = True,
            )

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["x", "y", "z"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]
    # PLOTTING TIME
    # Set plot figure sizes
    xsize = 10.0
    ysize = 10.0
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # slice_nH    = snap.get_Aslice("n_H", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    #
    # slice_B   = snap.get_Aslice("B", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    nprojections = 5
    # print(np.unique(snapGas.type))
    print("\n" + f"[@{int(snapNumber)}]: Projection 1 of {nprojections}")

    proj_T = snapGas.get_Aslice(
        "Tdens",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 2 of {nprojections}")

    proj_dens = snapGas.get_Aslice(
        "rho_rhomean",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 3 of {nprojections}")

    proj_nH = snapGas.get_Aslice(
        "n_H",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 4 of {nprojections}")

    proj_B = snapGas.get_Aslice(
        "B",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 5 of {nprojections}")

    proj_gz = snapGas.get_Aslice(
        "gz",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    return {"T":copy.deepcopy(proj_T), "dens":copy.deepcopy(proj_dens), "n_H":copy.deepcopy(proj_nH), "gz":copy.deepcopy(proj_gz), "B":copy.deepcopy(proj_B)}

def cr_quad_plot_averaging(
    quadPlotDict,
    CRPARAMS,
    snapRange,
):

    print("Quad plot averaging...")

    quadPlotDictAveraged = {}
    flatData = {}

    tmp = {}
    newKey = (f"{CRPARAMS['resolution']}", f"{CRPARAMS['CR_indicator']}")
    selectKey0 = (
        f"{CRPARAMS['resolution']}",
        f"{CRPARAMS['CR_indicator']}",
        f"{int(snapRange[0])}",
    )

    params = copy.deepcopy(list(quadPlotDict[selectKey0].keys()))

    for param in params:
        innertmp = {}
        for key in quadPlotDict[selectKey0][param].keys():
            stackList = []
            for snapNumber in snapRange:
                selectKey = (
                    f"{CRPARAMS['resolution']}",
                    f"{CRPARAMS['CR_indicator']}",
                    f"{int(snapNumber)}",
                )
                stackList.append(quadPlotDict[selectKey][param][key].copy())
            outvals = np.stack(stackList, axis=-1)
            innertmp.update({key : outvals})
        tmp.update({param: innertmp})
    flatData.update({newKey: tmp})

    for param in params:
        tmp ={}
        for arg in ["x","y"]:
            tmp.update({arg : np.nanmedian(flatData[newKey][param][arg],axis=-1)})

        tmp.update({"grid" : np.nansum(flatData[newKey][param]["grid"],axis=-1)/float(len(snapRange))})
        quadPlotDictAveraged.update({param : tmp})

    print("...averaging done!")
    # STOP1080
    return quadPlotDictAveraged

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
