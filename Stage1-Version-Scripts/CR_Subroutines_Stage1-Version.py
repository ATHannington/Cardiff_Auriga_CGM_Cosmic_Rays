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
import Tracers_Subroutines as tr
import Plotting_tools as apt
import h5py
import json
import copy
import os
import time
import multiprocessing as mp
import psutil
import math

def cr_analysis_radial(
    snapNumber,
    CRPARAMS,
    ylabel,
    xlimDict,
    DataSavepathBase,
    FullDataPathSuffix=".h5",
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

    saveDir = ( DataSavepathBase+f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}/"
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
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Starting Snap {snapNumber}"
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
        loadonlyhalo=int(CRPARAMS["HaloID"]),

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
        loadonlyhalo=int(CRPARAMS["HaloID"]),
    )

    snapGas.calc_sf_indizes(snap_subfind)
    if rotation_matrix is None:
        rotation_matrix = snapGas.select_halo(snap_subfind, do_rotation=True)
    else:
        snapGas.select_halo(snap_subfind, do_rotation=False)
        snapGas.rotateto(
            rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
        )

    snapStars.calc_sf_indizes(snap_subfind)
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
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
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
    snapGas.gima *= 1e10  # [Msol]

    snapStars.pos *= 1e3  # [kpc]
    snapStars.mass *= 1e10  # [Msol]
    snapStars.gima *= 1e10  # [Msol]

    snapGas.data["R"] = np.linalg.norm(snapGas.data["pos"], axis=1)
    snapStars.data["R"] = np.linalg.norm(snapStars.data["pos"], axis=1)

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select stars..."
    )


    whereWind = snapGas.data["age"] < 0.0

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereWind,
        errorString = "Remove Wind from Gas",
        verbose = verbose,
        )

    whereWindStars = snapStars.data["age"] < 0.0

    snapStars = remove_selection(
        snapStars,
        removalConditionMask = whereWindStars,
        errorString = "Remove Wind from Stars",
        verbose = verbose,
        )


    if analysisType == "cgm":
        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']},@{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select the CGM..."
        )
        whereNotCGM = (snapGas.data["R"] > CRPARAMS["Router"])

        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereNotCGM,
            errorString = "Remove NOT CGM from Gas",
            verbose = verbose,
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
            verbose = verbose,
            )

    elif analysisType == "ism":
        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select the ISM..."
        )

        whereNotISM = (snapGas.data["sfr"] < 0.0) \
        & (snapGas.data["R"] > CRPARAMS["Rinner"])

        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereNotISM,
            errorString = "Remove NOT ISM from Gas",
            verbose = verbose,
            )


        # select the CGM, acounting for variable ISM extent
        # whereISMSFR = np.where((snapGas.data["sfr"] > 0.0) & (snapGas.data["halo"]== 0) & (snapGas.data["subhalo"]== 0)) [0]
        # maxISMRadius = np.nanpercentile(snapGas.data["R"][whereISMSFR],97.72,axis=0)

        whereNotISMstars = (snapStars.data["R"] > CRPARAMS["Rinner"])

        snapStars = remove_selection(
            snapStars,
            removalConditionMask = whereNotISMstars,
            errorString = "Remove NOT ISM from Stars",
            verbose = verbose,
            )

    elif analysisType == "all":

        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select all of the halo..."
        )

        whereOutsideSelection = (snapGas.data["R"] > CRPARAMS["Router"])

        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereOutsideSelection,
            errorString = "Remove ALL Outside Selection from Gas",
            verbose = verbose,
            )

        # select the CGM, acounting for variable ISM extent
        # whereISMSFR = np.where((snapGas.data["sfr"] > 0.0) & (snapGas.data["halo"]== 0) & (snapGas.data["subhalo"]== 0)) [0]
        # maxISMRadius = np.nanpercentile(snapGas.data["R"][whereISMSFR],97.72,axis=0)

        whereOutsideSelectionStars = (snapStars.data["R"] > CRPARAMS["Router"])

        snapStars = remove_selection(
            snapStars,
            removalConditionMask = whereOutsideSelectionStars,
            errorString = "Remove ALL Outside Selection from Stars",
            verbose = verbose,
            )


    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select within R_virial..."
    )

    Rvir = (snap_subfind.data["frc2"] * 1e3)[int(CRPARAMS["HaloID"])]

    whereOutsideVirialStars = snapStars.data["R"] > Rvir

    snapStars = remove_selection(
        snapStars,
        removalConditionMask = whereOutsideVirialStars,
        errorString = "Remove Outside Virial from Stars",
        verbose = verbose,
        )

    whereOutsideVirial = snapGas.data["R"] > Rvir

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereOutsideVirial,
        errorString = "Remove Outside Virial from Gas",
        verbose = verbose,
        )

    rmax = np.max(CRPARAMS["Router"])
    boxmax = rmax
    box = [boxmax, boxmax, boxmax]

    # Calculate New Parameters and Load into memory others we want to track
    snapGas = tr.calculate_tracked_parameters(
        snapGas,
        oc.elements,
        oc.elements_Z,
        oc.elements_mass,
        oc.elements_solar,
        oc.Zsolar,
        oc.omegabaryon0,
        snapNumber,
        logParameters = CRPARAMS['logParameters'],
        paramsOfInterest=CRPARAMS["saveParams"],
        mappingBool=True,
        box=box,
        numthreads=CRPARAMS["numthreads"],
        DataSavepath = DataSavepath,
        verbose = verbose,
    )
    # snapGas = tr.calculate_tracked_parameters(snapGas,oc.elements,oc.elements_Z,oc.elements_mass,oc.elements_solar,oc.Zsolar,oc.omegabaryon0,100)
    if ("n_HI_col" in CRPARAMS["saveParams"])|("n_HI_col" in CRPARAMS["imageParams"]):

       
        ## Calculate n_HI_col for use as saveParam or image Param
        
        CRPARAMS["saveEssentials"].append("n_HI_colx")
        CRPARAMS["saveEssentials"].append("n_HI_coly")
        tmpdict = cr_calculate_projections(
            snapGas,
            CRPARAMS,
            ylabel,
            xlimDict,
            snapNumber=snapNumber,
            params = ["n_HI_col"],
            xsize = CRPARAMS["xsizeImages"],
            ysize = CRPARAMS["ysizeImages"],
            projection=True,
            Axes=CRPARAMS["Axes"],
            boxsize=CRPARAMS["boxsize"],
            boxlos=CRPARAMS["coldenslos"],
            pixres=CRPARAMS["pixres"],
            pixreslos=CRPARAMS["pixreslos"],
            fontsize = CRPARAMS["fontsize"],
            DPI=CRPARAMS["DPI"],
            numthreads=CRPARAMS["numthreads"],
            verbose = verbose,
        )

        ## Convert n_HI_col to per cm^-2 

        KpcTocm = 1e3 * c.parsec
        convert = float(CRPARAMS["pixreslos"])*KpcTocm
        snapGas.data["n_HI_col"] = copy.deepcopy(tmpdict["n_HI_col"]["grid"])*convert
        snapGas.data["n_HI_colx"] = copy.deepcopy(tmpdict["n_HI_col"]["x"])
        snapGas.data["n_HI_coly"] = copy.deepcopy(tmpdict["n_HI_col"]["y"])

        ## If n_HI_col in imageParams, remove it so we can calculate all others with none coldenslos, normal boxlos instead
        tmpimageParams = copy.deepcopy(CRPARAMS["imageParams"])
        tmpimageParams.pop("n_HI_col")

        quadPlotDict = cr_calculate_projections(
            snapGas,
            CRPARAMS,
            ylabel,
            xlimDict,
            snapNumber=snapNumber,
            params = tmpimageParams,
            xsize = CRPARAMS["xsizeImages"],
            ysize = CRPARAMS["ysizeImages"],
            projection=CRPARAMS["projection"],
            Axes=CRPARAMS["Axes"],
            boxsize=CRPARAMS["boxsize"],
            boxlos=CRPARAMS["boxlos"],
            pixres=CRPARAMS["pixres"],
            pixreslos=CRPARAMS["pixreslos"],
            fontsize = CRPARAMS["fontsize"],
            DPI=CRPARAMS["DPI"],
            numthreads=CRPARAMS["numthreads"],
            verbose = verbose,
        )

        ## If n_HI_col in imageParams, add it back in to be plotted and tracked for average plots
        ##   ~NOTE~ : boxlos is ignored after cr_calculate_projections call above, as code is configured such that when
        ##            apt.plot_slices receives a precalculated image dictionary (not Arepo snapshot) it will plot the
        ##            dictionary's contents without recalculating them, so pixres, boxlos, pixreslos etc are ignored.
        if ("n_HI_col" in CRPARAMS["imageParams"]):
            quadPlotDict.update(copy.deepcopy(tmpdict))

        if CRPARAMS["QuadPlotBool"] is True:
            apt.cr_plot_projections(
                quadPlotDict,
                CRPARAMS,
                ylabel,
                xlimDict,
                xsize = CRPARAMS["xsizeImages"],
                ysize = CRPARAMS["ysizeImages"],
                projection=CRPARAMS["projection"],
                Axes=CRPARAMS["Axes"],
                boxsize=CRPARAMS["boxsize"],
                boxlos=CRPARAMS["boxlos"],
                pixres=CRPARAMS["pixres"],
                pixreslos=CRPARAMS["pixreslos"],
                fontsize = CRPARAMS["fontsize"],
                DPI=CRPARAMS["DPI"],
                numthreads=CRPARAMS["numthreads"],
                verbose = verbose,
                savePathKeyword = snapNumber,
            )

        del tmpdict

    else:
        quadPlotDict = cr_calculate_projections(
            snapGas,
            CRPARAMS,
            ylabel,
            xlimDict,
            snapNumber=snapNumber,
            params = CRPARAMS["imageParams"],
            xsize = CRPARAMS["xsizeImages"],
            ysize = CRPARAMS["ysizeImages"],
            projection=CRPARAMS["projection"],
            Axes=CRPARAMS["Axes"],
            boxsize=CRPARAMS["boxsize"],
            boxlos=CRPARAMS["boxlos"],
            pixres=CRPARAMS["pixres"],
            pixreslos=CRPARAMS["pixreslos"],
            fontsize = CRPARAMS["fontsize"],
            DPI=CRPARAMS["DPI"],
            numthreads=CRPARAMS["numthreads"],
            verbose = verbose,
        )

        if CRPARAMS["QuadPlotBool"] is True:
            apt.cr_plot_projections(
                quadPlotDict,
                CRPARAMS,
                ylabel,
                xlimDict,
                xsize = CRPARAMS["xsizeImages"],
                ysize = CRPARAMS["ysizeImages"],
                projection=CRPARAMS["projection"],
                Axes=CRPARAMS["Axes"],
                boxsize=CRPARAMS["boxsize"],
                boxlos=CRPARAMS["boxlos"],
                pixres=CRPARAMS["pixres"],
                pixreslos=CRPARAMS["pixreslos"],
                fontsize = CRPARAMS["fontsize"],
                DPI=CRPARAMS["DPI"],
                numthreads=CRPARAMS["numthreads"],
                verbose = verbose,
                savePathKeyword = snapNumber,
            )

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Delete Dark Matter..."
    )

    whereDM = snapGas.data["type"] == 1

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereDM,
        errorString = "Remove DM from Gas",
        verbose = verbose,
        )

    whereStars = snapGas.data["type"] == 4
    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereStars,
        errorString = "Remove Stars from Gas",
        verbose = verbose
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

    print( 
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Ages: get_lookback_time_from_a() ..."
    )
    ages = snapStars.cosmology_get_lookback_time_from_a(snapStars.data["age"],is_flat=True)
    snapStars.data["age"] = ages

    snapGas.data["Redshift"] = np.array([redshift])
    snapGas.data["Lookback"] = np.array([lookback])
    snapGas.data["Snap"] = np.array([snapNumber])
    snapGas.data["Rvir"] = np.array([Rvir])

    snapStars.data["Redshift"] = np.array([redshift])
    snapStars.data["Lookback"] = np.array([lookback])
    snapStars.data["Snap"] = np.array([snapNumber])
    snapStars.data["Rvir"] = np.array([Rvir])

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
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
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                f"{int(snapNumber)}",
            ): inner
        }
    )

    out.update(
        {
            (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                f"{int(snapNumber)}",
                "Stars",
            ): innerStars
        }
    )

    quadPlotDictOut = { (
            f"{CRPARAMS['resolution']}",
            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
            f"{int(snapNumber)}",
        ): quadPlotDict
    }

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Finishing process..."
    )
    return out, rotation_matrix , quadPlotDictOut


def cr_parameters(CRPARAMSMASTER, simDict):
    CRPARAMS = copy.deepcopy(CRPARAMSMASTER)
    CRPARAMS.update(simDict)

    if CRPARAMS["with_CRs"] is True:
        CRPARAMS["CR_indicator"] = "with_CRs"
    else:
        CRPARAMS["CR_indicator"] = "no_CRs"

    if CRPARAMS["no-alfven"] is True:
        CRPARAMS["no-alfven_indicator"] = "_no_Alfven"
    else:
        CRPARAMS["no-alfven_indicator"] = ""


    return CRPARAMS


def cr_flatten_wrt_time(dataDict, CRPARAMS, snapRange):

    print("Flattening with respect to time...")
    flatData = {}

    print("Gas...")
    tmp = {}
    newKey = (f"{CRPARAMS['resolution']}",
              f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")
    selectKey0 = (
        f"{CRPARAMS['resolution']}",
        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
        f"{int(snapRange[0])}",
    )

    keys = copy.deepcopy(list(dataDict[selectKey0].keys()))

    for ii, subkey in enumerate(keys):
        print(f"{float(ii)/float(len(keys)):3.1%}")
        concatenateList = []
        for snapNumber in snapRange:
            selectKey = (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
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
    newKey = (f"{CRPARAMS['resolution']}", 
              f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
              "Stars")
    selectKey0 = (
        f"{CRPARAMS['resolution']}",
        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
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
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
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
    exclusions = ["Redshift", "Lookback", "Snap", "Rvir"],
    weightedStatsBool = True,
):
    if exclusions is None:
        exclusions = []

    print("[@cr_calculate_statistics]: Generate bins")
    if xParam in CRPARAMS["logParameters"]:
        xBins = np.logspace(
            start=xlimDict[xParam]["xmin"],
            stop=xlimDict[xParam]["xmax"],
            num=Nbins+1,
            base=10.0,
        )
    else:
        xBins = np.linspace(
            start=xlimDict[xParam]["xmin"], stop=xlimDict[xParam]["xmax"], num=Nbins+1
        )

    xmin, xmax = np.nanmin(xBins), np.nanmax(xBins)

    where_within = np.where((dataDict[xParam] >= xmin) & (dataDict[xParam] < xmax))[0]

    sort_ind = np.argsort(dataDict[xParam][where_within],axis=0)

    sortedData ={}
    print("[@cr_calculate_statistics]: Sort data by xParam")
    for param, values in dataDict.items():
        if (param in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]) & (
            param not in exclusions
        ):
            sortedData.update({param: copy.deepcopy(values[where_within][sort_ind])})

    xData = []
    datList = []
    printcount = 0.0
    # print(xBins)
    print("[@cr_calculate_statistics]: Calculate statistics from binned data")
    for (ii, (xmin, xmax)) in enumerate(zip(xBins[:-1], xBins[1:])):
        # print(xmin,xParam,xmax)
        percentage = (float(ii) / float(len(xBins[:-1]))) * 100.0
        if percentage >= printcount:
            print(f"{percentage:0.02f}% of statistics calculated!")
            printcount += printpercent
        xData.append((float(xmax) + float(xmin)) / 2.0)
        whereData = np.where((sortedData[xParam] >= xmin) & (sortedData[xParam] < xmax))[0]

        binnedData = {}
        for param, values in sortedData.items():
            if (param in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]) & (
                param not in exclusions
            ):
                binnedData.update({param: values[whereData]})
                sortedData.update({param: np.delete(values,whereData,axis=0)})

        dat = tr.calculate_statistics(
            binnedData,
            TRACERSPARAMS=CRPARAMS,
            saveParams=CRPARAMS["saveParams"],
            weightedStatsBool=weightedStatsBool,
        )

        # Fix values to arrays to remove concat error of 0D arrays
        dat = {key : val for key, val in dat.items()}
        datList.append(dat)

    statsData = {key: np.asarray([dd[key] for dd in datList if key in dd.keys()]) for key in datList[0].keys()}

    for param in exclusions:
        statsData.update({param: copy.deepcopy(dataDict[param])})

    statsData.update({f"{xParam}": np.asarray(xData)})
    return statsData

def map_params_to_types(snap, degeneracyBool = False):
    from itertools import combinations#
    import copy
    import pandas as pd
    from collections import Counter

    ADJUSTMENTFACTOR= 15

    #Unsorted unique using pandas!!
    snapType = False
    try:
        types = pd.unique(snap.data["type"])
        snapType = True
    except:
        try:
            types = pd.unique(snap["type"])
        except:
            raise Exception("[@remove_selection]: Unrecognised data format input! Data was neither Arepo snapshot format or Dictionary format!")
        snapType = False

    if snapType is True:
        lenTypes = [np.shape(np.where(snap.data["type"]==tp)[0])[0] for tp in types]
    else:
        lenTypes = [np.shape(np.where(snap["type"]==tp)[0])[0] for tp in types]

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

    if snapType is True:
        itrr = snap.data.items()
    else:
        itrr = snap.items()

    for key, value in itrr:
        if value is not None:
            whereValueShape = np.where(possibleValueLengthsSumTot == value.shape[0])[0]
            try:
                tmptypeCombos = possibleTypesCombos[whereValueShape][0]
            except:
                raise Exception(f"[@map_params_to_types]: FAILURE! CRITICAL! Parameter {key} could not be assigned a type!"
                                +"\n"
                                +"This usually means the parameter has an incorrect shape, and does not correspond to any type."
                                +"\n"
                                +"Check logic around parameter creation, or check that you meant to include this parameter into the data included in the call to this function."
                                +"\n"
                                +"e.g. you should only pass data with shapes corresponding to the included types in the passed data dict/snapshot.")
            paramToTypeMap.update({
                key: copy.deepcopy(tmptypeCombos),
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
                if snapType is True:
                    whereType = np.where(snap.data["type"]==tp)[0]
                    itrr = snap.data.items()
                else:
                    itrr = snap.items()
                    whereType = np.where(snap["type"]==tp)[0]

                for jj,(key, value) in enumerate(itrr):
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
                        if snapType is True:
                            snap.data[key] = value
                        else:
                            snap[key] = value
            #Recursive call to rerun this mapping and degeneracy detection
            snap,paramToTypeMap,degeneracyBool = map_params_to_types(snap,degeneracyBool = True)
        # You may be tempted to add the following:
        #else:
        #    degeneracyBool = False
        # DO NOT DO THIS! We want Degeneracy Bool to be "sitcky" such that if a degeneracy
        # is found, certain functions will complain! This is intentional to prevent, for example,
        # removal criteria being set on degenerate numbers of data for multiple types
        # where removal will fail!

    return snap, paramToTypeMap, degeneracyBool

def remove_selection(
    snap,
    removalConditionMask,
    errorString = "NOT SET",
    hush = False,
    verbose = False,
    ):
    """
    This function accepts as an input either an Arepo snapshot instance, or a dictionary along with a numpy boolean array of where to remove. It then
    removes from the data ~for ALL Arepo particle types~ whichever entries are True in this array.
    This function (and the function map_params_to_types) works on a combinatorics assumption. We assume that for every Arepo particle type, the number
    of data-points is unique. Ergo, for any given property of our data, we can infer which particle types have that property (e.g. position, age, etc)
    by producing every combination of sums of the number of data-points associated with each particle type and comparing it to the number of data-points
    for that property.
    This function ~does not care which Arepo particle types you have loaded into memory, nor their order~ but it ~does~ modify the data in the order
    of the types loaded in. Thus, much of the work of removing, for example, data of particle type 1 from loaded types [0, 1, 4] is involved with 
    determining the location of type 1 in the current parameter being modified, and adjusting the indices to be deleted accordingly.
    If a property has associated types [0,1] then we account for the length of type 0 before removing indices. 
    We modify the data of a property associated with types [0, 1] for example, by first removing the type 0 entries flagged for removal, and then
    the type 1 entries flagged for removal. Note: our function must keep track of the changes in shapes of each property and particle type as they
    are editted through the loops in this function.

    On a final note: we have included a (hopefully, rarely needed) degeneracy breaking functionality. It works by adding a small amount of data
    to any particle types that have equal number of data-points, such that the number of data-points of each type becomes unique again. This
    data is mostly in NaN form (with appropriate adjustment for dtypes where NaN isn't possible) and is removed at the end of this function
    """
    import copy
    import pandas as pd

    nRemovals = np.shape(np.where(removalConditionMask==True)[0])[0]
    if nRemovals==0:
        if not hush: print("[@remove_selection]: Number of data points to remove is zero! Skipping...")
        return snap

    snapType = False
    try:
        types = pd.unique(snap.data["type"])
        snapType = True
        if verbose: print("[@remove_selection]: snapshot type detected!")
    except:
        try:
            types = pd.unique(snap["type"])
        except:
            raise Exception("[@remove_selection]: Unrecognised data format input! Data was neither Arepo snapshot format or Dictionary format!")
        snapType = False
        if verbose: print("[@remove_selection]: dictionary type detected!")

    snap, paramToTypeMap, degeneracyBool = map_params_to_types(snap)

    if degeneracyBool is True:
        raise Exception(f"[remove_selection]: FAILURE! CRITICAL! Snapshot type lengths have been detected as degenerate by map_params_to_types() call in remove_selection()."+"\n"+"map_params_to_types() must be called seperately, prior to the evaluation of removalConditionMask in this call to remove_selection()"+"\n"+f"This error came from errorString {errorString} call to remove_selection()!")

    removedTruthy = np.full(types.shape,fill_value=False)
    if verbose is True: print("verbose!",errorString)

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
        skipBool = False
        if verbose is True: print(f" verbose! Type {tp}")
        if snapType is True:
            whereType = np.where(snap.data["type"]==tp)[0]
        else:
            whereType = np.where(snap["type"]==tp)[0]
        if verbose is True: print(f" verbose! START Shape of Type {np.shape(whereType)}")
        if(tp in typeCombosArray):
            if snapType is True:
                whereType = np.where(snap.data["type"]==tp)[0]
            else:
                whereType = np.where(snap["type"]==tp)[0]
            whereTypeInTypes = np.where(types==tp)[0][0]
            if verbose: print("whereTypeInTypes",whereTypeInTypes)
            locTypesOffset= np.array([jj for jj,tt in enumerate(types[:whereTypeInTypes])])# if tt not in typeCombosArray])

            if (len(locTypesOffset) == 0):
                # no types before this type so set offset to zero
                typesOffset = 0
            else:
                typesOffset = np.sum(np.array(paramToTypeMap["lty"])[locTypesOffset])
            if verbose: print("locTypesOffset",locTypesOffset)
            # Type specific removal which adjusts for any type in types that
            # aren't part of those included in removalConditionMask
            if snapType is True:
                whereType = np.where(snap.data["type"]==tp)[0]
            else:
                whereType = np.where(snap["type"]==tp)[0]
            if verbose:
                print(whereType)

            whereToRemove = np.where(removalConditionMask[whereType-typesOffset])[0] + typesOffset


            if verbose: print(typesOffset)
            if verbose: print(whereToRemove)
            if snapType is True:
                itrr = snap.data.items()
            else:
                itrr = snap.items()
            for jj,(key, value) in enumerate(itrr):
                if tp in paramToTypeMap[key]:
                    if value is not None:
                        if verbose: print(f"{jj}, {key}")

                        # For the key in snapshot data, retrieve types that
                        # contain that key (i.e. the types that have values
                        # for snap.data[key]). Find types in
                        # removalConditionMask relevant types that aren't
                        # used for this key, and remove the indexing
                        # of this unused data type (offset) from whereToRemove

                        locRemovalOffset= np.array([jj for jj,tt in enumerate(types[:whereTypeInTypes]) if tt not in paramToTypeMap[key]])
                        if verbose: print("locRemovalOffset",locRemovalOffset)
                        if verbose:
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
                        if verbose:
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
                            if len(whereToRemoveForKey)>0: 
                                removedTruthy[ii] = True
                            else:
                                removedTruthy[ii] = False
                            if newvalue.shape[0]>0:
                                if snapType is True:
                                    snap.data[key] = newvalue
                                else:
                                    snap[key] = newvalue
                            else:
                                # If no more data, set to None for cleaning
                                if snapType is True:
                                    snap.data[key] = None
                                else:
                                    snap[key] = None

                                # If no data for this type and key, remove this
                                # type from relevant to key mapping dict
                                remainingTypes = [tt for tt in paramToTypeMap[f"{key}"] if tt!=tp]

                                paramToTypeMap[f"{key}"] = copy.deepcopy(remainingTypes)

                        except Exception as e:
                            removedTruthy[ii] = False
                            if verbose:
                                print(f"[remove_selection]: verbose! Shape key: {np.shape(value)}")
                                print(f"[remove_selection]: verbose! WARNING! {str(e)}. Could not remove selection from {key} for particles of type {tp}")

            # Need to remove all entries (deleted (True) or kept (False))
            # of this type so that next type has
            # correct broadcast shape for removalConditionMask and whereType
            removalConditionMask = np.delete(removalConditionMask,(whereType-typesOffset),axis=0)
        else:
            removedTruthy[ii] = False
            # continue
        # typeCombosArray = np.delete(typeCombosArray,np.where(typeCombosArray==tp)[0])
        #update length of types
        if snapType is True:
            whereType = np.where(snap.data["type"]==tp)[0]
        else:
            whereType = np.where(snap["type"]==tp)[0]

        paramToTypeMap["lty"][ii] = whereType.shape[0]
        if verbose: print("paramToTypeMap['lty'][ii]",paramToTypeMap["lty"][ii])


    noneRemovedTruthy = np.all(~removedTruthy)

    if noneRemovedTruthy is True:
        print(f"[@remove_selection]: WARNING! Selection Criteria for error string = '{errorString}', has removed NO entries. Check logic! ")

    elif verbose is True:
        if np.any(~removedTruthy):
            print(f"[@remove_selection]: WARNING! verbose! Selection criteria for error string = '{errorString}' not applied to particles of type:")
            print(f"{types[np.where(removedTruthy==False)[0]]}")
        else:
            print(f"[@remove_selection]: verbose! Selection criteria for error string = '{errorString}' was ~successfully~ applied!")

    try:
        if snapType is True:
            nData = np.shape(snap.data["type"])[0]
        else:
            nData = np.shape(snap["type"])[0]
    except:
        raise Exception(f"[@remove_selection]: FAILURE! CRITICAL!"+
                        "\n"+f"Error String: {errorString} returned an empty snapShot!"
                        )

    #Remove None values and double check...
    snap = clean_snap_nones(snap)
    try:
        if snapType is True:
            nData = np.shape(snap.data["type"])[0]
        else:
            nData = np.shape(snap["type"])[0]
    except:
        raise Exception(f"[@remove_selection]: FAILURE! CRITICAL!"+
                        "\n"+f"Error String: {errorString} returned an empty snapShot!"
                        )
    return snap

def clean_snap_nones(snap):
    deleteKeys = []

    snapType = False
    try:
        types = pd.unique(snap.data["type"])
        snapType = True
    except:
        try:
            types = pd.unique(snap["type"])
        except:
            raise Exception("[@clean_snap_nones]: Unrecognised data format input! Data was neither Arepo snapshot format or Dictionary format!")
        snapType = False


    if snapType is True:
        itrr = snap.data.items()
    else:
        itrr = snap.items()
    for key, value in itrr:
        if value is None:
            deleteKeys.append(key)

    for key in deleteKeys:
        if snapType is True:
            del snap.data[key]
        else:
            del snap[key]
    return snap

def clean_snap_params(snap,paramsOfInterest):
    """
    Delete unwanted paramaters of snapshot data/dictionary data, and save only paramsOfInterest.
    """
    deleteKeys = []

    snapType = False
    try:
        types = pd.unique(snap.data["type"])
        snapType = True
    except:
        try:
            types = pd.unique(snap["type"])
        except:
            raise Exception("[@clean_snap_params]: Unrecognised data format input! Data was neither Arepo snapshot format or Dictionary format!")
        snapType = False


    if snapType is True:
        itrr = snap.data.items()
    else:
        itrr = snap.items()
    for key, value in itrr:
        if np.isin(np.asarray(key),np.asarray(paramsOfInterest)) == False:
            deleteKeys.append(key)

    for key in deleteKeys:
        if snapType is True:
            del snap.data[key]
        else:
            del snap[key]

    # Touch each data type we are interested in to load it into memory
    # prevents errors when remove_selection is called later
    if snapType is True:
        itrr = snap.data.items()
    else:
        itrr = snap.items()

    for key, value in itrr:
        if snapType is True:
            snap.data[key] *= 1
        else:
            snap[key] *= 1
    return snap

def cr_calculate_projections(
    snap,
    CRPARAMS,
    ylabel,
    xlimDict,
    snapNumber=None,
    params = ["T","n_H", "B", "gz"],
    xsize = 5.0,
    ysize = 5.0,
    fontsize=13,
    Axes=[0,1],
    boxsize=400.0,
    boxlos=50.0,
    pixreslos=0.3,
    pixres=0.3,
    projection=False,
    DPI=200,
    CMAP="inferno",
    numthreads=10,
 ):

    keys = list(CRPARAMS.keys())
    selectKey0 = keys[0]



    for param in params+["Tdens", "rho_rhomean"]:
        try:
            tmp = snap.data[param]
        except:
            snap = tr.calculate_tracked_parameters(
                snap,
                oc.elements,
                oc.elements_Z,
                oc.elements_mass,
                oc.elements_solar,
                oc.Zsolar,
                oc.omegabaryon0,
                snapNumber,
                paramsOfInterest=[param],
                mappingBool=True,
                numthreads=numthreads,
                verbose = False,
            )

    out = {}

    for sliceParam in params:
        tmpout = apt.plot_slices(snap,
            ylabel,
            xlimDict,
            logParameters = CRPARAMS['logParameters'],
            snapNumber=snapNumber,
            sliceParam = sliceParam,
            xsize = xsize,
            ysize = ysize,
            fontsize=fontsize,
            Axes=Axes,
            boxsize=boxsize,
            boxlos=boxlos,
            pixreslos=pixreslos,
            pixres=pixres,
            projection=projection,
            DPI=DPI,
            CMAP=CMAP,
            numthreads=numthreads,
            saveFigure = False,
        )
        out.update(tmpout)

    return out

def cr_slice_averaging(
    quadPlotDict,
    CRPARAMS,
    snapRange,
    averageType = 'median',
):

    print("Slice plot averaging...")

    quadPlotDictAveraged = {}
    flatData = {}

    tmp = {}
    newKey = (f"{CRPARAMS['resolution']}", 
              f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")
    selectKey0 = (
        f"{CRPARAMS['resolution']}",
        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
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
                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                    f"{int(snapNumber)}",
                )
                stackList.append(quadPlotDict[selectKey][param][key].copy())
            outvals = np.stack(stackList, axis=-1)
            innertmp.update({key : outvals})
            innertmp.update({"type": None})
        tmp.update({param: innertmp})
    flatData.update({newKey: tmp})

    for param in params:
        tmp ={}
        for arg in ["x","y"]:
            if averageType == "median":
               tmp.update({arg : np.nanmedian(flatData[newKey][param][arg],axis=-1)})
            elif averageType == "mean":
                tmp.update({arg : np.nanmean(flatData[newKey][param][arg],axis=-1)})
            else:
                raise Exception(f"[@cr_slice_averaging]: FAILURE! cr_slice_averaging given unknown averageType of {averageType}.")
        
        if averageType == "median":
            tmp.update({"grid" : np.nanmedian(flatData[newKey][param]["grid"],axis=-1)})
        elif averageType == "mean":
            tmp.update({"grid" : np.nanmean(flatData[newKey][param]["grid"],axis=-1)})
        else:
            raise Exception(f"[@cr_slice_averaging]: FAILURE! cr_slice_averaging given unknown averageType of {averageType}.")

        tmp.update({"type" : None})
        quadPlotDictAveraged.update({param : tmp})
        

    print("...averaging done!")
    # STOP1080
    return quadPlotDictAveraged

#def _map_cart_grid_to_cells_2d(pos_array, xx, yy):
#    nn = xx.shape[0]
#    return np.array(
#        [
#            np.ravel_multi_index(
#                [
#                    np.argmin(np.abs(xx - pos[0])),
#                    np.argmin(np.abs(yy - pos[1])),
#                ],
#                (nn, nn),
#            )
#            for pos in pos_array
#        ]
#    ).flatten()

#def _wrapper_map_cart_grid_to_cell_2d(pos_array, boxsize, gridres, center):

#    v_map_cart_grid_to_cells_2d = np.vectorize(
#        _map_cart_grid_to_cells_2d, signature="(m,2),(n),(n)->(m)"
#    )

#    halfbox = copy.copy(boxsize) / 2.0
#    coord_spacings = np.linspace(-1.0 * halfbox, halfbox, gridres)
#    xx = coord_spacings + center[0]
#    yy = coord_spacings + center[1]
#    #zz = coord_spacings + center[2]
#    out = v_map_cart_grid_to_cells_2d(pos_array, xx, yy)

#    return out

#def map_cart_grid_to_cells_2d(
#    snap,
#    param,
#    grid,
#    boxsize,
#    axes,
#    mapping=None,
#    ptype=0,
#    center=False,
#    box=None,
#    use_only_cells=None,
#    numthreads=8,
#    box_gt_one_mpc=False,
#    verbose=False,
#    nParentProcesses=1,
#    ):
#    import pylab

#    if use_only_cells is None:
#        use_only_cells = np.where(snap.type == ptype)[0]

#    if type(center) == list:
#        center = pylab.array(center)
#    elif type(center) != np.ndarray:
#        center = snap.center

#    if box is None:
#        if snap.boxsize >= 1.0:
#            print(f"[@map_cart_grid_to_cells_2d]: a maximum half box size (given by snap.boxsize) of {snap.boxsize:.5f} [Mpc] was detected." +
#                  "\n"+"User has not indicated box_gt_one_mpc so we are limiting to boxsize of 500 kpc (half box of 250 kpc). Remaining data will be NaN...")
#            box = np.array([0.5, 0.5, 0.5])
#        else:
#            bb = (snap.boxsize*2.)
#            box = pylab.array([bb for ii in range(0, 3)])
#    elif np.all(box == box[0]) is False:
#        raise Exception(
#            f"[@map_cart_grid_to_cells_2d]: WARNING! CRITICAL! FAILURE!"
#            + "\n"
#            + "Box not False, None, or all elements equal."
#            + "\n"
#            + "function @map_cart_grid_to_cells_2d not adapted for non-cube boxes."
#            + "\n"
#            + "All box sides must be equal, or snap.boxsize [Mpc] will be used."
#        )
#    elif (type(box) == list) | (type(box) == np.ndarray):
#        if (type(box) == list):
#            box = np.array(box)
#        if box_gt_one_mpc is False:
#            maxval = np.nanmax(box)
#            if maxval >= 1.0:
#                print(f"[@map_cart_grid_to_cells_2d]: a maximum box size of {maxval} was detected."+"\n" +
#                      "User has not indicated box_gt_one_mpc so we are assuming the box size has been given in kpc."+"\n"+"We will adjust to Mpc and continue...")
#                box = pylab.array([(bb*2.) / (1e3) for bb in box])
#            else:
#                box = pylab.array([(bb*2.) for bb in box])
#        else:
#            box = pylab.array([(bb*2.) for bb in box])
#    boxsize = box[0]

#    pos = snap.pos[use_only_cells, :].astype("float64").copy() / 1e3
#    px = np.abs(pos[:, 0] - center[0])
#    py = np.abs(pos[:, 1] - center[1])
#    pz = np.abs(pos[:, 2] - center[2])

#    (pp,) = np.where((px <= 0.5*box[0]) &
#                     (py <= 0.5*box[1]) & (pz <= 0.5*box[2]))
#    if verbose:
#        print("Selected %d of %d particles." % (pp.size, snap.npart))

    
#    #------------------------------------------------------------#    
#    #       Sort by image cart grid axis ordering and drop
#    #           projection axis from posdata
#    #------------------------------------------------------------#

#    posdata = pos[pp][:,tuple(axes)]


#    whereCGM = np.where((snap.data["R"][use_only_cells][pp] <= (boxsize/2.)*1e3) & (
#        snap.data["type"][use_only_cells][pp] == 0) & (snap.data["sfr"][use_only_cells][pp] <= 0.0))[0]

#    avgCellLength = (np.nanmean(
#        snap.data["vol"][use_only_cells][pp][whereCGM])/1e9)**(1/3)  # [Mpc]

#    gridres = int(math.floor(boxsize/avgCellLength))

#    #------------------------------------------------------------#
    

#    splitPos = np.array_split(posdata, numthreads)

#    args_list = [
#        [posSubset, boxsize, gridres, center]
#        for posSubset in splitPos
#    ]

#    if verbose:
#        print("Map...")
#    start = time.time()

#    if nParentProcesses > 1:
#        if verbose:
#            print("Daemon processes cannot spawn children...")
#            print("Starting single CPU analysis...")
#        output = _wrapper_map_cart_grid_to_cell_2d(
#            posdata, boxsize, gridres, center)
#        mapping = output.astype(np.int32)
#    else:
#        if verbose:
#            print(
#                f"Starting numthreads = {numthreads} mp pool with data split into {numthreads} chunks..."
#            )
#        pool = mp.Pool(processes=numthreads)
#        outputtmp = [pool.apply_async(_wrapper_map_cart_grid_to_cell_2d,
#                                        args=args, error_callback=tr.err_catcher) for args in args_list
#                        ]
#        pool.close()
#        pool.join()
#        output = [out.get() for out in outputtmp]
#        mapping = np.concatenate(
#            tuple(output), axis=0
#        ).astype(np.int32)

#    stop = time.time()

#    if verbose:
#        print("...done!")
#    if verbose:
#        print(f"Mapping took {stop-start:.2f}s")

#    oldshape = np.shape(snap.data["R"])[0]
#    # Perform mapping from Cart Grid back to approx. cells
    
#    tmp = ((grid.reshape(-1))[mapping]).copy()

#    snap.data[param] = snap.data[param].reshape(-1)

#    snap.data[param] = np.full(oldshape, fill_value=np.nan)

#    snap.data[param][use_only_cells[pp]] = tmp.copy()

#    del tmp

#    assert (
#        np.shape(snap.data[param])[0] == oldshape
#    ), f"[@map_cart_grid_to_cells_2d]: WARNING! CRITICAL! FAILURE! Output from Gradient Calc and subsequent mapping not equal in shape to input data! Check Logic!"

#    return snap, mapping

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
