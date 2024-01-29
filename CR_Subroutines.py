# coding=utf-8
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
import warnings

def cr_analysis_radial(
    snapNumber,
    CRPARAMS,
    ylabel,
    xlimDict,
    DataSavepathBase,
    FigureSavepathBase,
    types = [0, 1, 4],
    colImagexlimDict = None,
    imageCmapDict = {},
    FullDataPathSuffix=".h5",
    rotation_matrix=None,
    verbose = False,
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
    starsout = {}
    colout = {}
    quadPlotDict = {}
    innerColout = {}

    if colImagexlimDict is None:
        colImagexlimDict = copy.copy(xlimDict)


    saveDir = ( DataSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
        )
    saveDirFigures = ( FigureSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
        )

    # Generate halo directory
    tmp = ""
    for savePathChunk in saveDir.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    tmp = ""
    for savePathChunk in saveDirFigures.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass
        
    DataSavepath = saveDir

    FiguresSavepath = saveDirFigures

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
    snap = gadget_readsnap(
        snapNumber,
        loadpath,
        hdf5=True,
        loadonlytype=types,
        lazy_load=True,
        subfind=snap_subfind,
        #loadonlyhalo=int(CRPARAMS["HaloID"]),

    )

    # # load in the subfind group files
    # snap_subfind = load_subfind(100, dir="/home/universe/spxfv/Auriga/level4_cgm/h12_1kpc_CRs/output/")
    #
    # # load in the gas particles mass and position only for HaloID 0.
    # #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    # #       gas and stars (type 0 and 4) MUST be loaded first!!
    # snap = gadget_readsnap(100,"/home/universe/spxfv/Auriga/level4_cgm/h12_1kpc_CRs/output/",hdf5=True,loadonlytype=[0, 1, 4],lazy_load=True,subfind=snap_subfind)
    # snapStars = gadget_readsnap(
    #     100,
    #     "/home/universe/spxfv/Auriga/level4_cgm/h12_1kpc_CRs/output/",
    #     hdf5=True,
    #     loadonlytype=[4],
    #     lazy_load=True,
    #     subfind=snap_subfind,
    # )

    snap.calc_sf_indizes(snap_subfind)
    if rotation_matrix is None:
        rotation_matrix = snap.select_halo(snap_subfind, do_rotation=True)
        rotationsavepath = saveDir + f"rotation_matrix_{int(snapNumber)}.h5"
        tr.hdf5_save(rotationsavepath,{(f"{CRPARAMS['resolution']}",
        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}") : {"rotation_matrix" : rotation_matrix}})
        ## If we don't want to use the same rotation matrix for all snapshots, set rotation_matrix back to None
        if (CRPARAMS["constantRotationMatrix"] == False):
            rotation_matrix = None
    else:
        snap.select_halo(snap_subfind, do_rotation=False)
        snap.rotateto(
            rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
        )

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Clean SnapShot parameters..."
    )

    snap = clean_snap_params(
        snap,
        paramsOfInterest = CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]
    )

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}"
    )

    # --------------------------#
    ##    Units Conversion    ##
    # --------------------------#

    # Convert Units
    ## Make this a seperate function at some point??
    snap.pos *= 1e3  # [kpc]
    snap.vol *= 1e9  # [kpc^3]
    snap.mass *= 1e10  # [Msol]
    snap.hrgm *= 1e10  # [Msol]
    snap.gima *= 1e10  # [Msol]

    snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)
    rvir = (snap_subfind.data["frc2"] * 1e3)[CRPARAMS["HaloID"]]
    stellarType = 4
    rdisc = (snap_subfind.data["shmt"] * 1e3)[CRPARAMS["HaloID"]][stellarType]
    
    boxmax = max([CRPARAMS['boxsize'],CRPARAMS['boxlos'],CRPARAMS['coldenslos']])

    print(
        f"[@{int(snapNumber)}]: Remove beyond {boxmax:2.2f} kpc..."
    )

    whereOutsideBox = np.abs(snap.data["pos"]) > boxmax

    snap = remove_selection(
        snap,
        removalConditionMask = whereOutsideBox,
        errorString = "Remove Outside Box",
        verbose = verbose,
        )


    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Remove wind from stars..."
    )

    whereWind = snap.data["age"] < 0.0

    snap = remove_selection(
        snap,
        removalConditionMask = whereWind,
        errorString = "Remove Wind from Stars",
        verbose = verbose,
        )
    
    # Calculate New Parameters and Load into memory others we want to track
    snap = tr.calculate_tracked_parameters(
        snap,
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
        box=[boxmax,boxmax,boxmax],
        numthreads=CRPARAMS["numthreads"],
        DataSavepath = DataSavepath,
        verbose = verbose,
    )
    snap.data["R"] = snap.data["R"]/rvir
    # snap = tr.calculate_tracked_parameters(snap,oc.elements,oc.elements_Z,oc.elements_mass,oc.elements_solar,oc.Zsolar,oc.omegabaryon0,100)
    if len(CRPARAMS["colParams"])>0:

        # Create variant of xlimDict specifically for images of col params
        tmpxlimDict = copy.deepcopy(xlimDict)

        # Add the col param specific limits to the xlimDict variant
        for key, value in colImagexlimDict.items():
            tmpxlimDict[key] = value

        #---------------#
        # Check for any none-position-based parameters we need to track for col params:
        #       Start with mass (always needed!) and xParam:
        additionalColParams = ["mass"]
        if np.any(np.isin(np.asarray([CRPARAMS["xParam"]]),np.array(["R","x","y","z","pos"]))) == False:
            additionalColParams.append(CRPARAMS["xParam"])

        #       Now add in anything we needed to track for weights of col params in statistics
        cols = CRPARAMS["colParams"]
        for param in cols:
            additionalParam = CRPARAMS["nonMassWeightDict"][param]
            if (np.any(np.isin(np.asarray([additionalParam]),np.asarray(additionalColParams))) == False) \
            & (additionalParam is not None) & (additionalParam != "count"):
                additionalColParams.append(additionalParam)
        #---------------#

        # If there are other params to be tracked for col params, we need to create a projection
        # of them so as to be able to map these projection values back to the col param maps.
        # A side effect of this is that we will create "images" of any of these additional params.
        # Thus, we want to provide empty limits for the colourbars of these images as they will almost
        # certainly require different limits to those provided for the PDF plots, for example. 
        # In particular, params like mass will need very different limits to those used in the
        # PDF plots. We have left this side effect in this code version as it provides a useful
        # way of testing whether making a projection of unusual params to image (e.g. mass, or volume)
        # provide sensible, physical results.
        for key in additionalColParams:
            tmpxlimDict[key] = {}

        innerColout = {}
        cols = CRPARAMS["colParams"]+additionalColParams
        for param in cols:
            print(
                "\n"+f"[@{int(snapNumber)}]: Calculate {param} map..."
            )

            # By default, we set projection here to False. This ensures any weighting maps are
            # slices (projection versions were found to produce unphysical and unreliable results).
            # However, any _col parameters are forced into Projection=True inside apt.plot_slices().
            tmpdict = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=tmpxlimDict,
                logParameters = CRPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = param,
                Axes=CRPARAMS["Axes"],
                averageAcrossAxes = CRPARAMS["averageAcrossAxes"],
                saveAllAxesImages = CRPARAMS["saveAllAxesImages"],
                xsize = CRPARAMS["xsizeImages"],
                ysize = CRPARAMS["ysizeImages"],
                colourmapMain = CRPARAMS["colourmapMain"],
                colourmapsUnique = imageCmapDict,
                boxsize = CRPARAMS["boxsize"],
                boxlos = CRPARAMS["coldenslos"],
                pixreslos = CRPARAMS["pixreslos"],
                pixres = CRPARAMS["pixres"],
                projection = False,
                DPI = CRPARAMS["DPIimages"],
                numthreads=CRPARAMS["numthreads"],
                savePathBase = FiguresSavepath,
                savePathBaseFigureData = DataSavepath,
                saveFigureData = True,
                saveFigure = CRPARAMS["SaveImages"],
                inplace = False,
            )

            if tmpdict is not None:
                innerColout.update({param: (copy.deepcopy(tmpdict[param]["grid"])).reshape(-1)})
                
                # !!
                # You !! MUST !! provide type data for data that is no longer snapshot associated and thus
                # no longer has type data associated with it. This is to ensure any future selections made from
                # the dataset do not break the type length associated logic which is engrained in all of these
                # tools, primarily via the ' cr.remove_selection() ' function.
                # 
                # You may choose if you wish to discard/mask
                # a subset of your data from future figures by setting it to a non-zero integer value for type
                # but beware, this is an untested use-case and (especially for pre-existing types between 0-6)
                # the tools provided here may exhibit unexpected behaviours!
                # !!
                newShape = np.shape(innerColout[param])
                innerColout.update({"type": np.full(shape=newShape, fill_value=0)})

                if (CRPARAMS["xParam"] == "R") & (CRPARAMS["xParam"] not in list(innerColout.keys())):
                    xx = (copy.deepcopy(tmpdict[param]["x"])).reshape(-1)
                    xx = np.array(
                        [
                            (x1 + x2) / 2.0
                            for (x1, x2) in zip(xx[:-1], xx[1:])
                        ]
                    )
                    yy = (copy.deepcopy(tmpdict[param]["y"])).reshape(-1)
                    yy = np.array(
                        [
                            (x1 + x2) / 2.0
                            for (x1, x2) in zip(yy[:-1], yy[1:])
                        ]
                    )
                    values = np.linalg.norm(np.asarray(np.meshgrid(xx,yy)), axis=0).reshape(-1)
                    innerColout.update({"R": values/rvir})
    
    for param in CRPARAMS["imageParams"]+["Tdens", "rho_rhomean"]:
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
                logParameters = CRPARAMS['logParameters'],
                paramsOfInterest=[param],
                mappingBool=True,
                box=[boxmax,boxmax,boxmax],
                numthreads=CRPARAMS["numthreads"],
                DataSavepath = DataSavepath,
                verbose = verbose,
            )

        ## Check that radii are still being stored in units of Rvir...
    if np.all(snap.data["R"][np.where(np.linalg.norm(snap.data["pos"],axis=1)<=CRPARAMS["Router"]*rvir)[0]]<=CRPARAMS["Router"]): 
        pass
    else:
        ## if radii are not in units of rvir, set that now...
        snap.data["R"] = snap.data["R"]/rvir


    for projectionBool in [False, True]:
        for param in CRPARAMS["imageParams"]:
            tmpdict = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = CRPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = param,
                Axes=CRPARAMS["Axes"],
                xsize = CRPARAMS["xsizeImages"],
                ysize = CRPARAMS["ysizeImages"],
                colourmapMain = CRPARAMS["colourmapMain"],
                colourmapsUnique = imageCmapDict,
                boxsize = CRPARAMS["boxsize"],
                boxlos = CRPARAMS["boxlos"],
                pixreslos = CRPARAMS["pixreslos"],
                pixres = CRPARAMS["pixres"],
                projection = projectionBool,
                DPI = CRPARAMS["DPIimages"],
                numthreads=CRPARAMS["numthreads"],
                savePathBase = FiguresSavepath,
                savePathBaseFigureData = DataSavepath,
                saveFigureData = True,
                saveFigure = CRPARAMS["SaveImages"],
                inplace = False,
            )
            if tmpdict is not None:
                quadPlotDict.update(tmpdict)

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Delete unwanted data..."
    )


        ## Check that radii are still being stored in units of Rvir...
    if np.all(snap.data["R"][np.where(np.linalg.norm(snap.data["pos"],axis=1)<=CRPARAMS["Router"]*rvir)[0]]<=CRPARAMS["Router"]): 
        pass
    else:
        ## if radii are not in units of rvir, set that now...
        snap.data["R"] = snap.data["R"]/rvir

    whereSatellite = np.isin(snap.data["subhalo"],np.array([-1,int(CRPARAMS["HaloID"]),np.nan]))==False

    snap = remove_selection(
        snap,
        removalConditionMask = whereSatellite,
        errorString = "Remove Satellites",
        verbose = verbose,
    )

    # Redshift
    redshift = snap.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # Get lookback time in Gyrs
    # [0] to remove from numpy array for purposes of plot title
    lookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    print( 
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Ages: get_lookback_time_from_a() ..."
    )
    ages = snap.cosmology_get_lookback_time_from_a(snap.data["age"],is_flat=True)
    snap.data["age"] = ages

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Convert from SnapShot to Dictionary  ..."
    )


    # print(
    #     f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary  ..."
    # )
    # Make normal dictionary form of snap
    innerStars = {}
    for key, value in snap.data.items():
        if value is not None:
            innerStars.update({key: copy.deepcopy(value)})

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select stars and gas separately..."
    )

    whereNotStars = np.isin(innerStars["type"],np.array([0,1,2,3,5,6]))==True

    innerStars = remove_selection(
        innerStars,
        removalConditionMask = whereNotStars,
        errorString = "Remove Not Stars Types",
        verbose = verbose,
        )


    ##
    # Generate dataset clone containing all particle types in 'types' variable
    innerFull = {}
    for key, value in snap.data.items():
        if value is not None:
            innerFull.update({key: copy.deepcopy(value)})

    whereNotGas = np.isin(snap.data["type"],np.array([1,2,3,5,6]))==True

    snap = remove_selection(
        snap,
        removalConditionMask = whereNotGas,
        errorString = "Remove Not Gas Types (w/o stars)",
        verbose = verbose,
        )

    if analysisType == "cgm":
        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select the CGM..."
        )

    
        whereNotCGM = (snap.data["R"] > CRPARAMS["Router"])

        snap = remove_selection(
            snap,
            removalConditionMask = whereNotCGM,
            errorString = "Remove NOT CGM from Gas >Router",
            verbose = verbose,
            )

        whereNotStarsCGM = innerStars["R"] > CRPARAMS["Router"]

        innerStars = remove_selection(
            innerStars,
            removalConditionMask = whereNotStarsCGM,
            errorString = "Remove Stars Not within Router",
            verbose = verbose,
            )
        
        whereNotCGMFull = (innerFull["R"] > CRPARAMS["Router"])

        innerFull = remove_selection(
            innerFull,
            removalConditionMask = whereNotCGMFull,
            errorString = "Remove not within Router for full data",
            verbose = verbose,
            )    
        # whereNot CGM = (snap.data["R"] < CRPARAMS["Rinner"])

        # snap = remove_selection(
        #     snap,
        #     removalConditionMask = whereNotCGM,
        #     errorString = "Remove NOT CGM from Gas <Rinner",
        #     verbose = verbose,
        #     )
        
        # # whereNotCGM = (snap.data["sfr"] > 0.0)

        # # snap = remove_selection(
        # #     snap,
        # #     removalConditionMask = whereNotCGM,
        # #     errorString = "Remove NOT CGM from Gas <Rinner",
        # #     verbose = verbose,
        # #     )

        param = "ndens"

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
                logParameters = CRPARAMS['logParameters'],
                paramsOfInterest=[param],
                mappingBool=True,
                box=[boxmax,boxmax,boxmax],
                numthreads=CRPARAMS["numthreads"],
                DataSavepath = DataSavepath,
                verbose = verbose,
            )

        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Remove n >= 1.1e-1 cm^-3 gas..."
        )

        whereAboveCritDens = (snap.data["ndens"] >= 1.1e-1)

        snap = remove_selection(
            snap,
            removalConditionMask = whereAboveCritDens,
            errorString = "Remove above critical density for standard star formation",
            verbose = verbose,
            )
                
    elif analysisType == "ism":
        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select the ISM..."
        )

        whereNotISM = (snap.data["ndens"] < 1.1e-1) & (snap.data["R"] > CRPARAMS["Rinner"])

        snap = remove_selection(
            snap,
            removalConditionMask = whereNotISM,
            errorString = "Remove NOT ISM from Gas",
            verbose = verbose,
            )
        
        whereOutsideSelection = (innerFull["R"] > CRPARAMS["Rinner"])

        innerFull = remove_selection(
            innerFull,
            removalConditionMask = whereOutsideSelection,
            errorString = "Remove outside selection from full data",
            verbose = verbose,
            )  
        
        whereNotStarsCGM = (innerStars["ndens"] < 1.1e-1) & (innerStars["R"] > CRPARAMS["Rinner"])

        innerStars = remove_selection(
            innerStars,
            removalConditionMask = whereNotStarsCGM,
            errorString = "Remove NOT ISM from Stars",
            verbose = verbose,
            )
        
    elif analysisType == "all":

        print(
            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Select all of the halo..."
        )

        whereOutsideSelection = (snap.data["R"] > CRPARAMS["Router"])

        snap = remove_selection(
            snap,
            removalConditionMask = whereOutsideSelection,
            errorString = "Remove ALL Outside Selection from Gas",
            verbose = verbose,
            )
        
        
        whereOutsideSelectionFull = (innerFull["R"] > CRPARAMS["Router"])

        innerFull = remove_selection(
            innerFull,
            removalConditionMask = whereOutsideSelectionFull,
            errorString = "Remove ALL Outside Selection from full data",
            verbose = verbose,
            )  
        
        
        whereOutsideSelectionStars = (innerStars["R"] > CRPARAMS["Router"])

        innerStars = remove_selection(
            innerStars,
            removalConditionMask = whereOutsideSelectionStars,
            errorString = "Remove ALL Outside Selection from Stars",
            verbose = verbose,
            )  

    whereNotGas = snap.data["type"]==4

    snap = remove_selection(
        snap,
        removalConditionMask = whereNotGas,
        errorString = "Remove Stars",
        verbose = verbose,
        )

    inner = {}
    for key, value in snap.data.items():
        if value is not None:
            inner.update({key: copy.deepcopy(value)})

    inner["Redshift"] = np.array([redshift])
    inner["Lookback"] = np.array([lookback])
    inner["Snap"] = np.array([snapNumber])
    inner["Rvir"] = np.array([rvir])
    inner["Rdisc"] = np.array([rdisc])

    innerStars["Redshift"] = np.array([redshift])
    innerStars["Lookback"] = np.array([lookback])
    innerStars["Snap"] = np.array([snapNumber])
    innerStars["Rvir"] = np.array([rvir])
    innerStars["Rdisc"] = np.array([rdisc])

    innerColout["Redshift"] = np.array([redshift])
    innerColout["Lookback"] = np.array([lookback])
    innerColout["Snap"] = np.array([snapNumber])
    innerColout["Rvir"] = np.array([rvir])
    innerColout["Rdisc"] = np.array([rdisc])

    quadPlotDict["Redshift"] = np.array([redshift])
    quadPlotDict["Lookback"] = np.array([lookback])
    quadPlotDict["Snap"] = np.array([snapNumber])
    quadPlotDict["Rvir"] = np.array([rvir])
    quadPlotDict["Rdisc"] = np.array([rdisc])

    innerFull["Redshift"] = np.array([redshift])
    innerFull["Lookback"] = np.array([lookback])
    innerFull["Snap"] = np.array([snapNumber])
    innerFull["Rvir"] = np.array([rvir])
    innerFull["Rdisc"] = np.array([rdisc])

    # # Make normal dictionary form of snap
    # inner = {}
    # for key, value in snap.data.items():
    #     if key in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]:
    #         if value is not None:
    #             inner.update({key: copy.deepcopy(value)})

    # del snap

    # innerStars = {}
    # for key, value in snapStars.data.items():
    #     if key in CRPARAMS["saveParams"] + CRPARAMS["saveEssentials"]:
    #         if value is not None:
    #             innerStars.update({key: copy.deepcopy(value)})

    # del snapStars
    # # Add to final output
    out.update(
        {
            (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                f"{int(snapNumber)}",
            ): inner
        }
    )

    starsout.update(
        {
            (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                "Stars",
                f"{int(snapNumber)}",
            ): innerStars
        }
    )

    colout.update(
        {
            (
                f"{CRPARAMS['resolution']}",
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                "col",
                f"{int(snapNumber)}",
            ): innerColout
        }
    )

    quadPlotDictOut = { (
            f"{CRPARAMS['resolution']}",
            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
            f"{int(snapNumber)}",
        ): quadPlotDict
    }

    full = { (
            f"{CRPARAMS['resolution']}",
            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
            f"{int(snapNumber)}",
        ): innerFull
    }

    print(
        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Finishing process..."
    )
    return out, starsout, colout, quadPlotDictOut, full, rotation_matrix


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

# def cr_flatten_wrt_time(dataDict, CRPARAMS, snapRange):

#     print("Flattening with respect to time...")
#     flatData = {}

#     print("Gas...")
#     tmp = {}
#     newKey = (f"{CRPARAMS['resolution']}",
#               f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")
#     selectKey0 = (
#         f"{CRPARAMS['resolution']}",
#         f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
#         f"{int(snapRange[0])}",
#     )

#     keys = copy.deepcopy(list(dataDict[selectKey0].keys()))

#     for ii, subkey in enumerate(keys):
#         print(f"{float(ii)/float(len(keys)):3.1%}")
#         concatenateList = []
#         for snapNumber in snapRange:
#             selectKey = (
#                 f"{CRPARAMS['resolution']}",
#                 f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
#                 f"{int(snapNumber)}",
#             )
#             concatenateList.append(dataDict[selectKey][subkey].copy())

#             del dataDict[selectKey][subkey]
#             # # Fix values to arrays to remove concat error of 0D arrays
#             # for k, val in dataDict.items():
#             #     dataDict[k] = np.array([val]).flatten()
#         outvals = np.concatenate((concatenateList), axis=0)
#         tmp.update({subkey: outvals})
#     flatData.update({newKey: tmp})

#     print("Stars...")
#     tmp = {}
#     newKey = (f"{CRPARAMS['resolution']}", 
#               f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
#               "Stars")
#     selectKey0 = (
#         f"{CRPARAMS['resolution']}",
#         f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
#         f"{int(snapRange[0])}",
#         "Stars",
#     )

#     keys = copy.deepcopy(list(dataDict[selectKey0].keys()))

#     for ii, subkey in enumerate(keys):
#         print(f"{float(ii)/float(len(keys)):3.1%}")
#         concatenateList = []
#         for snapNumber in snapRange:
#             selectKey = (
#                 f"{CRPARAMS['resolution']}",
#                 f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
#                 f"{int(snapNumber)}",
#                 "Stars",
#             )
#             concatenateList.append(dataDict[selectKey][subkey].copy())

#             del dataDict[selectKey][subkey]
#             # # Fix values to arrays to remove concat error of 0D arrays
#             # for k, val in dataDict.items():
#             #     dataDict[k] = np.array([val]).flatten()
#         outvals = np.concatenate((concatenateList), axis=0)
#         tmp.update({subkey: outvals})
#     flatData.update({newKey: tmp})

#     print("...flattening done!")
#     return flatData



def cr_flatten_wrt_time(input, stack = True, verbose = False, hush = False):
    """
        For the combining of ~binned~ data across different snapshots.

        NOTE: 
        1)  This function cannot flatten data without tuple keys in format: keys = [(A, #1), (A, #2), ..., (B, #1), (B, #2), ...] or [(A,B, #1), (A,B, #2), ..., (C,D, #1), (C,D, #2), ...] etc.
        This is due to dictionary keys being unique. We cannot have more than one (A,B), as they will overwrite one another.

        2)  This will not work in default mode of stack = True if using snapshot data that has not been binned or summarised in some way.
        This is due to data from different snapshots having different numbers of data points, slty, flty, etc. 
       
        3)  Consequently, if you want to stack unsummarised snapshot data you cannot simply pass in a snapshot - snapshots aren't designed to hold data for more than one point of time from the simulations.
         a) Again, if you ~do~ wish to combine snapshot data (see below), then you must pass a dictionary where the keys have the last entry as numerals (these can be any, arbitrary numbering). And,
         b) Pass kwarg stack = False to concatenate data along 0th axis (see 2) from above as to why you cannot stack raw snapshot data).
        However, please use this with caution when applied to snapshot data, as you may run out of RAM rather quickly...

        4)  To clarify:
         a) the non-numeral entries before the last value in the tuple of the keys can be whatever you like. [(A, #1), (A, #2), ..., (B, #1), (B, #2), ...] or [(A,B, #1), (A,B, #2), ..., (C,D, #1), (C,D, #2), ...] 
            Will combine into [(A),(B)] and [(A,B),(C,D)]. But you could use [(A,B, #1), (A,B, #2), ..., (C,B, #1), (C,B, #2), ...] which would combine into [(A,B),(C,B)]
         b) The final entry in the tuple of each key can be a string, so long as each entry is unique. You can label the data however you wish, but it is the last element of the tuple that will be dropped when flattening the data. However,
            the data will be sorted into ascending order of the this last entry in the key tuple if it is numeric.
    """

    snapType = False
    try:
        tmp = input.data.keys()
        snapType = True
        raise Exception("[@cr_flatten_wrt_time]: ERROR! FATAL! Data format must be dictionary!"+"\n"+"Cannot flatten data without tuple keys in format: keys = [(A, #1), (A, #2), ..., (B, #1), (B, #2), ...] or [(A,B, #1), (A,B, #2), ..., (C,D, #1), (C,D, #2), ...] etc.")

    except:
        try:
            tmp = input.keys()
        except:
            raise Exception("[@cr_flatten_wrt_time]: Unrecognised data format input! Data was neither Arepo snapshot format or Dictionary format! Data format must be dictionary!")
        snapType = False

    if snapType == True:
        dataDict = copy.deepcopy(input.data)
    else:
        dataDict = copy.deepcopy(input)

    if hush is False: print("Flattening data...")
    flatData = {}

    keys = list(dataDict.keys())

    keysAreTuples = np.all(np.asarray([isinstance(kk, tuple) for kk in keys]))

    if keysAreTuples == False:
        raise Exception(f"[@cr_flatten_wrt_time]: ERROR! FATAL! Cannot flatten data without tuple keys in format keys: keys = [(A, #1), (A, #2), ..., (B, #1), (B, #2), ...] or [(A,B, #1), (A,B, #2), ..., (C,D, #1), (C,D, #2), ...] etc.")

    keysLastElements = [kk[-1] for kk in keys]
    if verbose: print(f"[@cr_flatten_wrt_time]: keysLastElements (the numerals...): {keysLastElements}")

    keysRestOfElements = [tuple(kk[:-1]) if len(kk[:-1])>1 else kk[0] for kk in keys]
    if verbose: print(f"[@cr_flatten_wrt_time]: keysRestOfElements (the rest of the original tuple keys...): {keysRestOfElements}")
    
    digitTruthy = np.asarray([str(kk).isdigit() for kk in keysLastElements])
    isNumericSequence = np.all(digitTruthy)
    if isNumericSequence == True:
        if hush is False: print("\n"+f"[@cr_flatten_wrt_time]: Last element of tuple in keys detected as being numeric! Will sort data by these numerals, in ascending order."+"\n")
        keysLastElements = [float(kk) for kk in keysLastElements]
        order = np.argsort(np.asarray(keysLastElements)).tolist()
        keysLastElements = [keysLastElements[ii] for ii in order]
        keysRestOfElements = [keysRestOfElements[ii] for ii in order]
        if verbose: print(f"[@cr_flatten_wrt_time]: Sorted keysLastElements (the numerals...): {keysLastElements}")
        if verbose: print(f"[@cr_flatten_wrt_time]: Sorted keysRestOfElements (the rest of the original tuple keys...): {keysRestOfElements}")
        


    combinedDataKeys = pd.unique(keysRestOfElements)
    if verbose: print(f"[@cr_flatten_wrt_time]: combinedDataKeys (the new unique tuple keys after data is flattened...): {combinedDataKeys}")

    for ii, key in enumerate(combinedDataKeys):
        if hush is False: print(f"{float(ii)/float(len(combinedDataKeys)):3.1%}")
        flattenList = []
        for kk in dataDict.keys():
            if ((len(kk[:-1])>1) & (tuple(kk[:-1])==key)) or ((len(kk[:-1])==1) & (kk[0]==key)):
                flattenList.append(kk)

        innerData = {}
        dataKeys = list(dataDict[flattenList[0]].keys())
        for dd in dataKeys:
            toCombine = []
            for kk in flattenList:
                try:
                    tmp = copy.copy(dataDict[kk][dd])
                except Exception as e:
                    print(str(e))
                    raise Exception(f"[@cr_flatten_wrt_time]: ERROR! FATAL! Data for {kk} {dd} not found! Cannot flatten data where data contained is inconsistent between snapshots/entries!")
                toCombine.append(np.asarray(tmp))

            if stack == True:
                outvals = np.stack((toCombine), axis=-1)
            else:
                outvals = np.concatenate((toCombine), axis=0)
            
            innerData.update({dd: outvals})

        if verbose: 
            for dd in innerData.keys():
                print(f"[@cr_flatten_wrt_time]: combinedDataKey {key} data {dd} has data of shape {innerData[dd].shape}")

        flatData.update({key: innerData})
    
    if hush is False: print("...flattening done!")
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
    exclusions = ["Redshift", "Lookback", "Snap", "Rvir", "Rdisc"],
    weightedStatsBool = True,
):

    if exclusions is None:
        exclusions = []

    ## Empty data checks ## 
    if bool(dataDict) is False:
        warnings.warn("\n"
                +f"[@cr_calculate_statistics]: dataDict is empty! Skipping plots ..."
                +"\n"
        )
        return
    
    print(f"[@cr_calculate_statistics]: Excluded properties (as passed into 'exclusions' kwarg): {exclusions}")

    print("[@cr_calculate_statistics]: Generate bins")
    if xParam in CRPARAMS["logParameters"]:
        xBins = np.logspace(
            start=np.log10(xlimDict[xParam]["xmin"]),
            stop=np.log10(xlimDict[xParam]["xmax"]),
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

    # # paramsInData = list(dataDict.keys())
    # # for param in exclusions:
    # #     if param in paramsInData:
    # #         statsData.update({param: copy.deepcopy(dataDict[param])})

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
    possibleTypesCombos = np.array(possibleTypesCombos,dtype=object)

    possibleValueLengths = []
    possibleValueLengthsSumTot = []
    for jj in range(1,len(types)+1):
        val = np.array(list(combinations(lenTypes,r=jj)))
        possibleValueLengths += val.tolist()
        possibleValueLengthsSumTot += np.sum(val,axis=-1).tolist()

    possibleValueLengths = np.array(possibleValueLengths,dtype=object)
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
                raise Exception(f"[@map_params_to_types]:  Parameter {key} could not be assigned a type!"
                                +"\n"
                                +"This usually means the parameter has an incorrect shape, and does not correspond to any type."
                                +"\n"
                                +"Check logic around parameter creation, or check that you meant to include this parameter into the data included in the call to this function."
                                +"\n"
                                +"e.g. you should only pass data with shapes corresponding to the included types in the passed data dict/snapshot."
                                +"\n"
                                +"Some parameters may be handled through exception via singleValueKeys or exclusions variables. Please check scripts from CR project for examples of usage.")
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
            warnings.warn(f"[map_params_to_types]: Type lengths are degenerate!"
                          +"\n"
                          +f"[map_params_to_types]: {countDict}"
                          +"\n"
                          +f"[map_params_to_types]: We will pad with np.nan by small amounts to try and break this degeneracy"
                    )

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
    WARNING: Do ~NOT~ enter indices (e.g. from a call to np.where() ) for the argument of removalConditionMask!
             Tempting though this may be, it will break the code in unexpected ways! I will try to add a catch for to throw an error when
             this is attempted.

    This function accepts as an input either an Arepo snapshot instance, or a dictionary along with a numpy boolean array of where to remove. It then
    removes from the data ~for ALL Arepo particle types~ whichever entries are True in this array.
    This function (and the function map_params_to_types) works on a combinatorics assumption. We assume that for every Arepo particle type, the number
    of data-points is unique. Ergo, for any given property of our data, we can infer which particle types have that property (e.g. position, age, etc)
    by producing every combination of sums of the number of data-points associated with each particle type and comparing it to the number of data-points
    for that property.
    This function ~does not care which Arepo particle types you have loaded into memory, nor their order~ but it ~does~ modify the data in the order
    of the types loaded in. Thus, much of the work of removing, for example, data of particle type 1 from loaded types [0, 1, 4] is involved with 
    determining the location of type 1 in the current parameter being modified, and adjusting the indices to be deleted accordingly.
    If a property has associated types [0, 1] then we account for the length of type 0 before removing indices. 
    We modify the data of a property associated with types [0, 1] for example, by first removing the type 0 entries flagged for removal, and then
    the type 1 entries flagged for removal. Note: our function must keep track of the changes in shapes of each property and particle type as they
    are editted through the loops in this function.

    On a final note: we have included a (hopefully, rarely needed) degeneracy breaking functionality. It works by adding a small amount of data
    to any particle types that have equal number of data-points, such that the number of data-points of each type becomes unique again. This
    data is mostly in NaN form (with appropriate adjustment for dtypes where NaN isn't possible) and is removed at the end of this function
    """
    import copy
    import pandas as pd

    if removalConditionMask.dtype is not np.dtype("bool"):
        raise TypeError(f"[@remove_selection]: removalConditionMask detected as dtype {removalConditionMask.dtype}!"
                            +"\n"
                            + "removalConditionMask can only be of dtype 'bool'."
                            +"\n"
                            +"Common causes of this error are using:"
                            +"\n"
                            +"removalConditionMask = np.where(condition==False)"
                            +"\n"
                            +"instead of removalConditionMask = condition==False"
                        )

    nRemovals = np.shape(np.where(removalConditionMask==True)[0])[0]
    if nRemovals==0:
        if not hush: print("[@remove_selection]: Number of data points to remove is zero! Skipping...")
        return snap

    snapType = False
    try:
        types = pd.unique(snap.data["type"])
        originalTypes, originalTypeCounts = np.unique(snap.data["type"],return_counts=True)
        snapType = True
        if verbose: print("[@remove_selection]: snapshot type detected!")
    except:
        try:
            types = pd.unique(snap["type"])
        except:
            raise Exception("[@remove_selection]: Unrecognised data format input! Data was neither Arepo snapshot format or Dictionary format!")
        snapType = False
        originalTypes, originalTypeCounts = np.unique(snap["type"],return_counts=True)
        if verbose: print("[@remove_selection]: dictionary type detected!")

    snap, paramToTypeMap, degeneracyBool = map_params_to_types(snap)

    if degeneracyBool is True:
        raise Exception(f"[remove_selection]:  Snapshot type lengths have been detected as degenerate by map_params_to_types() call in remove_selection()."+"\n"+"map_params_to_types() must be called seperately, prior to the evaluation of removalConditionMask in this call to remove_selection()"+"\n"+f"This error came from errorString {errorString} call to remove_selection()!")

    removedTruthy = np.full(types.shape,fill_value=False)

   
    if verbose is True: print("verbose!",errorString)

    # Find possible value length total that matches removalConditionMask
    # shape. From this, infer which parameters are involved in this
    # removalConditionMask. Save this possible type combination
    # as typeCombosArray, and use that to adjust how much the keys
    # in snap.data.keys() need to be offset based on which types
    # used in mask aren't relevant to that snap.data key.

    whereShapeMatch = np.where(paramToTypeMap["pvl_tot"] == removalConditionMask.shape[0])[0]

    if verbose: print("whereShapeMatch", whereShapeMatch)

    typeCombos = paramToTypeMap["ptyc"][whereShapeMatch]

    if verbose: print("typeCombos", typeCombos)

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
                print("whereType", whereType)

            whereToRemove = np.where(removalConditionMask[whereType-typesOffset])[0] + typesOffset


            if verbose: print("typesOffset",typesOffset)
            if verbose: print("whereToRemove",whereToRemove)
            
            if snapType is True:
                itrr = snap.data.items()
            else:
                itrr = snap.items()
            for jj,(key, value) in enumerate(itrr):
                if tp in paramToTypeMap[key]:
                    if value is not None:
                        if verbose: print("jj, key", f"{jj}, {key}")

                        # For the key in snapshot data, retrieve types that
                        # contain that key (i.e. the types that have values
                        # for snap.data[key]). Find types in
                        # removalConditionMask relevant types that aren't
                        # used for this key, and remove the indexing
                        # of this unused data type (offset) from whereToRemove

                        locRemovalOffset= np.array([jj for jj,tt in enumerate(types[:whereTypeInTypes]) if tt not in paramToTypeMap[key]])
                        if verbose: print("locRemovalOffset",locRemovalOffset)
                        if verbose:
                            print("tp",tp)
                            if len(locTypesOffset)>0:
                                print("types[locTypesOffset]",types[locTypesOffset])
                            else:
                                print("No locs type")

                            if len(locRemovalOffset)>0:
                                print("types[locRemovalOffset]",types[locRemovalOffset])
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
                                warnings.warn(f"[remove_selection]: Shape key: {np.shape(value)}"
                                              +"\n"
                                              +f"[remove_selection]: {str(e)}. Could not remove selection from {key} for particles of type {tp}")

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

    try:
        if snapType is True:
            nData = np.shape(snap.data["type"])[0]
        else:
            nData = np.shape(snap["type"])[0]
    except:
        raise Exception(f"[@remove_selection]: "+
                        "\n"+f"Error String: {errorString} returned an empty snapShot!"
                        )

    #Remove None values and double check...
    snap = clean_snap_nones(snap)
    try:
        if snapType is True:
            nData = np.shape(snap.data["type"])[0]        
            currentTypes, currentTypeCounts = np.unique(snap.data["type"],return_counts=True)
        else:
            nData = np.shape(snap["type"])[0]
            currentTypes, currentTypeCounts = np.unique(snap["type"],return_counts=True)

    except:
        raise Exception(f"[@remove_selection]: "+
                        "\n"+f"Error String: {errorString} returned an empty snapShot!"
                        )
    
    noneRemovedTruthy = np.all(~removedTruthy)

    if noneRemovedTruthy is True:
        if verbose: warnings.warn(f"[@remove_selection]: Selection Criteria for error string = '{errorString}', has removed NO entries. Check logic! ")
    else:
        if np.any(~removedTruthy):
            if verbose:
                warnings.warn(f"[@remove_selection]:  Selection criteria for error string = '{errorString}' not applied to particles of type:"
                              +"\n"
                              +f"{types[np.where(removedTruthy==False)[0]]}"
                              +"\n"
                              +f"Original types {originalTypes} with counts {originalTypeCounts}"
                              +"\n"
                              +f"New/current types {currentTypes} with counts {currentTypeCounts}")
        else:
            if verbose: warnings.warn(f"[@remove_selection]: Selection criteria for error string = '{errorString}' was ~successfully~ applied!")

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

def cr_stats_ratios(stats,comparisons,exclusions=[],verbose = False):
    comparisonDict = {}
    tmpstats = {}
    for sKey in stats.keys():
        tmpstats.update({sKey : stats[sKey][sKey]})
    inner = stats_ratios(
        stats = tmpstats,
        comparisons = comparisons,
        exclusions = exclusions,
        verbose = verbose, 
    )

    for compKey,val in inner.items():
        comparisonDict.update({compKey :{compKey : val}})
        
    return comparisonDict

def stats_ratios(stats, comparisons, exclusions=[], verbose = False):
    print(
        f"[@stats_ratios]: Beginning stats comparisons..."
    )
    comparisonDict = {}
    for comp in comparisons:
        numer = comp[0]
        denom = comp[1]
        comp = numer + "/" + denom
        print(
            f"[@stats_ratios]: Comparison: {comp}"
        )
        # comparisonDict[halo].update({compKey : {}})
        for sKey in stats.keys():
            listed = list(sKey)
            if numer in listed:
                denomKey = tuple([xx if xx != numer else denom for xx in listed])
                compKey = tuple([xx if xx != numer else comp for xx in listed])
                comparisonDict.update({compKey : {}})
                for key in stats[sKey].keys():
                    if key not in exclusions:
                        try:
                            val = stats[sKey][key]/stats[denomKey][key]
                        except Exception as e:
                            if verbose: 
                                warnings.warn(str(e)
                                +"\n"
                                +f"[@stats_ratios]: Variable {key} not found! Entering null data...")
                            val = np.full(shape=np.shape(stats[sKey][key]),fill_value=np.nan)
                        comparisonDict[compKey].update({key : copy.deepcopy(val)})
                    else:
                        comparisonDict[compKey].update({key : stats[sKey][key]})


    return comparisonDict


def cr_save_to_excel(
    statsDict,
    CRPARAMSHALO,
    savePathBase = "./",
    filename = "CR-Data.xlsx",
    replacements = [["high","hi"],["standard","std"],["no_CRs","MHD"],["with_CRs","CRs"],["_no_Alfven","-NA"]]
    ):

    replacements = replacements + [["/","__"]]

    selectKey0 = list(CRPARAMSHALO.keys())[0]

    if ("Stars" in selectKey0):
        simSavePath = f"type-{CRPARAMSHALO[selectKey0]['analysisType']}/{CRPARAMSHALO[selectKey0]['halo']}/Stars/"
    elif ("col" in selectKey0):
        simSavePath = f"type-{CRPARAMSHALO[selectKey0]['analysisType']}/{CRPARAMSHALO[selectKey0]['halo']}/Col-Projection-Mapped/"
    else:
        simSavePath = f"type-{CRPARAMSHALO[selectKey0]['analysisType']}/{CRPARAMSHALO[selectKey0]['halo']}/"

    savePath = savePathBase + simSavePath

    tmp = ""
    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass
            
    print(
        f"[@cr_save_to_excel]: Saving as",savePath+filename
    )
    excel = pd.ExcelWriter(path=savePath+filename,mode="w")
    with excel as writer:
        for (selectKey, simDict) in statsDict.items():
            df = pd.DataFrame.from_dict(simDict[selectKey])
            sheet = " ".join(list(selectKey))
            for replacement in replacements:
                sheet = sheet.replace(replacement[0],replacement[1])
            df.to_excel(writer,sheet_name=sheet)
        
    return


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
