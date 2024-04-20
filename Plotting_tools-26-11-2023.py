from select import select
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
import const as c
import OtherConstants as oc
from gadget import *
from gadget_subfind import *
import Tracers_Subroutines as tr
import CR_Subroutines as cr
import h5py
import json
import copy
import math
import os
import itertools 

plt.rcParams.update(matplotlib.rcParamsDefault)

#==========================================================#
##  General versions ...
#==========================================================#
def round_it(x, sig):
    """
        Minor adaptations made to the function taken from here https://www.delftstack.com/howto/python/round-to-significant-digits-python/
        Accessed: 21/04/2023
    """
    if x != 0:
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
    else:
        return 0.0

def check_params_are_in_xlimDict(xlimDict, params):
    if (type(params[0]) == list):
        params = list(itertools.chain.from_iterable(params))

    haslimits = []
    for param in params:
        try:
            values = xlimDict[param]
            try:
                tmp = values['xmin']
                tmp = values['xmax']
                haslimits.append(True)
            except:
                haslimits.append(False)
        except:
            haslimits.append(False)

    haslimits = np.asarray(haslimits)

    if not np.all(haslimits):
        raise Exception(f"[check_params_are_in_xlimDict]: FAILURE! All plotted properties must contain limits in xlimDict to allow for temporal averaging!"
                        +"\n"
                        +f"Properties {np.asarray(params)[np.where(haslimits==False)[0]]} do not have limits in xlimDict!"
                        )

    return

def phase_plot(
    dataDict,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber = None,
    yParams = ["T"],
    xParams = ["rho_rhomean","R"],
    colourBarKeys = ["mass","vol"],
    weightKeys = None,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    DPI=200,
    xsize=8.0,
    ysize=8.0,
    colourmapMain="plasma",
    Nbins=250,
    saveFigureData = False,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    allowPlotsWithoutxlimits = False,
    subfigures = False,
    inplace = False,
    replotFromData = False,
    verbose = False,
    hush = False,
):
    
    ## Empty data checks ## 
    if (bool(dataDict) is False)| (bool(xParams) is False)| (bool(yParams) is False)| (bool(colourBarKeys) is False):
        print("\n"
              +f"[@phase_plot]: WARNING! No data/no plots requested Skipping plot call and exiting ..."
              +"\n"
        )
        return

    if subfigures:
        tmpyParams, tmpxParams, tmpcolourBarKeys = np.asarray(yParams,dtype=object), np.asarray(xParams,dtype=object), np.asarray(colourBarKeys,dtype=object)
        assert np.shape(tmpxParams) == np.shape(tmpyParams), f"[@phase_plot]: FAILURE! xParams, yParams, colourBarKeys must all have the same shape for subFigures == True!"
        assert np.shape(tmpxParams) == np.shape(tmpcolourBarKeys), f"[@phase_plot]: FAILURE! xParams, yParams, colourBarKeys must all have the same shape for subFigures == True!"
        
        ncols = max([len(zz) for zz in tmpxParams])
        nrows = max([len(zz) for zz in tmpyParams])
        ncbars = max([len(zz) for zz in tmpcolourBarKeys])

        hasPlotMask = []
        for jj in range(0, nrows):
            row = []
            for ii in range(0,ncols):
                try:
                    tmp = xParams[ii][jj]
                    tmp = yParams[ii][jj]
                    tmp = colourBarKeys[ii][jj]
                    if tmp is not None:
                        hasPlot = True
                    else:
                        hasPlot = False
                except:
                    hasPlot = False
                    xParams[ii].insert(jj,None)
                    yParams[ii].insert(jj,None)
                    colourBarKeys[ii].insert(jj,None)

                row.append(copy.deepcopy(hasPlot))
            hasPlotMask.append(copy.deepcopy(row))
            
        hasPlotMask = np.asarray(hasPlotMask)
        figshape = np.shape(hasPlotMask)


    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    if allowPlotsWithoutxlimits == False:
        check_params_are_in_xlimDict(xlimDict, yParams)
        check_params_are_in_xlimDict(xlimDict, xParams)

    zlimDict = copy.deepcopy(xlimDict)

    if subfigures:
        if (type(xParams[0]) == list):
            xParams = list(itertools.chain.from_iterable(xParams))
        if (type(yParams[0]) == list):
            yParams = list(itertools.chain.from_iterable(yParams))
        if (type(colourBarKeys[0]) == list):
            colourBarKeys = list(itertools.chain.from_iterable(colourBarKeys))

    if inplace is True:
        simDict = dataDict
        print("\n"
              +f"[@phase_plot]: WARNING! inplace flag set to True. This will modify the data dictionary provided to this function call."
              +"\n"
              +f"Call details: yParams:{yParams}, xParams:{xParams}, colourBarKeys:{colourBarKeys}"
              +"\n"
              )
    else:
        simDict = copy.deepcopy(dataDict)

    savePath = savePathBase + "/Plots/Phases/"
    savePathFigureData = savePathBaseFigureData + "/Plots/Phases/"

    tmp = ""
    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    tmp = ""
    for savePathChunk in savePathFigureData.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if weightKeys == None:
        if hush is False:
            print("\n"
                +f"[@phase_plot]: WARNING! No weights dictionary provided! Will default to weighting by mass for all data provided to this call ..."
                +"\n"
            )        
        weightKeys = {kk: "mass" for kk in colourBarKeys}

    # ------------------------------------------------------------------------------#
    #               PLOTTING
    #
    # ------------------------------------------------------------------------------#

    subplotCount = -1
    if subfigures:
        fig, ax = plt.subplots(
            nrows=figshape[1],
            ncols=figshape[0],
            figsize=(xsize, ysize),
            dpi=DPI,
        )

    for ii, xParam in enumerate(xParams):
        subplotCount+=1
        print("\n"+"-----")
        print(f"Starting xParam {xParam}")
        for jj, yParam in enumerate(yParams):
            if subfigures:
                print(jj)
                yParam = yParams[ii]
                if (jj>0):
                    print("continue")
                    # Effectively "zip"'s the x, y, and colourbar params so combinations not desired aren't attempted
                    continue
            print("\n"+f"Starting yParam {yParam}")
            for kk, colourBarKey in enumerate(colourBarKeys):

                if subfigures == False:
                    fig, ax = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(xsize, ysize),
                        dpi=DPI,
                    )
                    currentAx = ax
                else:
                    axindex = np.unravel_index(subplotCount, shape=figshape)
                    currentAx = ax[axindex]

                plt.tick_params(axis="both", direction="in")
                if subfigures:
                    print(jj,kk)
                    colourBarKey = colourBarKeys[ii]
                    if ((jj>0)|(kk>0)):
                        print("continue")
                        # Effectively "zip"'s the x, y, and colourbar params so combinations not desired aren't attempted
                        continue
                    # axindex = np.unravel_index(subplotCount, shape=figshape)
                    # xParam = xParams[axindex[0]][axindex[1]]
                    # yParam = yParams[axindex[0]][axindex[1]]
                    # colourBarKey = colourBarKeys[axindex[0]][axindex[1]]

                    if hasPlotMask[axindex] == False:
                        ax[axindex].axis('off')
                        continue


            

                skipBool = False
                print("\n"+f"Starting colourBarKey {colourBarKey}")

                if colourBarKey == xParam:
                    print("\n" + f"colourBarKey same as xParam! Skipping...")
                    skipBool = True
                    continue

                if colourBarKey == yParam:
                    print("\n" + f"colourBarKey same as yParam! Skipping...")
                    skipBool = True
                    continue

                if xParam == yParam:
                    print("\n" + f"yParam same as xParam! Skipping...")
                    skipBool = True
                    continue

                if np.all(np.isin(np.array(["tcool","theat"]),np.array([xParam,yParam,colourBarKey]))) == True:
                    print("\n" + f"tcool and theat aren't compatible! Skipping...")
                    skipBool = True
                    continue

                try:
                    zmin = zlimDict[colourBarKey]["xmin"]
                    zmax = zlimDict[colourBarKey]["xmax"]
                    zlimBool = True
                except:
                    zlimBool = False

                try:
                    tmp = zlimDict[xParam]["xmin"]
                    tmp = zlimDict[xParam]["xmax"]
                    xlimBool = True
                except:
                    xlimBool = False

                try:
                    tmp = zlimDict[yParam]["xmin"]
                    tmp = zlimDict[yParam]["xmax"]
                    ylimBool = True
                except:
                    ylimBool = False

                if replotFromData is False:
                    try:
                        if subfigures:
                            tmpdataDict, paramToTypeMap, _ = cr.map_params_to_types(copy.deepcopy(simDict))
                        else:
                            tmpdataDict, paramToTypeMap, _ = cr.map_params_to_types(simDict)

                        typesUsedData = paramToTypeMap[xParam]

                        whereNotType = np.isin(simDict["type"],typesUsedData)==False
                        if verbose:
                            print(f"[@phase_plot]: typesUsedData {typesUsedData}")
                            print(f"[@phase_plot]: pd.unique(simDict['type']) {pd.unique(simDict['type'])}")
                        tmpdataDict = cr.remove_selection(
                            tmpdataDict,
                            removalConditionMask = whereNotType,
                            errorString = "Remove types not applicable to xParam",
                            hush = True,
                            verbose = verbose,
                        )

                        if xParam in logParameters:
                            xx = np.log10(tmpdataDict[xParam])
                        else:
                            xx = tmpdataDict[xParam]
                    except Exception as e:
                        # # raise Exception(e)
                        print(f"{str(e)}")
                        print("\n"+f"xParam of {xParam} data not found! Skipping...")
                        if replotFromData is False: skipBool = True
                        continue

                    try:
                        tmpdataDict, paramToTypeMap, _ = cr.map_params_to_types(tmpdataDict)
                        typesUsedData = paramToTypeMap[yParam]

                        whereNotType = np.isin(tmpdataDict["type"],typesUsedData)==False
                        if verbose:
                            print(f"[@phase_plot]: typesUsedData {typesUsedData}")
                            print(f"[@phase_plot]: pd.unique(tmpdataDict['type']) {pd.unique(tmpdataDict['type'])}")
                        tmpdataDict = cr.remove_selection(
                            tmpdataDict,
                            removalConditionMask = whereNotType,
                            errorString = "Remove types not applicable to yParam",
                            hush = True,
                            verbose = verbose,
                        )
                        if yParam in logParameters:
                            yy = np.log10(tmpdataDict[yParam])
                        else:
                            yy = tmpdataDict[yParam]
                    except Exception as e:
                        # # raise Exception(e)
                        print(f"{str(e)}")
                        print("\n"+f"yParam of {yParam} data not found! Skipping...")
                        if replotFromData is False: skipBool = True
                        continue
                    
                    try:
                        xmin, xmax =(
                            zlimDict[xParam]["xmin"], zlimDict[xParam]["xmax"]
                        )
                    except:
                        xmin, xmax, = ( np.nanmin(xx), np.nanmax(xx))

                    try:
                        ymin, ymax =(
                            zlimDict[yParam]["xmin"], zlimDict[yParam]["xmax"]
                        )
                    except:
                        ymin, ymax, = ( np.nanmin(yy), np.nanmax(yy))

                    xdataCells = xx[np.where((xx>=xmin)&(xx<=xmax)&(yy>=ymin)&(yy<=ymax)&(np.isfinite(xx)==True)&(np.isfinite(yy)==True)) [0]]
                    ydataCells = yy[np.where((xx>=xmin)&(xx<=xmax)&(yy>=ymin)&(yy<=ymax)&(np.isfinite(xx)==True)&(np.isfinite(yy)==True))[0]]

                    try:
                        weightKey = weightKeys[colourBarKey]
                    except:
                        if hush is False:
                            print(f"[@phase_plot]: WARNING! weightKey for {colourBarKey} not found! Will default to mass weighting")
                        weightKey = "mass"

                    weightCells = ( tmpdataDict[weightKey][
                        np.where((xx>=xmin)&(xx<=xmax)
                        &(yy>=ymin)&(yy<=ymax)&
                        (np.isfinite(xx)==True)&(np.isfinite(yy)==True))
                        [0]]
                    )

                    if colourBarKey != "count":
                        try:
                            weightDataCells = (
                                tmpdataDict[colourBarKey][
                                np.where((xx>=xmin)&(xx<=xmax)
                                &(yy>=ymin)&(yy<=ymax)&(np.isfinite(xx)==True)&(np.isfinite(yy)==True))
                                [0]] * weightCells
                            )
                            skipBool = False
                        except Exception as e:
                            print(f"{str(e)}")
                            print(
                                f"Variable {colourBarKey} not found. Skipping plot..."
                            )
                            if replotFromData is False: skipBool = True
                            continue

                    if (colourBarKey == weightKey):
                        finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                            xdataCells, ydataCells, bins=Nbins, weights=weightCells
                        )
                    elif colourBarKey == "count":
                        finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                            xdataCells, ydataCells, bins=Nbins, weights=None
                        )
                    else:
                        mhistCells, _, _ = np.histogram2d(
                            xdataCells, ydataCells, bins=Nbins, weights=weightCells
                        )
                        histCells, xedgeCells, yedgeCells = np.histogram2d(
                            xdataCells, ydataCells, bins=Nbins, weights=weightDataCells
                        )

                        finalHistCells = histCells / mhistCells

                    finalHistCells[finalHistCells == 0.0] = np.nan
                    try:
                        if colourBarKey in logParameters:
                            finalHistCells = np.log10(finalHistCells)
                    except Exception as e:
                        print(f"{str(e)}")
                        print(f"Variable {colourBarKey} not found. Skipping plot...")
                        if replotFromData is False: skipBool = True
                        continue
                    finalHistCells = finalHistCells.T

                    xcells, ycells = np.meshgrid(xedgeCells, yedgeCells)

                else:
                    # # out = {"data":{"x" : xcells, "y" : ycells, "hist": finalHistCells}}
                    selectKey = (xParam, yParam, colourBarKey)
                    try:
                        xcells = simDict[selectKey]["x"]
                        ycells = simDict[selectKey]["y"]
                        finalHistCells = simDict[selectKey]["hist"]
                    except Exception as e:
                        # raise Exception(e)
                        print(f"{str(e)}")
                        print(
                            f"selectKey {selectKey} not found! Skipping plot..."
                        )
                        if replotFromData is False: skipBool = True
                        continue                    

                if zlimBool is True:
                    img1 = currentAx.pcolormesh(
                        xcells,
                        ycells,
                        finalHistCells,
                        cmap=colourmapMain,
                        vmin=zmin,
                        vmax=zmax,
                        rasterized=True,
                    )
                else:
                    img1 = currentAx.pcolormesh(
                        xcells,
                        ycells,
                        finalHistCells,
                        cmap=colourmapMain,
                        rasterized=True,
                    )
                #
                # img1 = currentAx.imshow(finalHistCells,cmap=colourmapMain,vmin=xmin,vmax=xmax \
                # ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

                currentAx.set_xlabel(
                    ylabel[xParam],
                    fontsize=fontsize,
                )
                currentAx.set_ylabel(
                    ylabel[yParam],
                    fontsize=fontsize
                )

                if ylimBool is True:
                    currentAx.set_ylim(
                        zlimDict[yParam]["xmin"], zlimDict[yParam]["xmax"])
                else:
                    currentAx.set_ylim(np.nanmin(yedgeCells),np.nanmax(yedgeCells))

                if xlimBool is True:
                    if xParam == "vol":
                        currentAx.set_xlim(zlimDict[xParam]["xmax"],zlimDict[xParam]["xmin"])
                    else:
                        currentAx.set_xlim(zlimDict[xParam]["xmin"],zlimDict[xParam]["xmax"])
                else:
                    if xParam == "vol":
                        currentAx.set_xlim(np.nanmax(xcells),np.nanmin(xcells))
                    else:
                        currentAx.set_xlim(np.nanmin(xcells),np.nanmax(xcells))
                    # zlimDict["rho_rhomean"]["xmin"], zlimDict["rho_rhomean"]["xmax"])
                currentAx.tick_params(
                    axis="both", which="both", labelsize=fontsize)

                currentAx.set_aspect("auto")

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                #   Figure: Finishing up
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                if skipBool == True:
                    try:
                        tmp = finalHistCells
                    except Exception as e:
                        print(f"{str(e)}")
                        print(
                            f"Variable {colourBarKey} not found. Skipping plot..."
                        )
                        continue
                    else:
                        pass


                if snapNumber is not None:
                    if str(snapNumber).isdigit() is True:
                        SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                    else:
                        SaveSnapNumber = "_" + str(snapNumber)
                else:
                    SaveSnapNumber = ""

                if subfigures:

                    cb = plt.colorbar(img1, ax=currentAx, orientation="vertical", pad=0.05)
                    
                    cb.set_label(
                        label=ylabel[colourBarKey], size=fontsize
                    )
                    cax1 = cb.ax
                    cax1.yaxis.set_ticks_position("right")
                    cax1.yaxis.set_label_position("right")
                    cax1.yaxis.label.set_color("black")
                    cax1.tick_params(axis="y", colors="black", labelsize=fontsize)
                else:
                    #left, bottom, width, height
                    # x0,    y0,  delta x, delta y
                    cax1 = fig.add_axes([0.925, 0.10, 0.05, 0.80])

                    fig.colorbar(img1, cax=cax1, ax=currentAx, orientation="vertical", pad=0.05).set_label(
                        label=ylabel[colourBarKey], size=fontsize
                    )
                    cax1.yaxis.set_ticks_position("left")
                    cax1.yaxis.set_label_position("left")
                    cax1.yaxis.label.set_color("black")
                    cax1.tick_params(axis="y", colors="black", labelsize=fontsize)

                    if titleBool is True:
                        fig.suptitle(
                            f"{yParam} vs. {xParam} Diagram, weighted by {colourBarKey}",
                            fontsize=fontsizeTitle,
                        )

                    if titleBool is True:
                        plt.subplots_adjust(top=0.875, right=0.8, hspace=0.3, wspace=0.3)
                    else:
                        plt.subplots_adjust(right=0.8, hspace=0.3, wspace=0.3)

                    opslaan = (
                        savePath
                        + f"Phases-Plot_{yParam}-vs-{xParam}_weighted-by-{colourBarKey}{SaveSnapNumber}"
                    )
                    plt.savefig(opslaan+".pdf", dpi=DPI, transparent=False)
                    print(opslaan)
                    matplotlib.rc_file_defaults()
                    plt.close("all")

                    if saveFigureData is True:
                        opslaanData = savePathFigureData + f"Phases-Plot_{yParam}-vs-{xParam}_weighted-by-{colourBarKey}{SaveSnapNumber}"
                        out = {"data":{"x" : xcells, "y" : ycells, "hist": finalHistCells}}
                        tr.hdf5_save(opslaanData+"_data.h5",out)
    
    if subfigures:
        opslaan = (
            savePath
            + f"Phases-Plot{SaveSnapNumber}"
        )
        plt.gcf()
        plt.savefig(opslaan+".pdf", dpi=DPI, transparent=False)
        print(opslaan)
        matplotlib.rc_file_defaults()
        plt.close("all")
    return

def pdf_versus_plot(
    inputDict,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber = None,
    weightKeys = None,
    xParams = ["T"],
    titleBool=False,
    legendBool=True,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize=13,
    fontsizeTitle=14,
    linewidth=1.0,
    Nbins=250,
    ageWindow=None,
    cumulative = False,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    allSavePathsSuffix = "",
    saveFigureData = False,
    SFR = False,
    forceLogPDF = False,
    normalise = False,
    allowPlotsWithoutxlimits = False,
    inplace = False,
    replotFromData = False,
    combineMultipleOntoAxis = False,
    selectKeysList = None,
    verbose = False,
    styleDict = None,
    hush = False,
):

    ## Empty data checks ## 
    if (bool(inputDict) is False)| (bool(xParams) is False):
        print("\n"
              +f"[@pdf_versus_plot]: WARNING! No data/no plots requested Skipping plot call and exiting ..."
              +"\n"
        )
        return
    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    if weightKeys == None:
        if hush is False:
            print("\n"
                +f"[@pdf_versus_plot]: WARNING! No weights dictionary provided! Will default to weighting by mass for all data provided to this call ..."
                +"\n"
            )        
        weightKeys = {kk: "mass" for kk in xParams}

    savePath = savePathBase + allSavePathsSuffix + "Plots/PDFs/"
    savePathFigureData = savePathBaseFigureData + allSavePathsSuffix + "Plots/PDFs/"
    
    tmp = ""
    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    tmp = ""
    for savePathChunk in savePathFigureData.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if SFR is True:
        weightKeys = {"age" : "gima"}
        xParams = ["age"]
    elif allowPlotsWithoutxlimits == False:
        # # check_params_are_in_xlimDict(xlimDict, weightKeys) <-- Not needed, as y-axis not used when combining data
        check_params_are_in_xlimDict(xlimDict, xParams)
        

    if combineMultipleOntoAxis is False:
        selectKeysList = None
        if inplace is True:
            dataDict = inputDict
            print("\n"
                +f"[@pdf_versus_plot]: WARNING! inplace flag set to True. This will modify the data dictionary provided to this function call."
                +"\n"
                +f"Call details: xParams:{xParams}, weightKeys:{weightKeys}"
                +"\n"
                )
        else:
            dataDict = copy.deepcopy(inputDict)

        dataSources = {"": dataDict}
    else:

        if selectKeysList is None: 
            if hush is False:
                print(f"[@pdf_versus_plot]: WARNING! combineMultipleOntoAxis type plot called with no selectKeysList provided."
                    +"\n"
                    +f"Call details: xParams:{xParams}, weightKeys:{weightKeys}"
                    +"\n"
                    +"Will default to plotting all data provided to this call.")
                
            selectKeysList = list(inputDict.keys())           
        if inplace is True:
            print("\n"
                +f"[@pdf_versus_plot]: WARNING! inplace flag set to True. This will modify the data dictionary provided to this function call."
                +"\n"
                +f"Call details: xParams:{xParams}, weightKeys:{weightKeys}"
                +"\n"
                )

          
        dataSources = {}
        for selectKey, dataDict in inputDict.items():
            if selectKey in selectKeysList:
                if inplace is True:
                    dataSources.update({selectKey : dataDict})
                else:
                    dataSources.update({selectKey : copy.deepcopy(dataDict)})
            


        ## Empty data checks ## 
        if bool(dataSources) is False:
            print("\n"
                +f"[@pdf_versus_plot]: WARNING! dataSources dict is empty! Skipping plot call and exiting ..."
                +"\n"
            )
            
            return

    for analysisParam in xParams:
        print("")
        print(f"Starting {analysisParam} plots!")
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )
        
        plt.tick_params(axis="both", direction="in")

        skipBool = False

        try:
            weightKey = weightKeys[analysisParam]
        except:
            if hush is False:
                print(f"[@pdf_versus_plot]: WARNING! weightKey for {analysisParam} not found! Will default to mass weighting")
            weightKey = "mass"

        SFRBool = False
        if (weightKey == "gima")&(analysisParam=="age"):
            SFRBool = True


        yminList = []
        ymaxList = []
        for jj, (selectKey, dataDict) in enumerate(dataSources.items()):
        
            if bool(dataDict) is False:
                skipBool = True
                print("\n"
                    +f"[@pdf_versus_plot]: WARNING! dataDict is empty (plot #{int(jj)})! Skipping plots ..."
                    +"\n"
                )
                continue


            if replotFromData is False:
                try:
                    tmpdataDict, paramToTypeMap, _ = cr.map_params_to_types(dataDict)
                    try:
                        typesUsedData = paramToTypeMap[analysisParam]
                    except Exception as e:
                        # raise Exception(e)
                        print(f"{str(e)}")
                        print(
                            f"Variable {analysisParam} not found (plot #{int(jj)})! Skipping plot..."
                        )
                        skipBool = True
                        continue
                    
                    if verbose:
                        print(f"[@pdf_versus_plot]: typesUsedData {typesUsedData}")
                        print(f"[@pdf_versus_plot]: pd.unique(dataDict['type']) {pd.unique(dataDict['type'])}")
                    whereNotType = np.isin(dataDict["type"],typesUsedData)==False

                    tmpdataDict = cr.remove_selection(
                        tmpdataDict,
                        removalConditionMask = whereNotType,
                        errorString = "Remove types not applicable to analysisParam",
                        hush = True,
                        verbose = verbose,
                    )
                    if ((weightKey == "count")|(weightKey == None)):
                        pass
                    else:
                        tmpdataDict, paramToTypeMap, _ = cr.map_params_to_types(tmpdataDict)
                        typesUsedWeights = paramToTypeMap[weightKey]
                        
                        if verbose:
                            print(f"[@pdf_versus_plot]: typesUsedWeights {typesUsedWeights}")
                            print(f"[@pdf_versus_plot]: pd.unique(tmpdataDict['type']) {pd.unique(tmpdataDict['type'])}")
                        whereNotTypeWeights = np.isin(tmpdataDict["type"],typesUsedWeights)==False

                        tmpdataDict = cr.remove_selection(
                            tmpdataDict,
                            removalConditionMask = whereNotTypeWeights,
                            errorString = "Remove types not applicable to weightKey",
                            hush = True,
                            verbose = verbose,
                        )
                        weightsData = tmpdataDict[weightKey]

                    plotData = tmpdataDict[analysisParam]
                    skipBool = False
                except Exception as e:
                    # raise Exception(e)
                    print(f"{str(e)}")
                    print(
                        f"Variable {analysisParam} data subset selection failure (plot #{int(jj)})! Skipping plot..."
                    )
                    skipBool = True
                    continue

                if analysisParam in logParameters:
                    tmpPlot = np.log10(plotData).copy()
                else:
                    tmpPlot = plotData.copy()

                if ((weightKey != "count")&(weightKey != None)): tmpWeights = weightsData.copy()

                whereAgeBelowLimit = np.full(shape=np.shape(tmpPlot),fill_value=True)
                if ageWindow is not None:
                    if SFRBool is True:
                        print("Minimum age detected = ", np.nanmin(tmpPlot), "Gyr")
                        # minAge = np.nanmin(tmpPlot) + ((np.nanmax(tmpPlot) - np.nanmin(tmpPlot))*ageWindow)
                        maxAge = np.nanmin(tmpPlot)+ageWindow
                        print("Maximum age for plotting = ", maxAge, "Gyr")

                        whereAgeBelowLimit = tmpPlot<=maxAge
                        print("Number of data points meeting age = ",np.shape(np.where(whereAgeBelowLimit==True)[0])[0])
                    else:
                        print("[@pdf_versus_plot]: ageWindow not None, but SFR plot not detected. ageWindow will be ignored...")

                try:
                    xmin, xmax =(
                        xlimDict[analysisParam]["xmin"],
                        xlimDict[analysisParam]["xmax"]
                    )
                except:
                    xmin, xmax, = ( np.nanmin(tmpPlot[np.where(whereAgeBelowLimit==True)[0]]),
                        np.nanmax(tmpPlot[np.where(whereAgeBelowLimit==True)[0]]))

                if ((weightKey != "count")&(weightKey != None)):
                    try:
                        whereData = np.where((np.isfinite(tmpPlot)==True)
                        & (np.isfinite(tmpWeights)==True)
                        & (tmpPlot>=xmin)
                        & (tmpPlot<=xmax)
                        & (whereAgeBelowLimit == True)
                        )[0]
                    except:
                        whereData = np.where((np.isfinite(tmpPlot)==True)
                        & (np.isfinite(tmpWeights)==True)
                        & (whereAgeBelowLimit == True)
                        )[0]

                    plotData = tmpPlot[whereData]
                    weightsData = tmpWeights[whereData]
                else:
                    plotData = tmpPlot
                    weightsData = np.asarray([0.0])


                try:
                    tmpxmin = np.nanmin(plotData)
                    tmpxmax = np.nanmax(plotData)
                    skipBool = False
                except Exception as e:
                    # raise Exception(e)
                    print(f"{str(e)}")
                    print(
                        f"Variable {analysisParam} data all NaN (plot #{int(jj)})! Skipping plot...")
                    skipBool = True
                    continue

                if (
                    (np.isfinite(tmpxmin) == False)
                    or (np.isfinite(tmpxmax) == False)
                    or (np.isfinite(np.nanmin(weightsData)) == False)
                    or (np.isfinite(np.nanmin(weightsData)) == False)
                ):
                    # print()
                    print(f"Data All Inf/NaN! Skipping entry (plot #{int(jj)})!")
                    skipBool = True
                    continue
                
                xBins = np.linspace(
                    start=xmin, stop=xmax, num=Nbins)

                if ((weightKey == "count")|(weightKey == None)):
                    hist, bin_edges = np.histogram(
                        plotData,
                        bins=xBins,
                        density = normalise
                    )
                else:
                    hist, bin_edges = np.histogram(
                        plotData,
                        bins=xBins,
                        weights=weightsData,
                        density = normalise
                    )

                # Because SFR should increase over time, and thus as age decreases
                if (SFRBool is True):
                    hist = np.flip(hist)
                    delta = np.mean(np.diff(xBins))
                    hist /= (delta*1e9) # convert to SFR per yr

                if cumulative is True:
                    hist = np.cumsum(hist)
                    if normalise is True:
                        hist /= np.nanmax(hist)


                #Again, SFR needs to increase with decreasing age
                if (SFRBool is True):
                    xBins = np.flip(xBins)
                    bin_edges = np.flip(bin_edges)

                # If log10 desired: set zeros to nan for convenience of
                # not having to drop inf's later, and mask them from plots
                if ((weightKey in logParameters)&(forceLogPDF is True)):
                    hist[hist == 0.0] = np.nan

                if (weightKey == "mass"):
                    weightsSumTotal = np.sum(weightsData)

                    label = f"Sum total of {weightKey} = {weightsSumTotal:.2e}"
                else:
                    label= ""

                colour = "blue"
                linestyle = "solid"
                xFromBins = np.array(
                    [
                        (x1 + x2) / 2.0
                        for (x1, x2) in zip(bin_edges[:-1], bin_edges[1:])
                    ]
                )
            else:
                # out = {"data":{"x" : xFromBins, "y" : hist}}
                if combineMultipleOntoAxis is False:
                    selectKey = (analysisParam, weightKey)
                    try:
                        hist = dataDict[selectKey]["y"]
                        xFromBins = dataDict[selectKey]["x"]
                    except Exception as e:
                        # raise Exception(e)
                        print(f"{str(e)}")
                        print(
                            f"Variable {analysisParam} not found (plot #{int(jj)})! Skipping plot..."
                        )
                        skipBool = True
                        continue
                    label = "" # edit this to reflect data source if possible...
                    colour = "blue"
                    linestyle = "solid"
                else:
                    if selectKeysList is not None:
                        analysisSelectKey = selectKeysList[jj]
                        if ("Stars" in analysisSelectKey) | ("col" in analysisSelectKey) :
                            analysisSelectKeyShort = tuple([xx for xx in analysisSelectKey if (xx != "Stars") & (xx != "col")])
                        else:
                            analysisSelectKeyShort = analysisSelectKey
                        label = " ".join(list(analysisSelectKeyShort))
                        if styleDict is not None:
                            colour=styleDict[analysisSelectKeyShort]["colour"]
                            linestyle=styleDict[analysisSelectKeyShort]["linestyle"]
                        else:
                            colour = None #"blue"
                            linestyle = "solid"
                    else:
                        label = "" # edit this to reflect data source if possible...
                        colour = "blue"
                        linestyle = "solid"

                    selectKey = (analysisParam, weightKey)

                    try:
                        hist = dataDict[selectKey]["y"]
                        xFromBins = dataDict[selectKey]["x"]
                    except Exception as e:
                        # raise Exception(e)
                        print(f"{str(e)}")
                        print(
                            f"Variable {analysisParam} not found (plot #{int(jj)})! Skipping plot..."
                        )
                        if replotFromData is False: skipBool = True
                        continue
                
                if ((normalise is True)&(np.nanmax(hist)>=1.10)):
                    if cumulative is False: 
                        dx = np.diff(xFromBins,axis=0)
                        dx = np.concatenate((np.asarray([dx[0]]),dx),axis=0)
                        hist = hist / (dx*np.sum(hist)) #/ np.nanmax(hist)
                        if verbose: print(f"[@pdf_versus_plot]: Normalised sum total: {np.sum(hist*dx):.2f}. This should equal 1.")
                    elif ((np.all(np.diff(hist,axis=0)>=0.0)==False)|(np.all(np.diff(hist,axis=0)<=0.0)==False)):
                            
                        hist = np.cumsum(hist)
                        hist /= np.nanmax(hist) 

                        if verbose: print(f"[@pdf_versus_plot]: Adapting to normalised cumulative version!")

                elif (cumulative is True)&((np.all(np.diff(hist,axis=0)>=0.0)==False)|(np.all(np.diff(hist,axis=0)<=0.0)==False)):

                    hist = np.cumsum(hist) 

                    if verbose: print(f"[@pdf_versus_plot]: Adapting to cumulative version!")
                
            if np.all(np.isfinite(hist)==False) == True:
                print(f"Hist All Inf/NaN! Skipping entry (plot #{int(jj)})!")
                continue

            try:
                ymin = np.nanmin(hist[np.isfinite(hist)])
                ymax = np.nanmax(hist[np.isfinite(hist)])
                yminList.append(ymin)
                ymaxList.append(ymax)
                skipBool = False
            except Exception as e:
                # raise Exception(e)
                print(f"{str(e)}")
                print(
                    f"Variable {analysisParam} histogram all Inf/NaN (plot #{int(jj)})! Skipping plot...")
                skipBool = True
                continue



            ax.plot(
                xFromBins,
                hist,
                color=colour,
                linestyle=linestyle,
                linewidth = linewidth,
                label = label,
            )
            
            # # # if (analysisParam in logParameters):
            # # #     ax.set_xscale("log")
            if ((weightKey in logParameters)&(forceLogPDF == True)):
                ax.set_yscale("log")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(
            axis="both", which="both", labelsize=fontsize
        )

        ylabel_prefix = ""
        if cumulative is True:
            ylabel_prefix = "Cumulative "
            if normalise is True:
                ylabel_prefix = "Normalised " + ylabel_prefix

        if (normalise is True)&(cumulative is False):
            ax.set_ylabel("PDF", fontsize=fontsize)
            titleKeyword = "PDF"
        elif (normalise is True)&(cumulative is True):
            ax.set_ylabel("CDF", fontsize=fontsize)
            titleKeyword = "CDF"
        else:
            titleKeyword = "Histogram"
            if (forceLogPDF is False):
                adaptedylabel = ylabel_prefix + (copy.deepcopy(ylabel[weightKey])).replace(r"$\mathrm{Log_{10}}$ ", "")
            else:
                adaptedylabel = ylabel_prefix + ylabel[weightKey]
            ax.set_ylabel(adaptedylabel, fontsize=fontsize)

        if titleBool is True:
            fig.suptitle(
                    f"{titleKeyword} of {analysisParam}",
                fontsize=fontsizeTitle
            )

        # Only give 1 x-axis a label, as they sharex

        ax.set_xlabel(ylabel[analysisParam], fontsize=fontsize)

        if (skipBool == True):
            print(
                f"Variable {analysisParam} plot failed (reason should have been printed to stdout above). Skipping plot...")
            continue

        if replotFromData is False:

            #try:
            #    finalxmin = max(
            #        np.nanmin(xmin), xlimDict[analysisParam]["xmin"]
            #    )
            #    finalxmax = min(
            #        np.nanmax(xmax), xlimDict[analysisParam]["xmax"]
            #    )
            #except:
            # # # if analysisParam in logParameters:
            # # #     finalxmin = 10**(xmin)
            # # #     finalxmax = 10**(xmax)
            # # # else:
            finalxmin = xmin
            finalxmax = xmax


            if (
                (np.isinf(finalxmax) == True)
                or (np.isinf(finalxmin) == True)
                or (np.isnan(finalxmax) == True)
                or (np.isnan(finalxmin) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue

            if (forceLogPDF is False)&(SFRBool is False):
                try:
                    finalymin = 0.0
                    finalymax = np.nanmax(np.asarray(ymaxList))
                except:
                    print("Data All Inf/NaN! Skipping entry!")
                    continue
            else:
                try:
                    finalymin = np.nanmin(np.asarray(yminList))
                    finalymax = np.nanmax(np.asarray(ymaxList))
                except:
                    print("Data All Inf/NaN! Skipping entry!")
                    continue
            if (SFR is True)|(SFRBool is True):
                custom_xlim = (finalxmax, finalxmin)
            else:
                custom_xlim = (finalxmin, finalxmax)

            custom_ylim = (finalymin, finalymax)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            if ((label != "")&(legendBool == True)): ax.legend(loc="best", fontsize=fontsize)

        else:
            finalxmin = np.nanmin(xFromBins)
            finalxmax = np.nanmax(xFromBins)


            # # if analysisParam in logParameters:
            # #     finalxmin = 10**(finalxmin)
            # #     finalxmax = 10**(finalxmax)
            # # else:
            # #     finalxmin = xmin
            # #     finalxmax = xmax


            if (
                (np.isinf(finalxmax) == True)
                or (np.isinf(finalxmin) == True)
                or (np.isnan(finalxmax) == True)
                or (np.isnan(finalxmin) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue

            if (forceLogPDF is False)&(SFRBool is False):
                try:
                    finalymin = 0.0
                    finalymax = np.nanmax(np.asarray(ymaxList))
                except:
                    print("Data All Inf/NaN! Skipping entry!")
                    continue
            else:
                try:
                    finalymin = np.nanmin(np.asarray(yminList))
                    finalymax = np.nanmax(np.asarray(ymaxList))
                except:
                    print("Data All Inf/NaN! Skipping entry!")
                    continue

            if (SFR is True)|(SFRBool is True):
                custom_xlim = (finalxmax, finalxmin)
            else:
                custom_xlim = (finalxmin, finalxmax)
                
            custom_ylim = (finalymin, finalymax)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            if ((label != "")&(legendBool == True)): ax.legend(loc="best", fontsize=fontsize)

        # plt.tight_layout()
        if titleBool is True:
            plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
        else:
            plt.subplots_adjust(hspace=0.1, left=0.15)


        if normalise is True:
            tmp2 = savePath +"Normalised-"
            tmp2FigureData = savePathFigureData +"Normalised-"
        else:
            tmp2 = savePath
            tmp2FigureData = savePathFigureData

        if cumulative is True:
            tmp2 = tmp2 +"Cumulative-"
            tmp2FigureData = tmp2FigureData +"Cumulative-"

        if snapNumber is not None:
            if str(snapNumber).isdigit() is True:
                SaveSnapNumber = "_" + str(snapNumber).zfill(4)
            else:
                SaveSnapNumber = "_" + str(snapNumber)
        else:
            SaveSnapNumber = ""

        if SFRBool is True:
            opslaan = tmp2 + f"SFR{SaveSnapNumber}"
            opslaanFigureData = tmp2FigureData + f"SFR{SaveSnapNumber}"

        else:
            opslaan = tmp2 + f"{weightKey}-{analysisParam}-PDF{SaveSnapNumber}"
            opslaanFigureData = tmp2FigureData + f"{weightKey}-{analysisParam}-PDF{SaveSnapNumber}"

        if combineMultipleOntoAxis is True: opslaan = opslaan + "-Simulations-Combined"

        plt.savefig(opslaan + ".pdf", dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()
        matplotlib.rc_file_defaults()
        plt.close("all")

        if (saveFigureData is True)&(replotFromData is False)&(combineMultipleOntoAxis is False):
            print(f"Saving Figure Data as {opslaanFigureData}")
            out = {"data":{"x" : xFromBins, "y" : hist}}
            tr.hdf5_save(opslaanFigureData+"_data.h5",out)
            
        
    return

def load_pdf_versus_plot_data(
    snapRange,
    weightKeys = None,
    xParams = ["T"],
    cumulative = False,
    loadPathBase = "./",
    loadPathSuffix = "",
    SFR = False,
    normalise = False,
    stack = True,
    selectKeyLen=2,
    delimiter="-",
    verbose = False,
    hush = False,
    ):

    loadPath = loadPathBase + loadPathSuffix + "Plots/PDFs/" 
    out = {}

    if weightKeys == None:
        if hush is False:
            print(f"[@load_pdf_versus_plot_data]: WARNING! No weightKeys provided! Will default to attempting to load mass data for all datasets requested by this call...")
        weightKeys = {kk: "mass" for kk in xParams}

    if SFR is True:
        weightKeys = {"age": "gima"}
        xParams = ["age"]


    for analysisParam in xParams:
        if verbose:
            print("")
            print(f"Starting {analysisParam} load!")

        try:
            weightKey = weightKeys[analysisParam]
        except:
            if hush is False:
                print(f"[@load_pdf_versus_plot_data]: WARNING! No weightKey found for {analysisParam}! Defaulting to attempting 'weighted by mass' data load...")
            weightKey = "mass"

        SFRBool = False
        if ((weightKey == "gima")&(analysisParam=="age"))|(SFR==True):
            SFRBool = True

        if normalise is True:
            tmp2FigureData = loadPath +"Normalised-"
        else:
            tmp2FigureData = loadPath

        if cumulative is True:
            tmp2FigureData = tmp2FigureData +"Cumulative-"
        
    
        toCombine = {}
        for snapNumber in snapRange:
            if snapNumber is not None:
                if str(snapNumber).isdigit() is True:
                    SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                else:
                    SaveSnapNumber = "_" + str(snapNumber)
            else:
                SaveSnapNumber = ""

            if SFRBool is True:
                opslaanFigureData = tmp2FigureData + f"SFR{SaveSnapNumber}"

            else:
                opslaanFigureData = tmp2FigureData + f"{weightKey}-{analysisParam}-PDF{SaveSnapNumber}"

            # print(f"Loading Figure Data from {opslaanFigureData}")
            # out = {"data":{"x" : xFromBins, "y" : hist}}
            try:
                tmp = tr.hdf5_load(
                    opslaanFigureData+"_data.h5",
                    selectKeyLen = selectKeyLen,
                    delimiter = delimiter
                    )
                skipBool = False
                tmpkey = list(tmp.keys())[0]
            except Exception as e:
                if verbose: print(str(e))
                skipBool = True
                continue
            if skipBool is False: toCombine.update({("data",int(snapNumber)) : copy.deepcopy(tmp[tmpkey])})
        if skipBool is False:
            flattened = cr.cr_flatten_wrt_time(toCombine, stack = stack, verbose = verbose, hush = not verbose)
            out.update({(analysisParam, weightKey) : flattened["data"]})
        
    return out

def load_phase_plot_data(
    snapRange,
    yParams = ["T"],
    xParams = ["rho_rhomean","R"],
    weightKeys = ["mass","vol"],
    loadPathBase = "./",
    stack = True,
    selectKeyLen=2,
    delimiter="-",
    verbose = False,
    ):

    loadPath = loadPathBase + "Plots/Phases/" 
    out = {}

    if verbose:  print(f"Starting phase plot data load!")
    for yParam in yParams:
        if verbose:  print(f"{yParam}")
        for xParam in xParams:
            if verbose:  print(f"{xParam}")
            for weightKey in weightKeys:
                if verbose:  print(f"{weightKey}")

                if weightKey == xParam:
                    if verbose:  print("\n" + f"WeightKey same as xParam! Skipping...")
                    skipBool = True
                    continue

                if weightKey == yParam:
                    if verbose:  print("\n" + f"WeightKey same as yParam! Skipping...")
                    skipBool = True
                    continue

                if xParam == yParam:
                    if verbose:  print("\n" + f"yParam same as xParam! Skipping...")
                    skipBool = True
                    continue

                if np.all(np.isin(np.array(["tcool","theat"]),np.array([xParam,yParam,weightKey]))) == True:
                    if verbose:  print("\n" + f"tcool and theat aren't compatible! Skipping...")
                    skipBool = True
                    continue
            
            
                toCombine = {}
                for snapNumber in snapRange:
                    if snapNumber is not None:
                        if str(snapNumber).isdigit() is True:
                            SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                        else:
                            SaveSnapNumber = "_" + str(snapNumber)
                    else:
                        SaveSnapNumber = ""

                    opslaanData = (
                        loadPath
                        + f"Phases-Plot_{yParam}-vs-{xParam}_weighted-by-{weightKey}{SaveSnapNumber}"
                    )

                    # print(f"Loading Figure Data from {opslaanFigureData}")
                    # out = {"data":{"x" : xFromBins, "y" : hist}}
                    try:
                        tmp = tr.hdf5_load(
                            opslaanData+"_data.h5",
                            selectKeyLen = selectKeyLen,
                            delimiter = delimiter
                        )
                        skipBool = False
                        tmpkey = list(tmp.keys())[0]
                    except Exception as e:
                        if verbose:  print(str(e))
                        skipBool = True
                        continue
                    if skipBool is False: toCombine.update({("data",int(snapNumber)) : copy.deepcopy(tmp[tmpkey])})
                if skipBool is False:
                    flattened = cr.cr_flatten_wrt_time(toCombine, stack = stack, verbose = verbose, hush = not verbose)
                    out.update({(xParam, yParam, weightKey) : flattened["data"]})

    return out

def load_statistics_data(
    snapRange,
    loadPathBase = "./",
    loadFile = "statsDict",
    fileType = ".h5",
    stack = True,
    selectKeyLen=2,
    delimiter="-",
    verbose = False,
    ):

    loadPath = loadPathBase
    out = {}

    toCombine = {}
    for snapNumber in snapRange:
        if snapNumber is not None:
            SaveSnapNumber = "_" + str(snapNumber)
        else:
            SaveSnapNumber = ""

        opslaanData = (
            loadPath
            + f"{SaveSnapNumber}_"+loadFile+fileType
        )

        # print(f"Loading Figure Data from {opslaanFigureData}")
        # out = {"data":{"x" : xFromBins, "y" : hist}}
        try:
            tmp = tr.hdf5_load(
                opslaanData,
                selectKeyLen = selectKeyLen,
                delimiter = delimiter
            )
                
        except Exception as e:
            print(str(e))
            continue
        if ((list(tmp.keys())[0][-1]).isdigit() == True):
            toCombine.update(copy.deepcopy(tmp))
        else:
            updatedtmp = {}
            for key,val in tmp.items():
                updatedtmp.update({tuple(list(key)+[str(snapNumber)]) : copy.deepcopy(val)})
            toCombine.update(updatedtmp)
    flattened = cr.cr_flatten_wrt_time(toCombine, stack = stack, verbose = verbose, hush = not verbose)
    out.update(flattened)
            
    return out

def plot_slices(snap,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber = None,
    sliceParam = None,
    xsize = 5.0,
    ysize = 5.0,
    fontsize=13,
    Axes=[0,1],
    averageAcrossAxes = False,
    saveAllAxesImages = False,
    boxsize=400.0,
    boxlos=50.0,
    pixreslos=0.3,
    pixres=0.3,
    projection=False,
    DPI=200,
    colourmapMain="inferno",
    colourmapsUnique = {},
    numthreads=10,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    saveFigureData = False,
    saveFigure = True,
    selectKeysList = None,
    compareSelectKeysOn = "vertical",
    rasterized = True,
    subfigures = False,
    cbarscale = 0.4,
    sharex = True,
    sharey = True,
    verbose = False,
    inplace = False,
    replotFromData = False,
):

    from itertools import combinations

    if (sliceParam is None):
        print("\n"
              +f"[@plot_slices]: WARNING! No data/no plots requested Skipping plot call and exiting ..."
              +"\n"
        )
        return
    
    snapType = False
    try:
        types = pd.unique(snap.data["type"])
        snapType = True
        if verbose: print("[@plot_slices]: snapshot type detected!")
    except:
        snapType = False
        if verbose: print("[@plot_slices]: dictionary type detected!")

    if subfigures:
        if snapType == True:
            # Where outer keys do not exist (as is the case for a snapshot object)
            # we will skip iterating over select keys
            if selectKeysList is not None:
                print("\n"
                    +f"[@plot_slices]: WARNING! selectKeysList was provided for snapshot type object! This option is not implemented."
                    +"\n"
                    +"Please convert desired data into dictionary format (e.g. data = dict(snap.data) ) before making this call!"
                    +"\n"
                    +"selectKeysList will be ignored for current call..."
                    +"\n"
                )
            selectKeysList = None #range(0,len(sliceParam))
            inputDict = snap.data
        else:
            inputDict = snap


        nrows = len(sliceParam)
        ncols = max([len(zz) for zz in sliceParam])
        multrows = 1
        multcols = 1
        if selectKeysList is not None:
            if (compareSelectKeysOn.lower() == "vertical"):
                multrows = np.shape(selectKeysList)[0]
            elif (compareSelectKeysOn.lower() == "horizontal"):
                multrows = np.shape(selectKeysList)[0]
                multcols = np.shape(selectKeysList)[0]
            else:
                raise Exception(f"[@plot_slices]: FAILURE! Unkown compareSelectKeysOn = {compareSelectKeysOn} requested! Please use 'vertical' or 'horizontal'! ")

            tmp = []
            tmpprojection = []
            for ii in range(0,multrows):
                tmp += sliceParam
                tmpprojection += projection
            sliceParam = tmp
            projection = tmpprojection


        if selectKeysList is not None:
            tmpdataSources = list(inputDict.values())
        else:
            tmpdataSources = None

        tmpsliceParam = copy.deepcopy(sliceParam)
        tmpprojection = copy.deepcopy(projection)
        hasPlotMask = []
        for kk in range(0,multrows):
            for ii in range(0,nrows):
                rr = kk*nrows + ii
                row = []
                for ll in range(0,1):
                    for jj in range(0, ncols): 
                        cc = ll*ncols + jj
                        try:
                            tmp = projection[rr][cc]
                            tmp = sliceParam[rr][cc]
                            
                            key = sliceParam[rr][cc]
                            if selectKeysList is not None:
                                # if (compareSelectKeysOn.lower() == "vertical"):
                                hasData = tmpdataSources[kk][key]
                                # elif (compareSelectKeysOn.lower() == "horizontal"):
                                #     hasData = tmpdataSources[ll][key]
                            else:
                                hasData = inputDict[key]

                            # Will fail 'try' if data for sliceParam[ii][jj] is not present in provided input data
                            # hasData = inputDict[tmp]
                            if tmp is not None:
                                hasPlot = True
                            else:
                                hasPlot = False
                        except:
                        #     if (selectKeysList is not None) & (compareSelectKeysOn.lower() == "horizontal"):
                        #         hasPlot = False
                        #         tmp1 = copy.deepcopy(tmpsliceParam[cc])
                        #         tmp1[rr] = None
                        #         tmpsliceParam[cc] = tmp1

                        #         tmp2 = copy.deepcopy(tmpprojection[cc])
                        #         tmp2[rr] = None
                        #         tmpprojection[cc] = tmp2
                        #     else:
                            hasPlot = False
                            tmp1 = copy.deepcopy(tmpsliceParam[rr])
                            tmp1[cc] = None
                            tmpsliceParam[rr] = tmp1

                            tmp2 = copy.deepcopy(tmpprojection[rr])
                            tmp2[cc] = None
                            tmpprojection[rr] = tmp2
                        row.append(copy.deepcopy(hasPlot))
                hasPlotMask.append(copy.deepcopy(row))

        sliceParam = np.asarray(copy.deepcopy(tmpsliceParam))
        projection = np.asarray(copy.deepcopy(tmpprojection))
        hasPlotMask = np.asarray(hasPlotMask)
        figshape = np.shape(hasPlotMask)

        if (selectKeysList is not None) & (compareSelectKeysOn.lower() == "horizontal"):
            sliceParam = sliceParam.T
            projection = projection.T
            hasPlotMask = hasPlotMask.T
            figshape = figshape[::-1]
            multrows = figshape[0]

        dataSources = {}
        for selectKey, dataDict in inputDict.items():
            if selectKeysList is not None:
                if selectKey in selectKeysList:
                    dataSources.update({selectKey : dataDict})
            else:
                dataSources = {"": inputDict}
    # else:
    #     if selectKeysList is None: 
    #         if hush is False:
    #             print(f"[@plot_slices]: WARNING! Plot called with no selectKeysList provided."
    #                 +"\n"
    #                 +f"Call details: sliceParam:{xParam}, yParam:{yParam}"
    #                 +"\n"
    #                 +"Will default to plotting all data provided to this call.")
                
    #         selectKeysList = list(inputDict.keys())           


        ## Empty data checks ## 
        if bool(dataSources) is False:
            print("\n"
                +f"[@plot_slices]: WARNING! dataSources dict is empty! Skipping plot call and exiting ..."
                +"\n"
            )
            
            return



    savePath = savePathBase + "Plots/Slices/"
    savePathFigureData = savePathBaseFigureData + "Plots/Slices/"
    
    if subfigures:
        if (type(sliceParam[0]) == np.ndarray):
            sliceParam = sliceParam.flatten().tolist()
        if (type(projection[0]) == np.ndarray):
            projection = projection.flatten().tolist()
    else:
        if (type(sliceParam) != list):
            sliceParam = [sliceParam]
        if (type(projection) != list):
            projection = [projection]

    tmp = ""
    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    tmp = ""
    for savePathChunk in savePathFigureData.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["z","x","y"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#
    # PLOTTING TIME
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0


    subplotCount = -1
    if subfigures == True:
        dataSourcesValues = list(dataSources.values())
    else:
        dataSourcesValues = [snap]


    for ii,(param,proj) in enumerate(zip(sliceParam,projection)):
        subplotCount+=1
        if subfigures == True:
            axindex = np.unravel_index(subplotCount, shape=figshape)
            if hasPlotMask[axindex] == False:
                projection[subplotCount] = None
                continue
            if (selectKeysList is not None):
                if(compareSelectKeysOn.lower() == "horizontal"):
                    datasourceindex = axindex[1]
                else:
                    datasourceindex = axindex[0]
            snap = dataSourcesValues[datasourceindex]
        else:
            pass

        paramSplitList = param.split("_")

        if paramSplitList[-1] == "col":
            ## If _col variant is called we want to calculate a projection of the non-col parameter
            ## Thus, we force projection to now be true, and incorporate a dummy variable tmpsliceParam
            ## to force plots to generate non-col variants but save output as column density version

            tmpParam = "_".join(paramSplitList[:-1])
            projection[subplotCount] = True
        else:
            tmpParam = param

        if snapType == True:
            try:
                tmp = snap.data[tmpParam]
            except Exception as e:
                print(f"{str(e)}")
                print(
                f"[@plot_slices]: Variable {tmpParam} not found in snapshot data. Skipping plot..."
                )
                return
        else:
            # print(
            # f"[@plot_slices]: Assuming replot from data is desired. Setting replotFromData = True" 
            # )
            # replotFromData = True
            tmpParam = param
            try:
                slice = snap[param]
            except Exception as e:
                print(f"{str(e)}")
                print(
                f"[@plot_slices]: Variable {tmpParam} not found in data dictionary. Skipping plot..."
                )
                return

    if subfigures:
        newxsize = xsize
        newysize = ysize
        if (compareSelectKeysOn.lower() == "horizontal"):
            width = float(figshape[1])+cbarscale
            height = float(figshape[0])
            aspect_ratio = width/height
            newxsize = xsize*aspect_ratio
            newysize = ysize

            tmp = list(figshape)
            tmp[1] += 1
            figshape = tuple(tmp)

            # print(height,width)
            # print(newysize,newxsize)
            sliceParam = np.pad(np.asarray(sliceParam).reshape(figshape[0],-1),((0,0),(0,1)),constant_values=None)
            sliceParam = sliceParam.flatten().tolist()
            projection = np.pad(np.asarray(projection).reshape(figshape[0],-1),((0,0),(0,1)),constant_values=None)
            projection = projection.flatten().tolist()
            hasPlotMask = np.pad(hasPlotMask,((0,0),(0,1)),constant_values=False)
        elif (compareSelectKeysOn.lower() == "vertical"):
            width = float(figshape[1])
            height = float(figshape[0])+cbarscale
            aspect_ratio = height/width
            newxsize = xsize
            newysize = ysize*aspect_ratio

            tmp = list(figshape)
            tmp[0] += 1
            figshape = tuple(tmp)

            # print(height,width)
            # print(newysize,newxsize)
            sliceParam += [None]*multcols
            projection += [None]*multcols
            hasPlotMask = np.pad(hasPlotMask,((0,1),(0,0)),constant_values=False)

    if subfigures:
        if (compareSelectKeysOn.lower() == "horizontal"):
            fig, axes = plt.subplots(
                nrows=figshape[0],
                ncols=figshape[1],
                figsize=(newxsize, newysize),
                dpi=DPI,
                gridspec_kw={'width_ratios': [1]*multcols + [cbarscale]}
            )
        elif (compareSelectKeysOn.lower() == "vertical"):
            fig, axes = plt.subplots(
                nrows=figshape[0],
                ncols=figshape[1],
                figsize=(newxsize, newysize),
                dpi=DPI,
                gridspec_kw={'height_ratios': [1]*multrows + [cbarscale]}
            )


    halfbox = boxsize/2.0
    fullTicks = [xx for xx in np.linspace(-1.0 * halfbox, halfbox, 5)]
    fudgeTicks = fullTicks[1:]
    
    subplotCount = -1
    plotlist = []
    for (param,proj) in zip(sliceParam,projection):

        # print(param,subplotCount)

        subplotCount+=1
        if subfigures == True:

            axindex = np.unravel_index(subplotCount, shape=figshape)
            if (selectKeysList is not None):
                if(compareSelectKeysOn.lower() == "horizontal"):
                    datasourceindex = axindex[1]
                else:
                    datasourceindex = axindex[0]
            # print(axindex)
            if hasPlotMask[axindex] == False:
                if saveFigure:
                    currentAx = axes[axindex]
                    currentAx.axis('off')
                    if (axindex[0]==(figshape[0]-2))&(compareSelectKeysOn.lower() == "vertical"): 
                        cax = axes[axindex[0]+1,axindex[1]]
                        cax.axis("off")
                    elif (axindex[1]==(figshape[1]-2))&(compareSelectKeysOn.lower() == "horizontal"): 
                        cax = axes[axindex[0],axindex[1]+1]
                        cax.axis("off")
                    plotlist.append(None)
                    continue
                snap = dataSourcesValues[datasourceindex]
        else:
            pass

        try: 
            paramcmap = colourmapsUnique[param]
            cmap = plt.get_cmap(paramcmap)
        except:
            cmap = plt.get_cmap(colourmapMain)

        if snapType is True:
            if averageAcrossAxes is True:
                if subfigures:
                    raise Exception(
                        f"[@plot_slices]: FAILURE! Cannot combine subfigures and averageAcrossAxes=True!"
                    )
                axisCombinations = combinations(range(0,3,1),r=2)
                toCombine = {}
                for jj,currentAxes in enumerate(axisCombinations):
                    currentAxes = list(currentAxes)
                    requestedAxes = sorted(Axes)

                    if (currentAxes == requestedAxes):
                        plotAxes = Axes
                    else:
                        plotAxes = currentAxes 


                    if ((saveAllAxesImages is True)|((currentAxes == requestedAxes)&(saveFigure == True))):
                        saveCurrentFigure = True
                    else:
                        saveCurrentFigure = False

                    tmp = plot_slices(snap=snap,
                        ylabel=ylabel,
                        xlimDict=xlimDict,
                        logParameters=logParameters,
                        snapNumber = snapNumber,
                        sliceParam = param,
                        xsize = xsize,
                        ysize = ysize,
                        fontsize=fontsize,
                        Axes=plotAxes,
                        averageAcrossAxes = False,
                        saveAllAxesImages = False,
                        boxsize=boxsize,
                        boxlos=boxlos,
                        pixreslos=pixreslos,
                        pixres=pixres,
                        projection=proj,
                        DPI=DPI,
                        colourmapMain=colourmapMain,
                        numthreads=numthreads,
                        savePathBase = savePathBase,
                        savePathBaseFigureData = savePathBaseFigureData,
                        saveFigureData = saveFigureData,
                        saveFigure = saveCurrentFigure,
                        rasterized = rasterized,
                        verbose = verbose,
                        inplace = inplace,
                        replotFromData = False,
                    )
                    #out = {param: copy.deepcopy(slice)}
                    toCombine.update({(param, str(jj)): copy.deepcopy(tmp[param])})
                out = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = verbose, hush = not verbose)

                for sKey, data in out.items():
                    dataCopy = copy.deepcopy(data)
                    for key,value in data.items():
                        dataCopy.update({key: np.nanmedian(value,axis=-1)})
                    out[sKey].update(dataCopy)


                # Pad snapNumber with zeroes to enable easier video making

                if snapNumber is not None:
                    if str(snapNumber).isdigit() is True:
                        SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                    else:
                        SaveSnapNumber = "_" + str(snapNumber)
                else:
                    SaveSnapNumber = ""

                if proj is False:
                    savePath = savePath + f"Slice_Plot_AxAv_{param}{SaveSnapNumber}.pdf"
                    savePathFigureData = savePathFigureData + f"Slice_Plot_AxAv_{param}{SaveSnapNumber}"
                else:
                    savePath = savePath + f"Projection_Plot_AxAv_{param}{SaveSnapNumber}.pdf" 
                    savePathFigureData = savePathFigureData + f"Projection_Plot_AxAv_{param}{SaveSnapNumber}" 

                if (saveFigureData is True)&(replotFromData is False):
                    print(f"Saving Figure Data as {savePathFigureData}")
                    tr.hdf5_save(savePathFigureData+"_data.h5",out)

                return out

            paramSplitList = param.split("_")
            if paramSplitList[-1] == "col":
                ## If _col variant is called we want to calculate a projection of the non-col parameter
                ## Thus, we force projection to now be true, and incorporate a dummy variable tmpsliceParam
                ## to force plots to generate non-col variants but save output as column density version

                param = "_".join(paramSplitList[:-1])
                proj = True           
            # aspect = "equal"

            if proj is True:
                nz=int(boxlos / pixreslos)
                boxz=boxlos
            else:
                nz=None #int(boxlos / pixreslos)
                boxz= None #boxlos

            if param == "T":
                slice  = snap.get_Aslice(
                    "Tdens",
                    box=[boxsize, boxsize],
                    center=imgcent,
                    nx=int(boxsize / pixres),
                    ny=int(boxsize / pixres),
                    nz=nz,
                    boxz=boxz,
                    axes=Axes,
                    proj=proj,
                    numthreads=numthreads,
                )

                proj_dens = snap.get_Aslice(
                    "rho_rhomean",
                    box=[boxsize, boxsize],
                    center=imgcent,
                    nx=int(boxsize / pixres),
                    ny=int(boxsize / pixres),
                    nz=nz,
                    boxz=boxz,
                    axes=Axes,
                    proj=proj,
                    numthreads=numthreads,
                )

                slice["grid"] = slice["grid"]/proj_dens["grid"]
            else:
                slice = snap.get_Aslice(
                    param,
                    box=[boxsize, boxsize],
                    center=imgcent,
                    nx=int(boxsize / pixres),
                    ny=int(boxsize / pixres),
                    nz=nz,
                    boxz=boxz,
                    axes=Axes,
                    proj=proj,
                    numthreads=numthreads,
                )

                if paramSplitList[-1] == "col":
                    KpcTocm = 1e3 * c.parsec
                    convert = float(pixreslos)*KpcTocm
                    slice["grid"] = slice["grid"]*convert
                    param = "_".join(paramSplitList)
                elif proj is True:
                    slice["grid"] = slice["grid"]/ int(boxlos / pixreslos)
        elif replotFromData is True:
            slice = snap[param]
        
        if saveFigure is True:
            if subfigures == False:    
                fig, axes = plt.subplots(
                    nrows=1, ncols=1, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
                )
                currentAx = axes
                currentAx.tick_params(axis="both", direction="in")

            else:
                axindex = np.unravel_index(subplotCount, shape=figshape)
                currentAx = axes[axindex]
                currentAx.tick_params(axis="both", direction="in")
                if subfigures == True:
                    if (selectKeysList is not None):
                        if(compareSelectKeysOn.lower() == "horizontal"):
                            datasourceindex = axindex[1]
                        else:
                            datasourceindex = axindex[0]
                    snap = dataSourcesValues[datasourceindex]
                else:
                    pass

            # cmap = plt.get_cmap(colourmapMain)
            cmap = copy.copy(cmap)
            cmap.set_bad(color="grey")

            try:
                tmp = xlimDict[param]["xmin"]
                tmp = xlimDict[param]["xmax"]
                xlimBool = True
            except:
                xlimBool = False

            if xlimBool is True:
                if param in logParameters:
                    zmin = 10**(xlimDict[param]['xmin'])
                    zmax = 10**(xlimDict[param]['xmax'])
                    norm = matplotlib.colors.LogNorm(vmin = zmin, vmax = zmax,clip=True)

                    decs = np.arange(np.round(xlimDict[param]['xmin'],0),np.round(xlimDict[param]['xmax'],0)+1,1.0)

                    decs = decs[np.where((decs>=xlimDict[param]['xmin'])&(decs<=xlimDict[param]['xmax']))[0]]

                    cbarticks = [10**xx for xx in decs]

                    # while len(cbarticks)>=8:
                    #     cbarticks = cbarticks[::2]# + [cbarticks[-1]]

                    if (xlimDict[param]['xmin']%1 != 0): cbarticks = [zmin] + cbarticks
                    if (xlimDict[param]['xmax']%1 != 0): cbarticks = cbarticks + [zmax]
                    cbarticklabels = [r"$10^{%3.0f}$" % (xx) for xx in cbarticks]    
                    formatter = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=False, minor_thresholds=(np.inf,np.inf))
                    # formatter.set_locs(locs=cbarticks)
                    print(cbarticks)
                else:
                    zmin = xlimDict[param]['xmin']
                    zmax = xlimDict[param]['xmax']
                    norm = matplotlib.colors.Normalize(vmin = zmin, vmax = zmax,clip=True)
                    cbarticks = [xx for xx in np.linspace(zmin,zmax,5)]
                    cbarticklabels = [str(xx) for xx in cbarticks]
                    formatter = matplotlib.ticker.ScalarFormatter()
            else:
                if param in logParameters:
                    zmin = np.nanmax(slice["grid"])
                    zmax = np.nanmin(slice["grid"])
                    norm = matplotlib.colors.LogNorm(vmin = zmin, vmax = zmax,clip=True)

                    decs = np.arange(np.round(np.log10(zmin),0),np.round(np.log10(zmax),0)+1,1.0)

                    decs = decs[np.where((decs>=np.log10(zmin))&(decs<=np.log10(zmax)))[0]]

                    cbarticks = [10**xx for xx in decs]

                    # while len(cbarticks)>=8:
                    #     cbarticks = cbarticks[::2]# + [cbarticks[-1]]

                    if (np.log10(zmin)%1 != 0): cbarticks = [zmin] + cbarticks
                    if (np.log10(zmax)%1 != 0): cbarticks = cbarticks + [zmax]
                    cbarticklabels = [r"$10^{%3.0f}$" % (xx) for xx in cbarticks]
                    formatter = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=False, minor_thresholds=(np.inf,np.inf))
                    # formatter.set_locs(locs=cbarticks)
                    print(cbarticks)
                else:
                    zmin = np.nanmax(slice["grid"])
                    zmax = np.nanmin(slice["grid"])
                    norm = matplotlib.colors.Normalize(vmin = zmin, vmax = zmax,clip=True)
                    cbarticks = [xx for xx in np.linspace(zmin,zmax,5)]
                    cbarticklabels = [str(xx) for xx in cbarticks]
                    formatter = matplotlib.ticker.ScalarFormatter()
                    

            pcm = currentAx.pcolormesh(
                slice["x"],
                slice["y"],
                np.transpose(slice["grid"]),
                norm=norm,
                cmap=cmap,
                rasterized=rasterized,
            )

            plotlist.append(pcm)
            #currentAx.set_title(f"{param} Slice", fontsize=fontsize)


            # currentAx.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
            # currentAx.set_xlabel(f"{AxesLabels[Axes[0]]}" + " (kpc)", fontsize=fontsize)
            # Pad snapNumber with zeroes to enable easier video making
            if ((subfigures == True)&(sharex == True)&(sharey == True)):
                if (compareSelectKeysOn.lower() == "vertical"):
                    rowoffset = 2
                else:
                    rowoffset = 1

                if (axindex[1]==0):
                    if (axindex[0]==(figshape[0]-rowoffset)):
                        currentAx.set_yticks(fullTicks)
                        currentAx.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
                        currentAx.set_xticks(fullTicks)
                        currentAx.set_xlabel(f"{AxesLabels[Axes[0]]}" + " (kpc)", fontsize=fontsize)
                    else:
                        currentAx.set_yticks(fudgeTicks)
                        currentAx.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)

                        currentAx.set_xticks(ticks=[])
                else:
                    currentAx.set_yticks(ticks=[])
                    if (axindex[0]==(figshape[0]-rowoffset)):
                        currentAx.set_xticks(fudgeTicks)
                        currentAx.set_xlabel(f"{AxesLabels[Axes[0]]}" + " (kpc)", fontsize=fontsize)
                    else:
                        currentAx.set_xticks(ticks=[])

                if (axindex[0]==(figshape[0]-2))&(compareSelectKeysOn.lower() == "vertical"): 
                    caxorient = "horizontal"
                    caxtickloc = "bottom"
                    cax = axes[axindex[0]+1,axindex[1]]
                    cax.axis("off")


                    axindices = np.asarray([[ii,axindex[1]] for ii in range(0,figshape[0]-1)]).reshape(-1,2).T
                    axpcmsindices = np.ravel_multi_index(axindices,figshape)
                    axpcms = [plotlist[ii] for ii in axpcmsindices if ii<len(plotlist)]
                    wherenotnone = [xx for xx in axpcms if xx is not None]
                    
                    clb = plt.colorbar(wherenotnone[0], ax = axes[axindex[0]+1,axindex[1]], ticks=cbarticks, format=formatter, orientation = caxorient, ticklocation = caxtickloc, shrink=0.85)
                    clb.set_label(label=f"{ylabel[param]}", size=fontsize)
                    # clb.yaxis.set_ticks_position("left")
                    # clb.yaxis.set_ticks_position("left")
                    clb.ax.tick_params(axis="x", which="major", labelsize=fontsize)
                    clb.ax.tick_params(axis="x", which="minor", labelsize=fontsize)
                    clb.ax.xaxis.set_major_formatter(formatter)
                    clb.ax.xaxis.set_minor_formatter(formatter) 

                elif (axindex[1]==(figshape[1]-2))&(compareSelectKeysOn.lower() == "horizontal"): 
                    caxorient = "vertical"
                    caxtickloc = "left"
                    labellocation = "left"
                    cax = axes[axindex[0],axindex[1]+1]
                    cax.axis("off")

                    axindices = np.asarray([[axindex[0],ii] for ii in range(0,figshape[1]-1)]).reshape(-1,2).T
                    axpcmsindices = np.ravel_multi_index(axindices,figshape)
                    axpcms = [plotlist[ii] for ii in axpcmsindices if ii<len(plotlist)]
                    wherenotnone = [xx for xx in axpcms if xx is not None]

                    clb = plt.colorbar(wherenotnone[0], ax = axes[axindex[0],axindex[1]+1], ticks=cbarticks, format=formatter, orientation = caxorient, ticklocation = caxtickloc, shrink=0.85)
                    clb.set_label(label=f"{ylabel[param]}", size=fontsize)
                    clb.ax.yaxis.set_ticks_position(labellocation)
                    # clb.ax.yaxis.set_ticks_position(labellocation)
                    # clb.ax.tick_params(axis="y", which="major", labelsize=fontsize)
                    # clb.ax.tick_params(axis="y", which="minor", labelsize=fontsize)
                    # clb.ax.yaxis.set_major_formatter(formatter)
                    # clb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())                  


            else:
                currentAx.set_xticks(fullTicks)
                currentAx.set_yticks(fullTicks)
                currentAx.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
                currentAx.set_xlabel(f"{AxesLabels[Axes[0]]}" + " (kpc)", fontsize=fontsize)
                cax1 = inset_axes(currentAx, width="5%", height="95%", loc="right")
                fig.colorbar(pcm, cax=cax1, ticks=cbarticks, format=formatter, orientation="vertical").set_label(
                    label=f"{ylabel[param]}", size=fontsize, weight="bold"
                )
                cax1.yaxis.set_ticks_position("left")
                cax1.yaxis.set_label_position("left")
                cax1.yaxis.label.set_color("white")
                cax1.tick_params(axis="y", which="major", colors="white", labelsize=fontsize)
                cax1.tick_params(axis="y", which="minor", colors="white", labelsize=fontsize)                  

            currentAx.set_aspect("equal")
            if snapNumber is not None:
                if str(snapNumber).isdigit() is True:
                    SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                else:
                    SaveSnapNumber = "_" + str(snapNumber)
            else:
                SaveSnapNumber = ""

            if proj is False:
                opslaan = savePath + f"Slice_Plot_{AxesLabels[Axes[0]]}-{AxesLabels[Axes[1]]}_{param}{SaveSnapNumber}.pdf"
            else:
                opslaan = savePath + f"Projection_Plot_{AxesLabels[Axes[0]]}-{AxesLabels[Axes[1]]}_{param}{SaveSnapNumber}.pdf" 

            if subfigures == False:
                print(f" Save {opslaan}")
                plt.savefig(opslaan, transparent=False)
                plt.close()

                matplotlib.rc_file_defaults()
                plt.close("all")
                print(f" ...done!")
            else:
                if ((sharex == True)&(sharey == True)):plt.subplots_adjust(hspace=0.0, wspace=0.0)
                elif (sharex == True):plt.subplots_adjust(wspace=0.0)
                elif (sharey == True):plt.subplots_adjust(hspace=0.0)

    # Pad snapNumber with zeroes to enable easier video making

    if snapNumber is not None:
        if str(snapNumber).isdigit() is True:
            SaveSnapNumber = "_" + str(snapNumber).zfill(4)
        else:
            SaveSnapNumber = "_" + str(snapNumber)
    else:
        SaveSnapNumber = ""

    if subfigures:
        opslaan = savePath + f"Images_Plot{SaveSnapNumber}.pdf"
      
        plt.gcf()
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        matplotlib.rc_file_defaults()
        plt.close("all")

        print(
            f"[@plot_slices]: WARNING! Not implemented! Cannot return output dictionary from subfigures==True version! Output dictionary will be empty..."
        )
        out = {}
    else:
        if projection[0] is False:
            savePathFigureData = savePathFigureData + f"Slice_Plot_{AxesLabels[Axes[0]]}-{AxesLabels[Axes[1]]}_{param}{SaveSnapNumber}"
        else:
            savePathFigureData = savePathFigureData + f"Projection_Plot_{AxesLabels[Axes[0]]}-{AxesLabels[Axes[1]]}_{param}{SaveSnapNumber}" 

        out = {param: copy.deepcopy(slice)}
        if (saveFigureData is True)&(replotFromData is False):
            print(f"Saving Figure Data as {savePathFigureData}")
            tr.hdf5_save(savePathFigureData+"_data.h5",out)


    return out

def medians_versus_plot(
    inputDict,
    PARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    yParam=None,
    xParam="R",
    titleBool=False,
    legendBool = True,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize = 13,
    fontsizeTitle = 14,
    linewidth=1.0,
    opacityPercentiles=0.25,
    colourmapMain="tab10",
    savePathBase ="./",
    savePathBaseFigureData = "./",
    allowPlotsWithoutxlimits = False,
    inplace = False,
    saveFigureData = False,
    replotFromData = False,
    combineMultipleOntoAxis = False,
    selectKeysList = None,
    styleDict = None,
    hush = False,
    ):

    if (bool(inputDict) is False)| (bool(xParam) is False)| (bool(yParam) is False):
        print("\n"
              +f"[@medians_versus_plot]: WARNING! No data/no plots requested Skipping plot call and exiting ..."
              +"\n"
        )
        return
    
    keys = list(PARAMS.keys())
    selectKey0 = keys[0]

    if yParam is None:
        plotParams = PARAMS[selectKey0]["saveParams"]
        if hush is False:
            print(f"[@medians_versus_plot]: WARNING! No yParam provided so default of all 'saveParams' being used. This may cause errors if any of 'saveParams' do not have limits set in xlimDict...")
    else:
        plotParams = yParam

    if allowPlotsWithoutxlimits == False:
        # # check_params_are_in_xlimDict(xlimDict, weightKeys) <-- Not needed, as y-axis not used when combining data
        check_params_are_in_xlimDict(xlimDict, [xParam])

    savePath = savePathBase + "/Plots/Medians/"
 
    tmp = ""
    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass


    if combineMultipleOntoAxis is False:
        selectKeysList = list(inputDict.keys())  
        if inplace is True:
            dataDict = inputDict
            print("\n"
                +f"[@medians_versus_plot]: WARNING! inplace flag set to True. This will modify the data dictionary provided to this function call."
                +"\n"
                +f"Call details: xParam:{xParam}, yParam:{yParam}"
                +"\n"
                )
        else:
            dataDict = copy.deepcopy(inputDict)

        dataSources = dataDict
    else:

        if selectKeysList is None: 
            if hush is False:
                print(f"[@medians_versus_plot]: WARNING! combineMultipleOntoAxis type plot called with no selectKeysList provided."
                    +"\n"
                    +f"Call details: xParam:{xParam}, yParam:{yParam}"
                    +"\n"
                    +"Will default to plotting all data provided to this call.")
                
            selectKeysList = list(inputDict.keys())           
        if inplace is True:
            print("\n"
                +f"[@medians_versus_plot]: WARNING! inplace flag set to True. This will modify the data dictionary provided to this function call."
                +"\n"
                +f"Call details: xParam:{xParam}, yParam:{yParam}"
                +"\n"
                )


          
        dataSources = {}
        for selectKey, dataDict in inputDict.items():
            if selectKey in selectKeysList:
                if inplace is True:
                    dataSources.update({selectKey : dataDict})
                else:
                    dataSources.update({selectKey : copy.deepcopy(dataDict)})
            


        ## Empty data checks ## 
        if bool(dataSources) is False:
            print("\n"
                +f"[@medians_versus_plot]: WARNING! dataSources dict is empty! Skipping plot call and exiting ..."
                +"\n"
            )
            
            return




    for analysisParam in plotParams:
        if analysisParam != xParam:
            print("")
            print(f"Starting {analysisParam} plots!")
  
            fig, ax = plt.subplots(

            nrows=1,
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
            )

            plt.tick_params(axis="both", direction="in")
            yminlist = []
            ymaxlist = []

            skipBool = False

            for jj, (selectKey, simDict) in enumerate(dataSources.items()):
                skipBool = False
                Nkeys = len(list(selectKeysList))

                if ("Stars" in selectKey) | ("col" in selectKey) :
                    selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
                else:
                    selectKeyShort = selectKey

                ## Empty data checks ## 
                if bool(simDict) is False:
                    skipBool = True
                    print("\n"
                        +f"[@medians_versus_plot]: WARNING! simDict is empty! Skipping plots ..."
                        +"\n"
                    )

                print(f"Starting {selectKey} plot")     


                label = " ".join(list(selectKeyShort))

                if combineMultipleOntoAxis is False:
                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if colourmapMain == "tab10":
                        colour = cmap(float(jj) / 10.0)
                    else:
                        colour = cmap(float(jj) / float(Nkeys))
                    linestyle = "solid"
                    plotData = copy.deepcopy(simDict)
                    xData = np.array(copy.deepcopy(simDict[xParam]))
                else:
                    if selectKeysList is not None:
                        label = " ".join(list(selectKeyShort))
                        if styleDict is not None:
                            colour=styleDict[selectKeyShort]["colour"]
                            linestyle=styleDict[selectKeyShort]["linestyle"]
                        else:
                            cmap = matplotlib.cm.get_cmap(colourmapMain)
                            if colourmapMain == "tab10":
                                colour = cmap(float(jj) / 10.0)
                            else:
                                colour = cmap(float(jj) / float(Nkeys))
                            linestyle = "solid"
                    else:
                        cmap = matplotlib.cm.get_cmap(colourmapMain)
                        if colourmapMain == "tab10":
                            colour = cmap(float(jj) / 10.0)
                        else:
                            colour = cmap(float(jj) / float(Nkeys))
                        linestyle = "solid"


                    plotData = copy.deepcopy(simDict[selectKey])
                    xData = np.array(copy.deepcopy(simDict[selectKey][xParam]))



                loadPercentilesTypes = [
                    analysisParam + "_" + str(percentile) + "%"
                    for percentile in PARAMS[selectKeyShort]["percentiles"]
                ]


                ## Check that any percentiles other than median (50th) have been requested ##
                LO = (
                    analysisParam
                    + "_"
                    + str(min(PARAMS[selectKeyShort]["percentiles"]))
                    + "%"
                )
                UP = (
                    analysisParam
                    + "_"
                    + str(max(PARAMS[selectKeyShort]["percentiles"]))
                    + "%"
                )
                median = analysisParam + "_" + "50.00%"

            
                try:
                    tmpPerData = plotData[LO]
                except Exception as e:
                    print(f"{str(e)}")
                    print(
                        f"[@medians_versus_plot]: Lower percentile {analysisParam} not found! Omitting this from plot..."
                    )

                try:
                    tmpPerData = plotData[UP]
                except Exception as e:
                    print(f"{str(e)}")
                    print(
                        f"[@medians_versus_plot]: Upper percentile {analysisParam} not found! Omitting this from plot..."
                    )

                try:
                    tmpPerData = plotData[median]
                except Exception as e:
                    print(f"{str(e)}")
                    print(
                        f"[@medians_versus_plot]: Median value for {analysisParam} not found! Skipping plot entirely..."
                    )
                    if replotFromData is False: skipBool = True
                    continue

                if skipBool == True: continue

                tmpPerData = np.isfinite(plotData[median])

                if np.all(~tmpPerData) == True:
                    print(
                        f"[@medians_versus_plot]: Median values for {analysisParam} all not finite! Skipping plot entirely..."
                    )
                    if replotFromData is False: skipBool = True
                    continue

                if skipBool == True: continue

                if analysisParam in PARAMS[selectKeyShort]["logParameters"]:
                    for k, v in plotData.items():
                        plotData.update({k: np.log10(v)})

                loExists = True
                try:
                    ymin = np.nanmin(
                        plotData[LO][np.isfinite(plotData[LO])])
                    yminlist.append(ymin)
                except Exception as e:
                    print(f"{str(e)}")
                    print(
                        f"[@medians_versus_plot]: Lower percentile {analysisParam} has no finite values. Omitting this from plot...")
                    loExists = False    
                    ymin = np.nanmin(plotData[median][np.isfinite(plotData[median])])
                    yminlist.append(ymin)

                upExists = True
                try:
                    ymax = np.nanmax(
                        plotData[UP][np.isfinite(plotData[UP])])
                    ymaxlist.append(ymax)

                except Exception as e:
                    print(f"{str(e)}")
                    print(
                        f"[@medians_versus_plot]: Upper percentile {analysisParam} has no finite values. Omitting this from plot...")
                    upExists = False
                    ymax = np.nanmax(plotData[median][np.isfinite(plotData[median])])
                    ymaxlist.append(ymax)

                currentAx = ax
                # # # path = copy.copy(savePathBase)

                # # # splitbase = path.split("/")
                # # # # print(splitbase)
                # # # if "" in splitbase:
                # # #     splitbase.remove("")
                # # # if "." in splitbase:
                # # #     splitbase.remove(".")
                # # # if "Plots" in splitbase:
                # # #     splitbase.remove("Plots")
                # # # # print(splitbase)

                # # # if len(splitbase)>2:
                # # #     label = f'{splitbase[0]}: {"_".join(((splitbase[-2]).split("_"))[:2])} ({splitbase[-1]})'
                # # # elif len(splitbase)>1:
                # # #     label = f'{splitbase[0]}: {"_".join(((splitbase[-1]).split("_"))[:2])}'
                # # # else:
                # # #     label = f'Original: {"_".join(((splitbase[-1]).split("_"))[:2])}'



                ## Load in all percentiles requested ##
                midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)
                percentilesPairs = zip(
                    loadPercentilesTypes[:midPercentile],
                    loadPercentilesTypes[midPercentile + 1:],
                )
                if (loExists==True)&(upExists==True):
                    percentilesPairs = zip(
                        loadPercentilesTypes[:midPercentile],
                        loadPercentilesTypes[midPercentile + 1:],
                    )
                    for (LO, UP) in percentilesPairs:
                        currentAx.fill_between(
                            xData,
                            plotData[UP],
                            plotData[LO],
                            facecolor=colour,
                            alpha=opacityPercentiles,
                            interpolate=False,
                        )
                elif(loExists==True)&(upExists==False):
                    percentilesPairs = list(loadPercentilesTypes[:-1])
                    for LO in percentilesPairs:
                        currentAx.fill_between(
                            xData,
                            plotData[median],
                            plotData[LO],
                            facecolor=colour,
                            alpha=opacityPercentiles,
                            interpolate=False,
                        )
                elif(loExists==False)&(upExists==True):
                    percentilesPairs = list(loadPercentilesTypes[1:])
                    for UP in percentilesPairs:
                        currentAx.fill_between(
                            xData,
                            plotData[UP],
                            plotData[median],
                            facecolor=colour,
                            alpha=opacityPercentiles,
                            interpolate=False,
                        )
                else:
                    pass

                currentAx.plot(
                    xData,
                    plotData[median],
                    label= label,
                    color=colour,
                    linestyle = linestyle,
                    linewidth = linewidth,
                )

                currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                currentAx.tick_params(
                    axis="both", which="both", labelsize=fontsize)

                currentAx.set_ylabel(
                    ylabel[analysisParam], fontsize=fontsize)

                if titleBool is True:
                    if "Stars" in selectKey:
                        fig.suptitle(
                            f"Median and Percentiles of"
                            + "\n"
                            + f" Stellar-{analysisParam} vs {xParam}",
                            fontsize=fontsizeTitle,
                        )

                    elif "col" in selectKey:
                        fig.suptitle(
                            f"Median and Percentiles of"
                            + "\n"
                            + f" Projected Maps of {analysisParam} vs {xParam}",
                            fontsize=fontsizeTitle,
                        )    

                    else:
                        fig.suptitle(
                            f"Median and Percentiles of"
                            + "\n"
                            + f" {analysisParam} vs {xParam}",
                            fontsize=fontsizeTitle,
                        )

            ax.set_xlabel(ylabel[xParam], fontsize=fontsize)

            if (skipBool == True):
                print(
                    f"Variable {analysisParam} plot failed (reason should have been printed to stdout above). Skipping plot...")
                continue


            if (len(yminlist) == 0) | (len(ymaxlist) == 0):
                print(
                    f"[@medians_versus_plot]: Variable {analysisParam} has no ymin or ymax. Skipping plot..."
                )
                continue

            try:
                xmin, xmax =(
                xlimDict[xParam]["xmin"], xlimDict[xParam]["xmax"]
                )
            except:
                xmin, xmax, = ( np.nanmin(xData), np.nanmax(xData))

            try:
                finalymin, finalymax =(
                xlimDict[analysisParam]["xmin"], xlimDict[analysisParam]["xmax"]
                )
            except:
                finalymin, finalymax, = ( np.nanmin(yminlist), np.nanmax(ymaxlist))


            if (
                (np.isinf(finalymin) == True)
                or (np.isinf(finalymax) == True)
                or (np.isnan(finalymin) == True)
                or (np.isnan(finalymax) == True)
            ):
                print("[@medians_versus_plot]: Data All Inf/NaN! Skipping entry!")
                continue

            custom_xlim = (xmin, xmax)
            custom_ylim = (finalymin, finalymax)
            # xticks = [round_it(xx,2) for xx in np.linspace(min(xData),max(xData),5)]
            # custom_xlim = (min(xData),max(xData)*1.05)
            # if xParam == "R":
            #     if PARAMS[selectKeyShort]['analysisType'] == "cgm":
            #         ax.fill_betweenx([finalymin,finalymax],0,min(xData), color="tab:gray",alpha=opacityPercentiles)
            #         custom_xlim = (0,max(xData)*1.05)
            #     else:
            #         custom_xlim = (0,max(xData)*1.05)
            # ax.set_xticks(xticks)
            if ((label != "")&(legendBool == True)): ax.legend(loc="best", fontsize=fontsize)

            plt.setp(
                ax,
                ylim=custom_ylim,
                xlim=custom_xlim
            )
            # plt.tight_layout()

            if snapNumber is not None:
                if str(snapNumber).isdigit() is True:
                    SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                else:
                    SaveSnapNumber = "_" + str(snapNumber)
            else:
                SaveSnapNumber = ""

            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            if "Stars" in selectKey:
                opslaan = savePath + \
                    f"Stellar-{analysisParam}_Medians{SaveSnapNumber}"
            elif "col" in selectKey:
                opslaan = savePath + \
                    f"Projection-Mapped-{analysisParam}_Medians{SaveSnapNumber}"
            else:
                opslaan = savePath + f"{analysisParam}_Medians{SaveSnapNumber}"

            if combineMultipleOntoAxis is True: opslaan = opslaan + "-Simulations-Combined"

            plt.savefig(opslaan + ".pdf", dpi=DPI, transparent=False)
            matplotlib.rc_file_defaults()
            plt.close("all")
            print(opslaan)
            plt.close()

    if (saveFigureData is True)&(replotFromData is False)&(combineMultipleOntoAxis is False):
        print("\n"+"[@medians_versus_plot]: saveFigureData is redundant for this plot type. Save data from calculate_statistics calls instead!")

    return

#==========================================================#
##  Project Specific variants ...
#==========================================================#

def hy_plot_slices(snap,
    snapNumber,
    xsize = 10.0,
    ysize = 5.0,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    Axes=[0, 1],
    boxsize=400.0,
    pixres=0.1,
    DPI=200,
    colourmapMain=None,
    numthreads=10,
    savePathBase = "./",
):
    savePath = savePathBase + f"Plots/Slices/"
    tmp = ""

    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if colourmapMain is None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = plt.get_cmap(colourmapMain)

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["z","x","y"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # slice_nH    = snap.get_Aslice("n_H", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    #
    # slice_B   = snap.get_Aslice("B", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    nprojections = 2
    # print(np.unique(snap.type))
    print("\n" + f"Projection 1 of {nprojections}")

    slice_T = snap.get_Aslice(
        "T",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f" Projection 2 of {nprojections}")

    slice_vol = snap.get_Aslice(
        "vol",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    # ------------------------------------------------------------------------------#
    # PLOTTING TIME

    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    if titleBool is True:
        # Redshift
        redshift = snap.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
            0
        ]  # [Gyrs]
    # ==============================================================================#
    #
    #           Quad Plot for standard video
    #
    # ==============================================================================#
    print(f" Quad Plot...")

    fullTicks = [xx for xx in np.linspace(-1.0 * halfbox, halfbox, 9)]
    fudgeTicks = fullTicks[1:]

    aspect = "equal"

    # DPI Controlled by user as lower res needed for videos #
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
    )

    if titleBool is True:
        # Add overall figure plot
        TITLE = (
            r"Redshift $(z) =$"
            + f"{redshift:0.03f} "
            + " "
            + r"$t_{Lookback}=$"
            + f"{tlookback :0.03f} Gyr"
        )
        fig.suptitle(TITLE, fontsize=fontsizeTitle)

    # cmap = plt.get_cmap(colourmapMain)
    cmap = copy.copy(cmap)
    cmap.set_bad(color="grey")

    # -----------#
    # Plot Temperature #
    # -----------#
    # print("pcm1")
    ax1 = axes[0]

    pcm1 = ax1.pcolormesh(
        slice_T["x"],
        slice_T["y"],
        np.transpose(slice_T["grid"]),
        norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=10 ** (6.5)),
        cmap=cmap,
        rasterized=True,
    )

    ax1.set_title(f"Temperature Slice", fontsize=fontsize)
    cax1 = inset_axes(ax1, width="5%", height="95%", loc="right")
    fig.colorbar(pcm1, cax=cax1, orientation="vertical").set_label(
        label="T (K)", size=fontsize, weight="bold"
    )
    cax1.yaxis.set_ticks_position("left")
    cax1.yaxis.set_label_position("left")
    cax1.yaxis.label.set_color("white")
    cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax1.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
    # ax1.set_xlabel(f'{AxesLabels[Axes[0]]}"+" [kpc]"', fontsize = fontsize)
    # ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[1]

    cmapVol = cm.get_cmap("seismic")
    norm = matplotlib.colors.LogNorm(vmin = 5e-3, vmax = 5e3, clip=True)
    pcm2 = ax2.pcolormesh(
        slice_vol["x"],
        slice_vol["y"],
        np.transpose(slice_vol["grid"]),
        norm=norm,
        cmap=cmapVol,
        rasterized=True,
    )

    # cmapVol = cm.get_cmap("seismic")
    # bounds = [0.5, 2.0, 4.0, 16.0]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmapVol.N, extend="both")
    # pcm2 = ax2.pcolormesh(
    #     slice_vol["x"],
    #     slice_vol["y"],
    #     np.transpose(slice_vol["grid"]),
    #     norm=norm,
    #     cmap=cmapVol,
    #     rasterized=True,
    # )

    ax2.set_title(r"Volume Slice", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"V (kpc$^{3}$)", size=fontsize, weight="bold"
    )
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.yaxis.label.set_color("white")
    cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
    # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax2)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)


    # print("snapnum")
    # Pad snapnum with zeroes to enable easier video making
    if titleBool is True:
        fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.90)
    else:
        fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.95)

    # fig.tight_layout()

    SaveSnapNumber = str(snapNumber).zfill(4)
    savePath = savePath + f"Slice_Plot_{int(SaveSnapNumber)}.pdf" #_binary-split

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()
    matplotlib.rc_file_defaults()
    plt.close("all")
    print(f" ...done!")

    return

def hy_plot_slices_quad(snap,
    snapNumber,
    xsize = 10.0,
    ysize = 10.0,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    Axes=[0, 1],
    boxsize=400.0,
    pixres=0.1,
    DPI=200,
    colourmapMain=None,
    numthreads=10,
    savePathBase = "./",
):
    savePath = savePathBase + "Plots/Slices/"

    tmp = ""

    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if colourmapMain is None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = plt.get_cmap(colourmapMain)

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["z","x","y"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#
    # PLOTTING TIME
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    if titleBool is True:
        # Redshift
        redshift = snap.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
            0
        ]  # [Gyrs]
    # ==============================================================================#
    #
    #           Quad Plot for standard video
    #
    # ==============================================================================#
    print(f" Quad Plot...")

    fullTicks = [xx for xx in np.linspace(-1.0 * halfbox, halfbox, 9)]
    fudgeTicks = fullTicks[1:]

    aspect = "equal"

    # DPI Controlled by user as lower res needed for videos #
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
    )

    nprojections = 4
    # print(np.unique(snap.type))
    print("\n" + f"[@{int(snapNumber)}]: Projection 1 of {nprojections}")
    slice_T = snap.get_Aslice(
        "T",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 2 of {nprojections}")

    slice_tcool = snap.get_Aslice(
        "tcool",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 3 of {nprojections}")

    slice_nH = snap.get_Aslice(
        "n_H",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 4 of {nprojections}")

    # slice_gz = snap.get_Aslice(
    #     "gz",
    #     box=[boxsize, boxsize],
    #     center=imgcent,
    #     nx=int(boxsize / pixres),
    #     ny=int(boxsize / pixres),
    #     axes=Axes,
    #     proj=False,
    #     numthreads=numthreads,
    # )

    slice_cl = snap.get_Aslice(
        "cool_length",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )


    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
    )


    if titleBool is True:
        # Add overall figure plot
        TITLE = (
            r"Redshift $(z) =$"
            + f"{redshift:0.03f} "
            + " "
            + r"$t_{Lookback}=$"
            + f"{tlookback :0.03f} Gyr"
        )
        fig.suptitle(TITLE, fontsize=fontsizeTitle)

    # cmap = plt.get_cmap(colourmapMain)
    cmap = copy.copy(cmap)
    cmap.set_bad(color="grey")

    # -----------#
    # Plot Temperature #
    # -----------#
    # print("pcm1")
    ax1 = axes[0,0]

    pcm1 = ax1.pcolormesh(
        slice_T["x"],
        slice_T["y"],
        np.transpose(slice_T["grid"]),
        norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=10 ** (6.5)),
        cmap=cmap,
        rasterized=True,
    )

    ax1.set_title(f"Temperature Slice", fontsize=fontsize)
    cax1 = inset_axes(ax1, width="5%", height="95%", loc="right")
    fig.colorbar(pcm1, cax=cax1, orientation="vertical").set_label(
        label="T (K)", size=fontsize, weight="bold"
    )
    cax1.yaxis.set_ticks_position("left")
    cax1.yaxis.set_label_position("left")
    cax1.yaxis.label.set_color("white")
    cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax1.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
    # ax1.set_xlabel(f'{AxesLabels[Axes[0]]}"+" [kpc]"', fontsize = fontsize)
    # ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[0,1]

    pcm2 = ax2.pcolormesh(
        slice_tcool["x"],
        slice_tcool["y"],
        np.transpose(slice_tcool["grid"]),
        norm=matplotlib.colors.LogNorm(vmin = (10)**(-3.5), vmax = 1e2),
        cmap=cmap,
        rasterized=True,
    )

    # cmapVol = cm.get_cmap("seismic")
    # bounds = [0.5, 2.0, 4.0, 16.0]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmapVol.N, extend="both")
    # pcm2 = ax2.pcolormesh(
    #     slice_vol["x"],
    #     slice_vol["y"],
    #     np.transpose(slice_vol["grid"]),
    #     norm=norm,
    #     cmap=cmapVol,
    #     rasterized=True,
    # )

    ax2.set_title(r"Cooling Time Slice", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"t$_{\mathrm{Cool}}$ (Gyr)", size=fontsize, weight="bold"
    )
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.yaxis.label.set_color("white")
    cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
    # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax2)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)

    # -----------#
    # Plot Metallicity #
    # -----------#
    # print("pcm3")
    ax3 = axes[1, 0]

    pcm3 = ax3.pcolormesh(
        slice_cl["x"],
        slice_cl["y"],
        np.transpose(slice_cl["grid"]),
        norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e4),
        cmap=cmap,
        rasterized=True,
    )

    # ax3.set_title(f"Metallicity Slice", y=-0.2, fontsize=fontsize)
    #
    # cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    # fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
    #     label=r"$Z/Z_{\odot}$", size=fontsize, weight="bold"
    # )
    ax3.set_title(f"Cooling Length Slice", y=-0.2, fontsize=fontsize)

    cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
        label=r"$l_{cool}$ (kpc)", size=fontsize, weight="bold"
    )

    cax3.yaxis.set_ticks_position("left")
    cax3.yaxis.set_label_position("left")
    cax3.yaxis.label.set_color("white")
    cax3.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax3.set_ylabel(f"{AxesLabels[Axes[1]]} " + r" (kpc)", fontsize=fontsize)
    ax3.set_xlabel(f"{AxesLabels[Axes[0]]} " + r" (kpc)", fontsize=fontsize)

    # ax3.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax3)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)

    # -----------#
    # Plot Magnetic Field Projection #
    # -----------#
    # print("pcm4")
    ax4 = axes[1, 1]

    pcm4 = ax4.pcolormesh(
        slice_nH["x"],
        slice_nH["y"],
        np.transpose(slice_nH["grid"]),
        norm=matplotlib.colors.LogNorm(vmin=1e-7, vmax=1e-1),
        cmap=cmap,
        rasterized=True,
    )

    ax4.set_title(r"HI Number Density Slice",
                  y=-0.2, fontsize=fontsize)

    cax4 = inset_axes(ax4, width="5%", height="95%", loc="right")
    fig.colorbar(pcm4, cax=cax4, orientation="vertical").set_label(
        label=r"n$_{H}$ (cm$^{-3}$)", size=fontsize, weight="bold"
    )
    cax4.yaxis.set_ticks_position("left")
    cax4.yaxis.set_label_position("left")
    cax4.yaxis.label.set_color("white")
    cax4.tick_params(axis="y", colors="white", labelsize=fontsize)

    # ax4.set_ylabel(f'{AxesLabels[Axes[1]]} "+r" (kpc)"', fontsize=fontsize)
    ax4.set_xlabel(f"{AxesLabels[Axes[0]]} " + r" (kpc)", fontsize=fontsize)
    # ax4.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax4)
    plt.xticks(fudgeTicks)
    plt.yticks(fullTicks)

    # print("snapnum")
    # Pad snapnum with zeroes to enable easier video making
    if titleBool is True:
        fig.subplots_adjust(wspace=0.0 ,hspace=0.0, top=0.90)
    else:
        fig.subplots_adjust(wspace=0.0 ,hspace=0.0, top=0.95)

    # fig.tight_layout()

    SaveSnapNumber = str(snapNumber).zfill(4)
    savePath = savePath + f"Slice_Plot_Quad_{int(SaveSnapNumber)}.pdf" #_binary-split

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()
    matplotlib.rc_file_defaults()
    plt.close("all")
    print(f" ...done!")

    return

def hy_plot_projections(snap,
    snapNumber,
    xsize = 10.0,
    ysize = 5.0,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    Axes=[0, 1],
    zAxis = [2],
    boxsize=400.0,
    boxlos=50.0,
    pixreslos=0.3,
    pixres=0.3,
    DPI=200,
    colourmapMain=None,
    numthreads=10,
    savePathBase = "./",
):

    savePath = savePathBase + "Plots/Projections/"
    tmp = ""

    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if colourmapMain is None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = plt.get_cmap(colourmapMain)

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["z","x","y"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    nprojections = 3
    # print(np.unique(snap.type))
    print("\n" + f"Projection 1 of {nprojections}")

    proj_T = snap.get_Aslice(
        "Tdens",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numthreads,
    )

    print("\n" + f"Projection 2 of {nprojections}")

    proj_dens = snap.get_Aslice(
        "rho_rhomean",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numthreads,
    )

    print("\n" + f" Projection 3 of {nprojections}")

    proj_vol = snap.get_Aslice(
        "vol",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numthreads,
    )

    # ------------------------------------------------------------------------------#
    # PLOTTING TIME
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    if titleBool is True:
        # Redshift
        redshift = snap.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
            0
        ]  # [Gyrs]
    # ==============================================================================#
    #
    #           Quad Plot for standard video
    #
    # ==============================================================================#
    print(f" Quad Plot...")

    fullTicks = [xx for xx in np.linspace(-1.0 * halfbox, halfbox, 9)]
    fudgeTicks = fullTicks[1:]

    aspect = "equal"

    # DPI Controlled by user as lower res needed for videos #
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
    )

    if titleBool is True:
        # Add overall figure plot
        TITLE = (
            r"Redshift $(z) =$"
            + f"{redshift:0.03f} "
            + " "
            + r"$t_{Lookback}=$"
            + f"{tlookback :0.03f} Gyr"
        )
        fig.suptitle(TITLE, fontsize=fontsizeTitle)

    # cmap = plt.get_cmap(colourmapMain)
    cmap = copy.copy(cmap)
    cmap.set_bad(color="grey")

    # -----------#
    # Plot Temperature #
    # -----------#
    # print("pcm1")
    ax1 = axes[0]

    pcm1 = ax1.pcolormesh(
        proj_T["x"],
        proj_T["y"],
        np.transpose(proj_T["grid"] / proj_dens["grid"]),
        norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=10 ** (6.5)),
        cmap=cmap,
        rasterized=True,
    )

    ax1.set_title(f"Temperature Projection", fontsize=fontsize)
    cax1 = inset_axes(ax1, width="5%", height="95%", loc="right")
    fig.colorbar(pcm1, cax=cax1, orientation="vertical").set_label(
        label="T (K)", size=fontsize, weight="bold"
    )
    cax1.yaxis.set_ticks_position("left")
    cax1.yaxis.set_label_position("left")
    cax1.yaxis.label.set_color("white")
    cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax1.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
    # ax1.set_xlabel(f'{AxesLabels[Axes[0]]}"+" [kpc]"', fontsize = fontsize)
    # ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[1]

    cmapVol = cm.get_cmap("seismic")
    norm = matplotlib.colors.LogNorm(vmin = 5e-3, vmax = 5e3, clip=True)
    pcm2 = ax2.pcolormesh(
        proj_vol["x"],
        proj_vol["y"],
        np.transpose(proj_vol["grid"]) / int(boxlos / pixreslos),
        norm=norm,
        cmap=cmapVol,
        rasterized=True,
    )

    # cmapVol = cm.get_cmap("seismic")
    # bounds = [0.5, 2.0, 4.0, 16.0]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmapVol.N, extend="both")
    # pcm2 = ax2.pcolormesh(
    #     proj_vol["x"],
    #     proj_vol["y"],
    #     np.transpose(proj_vol["grid"]),
    #     norm=norm,
    #     cmap=cmapVol,
    #     rasterized=True,
    # )

    ax2.set_title(r"Volume Projection", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"V (kpc$^{3}$)", size=fontsize, weight="bold"
    )
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.yaxis.label.set_color("white")
    cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
    # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax2)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)


    # print("snapnum")
    # Pad snapnum with zeroes to enable easier video making
    if titleBool is True:
        fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.90)
    else:
        fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.95)

    # fig.tight_layout()

    SaveSnapNumber = str(snapNumber).zfill(4)
    savePath = savePath + f"Projection_Plot_{int(SaveSnapNumber)}.pdf" #_binary-split

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()

    matplotlib.rc_file_defaults()
    plt.close("all")
    print(f" ...done!")

    return

def hy_load_pdf_versus_plot_data(
    selectKeysList,
    loadPathList,
    snapRange,
    weightKeys = None,
    xParams = ["T"],
    cumulative = False,
    loadPathBase = "./",
    loadPathSuffix = "",
    SFR = False,
    normalise = False,
    stack = True,
    verbose = False,
    hush = False,
    ):

    out = {}

    for loadpath,selectKey in zip(loadPathList,selectKeysList):
        print(selectKey)

        datapath = loadPathBase + loadpath

        tmp = load_pdf_versus_plot_data(
            snapRange,
            weightKeys = weightKeys,
            xParams = xParams,
            cumulative = cumulative,
            loadPathBase = datapath,
            loadPathSuffix= loadPathSuffix,
            SFR = SFR,
            normalise = normalise,
            stack = stack,
            verbose = verbose,
            hush = hush,
            )
        
        out.update({selectKey : copy.deepcopy(tmp)})

    return out

def hy_load_phase_plot_data(
    selectKeysList,
    loadPathList,
    snapRange,
    yParams = ["T"],
    xParams = ["rho_rhomean","R"],
    weightKeys = ["mass","vol"],
    loadPathBase = "./",
    stack = True,
    verbose = False,
    ):

    out = {}

    for loadpath,selectKey in zip(loadPathList,selectKeysList):
        print(selectKey)

        datapath = loadPathBase + loadpath

        tmp = load_phase_plot_data(
            snapRange,
            yParams = yParams,
            xParams = xParams,
            weightKeys = weightKeys,
            loadPathBase = datapath,
            stack = stack,
            verbose = verbose,
            )
        
        out.update({selectKey : copy.deepcopy(tmp)})

    return out

def hy_load_statistics_data(
    selectKeysList,
    loadPathList,
    snapRange,
    loadPathBase = "./",
    loadFile = "statsDict",
    fileType = ".h5",
    stack = True,
    verbose = False,
    ):

    out = {}

    for loadpath,selectKey in zip(loadPathList,selectKeysList):
        print(selectKey)

        datapath = loadPathBase + loadpath + "Data"

        tmp = load_statistics_data(
            snapRange,
            loadPathBase = datapath,
            loadFile = loadFile,
            fileType = fileType,
            stack = stack,
            verbose = verbose,
            )
        updatedtmp = {}
        for key,val in tmp.items():
            updatedtmp.update({selectKey :copy.deepcopy(val)})
        out.update({selectKey : copy.deepcopy(updatedtmp)})

    return out

def get_linestyles_and_colours(selectKeysList,colourmapMain,colourGroupBy,linestyleGroupBy,lastColourOffset = 0.10):
    from itertools import cycle

    cmap = matplotlib.cm.get_cmap(colourmapMain)
    Ncolours = len(colourGroupBy)

    colourList = []

    if (Ncolours>1):
        for ii in range(0,Ncolours):
            colour = cmap((float(ii) / float(Ncolours))-lastColourOffset)
            colourList.append(colour)
        colourList.append(cmap(1.00))
    else:
        for ii in range(0,len(selectKeysList)+1):
            colour = cmap((float(ii) / float(len(selectKeysList)+1))-lastColourOffset)
            colourList.append(colour)
        colourList.append(cmap(1.00))

    lineStyles = ["solid","dotted","dashed","dashdot"]
    linecycler = cycle(lineStyles)
    linestyleList = [next(linecycler) for ii in range(0,len(linestyleGroupBy)+1)]

    styleDict = {skey : {"colour" : colourList[-1], "linestyle" : linestyleList[-1]} for skey in selectKeysList}

    if (Ncolours>1):
        for colour,colourGroup in zip(colourList[:-1],colourGroupBy):
            for selectKey in selectKeysList:
                listed = list(selectKey)
                if colourGroup in listed:
                    styleDict[selectKey].update({"colour" : colour})
    else:
        for colour,selectKey in zip(colourList[:-1],selectKeysList):
            styleDict[selectKey].update({"colour" : colour})

    for linestyle,linestyleGroup in zip(linestyleList[:-1],linestyleGroupBy):
        for selectKey in selectKeysList:
            listed = list(selectKey)
            if linestyleGroup in listed:
                styleDict[selectKey].update({"linestyle" : linestyle})

    return styleDict

def cr_medians_versus_plot(
    statsDict,
    CRPARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    yParam=None,
    xParam="R",
    titleBool=False,
    legendBool = True,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize = 13,
    fontsizeTitle = 14,
    linewidth=1.0,
    opacityPercentiles=0.25,
    colourmapMain="tab10",
    savePathBase ="./",
    savePathBaseFigureData = "./",
    inplace = False,
    saveFigureData = False,
    allowPlotsWithoutxlimits = False,
    replotFromData = False,
    combineMultipleOntoAxis = False,
    selectKeysList = None,
    styleDict = None,
    hush = False,
    ):

    if combineMultipleOntoAxis is False:
        for (ii, (selectKey, simDict)) in enumerate(statsDict.items()):
            if ("Stars" in selectKey) | ("col" in selectKey) :
                selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
            else:
                selectKeyShort = selectKey

            selectKeyShortest = selectKeyShort[:-1]

            if yParam is None:
                yParam = CRPARAMS[selectKeyShort]["saveParams"]
                if hush is False:
                    print(f"[cr_medians_versus_plot]: WARNING! No yParam provided so default of all 'saveParams' being used. This may cause errors if any of 'saveParams' do not have limits set in xlimDict...")
            else:
                pass

            if (allowPlotsWithoutxlimits == False) & (replotFromData == False):
                ##check_params_are_in_xlimDict(xlimDict, yParam) <-- Not needed, as y-axis range not used when combining data
                check_params_are_in_xlimDict(xlimDict, [xParam])

            fontsize = CRPARAMS[selectKeyShort]["fontsize"]
            fontsizeTitle = CRPARAMS[selectKeyShort]["fontsizeTitle"]

            loadpath = CRPARAMS[selectKeyShort]["simfile"]
            if loadpath is not None:
                print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")

                plotableDict = {selectKeyShortest : simDict}
                tmpCRPARAMS = {selectKeyShortest : copy.deepcopy(CRPARAMS[selectKeyShort])}

                if ("Stars" in selectKey):
                    simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}/"+f"{tmpCRPARAMS[selectKeyShortest]['resolution']}/{tmpCRPARAMS[selectKeyShortest]['CR_indicator']}"+f"{tmpCRPARAMS[selectKeyShortest]['no-alfven_indicator']}/Stars/"
                elif ("col" in selectKey):
                    simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}/"+f"{tmpCRPARAMS[selectKeyShortest]['resolution']}/{tmpCRPARAMS[selectKeyShortest]['CR_indicator']}"+f"{tmpCRPARAMS[selectKeyShortest]['no-alfven_indicator']}/Col-Projection-Mapped/"
                else:
                    simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}/"+f"{tmpCRPARAMS[selectKeyShortest]['resolution']}/{tmpCRPARAMS[selectKeyShortest]['CR_indicator']}"+f"{tmpCRPARAMS[selectKeyShortest]['no-alfven_indicator']}/"


                savePath = savePathBase + simSavePath
                tmp = ""

                for savePathChunk in savePath.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass

                savePathData = savePathBaseFigureData + simSavePath

                tmp = ""
                for savePathChunk in savePathData.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass
                    else:
                        pass

                medians_versus_plot(
                    plotableDict,
                    tmpCRPARAMS,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    yParam=yParam,
                    xParam=xParam,
                    titleBool=titleBool,
                    legendBool = legendBool,
                    DPI=DPI,
                    xsize=xsize,
                    ysize=ysize,
                    fontsize = fontsize,
                    fontsizeTitle = fontsizeTitle,
                    linewidth=linewidth,
                    opacityPercentiles=opacityPercentiles,
                    colourmapMain=colourmapMain,
                    savePathBase = savePath,
                    savePathBaseFigureData = savePathBaseFigureData,
                    allowPlotsWithoutxlimits = allowPlotsWithoutxlimits,
                    inplace = inplace,
                    saveFigureData=saveFigureData,
                    replotFromData = replotFromData,
                    combineMultipleOntoAxis = combineMultipleOntoAxis,
                    selectKeysList = selectKeysList,
                    styleDict = styleDict,
                    hush = hush,
                )
    else:
        for (ii, (selectKey, simDict)) in enumerate(statsDict.items()):
            #Only want to plot combined version once
            if (ii>0):return

            if ("Stars" in selectKey) | ("col" in selectKey) :
                selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
            else:
                selectKeyShort = selectKey

            if len(selectKeyShort)>2:
                selectKeyShortest = selectKeyShort[:-1]
            else:
                selectKeyShortest = selectKeyShort

            if yParam is None:
                yParam = CRPARAMS[selectKeyShort]["saveParams"]
                if hush is False:
                    print(f"[cr_medians_versus_plot]: WARNING! No yParam provided so default of all 'saveParams' being used. This may cause errors if any of 'saveParams' do not have limits set in xlimDict...")
            else:
                pass

            if (allowPlotsWithoutxlimits == False) & (replotFromData == False):
                ##check_params_are_in_xlimDict(xlimDict, yParam) <-- Not needed, as y-axis range not used when combining data
                check_params_are_in_xlimDict(xlimDict, [xParam])

            fontsize = CRPARAMS[selectKeyShort]["fontsize"]
            fontsizeTitle = CRPARAMS[selectKeyShort]["fontsizeTitle"]

            loadpath = CRPARAMS[selectKeyShort]["simfile"]
            if loadpath is not None:
                print(f"Starting combined Medians profile plot")
                
                #pass in the plotable version to Medians plot call without changes made...
                plotableDict = statsDict


                tmpCRPARAMS = {selectKeyShortest : copy.deepcopy(CRPARAMS[selectKeyShort])}
                if ("Stars" in selectKey):
                    simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}/Stars/"
                elif ("col" in selectKey):
                    simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}/Col-Projection-Mapped/"
                else:
                    simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}"


                savePath = savePathBase + simSavePath
                tmp = ""

                for savePathChunk in savePath.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass

                savePathData = savePathBaseFigureData + simSavePath

                tmp = ""
                for savePathChunk in savePathData.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass
                    else:
                        pass

                medians_versus_plot(
                    plotableDict,
                    CRPARAMS,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    yParam=yParam,
                    xParam=xParam,
                    titleBool=titleBool,
                    legendBool = legendBool,
                    DPI=DPI,
                    xsize=xsize,
                    ysize=ysize,
                    fontsize = fontsize,
                    fontsizeTitle = fontsizeTitle,
                    linewidth=linewidth,
                    opacityPercentiles=opacityPercentiles,
                    colourmapMain=colourmapMain,
                    savePathBase = savePath,
                    savePathBaseFigureData = savePathBaseFigureData,
                    allowPlotsWithoutxlimits = allowPlotsWithoutxlimits,
                    inplace = inplace,
                    saveFigureData=saveFigureData,
                    replotFromData = replotFromData,
                    combineMultipleOntoAxis = combineMultipleOntoAxis,
                    selectKeysList = selectKeysList,
                    styleDict = styleDict,
                    hush = hush,
                )
    return

def cr_pdf_versus_plot(
    dataDict,
    CRPARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    weightKeys = None,
    xParams = ["T"],
    titleBool=False,
    legendBool=True,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize=13,
    fontsizeTitle=14,
    linewidth=1.0,
    Nbins=250,
    ageWindow=None,
    cumulative = False,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    allSavePathsSuffix = "",
    saveFigureData = True,
    SFR = False,    
    forceLogPDF = False,
    normalise = False,
    verbose = False,
    inplace = False,
    allowPlotsWithoutxlimits = False,
    exclusions = ["Redshift", "Lookback", "Snap", "Rvir"],
    replotFromData = False,
    combineMultipleOntoAxis = False,
    selectKeysList = None,
    styleDict = None,
    hush = False,
):

    if combineMultipleOntoAxis is False:
        for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
            if ("Stars" in selectKey) | ("col" in selectKey) :
                selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
            else:
                selectKeyShort = selectKey

            if (allowPlotsWithoutxlimits == False) & (SFR == False) & (replotFromData == False):
                ##check_params_are_in_xlimDict(xlimDict, weightKeys) <-- Not needed, as y-axis range not used when combining data
                check_params_are_in_xlimDict(xlimDict, xParams)

            fontsize = CRPARAMS[selectKeyShort]["fontsize"]
            fontsizeTitle = CRPARAMS[selectKeyShort]["fontsizeTitle"]

            loadpath = CRPARAMS[selectKeyShort]["simfile"]
            if loadpath is not None:
                print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")
                
                if replotFromData is False:
                    plotableDict = copy.deepcopy(simDict)
                    for excl in exclusions:
                        if excl in list(plotableDict.keys()):
                            plotableDict.pop(excl)
                else:
                    plotableDict = copy.deepcopy(simDict)


                if ("Stars" in selectKey):
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
                elif ("col" in selectKey):
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
                else:
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"

                savePath = savePathBase + simSavePath
                tmp = ""

                for savePathChunk in savePath.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass

                savePathData = savePathBaseFigureData + simSavePath

                tmp = ""
                for savePathChunk in savePathData.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass
                    else:
                        pass

                pdf_versus_plot(
                    inputDict = plotableDict,
                    ylabel = ylabel,
                    xlimDict = xlimDict,
                    logParameters = CRPARAMS[selectKeyShort]["logParameters"],
                    snapNumber = snapNumber,
                    weightKeys = weightKeys,
                    xParams = xParams,
                    titleBool = titleBool,
                    legendBool = legendBool,
                    DPI = DPI,
                    xsize = xsize,
                    ysize = ysize,
                    fontsize = fontsize,
                    fontsizeTitle = fontsizeTitle,
                    linewidth=linewidth,
                    Nbins = Nbins,
                    ageWindow = ageWindow,
                    cumulative = cumulative,
                    savePathBase = savePath,#"./",
                    savePathBaseFigureData = savePathData,#False,
                    allSavePathsSuffix = allSavePathsSuffix,
                    saveFigureData = saveFigureData,
                    SFR = SFR,
                    forceLogPDF = forceLogPDF,
                    normalise = normalise,
                    replotFromData = replotFromData,
                    combineMultipleOntoAxis = combineMultipleOntoAxis,
                    selectKeysList = selectKeysList,
                    verbose = verbose,
                    allowPlotsWithoutxlimits = allowPlotsWithoutxlimits,
                    inplace = inplace,
                    styleDict = styleDict,
                    hush = hush,
                )
    else:
        for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
            #Only want to plot combined version once
            if (ii>0):return

            if ("Stars" in selectKey) | ("col" in selectKey) :
                selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
            else:
                selectKeyShort = selectKey

            if (allowPlotsWithoutxlimits == False) & (SFR == False) & (replotFromData == False):
                ##check_params_are_in_xlimDict(xlimDict, weightKeys) <-- Not needed, as y-axis range not used when combining data
                check_params_are_in_xlimDict(xlimDict, xParams)

            fontsize = CRPARAMS[selectKeyShort]["fontsize"]
            fontsizeTitle = CRPARAMS[selectKeyShort]["fontsizeTitle"]

            loadpath = CRPARAMS[selectKeyShort]["simfile"]
            if loadpath is not None:
                print(f"Starting combined pdf")
                
                #pass in the plotable version to pdf versus plot call without changes made...
                plotableDict = dataDict

                if ("Stars" in selectKey):
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/Stars/"
                elif ("col" in selectKey):
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/Col-Projection-Mapped/"
                else:
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"


                savePath = savePathBase + simSavePath
                tmp = ""

                for savePathChunk in savePath.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass

                savePathData = savePathBaseFigureData + simSavePath

                tmp = ""
                for savePathChunk in savePathData.split("/")[:-1]:
                    tmp += savePathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass
                    else:
                        pass

                pdf_versus_plot(
                    inputDict = plotableDict,
                    ylabel = ylabel,
                    xlimDict = xlimDict,
                    logParameters = CRPARAMS[selectKeyShort]["logParameters"],
                    snapNumber = snapNumber,
                    weightKeys = weightKeys,
                    xParams = xParams,
                    titleBool = titleBool,
                    legendBool = legendBool,
                    DPI = DPI,
                    xsize = xsize,
                    ysize = ysize,
                    fontsize = fontsize,
                    fontsizeTitle = fontsizeTitle,
                    linewidth=linewidth,
                    Nbins = Nbins,
                    ageWindow = ageWindow,
                    cumulative = cumulative,
                    savePathBase = savePath,#"./",
                    savePathBaseFigureData = savePathData,
                    allSavePathsSuffix = allSavePathsSuffix,
                    saveFigureData = saveFigureData,
                    SFR = SFR,
                    forceLogPDF = forceLogPDF,
                    normalise = normalise,
                    replotFromData = replotFromData,
                    combineMultipleOntoAxis = combineMultipleOntoAxis,
                    selectKeysList = selectKeysList,
                    verbose = verbose,
                    allowPlotsWithoutxlimits = allowPlotsWithoutxlimits,
                    inplace = inplace,
                    styleDict = styleDict,
                    hush = hush,
                )
    return

def cr_load_pdf_versus_plot_data(
    selectKeysList,
    CRPARAMS,
    snapRange,
    weightKeys = None,
    xParams = ["T"],
    cumulative = False,
    loadPathBase = "./",
    loadPathSuffix = "",
    SFR = False,
    normalise = False,
    selectKeyLen=3,
    delimiter="-",
    stack = True,
    verbose = False,
    hush = False,
    ):

    out = {}

    for selectKey in selectKeysList:
        if ("Stars" in selectKey) | ("col" in selectKey) :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
        else:
            selectKeyShort = selectKey

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")
            
            # for excl in exclusions:
            #     plotableDict.pop(excl)

            if ("Stars" in selectKey):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            elif ("col" in selectKey):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
            else:
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"


            datapath = loadPathBase + simSavePath

            tmp = load_pdf_versus_plot_data(
                snapRange,
                weightKeys = weightKeys,
                xParams = xParams,
                cumulative = cumulative,
                loadPathBase = datapath,
                loadPathSuffix= loadPathSuffix,
                SFR = SFR,
                normalise = normalise,
                stack = stack,
                selectKeyLen = selectKeyLen,
                delimiter = delimiter,
                verbose = verbose,
                hush = hush,
                )
            
            out.update({selectKey : copy.deepcopy(tmp)})

    return out


def cr_load_slice_plot_data(
    selectKeysList,
    CRPARAMS,
    snapNumber,
    sliceParam = None,
    Axes = [0,1],
    averageAcrossAxes = False,
    projection=False,
    loadPathBase = "./",
    loadPathSuffix = "",
    selectKeyLen=3,
    delimiter="-",
    stack = None,
    allowFindOtherAxesData = False,
    verbose = False,
    hush = False,
    ):

    from itertools import permutations

    out = {}

    #Legacy labelling... 
    AxesLabels = ["x","y","z"]
    

    for selectKey in selectKeysList:
        if ("Stars" in selectKey) | ("col" in selectKey) :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
        else:
            selectKeyShort = selectKey

        if snapNumber is not None:
            if type(snapNumber) == int:
                SaveSnapNumber = "_" + str(snapNumber).zfill(4)
            else:
                SaveSnapNumber = "_" + str(snapNumber)
        else:
            SaveSnapNumber = ""
    
        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            inner = {}
            for param in sliceParam:
                fileFound = False
                if verbose: print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")

                paramSplitList = param.split("_")


                if paramSplitList[-1] == "col":
                    ## If _col variant is called we want to calculate a projection of the non-col parameter
                    ## Thus, we force projection to now be true, and incorporate a dummy variable tmpsliceParam
                    ## to force plots to generate non-col variants but save output as column density version

                    tmpsliceParam = "_".join(paramSplitList[:-1])
                    projection = True
                else:
                    tmpsliceParam = param
                    projection = False

                if ("Stars" in selectKey):
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
                elif ("col" in selectKey):
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
                else:
                    simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"

                if projection is False:
                    loadPathFigureData = simSavePath + "/Plots/Slices/" + f"Slice_Plot_{AxesLabels[Axes[0]]}-{AxesLabels[Axes[1]]}_{param}{SaveSnapNumber}"
                else:
                    loadPathFigureData = simSavePath + "/Plots/Slices/" + f"Projection_Plot_{AxesLabels[Axes[0]]}-{AxesLabels[Axes[1]]}_{param}{SaveSnapNumber}" 

                datapath = loadPathBase + loadPathFigureData
                if verbose: print("\n"+f"[@{int(snapNumber)}]: Loading {loadPathFigureData}")
                try:
                    tmp = tr.hdf5_load(
                        datapath+"_data.h5",
                        selectKeyLen = selectKeyLen,
                        delimiter=delimiter,
                        )
                    inner.update(tmp)
                    fileFound = True
                except Exception as e:
                    if hush == False: print(str(e))
                    if hush == False: print("File not found! Skipping ...")
                    fileFound = False
                    if allowFindOtherAxesData == False :
                        continue

                    
                if ((fileFound == False) & (allowFindOtherAxesData == True)):
                        possibleAxes = permutations(range(len(AxesLabels)),2)
                        print(f"[@cr_load_slice_plot_data]: Data not found for Axes selection provided for"
                              + "\n"
                              + f"{loadPathFigureData}"
                              + "\n"
                              +"Attempting to recover other single-axis data!")
                        for ax0, ax1 in possibleAxes:
                            print(f"[@cr_load_slice_plot_data]: Attempting {AxesLabels[ax0]} {AxesLabels[ax1]} data load...")
                            paramSplitList = param.split("_")


                            if paramSplitList[-1] == "col":
                                ## If _col variant is called we want to calculate a projection of the non-col parameter
                                ## Thus, we force projection to now be true, and incorporate a dummy variable tmpsliceParam
                                ## to force plots to generate non-col variants but save output as column density version

                                tmpsliceParam = "_".join(paramSplitList[:-1])
                                projection = True
                            else:
                                tmpsliceParam = param
                                projection = False

                            if ("Stars" in selectKey):
                                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
                            elif ("col" in selectKey):
                                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
                            else:
                                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"

                            if projection is False:
                                loadPathFigureData = simSavePath + "/Plots/Slices/" + f"Slice_Plot_{AxesLabels[ax0]}-{AxesLabels[ax1]}_{param}{SaveSnapNumber}"
                            else:
                                loadPathFigureData = simSavePath + "/Plots/Slices/" + f"Projection_Plot_{AxesLabels[ax0]}-{AxesLabels[ax1]}_{param}{SaveSnapNumber}" 

                            datapath = loadPathBase + loadPathFigureData
                            if verbose: print("\n"+f"[@{int(snapNumber)}]: Loading {loadPathFigureData}")
                            try:
                                tmp = tr.hdf5_load(
                                    datapath+"_data.h5",
                                    selectKeyLen = selectKeyLen,
                                    delimiter=delimiter,
                                    )
                                inner.update(tmp)
                                print(f"[@cr_load_slice_plot_data]: {AxesLabels[ax0]} {AxesLabels[ax1]} data load successful!")
                                fileFound = True
                                break
                            except Exception as e:
                                fileFound = False
                                if hush == False: print(str(e))
                                if hush == False: print("File not found! Skipping ...")
                                
                            
            out.update({selectKey : copy.deepcopy(inner)})

    return out

def cr_load_phase_plot_data(
    selectKeysList,
    CRPARAMS,
    snapRange,
    yParams = ["T"],
    xParams = ["rho_rhomean","R"],
    weightKeys = ["mass","vol"],
    loadPathBase = "./",
    stack = True,
    selectKeyLen=3,
    delimiter="-",
    verbose = False,
    ):

    out = {}

    for selectKey in selectKeysList:
        if ("Stars" in selectKey) | ("col" in selectKey) :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
        else:
            selectKeyShort = selectKey

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")
            
            # for excl in exclusions:
            #     plotableDict.pop(excl)

            if ("Stars" in selectKey):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            elif ("col" in selectKey):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
            else:
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"


            datapath = loadPathBase + simSavePath

            tmp = load_phase_plot_data(
                snapRange,
                yParams = yParams,
                xParams = xParams,
                weightKeys = weightKeys,
                loadPathBase = datapath,
                stack = stack,
                selectKeyLen = selectKeyLen,
                delimiter = delimiter,
                verbose = verbose,
                )
            
            out.update({selectKey : copy.deepcopy(tmp)})

    return out

def cr_load_statistics_data(
    selectKeysList,
    CRPARAMS,
    snapRange,
    loadPathBase = "./",
    loadFile = "statsDict",
    fileType = ".h5",
    stack = True,
    selectKeyLen=3,
    delimiter="-",
    verbose = False,
    ):

    out = {}

    for selectKey in selectKeysList:
        if ("Stars" in selectKey) | ("col" in selectKey) :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
        else:
            selectKeyShort = selectKey

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")
            
            # for excl in exclusions:
            #     plotableDict.pop(excl)

            # if ("Stars" in selectKey):
            #     simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            # elif ("col" in selectKey):
            #     simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
            # else:
            simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"


            datapath = loadPathBase + simSavePath + "CR-Data"

            tmp = load_statistics_data(
                snapRange,
                loadPathBase = datapath,
                loadFile = loadFile,
                fileType = fileType,
                stack = stack,
                selectKeyLen = selectKeyLen,
                delimiter = delimiter,
                verbose = verbose,
                )
            
            out.update({selectKey : copy.deepcopy(tmp)})

    return out

def cr_phase_plot(
    dataDict,
    CRPARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    yParams = ["T"],
    xParams = ["rho_rhomean","R"],
    colourBarKeys = ["mass","vol"],
    weightKeys = None,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    DPI=200,
    xsize=8.0,
    ysize=8.0, #lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
    colourmapMain="plasma",
    Nbins=250,
    saveFigureData = True,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    verbose = False,
    inplace = False,
    exclusions = ["Redshift", "Lookback", "Snap", "Rvir"],
    allowPlotsWithoutxlimits = False,
    replotFromData = False,
    hush = False,
):


    # ------------------------------------------------------------------------------#
    #               PLOTTING
    #
    # ------------------------------------------------------------------------------#
    for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
        if ("Stars" in selectKey) | ("col" in selectKey) :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
            # print("[@phases_plot]: WARNING! Stars not supported! Skipping!")
            # continue
        else:
            selectKeyShort = selectKey

        if allowPlotsWithoutxlimits == False:
            check_params_are_in_xlimDict(xlimDict, xParams)
            check_params_are_in_xlimDict(xlimDict, yParams)

        fontsize = CRPARAMS[selectKeyShort]["fontsize"]
        fontsizeTitle = CRPARAMS[selectKeyShort]["fontsizeTitle"]

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            if verbose: print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")                    # Create a plot for each Temperature

            plotableDict = copy.deepcopy(simDict)
            for excl in exclusions:
                if excl in list(plotableDict.keys()):
                    plotableDict.pop(excl)

            if ("Stars" in selectKey):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            elif ("col" in selectKey):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
            else:
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"

            savePath = savePathBase + simSavePath
            tmp = ""
            for savePathChunk in savePath.split("/")[:-1]:
                tmp += savePathChunk + "/"
                try:
                    os.mkdir(tmp)
                except:
                    pass

            savePathData = savePathBaseFigureData + simSavePath
            tmp = ""
            for savePathChunk in savePathData.split("/")[:-1]:
                tmp += savePathChunk + "/"
                try:
                    os.mkdir(tmp)
                except:
                    pass

            phase_plot(
                plotableDict,
                ylabel,
                xlimDict,
                CRPARAMS[selectKeyShort]["logParameters"],
                snapNumber = snapNumber,
                yParams = yParams,
                xParams = xParams,                    
                colourBarKeys = colourBarKeys,
                weightKeys = weightKeys,
                fontsize=fontsize,#13,
                fontsizeTitle=fontsizeTitle,#14
                titleBool=titleBool,#False,
                DPI=DPI,
                xsize=xsize,#6.0,
                ysize=ysize,#6.0,
                colourmapMain=colourmapMain,
                Nbins=Nbins,
                saveFigureData = saveFigureData,
                savePathBase = savePath,
                savePathBaseFigureData = savePathData,
                verbose=verbose,
                inplace=inplace,
                allowPlotsWithoutxlimits = allowPlotsWithoutxlimits,
                replotFromData = replotFromData,
                hush = hush,
            )


    return