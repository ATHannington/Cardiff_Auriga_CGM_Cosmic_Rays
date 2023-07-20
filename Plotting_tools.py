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
    weightKeys = ["mass","vol"],
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
    verbose = False,
    inplace = False,
):

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    zlimDict = copy.deepcopy(xlimDict)

    if inplace is True:
        simDict = dataDict
        print("\n"
              +f"[@phase_plot]: WARNING! inplace flag set to True. This will modify the data dictionary provided to this function call."
              +"\n"
              +f"Call details: yParams:{yParams}, xParams:{xParams}, weightKeys:{weightKeys}"
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
    # ------------------------------------------------------------------------------#
    #               PLOTTING
    #
    # ------------------------------------------------------------------------------#
    for yParam in yParams:
        print("\n"+"-----")
        print(f"Starting yParam {yParam}")
        for xParam in xParams:
            print("\n"+f"Starting xParam {xParam}")
            for weightKey in weightKeys:
                print("\n"+f"Starting weightKey {weightKey}")

                if weightKey == xParam:
                    print("\n" + f"WeightKey same as xParam! Skipping...")
                    skipBool = True
                    continue

                if weightKey == yParam:
                    print("\n" + f"WeightKey same as yParam! Skipping...")
                    skipBool = True
                    continue

                if xParam == yParam:
                    print("\n" + f"yParam same as xParam! Skipping...")
                    skipBool = True
                    continue

                if np.all(np.isin(np.array(["tcool","theat"]),np.array([xParam,yParam,weightKey]))) == True:
                    print("\n" + f"tcool and theat aren't compatible! Skipping...")
                    skipBool = True
                    continue

                try:
                    zmin = zlimDict[weightKey]["xmin"]
                    zmax = zlimDict[weightKey]["xmax"]
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


                fig, ax = plt.subplots(
                    nrows=1,
                    ncols=1,
                    figsize=(xsize, ysize),
                    dpi=DPI,
                    sharey=True,
                    sharex=True,
                )

                currentAx = ax

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                #   Figure 1: Full Cells Data
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                try:
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
                    skipBool = True
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
                    skipBool = True
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

                massCells = ( tmpdataDict["mass"][
                    np.where((xx>=xmin)&(xx<=xmax)
                    &(yy>=ymin)&(yy<=ymax)&
                    (np.isfinite(xx)==True)&(np.isfinite(yy)==True))
                    [0]]
                )

                if weightKey != "count":
                    try:
                        weightDataCells = (
                            tmpdataDict[weightKey][
                            np.where((xx>=xmin)&(xx<=xmax)
                            &(yy>=ymin)&(yy<=ymax)&(np.isfinite(xx)==True)&(np.isfinite(yy)==True))
                            [0]] * massCells
                        )
                        skipBool = False
                    except Exception as e:
                        print(f"{str(e)}")
                        print(
                            f"Variable {weightKey} not found. Skipping plot..."
                        )
                        skipBool = True
                        continue

                if weightKey == "mass":
                    finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=massCells
                    )
                elif weightKey == "count":
                    finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=None
                    )
                else:
                    mhistCells, _, _ = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=massCells
                    )
                    histCells, xedgeCells, yedgeCells = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=weightDataCells
                    )

                    finalHistCells = histCells / mhistCells

                finalHistCells[finalHistCells == 0.0] = np.nan
                try:
                    if weightKey in logParameters:
                        finalHistCells = np.log10(finalHistCells)
                except Exception as e:
                    print(f"{str(e)}")
                    print(f"Variable {weightKey} not found. Skipping plot...")
                    skipBool = True
                    continue
                finalHistCells = finalHistCells.T

                xcells, ycells = np.meshgrid(xedgeCells, yedgeCells)

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
                        currentAx.set_xlim(np.nanmax(xedgeCells),np.nanmin(xedgeCells))
                    else:
                        currentAx.set_xlim(np.nanmin(xedgeCells),np.nanmax(xedgeCells))
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
                            f"Variable {weightKey} not found. Skipping plot..."
                        )
                        continue
                    else:
                        pass

                    #left, bottom, width, height
                    # x0,    y0,  delta x, delta y
                cax1 = fig.add_axes([0.925, 0.10, 0.05, 0.80])

                fig.colorbar(img1, cax=cax1, ax=ax, orientation="vertical", pad=0.05).set_label(
                    label=ylabel[weightKey], size=fontsize
                )
                cax1.yaxis.set_ticks_position("left")
                cax1.yaxis.set_label_position("left")
                cax1.yaxis.label.set_color("black")
                cax1.tick_params(axis="y", colors="black", labelsize=fontsize)

                if titleBool is True:
                    fig.suptitle(
                        f"{yParam} vs. {xParam} Diagram, weighted by {weightKey}",
                        fontsize=fontsizeTitle,
                    )

                if titleBool is True:
                    plt.subplots_adjust(top=0.875, right=0.8, hspace=0.3, wspace=0.3)
                else:
                    plt.subplots_adjust(right=0.8, hspace=0.3, wspace=0.3)

                if snapNumber is not None:
                    if type(snapNumber) == int:
                        SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                    else:
                        SaveSnapNumber = "_" + str(snapNumber)
                else:
                    SaveSnapNumber = ""

                opslaan = (
                    savePath
                    + f"Phases-Plot_{yParam}-vs-{xParam}_weighted-by-{weightKey}{SaveSnapNumber}"
                )
                plt.savefig(opslaan+".pdf", dpi=DPI, transparent=False)
                print(opslaan)
                matplotlib.rc_file_defaults()
                plt.close("all")

                if saveFigureData is True:
                    osplaanData = savePathFigureData + f"Phases-Plot_{yParam}-vs-{xParam}_weighted-by-{weightKey}{SaveSnapNumber}"
                    out = {"data":{"x" : xcells, "y" : ycells, "hist": finalHistCells}}
                    tr.hdf5_save(osplaanData+"_data.h5",out)

    return

def pdf_versus_plot(
    inputDict,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber = None,
    weightKeys = ['mass'],
    xParams = ["T"],
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize=13,
    fontsizeTitle=14,
    Nbins=250,
    binByxParam = False,
    NxParamBins = None,
    ageWindow=None,
    cumulative = False,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    saveFigureData = False,
    SFR = False,
    byType = False,
    forceLogMass = False,
    normalise = False,
    verbose = False,
    inplace = False,
):

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})


    savePath = savePathBase + "Plots/PDFs/"
    savePathFigureData = savePathBaseFigureData + "Plots/PDFs/"
    
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
        weightKeys = ["gima"]
        xParams = ["age"]

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


    if byType is True:
        uniqueTypes = np.unique(dataDict["type"])
        for tp in uniqueTypes:
            print("Starting type ",tp)
            whereNotType = dataDict["type"] != tp

            tpData = cr.remove_selection(
                copy.deepcopy(dataDict),
                removalConditionMask = whereNotType,
                errorString = "byType PDF whereNotType",
                hush = True,
                verbose = verbose,
                )

            pdf_versus_plot(
                inputDict = tpData,
                ylabel = ylabel,
                xlimDict = xlimDict,
                logParameters = logParameters,
                snapNumber = snapNumber,
                weightKeys = weightKeys,
                xParams = xParams,
                titleBool = titleBool,
                DPI = DPI,
                xsize = xsize,
                ysize = ysize,
                fontsize = fontsize,
                fontsizeTitle = fontsizeTitle,
                Nbins = Nbins,
                binByxParam = binByxParam,
                NxParamBins = NxParamBins,
                ageWindow = ageWindow,
                cumulative = cumulative,
                savePathBase = savePathBase+f"type{int(tp)}/",
                savePathBaseFigureData = savePathBaseFigureData+f"type{int(tp)}/",
                saveFigureData = saveFigureData,
                SFR = SFR,
                forceLogMass = forceLogMass,
                byType = False,
                normalise = normalise,
                verbose = verbose,
                inplace = False,
            )
        return
    
    if binByxParam is True:
        binIndices = range(0,NxParamBins+1,1)
        for ii,(lowerIndex,upperIndex) in enumerate(zip(binIndices[1:],binIndices[:-1])):
            print("Starting Binned PDF plot ",ii," of ",NxParamBins)
            
            binnedxlimDict = copy.deepcopy(xlimDict)
            for key,value in xlimDict.items():
                if key in xParams:
                    
                    bins = np.linspace(start=value["xmin"],stop=value["xmax"],num=NxParamBins,endpoint=True)
                    binnedxlimDict.update({key:{"xmin":bins[lowerIndex],"xmax":bins[upperIndex]}})
                else:
                    pass
        
            pdf_versus_plot(
                inputDict = inputDict,
                ylabel = ylabel,
                xlimDict = binnedxlimDict,
                logParameters = logParameters,
                snapNumber = snapNumber,
                weightKeys = weightKeys,
                xParams = xParams,
                titleBool = titleBool,
                DPI = DPI,
                xsize = xsize,
                ysize = ysize,
                fontsize = fontsize,
                fontsizeTitle = fontsizeTitle,
                Nbins = Nbins,
                binByxParam = False,
                NxParamBins = NxParamBins,
                ageWindow = ageWindow,
                cumulative = cumulative,
                savePathBase = savePathBase+f"type{int(tp)}/",
                savePathBaseFigureData = savePathBaseFigureData+f"type{int(tp)}/",
                saveFigureData = saveFigureData,
                SFR = SFR,
                forceLogMass = forceLogMass,
                byType = byType,
                normalise = normalise,
                verbose = verbose,
                inplace = False,
            )


        return



    for weightKey in weightKeys:
        print("-----")
        print("")
        print(f"Starting {weightKey} weighted!")
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
            
            skipBool = False
            try:
                tmpdataDict, paramToTypeMap, _ = cr.map_params_to_types(dataDict)
                typesUsedData = paramToTypeMap[analysisParam]
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
                
                plotData = tmpdataDict[analysisParam]
                weightsData = tmpdataDict[weightKey]
                skipBool = False
            except Exception as e:
                # raise Exception(e)
                print(f"{str(e)}")
                print(
                    f"Variable {analysisParam} data subset selection failure. Skipping plot..."
                )
                skipBool = True
                continue


            colour = "blue"

            if analysisParam in logParameters:
                tmpPlot = np.log10(plotData).copy()
            else:
                tmpPlot = plotData.copy()

            tmpWeights = weightsData.copy()

            SFRBool = False
            if (weightKey == "gima")&(analysisParam=="age"):
                SFRBool = True

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

            try:
                xmin = np.nanmin(plotData)
                xmax = np.nanmax(plotData)
                skipBool = False
            except Exception as e:
                # raise Exception(e)
                print(f"{str(e)}")
                print(
                    f"Variable {analysisParam} data all NaN. Skipping plot...")
                skipBool = True
                continue

            if (
                (np.isfinite(xmin) == False)
                or (np.isfinite(xmax) == False)
                or (np.isfinite(np.nanmin(weightsData)) == False)
                or (np.isfinite(np.nanmin(weightsData)) == False)
            ):
                # print()
                print("Data All Inf/NaN! Skipping entry!")
                skipBool = True
                continue
            
            xBins = np.linspace(
                start=xmin, stop=xmax, num=Nbins)

            currentAx = ax


            hist, bin_edges = np.histogram(
                plotData,
                bins=xBins,
                weights=weightsData,
                density = normalise
            )

            # Because SFR should increase over time, and thus as age decreases
            if (SFRBool is True):
                hist = np.flip(hist)

            if cumulative is True:
                hist = np.cumsum(hist)
                if normalise is True:
                    hist /= np.nanmax(hist)


            if SFRBool is True:
                delta = np.mean(np.diff(xBins))
                hist /= (delta*1e9) # convert to SFR per yr

            #Again, SFR needs to increase with decreasing age
            if (SFRBool is True):
                xBins = np.flip(xBins)
                bin_edges = np.flip(bin_edges)

            # Take the log10 if desired. Set zeros to nan for convenience of
            # not having to drop inf's later, and mask them from plots
            if weightKey in logParameters:
                if weightKey != "mass":
                    hist[hist == 0.0] = np.nan
                    hist = np.log10(hist)
            elif (forceLogMass is True)&(weightKey=="mass"):
                hist[hist == 0.0] = np.nan
                hist = np.log10(hist)

            weightsSumTotal = np.sum(weightsData)

            if np.all(np.isfinite(hist)==False) == True:
                print("Hist All Inf/NaN! Skipping entry!")
                continue

            try:
                ymin = np.nanmin(hist[np.isfinite(hist)])
                ymax = np.nanmax(hist[np.isfinite(hist)])
                skipBool = False
            except Exception as e:
                # raise Exception(e)
                print(f"{str(e)}")
                print(
                    f"Variable {analysisParam} histogram all Inf/NaN. Skipping plot...")
                skipBool = True
                continue
            xFromBins = np.array(
                [
                    (x1 + x2) / 2.0
                    for (x1, x2) in zip(bin_edges[:-1], bin_edges[1:])
                ]
            )

            currentAx.plot(
                xFromBins,
                hist,
                color=colour,
                linestyle="solid",
                label = f"Sum total of {weightKey} = {weightsSumTotal:.2e}"
            )

            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(
                axis="both", which="both", labelsize=fontsize
            )

            ylabel_prefix = ""
            if cumulative is True:
                ylabel_prefix = "Cumulative "
            if normalise is True:
                ylabel_prefix = "Normalised " + ylabel_prefix

            if weightKey == "mass":
                if (forceLogMass is True):
                    currentAx.set_ylabel(r"$Log_{10}$ "+ylabel_prefix+"Mass (M$_{\odot}$)", fontsize=fontsize)
                else:
                    currentAx.set_ylabel(ylabel_prefix+r"Mass (M$_{\odot}$)", fontsize=fontsize)
            else:
                currentAx.set_ylabel(
                ylabel_prefix+ylabel[weightKey], fontsize=fontsize)


            if titleBool is True:
                fig.suptitle(
                    ylabel_prefix + f"PDF of"
                    + "\n"
                    + f" {weightKey} vs {analysisParam}",
                    fontsize=fontsizeTitle,
                )

            # Only give 1 x-axis a label, as they sharex

            ax.set_xlabel(ylabel[analysisParam], fontsize=fontsize)

            if (skipBool == True):
                print(
                    f"Variable {analysisParam} plot failed (reason should have been printed to stdout above). Skipping plot...")
                continue

            #try:
            #    finalxmin = max(
            #        np.nanmin(xmin), xlimDict[analysisParam]["xmin"]
            #    )
            #    finalxmax = min(
            #        np.nanmax(xmax), xlimDict[analysisParam]["xmax"]
            #    )
            #except:
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

            if (weightKey == "mass")&(forceLogMass is False)&(SFRBool is False):
                try:
                    finalymin = 0.0
                    finalymax = np.nanmax(ymax)
                except:
                    print("Data All Inf/NaN! Skipping entry!")
                    continue
            else:
                try:
                    finalymin = np.nanmin(ymin)
                    finalymax = np.nanmax(ymax)
                except:
                    print("Data All Inf/NaN! Skipping entry!")
                    continue

            custom_xlim = (finalxmin, finalxmax)
            custom_ylim = (finalymin, finalymax)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            ax.legend(loc="best", fontsize=fontsize)

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
                if type(snapNumber) == int:
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

            plt.savefig(opslaan + ".pdf", dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()
            matplotlib.rc_file_defaults()
            plt.close("all")

            if saveFigureData is True:
                print(f"Saving Figure Data as {opslaanFigureData}")
                out = {"data":{"x" : xFromBins, "y" : hist}}
                tr.hdf5_save(opslaanFigureData+"_data.h5",out)
                

            
    return

def plot_slices(snap,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber = None,
    sliceParam = "T",
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
    colourmapMain="inferno",
    numthreads=10,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    saveFigureData = False,
    saveFigure = True,
    rasterized = True,
    verbose = False,
    inplace = False,
):
    savePath = savePathBase + "Plots/Slices/"
    savePathFigureData = savePathBaseFigureData + "Plots/Slices/"
    
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

    cmap = plt.get_cmap(colourmapMain)

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["x", "y", "z"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#
    # PLOTTING TIME
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    snapType = False
    try:
        types = pd.unique(snap.data["type"])
        snapType = True
        if verbose: print("[@plot_slices]: snapshot type detected!")
    except:
        snapType = False
        if verbose: print("[@plot_slices]: dictionary type detected!")

    paramSplitList = sliceParam.split("_")

    if paramSplitList[-1] == "col":
        ## If _col variant is called we want to calculate a projection of the non-col parameter
        ## Thus, we force projection to now be true, and incorporate a dummy variable tmpsliceParam
        ## to force plots to generate non-col variants but save output as column density version

        tmpsliceParam = "_".join(paramSplitList[:-1])
        projection = True
    else:
        tmpsliceParam = sliceParam

    if snapType == True:
        try:
            tmp = snap.data[tmpsliceParam]
        except Exception as e:
            print(f"{str(e)}")
            print(
            f"[@plot_slices]: Variable {tmpsliceParam} not found in snapshot data. Skipping plot..."
            )
            return
    else:
        try:
            slice = snap[tmpsliceParam]
        except Exception as e:
            print(f"{str(e)}")
            print(
            f"[@plot_slices]: Variable {tmpsliceParam} not found in data dictionary. Skipping plot..."
            )
            return

    if snapType is True:
        # ==============================================================================#
        #
        #           Quad Plot for standard video
        #
        # ==============================================================================#

        #fullTicks = [xx for xx in np.linspace(-1.0 * halfbox, halfbox, 9)]
        #fudgeTicks = fullTicks[1:]

        aspect = "equal"

        if projection is True:
            nz=int(boxlos / pixreslos)
            boxz=boxlos
        else:
            nz=None #int(boxlos / pixreslos)
            boxz= None #boxlos

        if tmpsliceParam == "T":
            slice  = snap.get_Aslice(
                "Tdens",
                box=[boxsize, boxsize],
                center=imgcent,
                nx=int(boxsize / pixres),
                ny=int(boxsize / pixres),
                nz=nz,
                boxz=boxz,
                axes=Axes,
                proj=projection,
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
                proj=projection,
                numthreads=numthreads,
            )

            slice["grid"] = slice["grid"]/proj_dens["grid"]
        else:
            slice = snap.get_Aslice(
                tmpsliceParam,
                box=[boxsize, boxsize],
                center=imgcent,
                nx=int(boxsize / pixres),
                ny=int(boxsize / pixres),
                nz=nz,
                boxz=boxz,
                axes=Axes,
                proj=projection,
                numthreads=numthreads,
            )

            paramSplitList = sliceParam.split("_")

            if paramSplitList[-1] == "col":
                KpcTocm = 1e3 * c.parsec
                convert = float(pixreslos)*KpcTocm
                slice["grid"] = slice["grid"]*convert
            elif projection is True:
                slice["grid"] = slice["grid"]/ int(boxlos / pixreslos)
    
    if saveFigure is True:
            
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
        )

        # cmap = plt.get_cmap(colourmapMain)
        cmap = copy.copy(cmap)
        cmap.set_bad(color="grey")

        try:
            tmp = xlimDict[sliceParam]["xmin"]
            tmp = xlimDict[sliceParam]["xmax"]
            xlimBool = True
        except:
            xlimBool = False

        if xlimBool is True:
            if sliceParam in logParameters:
                zmin = 10**(xlimDict[sliceParam]['xmin'])
                zmax = 10**(xlimDict[sliceParam]['xmax'])
            else:
                zmin = xlimDict[sliceParam]['xmin']
                zmax = xlimDict[sliceParam]['xmax']
        else:
            zmin, zmax = None, None

        norm = matplotlib.colors.LogNorm(vmin = zmin, vmax = zmax,clip=True)
        pcm = axes.pcolormesh(
            slice["x"],
            slice["y"],
            np.transpose(slice["grid"]),
            norm=norm,
            cmap=cmap,
            rasterized=rasterized,
        )

        #axes.set_title(f"{sliceParam} Slice", fontsize=fontsize)
        cax1 = inset_axes(axes, width="5%", height="95%", loc="right")
        fig.colorbar(pcm, cax=cax1, orientation="vertical").set_label(
            label=f"{ylabel[sliceParam]}", size=fontsize, weight="bold"
        )
        cax1.yaxis.set_ticks_position("left")
        cax1.yaxis.set_label_position("left")
        cax1.yaxis.label.set_color("white")
        cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

        axes.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
        axes.set_xlabel(f"{AxesLabels[Axes[0]]}" + " (kpc)", fontsize = fontsize)
        axes.set_aspect("equal")

        # Pad snapNumber with zeroes to enable easier video making

        if snapNumber is not None:
            if type(snapNumber) == int:
                SaveSnapNumber = "_" + str(snapNumber).zfill(4)
            else:
                SaveSnapNumber = "_" + str(snapNumber)
        else:
            SaveSnapNumber = ""

        if projection is False:
            savePath = savePath + f"Slice_Plot_{sliceParam}{SaveSnapNumber}.pdf"
            savePathFigureData = savePathFigureData + f"Slice_Plot_{sliceParam}{SaveSnapNumber}"
        else:
            savePath = savePath + f"Projection_Plot_{sliceParam}{SaveSnapNumber}.pdf" 
            savePathFigureData = savePathFigureData + f"Projection_Plot_{sliceParam}{SaveSnapNumber}" 

        print(f" Save {savePath}")
        plt.savefig(savePath, transparent=False)
        plt.close()

        matplotlib.rc_file_defaults()
        plt.close("all")
        print(f" ...done!")

        out = {sliceParam: copy.deepcopy(slice)}
        if saveFigureData is True:
            print(f"Saving Figure Data as {savePathFigureData}")
            tr.hdf5_save(savePathFigureData+"_data.h5",out)

    return out

def combined_pdf_versus_plot(
    loadPaths,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber = None,
    weightKeys = ['mass'],
    xParams = ["T"],
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize=13,
    fontsizeTitle=14,
    colourmapMain = "plasma",
    savePathBase = "./Combined-Plots/",
    cumulative = False,
    SFR = False,
    byType = False,
    forceLogMass = False,
    normalise = False,
    verbose = False,
    ):

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    check_params_are_in_xlimDict(xlimDict, xParams)

    if SFR is True:
        weightKeys = ["gima"]
        xParams = ["age"]

    if byType is True:
        uniqueTypes = [0,1,2,3,4,5]
        for tp in uniqueTypes:
            print("Starting type ",tp)

            tmploadPaths = []
            for loadPath in loadPaths:
                path = loadPath + f"type{int(tp)}/"
                tmploadPaths.append(path)

            combined_pdf_versus_plot(
                loadPaths = tmploadPaths,
                ylabel = ylabel,
                xlimDict = xlimDict,
                logParameters = logParameters,
                snapNumber = snapNumber,
                weightKeys = weightKeys,
                xParams = xParams,
                titleBool = titleBool,
                DPI = DPI,
                xsize = xsize,
                ysize = ysize,
                fontsize = fontsize,
                fontsizeTitle = fontsizeTitle,
                colourmapMain = colourmapMain,
                savePathBase = savePathBase,
                cumulative = cumulative,
                SFR = SFR,
                byType = byType,
                forceLogMass = forceLogMass,
                normalise = normalise,
                verbose = verbose,
            )
        return

    skipBool = False

    for weightKey in weightKeys:
        print("-----")
        print("")
        print(f"Starting {weightKey} weighted!")
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
            Nsims = len(loadPaths)
            for (ii,loadPathBase) in enumerate(loadPaths):
                print("loadPathBase= ",loadPathBase)

                loadPath = loadPathBase + "Plots/PDFs/"
                tmp = ""

                for loadPathChunk in loadPath.split("/")[:-1]:
                    tmp += loadPathChunk + "/"
                    try:
                        os.mkdir(tmp)
                    except:
                        pass
                    else:
                        pass


                SFRBool = False
                if (weightKey == "gima")&(analysisParam=="age"):
                    SFRBool = True

                if normalise is True:
                    tmp2 = loadPath +"Normalised-"
                else:
                    tmp2 = loadPath

                if cumulative is True:
                    tmp2 = tmp2 +"Cumulative-"

                if snapNumber is not None:
                    if type(snapNumber) == int:
                        SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                    else:
                        SaveSnapNumber = "_" + str(snapNumber)
                else:
                    SaveSnapNumber = ""

                if SFRBool is True:
                    opslaan = tmp2 + f"SFR{SaveSnapNumber}"

                else:
                    opslaan = tmp2 + f"{weightKey}-{analysisParam}-PDF{SaveSnapNumber}"

                print(opslaan)

                # out = {"data":{"x" : xFromBins, "y" : hist}}
                try:
                    dataDict = tr.hdf5_load(opslaan+"_data.h5")
                    skipBool = False
                except:
                    print(f"Load path of {opslaan+'_data.h5'} not found! Skipping...")
                    skipBool = True
                    
                    continue

                if normalise is True:
                    tmp2 = "Normalised-"
                else:
                    tmp2 = ""

                if cumulative is True:
                    tmp2 = tmp2 +"Cumulative-"

                if SFRBool is True:
                    savePath = tmp2 + f"SFR"
                else:
                    savePath = tmp2 + f"{weightKey}-{analysisParam}-PDF"

                cmap = matplotlib.cm.get_cmap(colourmapMain)
                if colourmapMain == "tab10":
                    colour = cmap(float(ii) / 10.0)
                else:
                    colour = cmap(float(ii) / float(Nsims))

                splitbase = loadPathBase.split("/")
                # print(splitbase)
                if "" in splitbase:
                    splitbase.remove("")
                if "." in splitbase:
                    splitbase.remove(".")
                if "Plots" in splitbase:
                    splitbase.remove("Plots")

                    # print(splitbase)

                if len(splitbase)>2:
                    label = f'{splitbase[0]}: {"_".join(((splitbase[-2]).split("_"))[:2])} ({splitbase[-1]})'
                elif len(splitbase)>1:
                    label = f'{splitbase[0]}: {"_".join(((splitbase[-1]).split("_"))[:2])}'
                else:
                    label = f'Original: {"_".join(((splitbase[-1]).split("_"))[:2])}'
                # print("label= ",label)
                ax.plot(
                    dataDict["data"]["x"],
                    dataDict["data"]["y"],
                    color=colour,
                    linestyle="solid",
                    label = label
                )

                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(
                    axis="both", which="both", labelsize=fontsize
                )

            if skipBool: continue

            ylabel_prefix = ""
            if cumulative is True:
                ylabel_prefix = "Cumulative "
            if normalise is True:
                ylabel_prefix = "Normalised " + ylabel_prefix

            if weightKey == "mass":
                if forceLogMass is False:
                    ax.set_ylabel(ylabel_prefix+r"Mass (M$_{\odot}$)", fontsize=fontsize)
                else:
                    ax.set_ylabel(r"$Log_{10}$ "+ylabel_prefix+"Mass (M$_{\odot}$)", fontsize=fontsize)
            else:
                ax.set_ylabel(
                ylabel_prefix+ylabel[weightKey], fontsize=fontsize)


            if titleBool is True:
                fig.suptitle(
                    ylabel_prefix + f"PDF of"
                    + "\n"
                    + f" {weightKey} vs {analysisParam}",
                    fontsize=fontsizeTitle,
                )

            # Only give 1 x-axis a label, as they sharex

            ax.set_xlabel(ylabel[analysisParam], fontsize=fontsize)
            ax.legend(loc="best", fontsize=fontsize)

            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            savePath = savePathBase + "Plots/PDFs/" + savePath
            tmp = ""

            for savePathChunk in savePath.split("/")[:-1]:
                tmp += savePathChunk + "/"
                try:
                    os.mkdir(tmp)
                except:
                    pass
                else:
                    pass

            plt.savefig(savePath + "_sims-combined" + ".pdf", dpi=DPI, transparent=False)
            print("Saved as: ",savePath + "_sims-combined" + ".pdf")
            plt.close()
    return

def medians_versus_plot(
    inputDict,
    PARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    yParam=None,
    xParam="R",
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize = 13,
    fontsizeTitle = 14,
    opacityPercentiles=0.25,
    colourmapMain="tab10",
    savePathBase ="./",
    savePathBaseFigureData = "./",
    inplace = False,
    saveFigureData = False,
    ):

 
    keys = list(PARAMS.keys())
    selectKey0 = keys[0]

    if yParam is None:
        plotParams = PARAMS[selectKey0]["saveParams"]
        print(f"[medians_versus_plot]: WARNING! No yParam provided so default of all 'saveParams' being used. This may cause errors if any of 'saveParams' do not have limits set in xlimDict...")
    else:
        plotParams = yParam

    #check_params_are_in_xlimDict(xlimDict, yParam)
    #check_params_are_in_xlimDict(xlimDict, [xParam])

    if inplace is True:
        statsDict = inputDict
        print("\n"
              +f"[@medians_versus_plot]: WARNING! inplace flag set to True. This will modify the data dictionary provided to this function call."
              +"\n"
              +f"Call details: xParam:{xParam}, yParam:{yParam}"
              +"\n"
              )
    else:
        statsDict = copy.deepcopy(inputDict)

    savePath = savePathBase + "/Plots/Medians/"
 
    tmp = ""
    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass

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

            yminlist = []
            ymaxlist = []

            Nkeys = len(list(statsDict.keys()))

            for (ii, (selectKey, simDict)) in enumerate(statsDict.items()):
                if (selectKey[-2] == "Stars") | (selectKey[-2] == "col") :
                    selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
                else:
                    selectKeyShort = selectKey

                try:
                    HASDATA = simDict.values()
                    if HASDATA is None:
                        print(f"[@medians_versus_plot]: No data found in simDict entry for {selectKey} (or for {selectKeyShort}). Skipping...")
                        continue
                except Exception as e:
                    print(f"{str(e)}")
                    print(f"[@medians_versus_plot]: No data found in simDict entry for {selectKey} (or for {selectKeyShort}). Skipping...")
                    continue

                print(f"Starting {selectKey} plot")             


                plotData = simDict.copy()
                xData = np.array(simDict[xParam].copy())

                cmap = matplotlib.cm.get_cmap(colourmapMain)
                if colourmapMain == "tab10":
                    colour = cmap(float(ii) / 10.0)
                else:
                    colour = cmap(float(ii) / float(Nkeys))

                try:
                    loadPercentilesTypes = [
                        analysisParam + "_" + str(percentile) + "%"
                        for percentile in PARAMS[selectKeyShort]["percentiles"]
                    ]
                except Exception as e:
                    print(f"{str(e)}")
                    print(
                        f"[@medians_versus_plot]: Variable {analysisParam} not found in PARAMS. Skipping plot..."
                    )
                    continue
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

                if analysisParam in PARAMS[selectKeyShort]["logParameters"]:
                    for k, v in plotData.items():
                        plotData.update({k: np.log10(v)})

                try:
                    ymin = np.nanmin(
                        plotData[LO][np.isfinite(plotData[LO])])
                    ymax = np.nanmax(
                        plotData[UP][np.isfinite(plotData[UP])])
                except Exception as e:
                    print(f"{str(e)}")
                    print(
                        f"[@medians_versus_plot]: Variable {analysisParam} has no finite values. Skipping plot...")
                    continue
                yminlist.append(ymin)
                ymaxlist.append(ymax)

                if (
                    (np.isinf(ymin) == True)
                    or (np.isinf(ymax) == True)
                    or (np.isnan(ymin) == True)
                    or (np.isnan(ymax) == True)
                ):
                    # print()
                    print("[@medians_versus_plot]: Data All Inf/NaN! Skipping entry!")
                    continue

                currentAx = ax

                path = copy.copy(savePathBase)

                splitbase = path.split("/")
                # print(splitbase)
                if "" in splitbase:
                    splitbase.remove("")
                if "." in splitbase:
                    splitbase.remove(".")
                if "Plots" in splitbase:
                    splitbase.remove("Plots")
                # print(splitbase)

                if len(splitbase)>2:
                    label = f'{splitbase[0]}: {"_".join(((splitbase[-2]).split("_"))[:2])} ({splitbase[-1]})'
                elif len(splitbase)>1:
                    label = f'{splitbase[0]}: {"_".join(((splitbase[-1]).split("_"))[:2])}'
                else:
                    label = f'Original: {"_".join(((splitbase[-1]).split("_"))[:2])}'


                midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)
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
                currentAx.plot(
                    xData,
                    plotData[median],
                    label= label,
                    color=colour,
                )

                currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                currentAx.tick_params(
                    axis="both", which="both", labelsize=fontsize)

                currentAx.set_ylabel(
                    ylabel[analysisParam], fontsize=fontsize)

                if titleBool is True:
                    if selectKey[-2] == "Stars":
                        fig.suptitle(
                            f"Median and Percentiles of"
                            + "\n"
                            + f" Stellar-{analysisParam} vs {xParam}",
                            fontsize=fontsizeTitle,
                        )

                    elif selectKey[-2] == "col":
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
            ax.legend(loc="best", fontsize=fontsize)

            plt.setp(
                ax,
                ylim=custom_ylim,
                xlim=custom_xlim
            )
            # plt.tight_layout()

            if snapNumber is not None:
                if type(snapNumber) == int:
                    SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                else:
                    SaveSnapNumber = "_" + str(snapNumber)
            else:
                SaveSnapNumber = ""

            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            if selectKey[-2] == "Stars":
                opslaan = savePath + \
                    f"Stellar-{analysisParam}_Medians{SaveSnapNumber}.pdf"
            elif selectKey[-2] == "col":
                opslaan = savePath + \
                    f"Projection-Mapped-{analysisParam}_Medians{SaveSnapNumber}.pdf"
            else:
                opslaan = savePath + f"{analysisParam}_Medians{SaveSnapNumber}.pdf"
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            matplotlib.rc_file_defaults()
            plt.close("all")
            print(opslaan)
            plt.close()

    if saveFigureData is True:
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
    AxesLabels = ["x", "y", "z"]

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
    AxesLabels = ["x", "y", "z"]

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
        label=r"t$_{cool}$ (Gyr)", size=fontsize, weight="bold"
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
    AxesLabels = ["x", "y", "z"]

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

def cr_medians_versus_plot(
    statsDict,
    CRPARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    yParam=None,
    xParam="R",
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize = 13,
    fontsizeTitle = 14,
    opacityPercentiles=0.25,
    colourmapMain="tab10",
    savePathBase ="./",
    savePathBaseFigureData = "./",
    inplace = False,
    saveFigureData = False,
    forcePlotsWithoutxlimits = False,
    ):

    keys = list(CRPARAMS.keys())
    selectKey0 = keys[0]

    if yParam is None:
        plotParams = CRPARAMS[selectKey0]["saveParams"]
        print(f"[cr_medians_versus_plot]: WARNING! No yParam provided so default of all 'saveParams' being used. This may cause errors if any of 'saveParams' do not have limits set in xlimDict...")
    else:
        plotParams = yParam

    if forcePlotsWithoutxlimits == False:
        check_params_are_in_xlimDict(xlimDict, yParam)
        check_params_are_in_xlimDict(xlimDict, [xParam])


    fontsize = CRPARAMS[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMS[selectKey0]["fontsizeTitle"]

    for (ii, (selectKey, simDict)) in enumerate(statsDict.items()):
        if (selectKey[-2] == "Stars") | (selectKey[-2] == "col") :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
        else:
            selectKeyShort = selectKey

        selectKeyShortest = selectKeyShort[:-1]

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")
            if snapNumber is None:
                snapNumber = "Averaged"

            plotableDict = {selectKeyShortest : simDict}
            tmpCRPARAMS = {selectKeyShortest : copy.deepcopy(CRPARAMS[selectKeyShort])}

            if (selectKey[-2] == "Stars"):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            elif (selectKey[-2] == "col"):
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
            for savePathChunk in savePathData.split("/")[1:-1]:
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
                yParam=plotParams,
                xParam=xParam,
                titleBool=titleBool,
                DPI=DPI,
                xsize=xsize,
                ysize=ysize,
                fontsize = fontsize,
                fontsizeTitle = fontsizeTitle,
                opacityPercentiles=opacityPercentiles,
                colourmapMain=colourmapMain,
                savePathBase = savePath,
                savePathBaseFigureData = savePathBaseFigureData,
                inplace = inplace,
                saveFigureData=saveFigureData
            )

    return

def cr_pdf_versus_plot(
    dataDict,
    CRPARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    weightKeys = ['mass'],
    xParams = ["T"],
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize=13,
    fontsizeTitle=14,
    Nbins=250,
    binByxParam = False,
    NxParamBins = None,
    ageWindow=None,
    cumulative = False,
    savePathBase = "./",
    savePathBaseFigureData = "./",
    saveFigureData = True,
    SFR = False,
    byType = False,
    forceLogMass = False,
    normalise = False,
    verbose = False,
    inplace = False,
    forcePlotsWithoutxlimits = False,
    exclusions = ["Redshift", "Lookback", "Snap", "Rvir"],
    #lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
):
    
    if (forcePlotsWithoutxlimits == False) & (SFR == False):
        check_params_are_in_xlimDict(xlimDict, xParams)

    keys = list(CRPARAMS.keys())
    selectKey0 = keys[0]

    fontsize = CRPARAMS[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMS[selectKey0]["fontsizeTitle"]

    for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
        if (selectKey[-2] == "Stars") | (selectKey[-2] == "col") :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
        else:
            selectKeyShort = selectKey

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")
            snapNumber = copy.deepcopy(simDict["Snap"])
            if len(snapNumber)>1:
                snapNumber = "Averaged"
            elif (type(snapNumber) is list)|(type(snapNumber) is np.ndarray):
                snapNumber = snapNumber[0]


            plotableDict = copy.deepcopy(simDict)
            for excl in exclusions:
                plotableDict.pop(excl)

            if (selectKey[-2] == "Stars"):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            elif (selectKey[-2] == "col"):
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
            for savePathChunk in savePathData.split("/")[1:-1]:
                tmp += savePathChunk + "/"
                try:
                    os.mkdir(tmp)
                except:
                    pass
                else:
                    pass

            pdf_versus_plot(
                plotableDict,
                ylabel,
                xlimDict,
                CRPARAMS[selectKeyShort]["logParameters"],
                snapNumber = snapNumber,
                weightKeys = weightKeys,#['mass'],
                xParams = xParams, #["T"],
                titleBool=titleBool,#False,
                DPI=DPI,#150,
                xsize=xsize,#6.0,
                ysize=ysize,#6.0,
                fontsize=fontsize,#13,
                fontsizeTitle=fontsizeTitle,#14,
                Nbins=Nbins,#250,
                binByxParam = binByxParam,
                NxParamBins = NxParamBins,
                ageWindow=ageWindow,#None,
                cumulative = cumulative,#False,
                savePathBase = savePath,#"./",
                savePathBaseFigureData = savePathData,#False,
                SFR = SFR,#False,
                byType = byType,#False,
                forceLogMass = forceLogMass,#False,
                normalise = normalise,#False,
                verbose = verbose,#False,
                inplace = inplace, #False,
            )
    return

def cr_combined_pdf_versus_plot(
    dataDict,
    CRPARAMS,
    ylabel,
    xlimDict,
    weightKeys = ['mass'],
    xParams = ["T"],
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize=13,
    fontsizeTitle=14,
    cumulative = False,
    savePathBase = "./",
    SFR = False,
    byType = False,
    forceLogMass = False,
    normalise = False,
    verbose = False,
    forcePlotsWithoutxlimits = False,
    exclusions = ["Redshift", "Lookback", "Snap", "Rvir"],
    ):

    raise Exception("UPDATE THIS FUNCTION TO NEW CR_PDF_VERSUS_PLOT CALL SIGNIATURES")
    cr_pdf_versus_plot(
        dataDict,
        CRPARAMS,
        ylabel,
        xlimDict,
        snapNumber = None,
        weightKeys = ['mass'],
        xParams = ["T"],
        titleBool=False,
        DPI=150,
        xsize=6.0,
        ysize=6.0,
        fontsize=13,
        fontsizeTitle=14,
        Nbins=250,
        binByxParam = False,
        NxParamBins = None,
        ageWindow=None,
        cumulative = False,
        savePathBase = "./",
        savePathBaseFigureData = "./",
        saveFigureData = True,
        SFR = False,
        byType = False,
        forceLogMass = False,
        normalise = False,
        verbose = False,
        inplace = False,
        exclusions = ["Redshift", "Lookback", "Snap", "Rvir"],
        #lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
    )

    #lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
    check_params_are_in_xlimDict(xlimDict, xParams)



    keys = list(CRPARAMS.keys())
    selectKey0 = keys[0]

    fontsize = CRPARAMS[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMS[selectKey0]["fontsizeTitle"]


    pdfLoadPaths = []
    for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
        if (selectKey[-2] == "Stars") | (selectKey[-2] == "col") :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
        else:
            selectKeyShort = selectKey

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:

            if (selectKey[-2] == "Stars"):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            elif (selectKey[-2] == "col"):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Col-Projection-Mapped/"
            else:
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/"

            path = savePathBase + simSavePath
            pdfLoadPaths.append(path)

    if (selectKey[-2] == "Stars"):
        spath = f"type-{CRPARAMS[selectKey0]['analysisType']}/{CRPARAMS[selectKey0]['halo']}/Stars/"
    elif (selectKey[-2] == "col"):
        spath = f"type-{CRPARAMS[selectKey0]['analysisType']}/{CRPARAMS[selectKey0]['halo']}/Col-Projection-Mapped/"
    else:
        spath = f"type-{CRPARAMS[selectKey0]['analysisType']}/{CRPARAMS[selectKey0]['halo']}/"

    savePath = savePathBase + spath

    snapNumber = copy.deepcopy(simDict["Snap"])
    if len(snapNumber)>1:
        snapNumber = "Averaged"
    elif (type(snapNumber) is list)|(type(snapNumber) is np.ndarray):
        snapNumber = snapNumber[0]

    tmp = ""

    for savePathChunk in savePath.split("/")[:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass

    combined_pdf_versus_plot(
        pdfLoadPaths,
        ylabel,
        xlimDict,
        CRPARAMS[selectKey0]["logParameters"],
        snapNumber = snapNumber,
        weightKeys = weightKeys,#['mass'],
        xParams = xParams, #["T"],
        titleBool=titleBool,#False,
        DPI=DPI,#150,
        xsize=xsize,#6.0,
        ysize=ysize,#6.0,
        fontsize=fontsize,#13,
        fontsizeTitle=fontsizeTitle,#14,
        colourmapMain = CRPARAMS[selectKey0]["colourmapMain"],
        savePathBase = savePath,
        cumulative = cumulative,#False,
        SFR = SFR,#False,
        byType = byType,#False,
        forceLogMass = forceLogMass,#False,
        normalise = normalise,#False,
        verbose = verbose,#False,
    )
    return

def cr_phase_plot(
    dataDict,
    CRPARAMS,
    ylabel,
    xlimDict,
    snapNumber = None,
    yParams = ["T"],
    xParams = ["rho_rhomean","R"],
    weightKeys = ["mass","vol"],
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
    forcePlotsWithoutxlimits = False,
):

    if forcePlotsWithoutxlimits == False:
        check_params_are_in_xlimDict(xlimDict, xParams)
        check_params_are_in_xlimDict(xlimDict, yParams)

    keys = list(CRPARAMS.keys())
    selectKey0 = keys[0]

    fontsize = CRPARAMS[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMS[selectKey0]["fontsizeTitle"]
    # ------------------------------------------------------------------------------#
    #               PLOTTING
    #
    # ------------------------------------------------------------------------------#
    for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
        if (selectKey[-2] == "Stars") | (selectKey[-2] == "col") :
            selectKeyShort = tuple([xx for xx in selectKey if (xx != "Stars") & (xx != "col")])
            # print("[@phases_plot]: WARNING! Stars not supported! Skipping!")
            # continue
        else:
            selectKeyShort = selectKey

        loadpath = CRPARAMS[selectKeyShort]["simfile"]
        if loadpath is not None:
            print(f"{CRPARAMS[selectKeyShort]['resolution']}, {CRPARAMS[selectKeyShort]['CR_indicator']}{CRPARAMS[selectKeyShort]['no-alfven_indicator']}")                    # Create a plot for each Temperature


            snapNumber = copy.deepcopy(simDict["Snap"])
            if len(snapNumber)>1:
                snapNumber = "Averaged"
            elif (type(snapNumber) is list)|(type(snapNumber) is np.ndarray):
                snapNumber = snapNumber[0]

            plotableDict = copy.deepcopy(simDict)
            for excl in exclusions:
                plotableDict.pop(excl)

            if (selectKey[-2] == "Stars"):
                simSavePath = f"type-{CRPARAMS[selectKeyShort]['analysisType']}/{CRPARAMS[selectKeyShort]['halo']}/"+f"{CRPARAMS[selectKeyShort]['resolution']}/{CRPARAMS[selectKeyShort]['CR_indicator']}"+f"{CRPARAMS[selectKeyShort]['no-alfven_indicator']}/Stars/"
            elif (selectKey[-2] == "col"):
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
                weightKeys=weightKeys,
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
                inplace=inplace
            )

    return