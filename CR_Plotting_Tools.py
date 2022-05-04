import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
import const as c
import OtherConstants as oc
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
from CR_Subroutines import *
import copy
import h5py
import json
import math
from random import sample
import sys
import logging

fontsize = 13
fontsizeTitle = 14

def round_it(x, sig):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

def medians_versus_plot(
    statsDict,
    CRPARAMSHALO,
    halo,
    ylabel,
    xlimDict,
    xParam = "R",
    titleBool=False,
    DPI=150,
    xsize = 6.0,
    ysize = 6.0,
    opacityPercentiles = 0.25,
    lineStyleDict = {"with_CRs": "solid", "no_CRs": "-."},
    colourmapMain = "tab10",
):

    savePath = f"./Plots/{halo}/Medians/"
    try:
        os.mkdir(savePath)
    except:
        pass
    else:
        pass

    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]
    for analysisParam in CRPARAMSHALO[selectKey0]['saveParams']:
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
            Nkeys = len(list(statsDict.items()))
            for (ii, (selectKey, simDict)) in enumerate(statsDict.items()):
                loadpath = CRPARAMSHALO[selectKey]['simfile']
                if loadpath is not None :
                    print(f"{CRPARAMSHALO[selectKey]['resolution']}, @{CRPARAMSHALO[selectKey]['CR_indicator']}")
                    # Create a plot for each Temperature
                    yminlist = []
                    ymaxlist = []
                    patchList = []
                    labelList = []

                    plotData = simDict.copy()
                    xData = simDict[xParam].copy()

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if colourmapMain == "tab10":
                        colour = cmap(float(ii) / 10.)
                    else:
                        colour = cmap(float(ii) / float(Nkeys))

                    lineStyle = lineStyleDict[CRPARAMSHALO[selectKey]['CR_indicator']]

                    loadPercentilesTypes = [
                        analysisParam + "_" + str(percentile) + "%"
                        for percentile in CRPARAMSHALO[selectKey]["percentiles"]
                    ]
                    LO = analysisParam + "_" + str(min(CRPARAMSHALO[selectKey]["percentiles"])) + "%"
                    UP = analysisParam + "_" + str(max(CRPARAMSHALO[selectKey]["percentiles"])) + "%"
                    median = analysisParam + "_" + "50.00%"

                    if analysisParam in CRPARAMSHALO[selectKey]['logParameters']:
                        for k, v in plotData.items():
                            plotData.update({k: np.log10(v)})

                    try:
                        ymin = np.nanmin(plotData[LO][np.isfinite(plotData[LO])])
                        ymax = np.nanmax(plotData[UP][np.isfinite(plotData[UP])])
                    except:
                        print(f"Variable {analysisParam} not found. Skipping plot...")
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
                        print("Data All Inf/NaN! Skipping entry!")
                        continue

                    currentAx = ax


                    midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)
                    percentilesPairs = zip(
                        loadPercentilesTypes[:midPercentile],
                        loadPercentilesTypes[midPercentile + 1 :],
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
                        label=f"{CRPARAMSHALO[selectKey]['resolution']}: {CRPARAMSHALO[selectKey]['CR_indicator']}",
                        color=colour,
                        lineStyle=lineStyle,
                    )

                    currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

                    currentAx.set_ylabel(ylabel[analysisParam], fontsize=fontsize)


                    if titleBool is True:
                        fig.suptitle(
                            f"Median and Percentiles of"+"\n"+f" {analysisParam} vs {xParam}",
                            fontsize=fontsizeTitle,
                        )

                # Only give 1 x-axis a label, as they sharex


            ax.set_xlabel(ylabel[xParam], fontsize=fontsize)

            try:
                finalymin = min(np.nanmin(yminlist),xlimDict[analysisParam]['xmin'])
                finalymax = max(np.nanmax(ymaxlist),xlimDict[analysisParam]['xmax'])
            except:
                finalymin = np.nanmin(yminlist)
                finalymax = np.nanmax(ymaxlist)
            else:
                pass

            if (
                (np.isinf(finalymin) == True)
                or (np.isinf(finalymax) == True)
                or (np.isnan(finalymin) == True)
                or (np.isnan(finalymax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue
            finalymin = numpy.round_(finalymin, decimals = 2)
            finalymax = numpy.round_(finalymax, decimals = 2)

            custom_ylim = (finalymin, finalymax)

            xticks = [round_it(xx,2) for xx in np.linspace(min(xData),max(xData),5)]
            custom_xlim = (min(xData),max(xData)*1.05)
            if xParam == "R":
                ax.fill_betweenx([finalymin,finalymax],0,min(xData), color="tab:gray",alpha=opacityPercentiles)
                custom_xlim = (0,max(xData)*1.05)
            ax.set_xticks(xticks)
            ax.legend(loc="upper right",fontsize=fontsize)

            plt.setp(
                ax,
                ylim=custom_ylim,
                xlim=custom_xlim
            )
            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1,left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1,left=0.15)

            opslaan = (savePath+f"CR_{halo}_{analysisParam}_Medians.pdf"
            )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return


def mass_pdf_versus_plot(
    dataDict,
    CRPARAMSHALO,
    halo,
    ylabel,
    xlimDict,
    snapRange,
    titleBool=False,
    DPI=150,
    Nbins = 150,
    xsize = 6.0,
    ysize = 6.0,
    colourmapMain = "tab10",
    lineStyleDict = {"with_CRs": "solid", "no_CRs": "-."},
):

    savePath = f"./Plots/{halo}/PDFs/"
    try:
        os.mkdir(savePath)
    except:
        pass
    else:
        pass

    Nsnaps = float(len(snapRange))

    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]
    for analysisParam in CRPARAMSHALO[selectKey0]['saveParams']:
        if analysisParam != "mass":
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
            Nkeys = len(list(dataDict.items()))
            for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
                loadpath = CRPARAMSHALO[selectKey]['simfile']
                if loadpath is not None :
                    print(f"{CRPARAMSHALO[selectKey]['resolution']}, @{CRPARAMSHALO[selectKey]['CR_indicator']}")
                    # Create a plot for each Temperature
                    xminlist = []
                    xmaxlist = []
                    yminlist = []
                    ymaxlist = []
                    patchList = []
                    labelList = []

                    try:
                        plotData = simDict[analysisParam].copy()
                        weightsData = simDict["mass"].copy()
                    except:
                        print(f"Variable {analysisParam} not found. Skipping plot...")
                        continue
                        
                    lineStyle = lineStyleDict[CRPARAMSHALO[selectKey]['CR_indicator']]

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if colourmapMain == "tab10":
                        colour = cmap(float(ii) / 10.)
                    else:
                        colour = cmap(float(ii) / float(Nkeys))

                    if analysisParam in CRPARAMSHALO[selectKey]['logParameters']:
                        plotData = np.log10(plotData)

                    xmin = np.nanmin(plotData[np.isfinite(plotData)])
                    xmax = np.nanmax(plotData[np.isfinite(plotData)])
                    xminlist.append(xmin)
                    xmaxlist.append(xmax)

                    if (
                        (np.isinf(xmin) == True)
                        or (np.isinf(xmax) == True)
                        or (np.isnan(xmin) == True)
                        or (np.isnan(xmax) == True)
                    ):
                        # print()
                        print("Data All Inf/NaN! Skipping entry!")
                        continue

                    try:
                        xBins = np.linspace(start=xlimDict[analysisParam]['xmin'], stop=xlimDict[analysisParam]['xmax'], num=Nbins)
                    except:
                        xBins = np.linspace(start=xmin, stop=xmax, num=Nbins)
                    else:
                        pass

                    currentAx = ax

                    hist, bin_edges = np.histogram(plotData,bins=xBins, weights = weightsData)

                    hist = hist/Nsnaps
                    hist = np.log10(hist)

                    yminlist.append(np.nanmin(hist[np.isfinite(hist)]))
                    ymaxlist.append(np.nanmax(hist[np.isfinite(hist)]))

                    xFromBins = np.array([(x1+x2)/2. for (x1,x2) in zip(bin_edges[:-1],bin_edges[1:])])

                    currentAx.plot(xFromBins,hist,label=f"{CRPARAMSHALO[selectKey]['resolution']}: {CRPARAMSHALO[selectKey]['CR_indicator']}", color=colour, linestyle= lineStyle)

                    currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

                    currentAx.set_ylabel(ylabel["mass"], fontsize=fontsize)


                    if titleBool is True:
                        fig.suptitle(
                            f"PDF of"+"\n"+f" mass vs {analysisParam}",
                            fontsize=fontsizeTitle,
                        )

                # Only give 1 x-axis a label, as they sharex


            ax.set_xlabel(ylabel[analysisParam], fontsize=fontsize)

            try:
                finalxmin = xlimDict[analysisParam]['xmin']
                finalxmax = xlimDict[analysisParam]['xmax']
            except:
                finalxmin = np.nanmin(xminlist)
                finalxmax = np.nanmax(xmaxlist)
            else:
                pass

            finalymin = np.nanmin(yminlist)
            finalymax = np.nanmax(ymaxlist)

            if (
                (np.isinf(finalxmin) == True)
                or (np.isinf(finalxmax) == True)
                or (np.isnan(finalxmin) == True)
                or (np.isnan(finalxmax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue
            finalxmin = numpy.round_(finalxmin, decimals = 2)
            finalxmax = numpy.round_(finalxmax, decimals = 2)
            finalymin = math.floor(finalymin)
            finalymax = math.ceil(finalymax)

            custom_xlim = (finalxmin, finalxmax)
            custom_ylim = (finalymin, finalymax)
            plt.setp(
                ax,
                xlim=custom_xlim,
                ylim=custom_ylim
            )
            ax.legend(loc="upper right",fontsize=fontsize)

            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1,left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1,left=0.15)

            opslaan = (savePath+f"CR_{halo}_{analysisParam}_PDF.pdf"
            )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return
