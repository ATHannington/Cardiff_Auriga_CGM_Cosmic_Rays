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

def round_it(x, sig):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

def medians_versus_plot(
    statsDict,
    CRPARAMSHALO,
    halo,
    ylabel,
    xlimDict,
    yParam = None,
    xParam = "R",
    titleBool=False,
    DPI=150,
    xsize = 6.0,
    ysize = 6.0,
    opacityPercentiles = 0.25,
    lineStyleDict = {"with_CRs": "solid", "no_CRs": "-."},
    colourmapMain = "tab10",
):

    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]


    savePath = f"./Plots/{halo}/{CRPARAMSHALO[selectKey0]['analysisType']}/Medians/"
    tmp = "./"
    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass


    if yParam is None:
        plotParams = CRPARAMSHALO[selectKey0]['saveParams']
    else:
        plotParams = [yParam]

    fontsize = CRPARAMSHALO[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMSHALO[selectKey0]["fontsizeTitle"]


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
            patchList = []
            labelList = []

            Nkeys = len(list(statsDict.items()))
            for (ii, (selectKey, simDict)) in enumerate(statsDict.items()):
                if selectKey[-1] == "Stars":
                    selectKeyShort = selectKey[:-1]
                else:
                    selectKeyShort = selectKey

                loadpath = CRPARAMSHALO[selectKeyShort]['simfile']
                if loadpath is not None :
                    print(f"{CRPARAMSHALO[selectKeyShort]['resolution']}, @{CRPARAMSHALO[selectKeyShort]['CR_indicator']}")
                    # Create a plot for each Temperature

                    plotData = simDict.copy()
                    xData = simDict[xParam].copy()

                    if CRPARAMSHALO[selectKeyShort]['analysisType'] == 'cgm':
                        xlimDict["R"]['xmin'] = simDict["maxDiskRadius"]
                        xlimDict["R"]['xmax'] = CRPARAMSHALO[selectKeyShort]['Router']

                    elif CRPARAMSHALO[selectKeyShort]['analysisType'] == 'disk':
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] = simDict["maxDiskRadius"]
                    else:
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] =  CRPARAMSHALO[selectKeyShort]['Router']

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if colourmapMain == "tab10":
                        colour = cmap(float(ii) / 10.)
                    else:
                        colour = cmap(float(ii) / float(Nkeys))

                    lineStyle = lineStyleDict[CRPARAMSHALO[selectKeyShort]['CR_indicator']]

                    loadPercentilesTypes = [
                        analysisParam + "_" + str(percentile) + "%"
                        for percentile in CRPARAMSHALO[selectKeyShort]["percentiles"]
                    ]
                    LO = analysisParam + "_" + str(min(CRPARAMSHALO[selectKeyShort]["percentiles"])) + "%"
                    UP = analysisParam + "_" + str(max(CRPARAMSHALO[selectKeyShort]["percentiles"])) + "%"
                    median = analysisParam + "_" + "50.00%"

                    if analysisParam in CRPARAMSHALO[selectKeyShort]['logParameters']:
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
                        label=f"{CRPARAMSHALO[selectKeyShort]['resolution']}: {CRPARAMSHALO[selectKeyShort]['CR_indicator']}",
                        color=colour,
                        lineStyle=lineStyle,
                    )

                    currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

                    currentAx.set_ylabel(ylabel[analysisParam], fontsize=fontsize)


                    if titleBool is True:
                        if selectKey[-1] == "Stars":
                            fig.suptitle(
                                f"Median and Percentiles of"+"\n"+f" Stellar-{analysisParam} vs {xParam}",
                                fontsize=fontsizeTitle,
                            )

                        else:
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

            custom_ylim = (finalymin, finalymax)

            xticks = [round_it(xx,2) for xx in np.linspace(min(xData),max(xData),5)]
            custom_xlim = (min(xData),max(xData)*1.05)
            if xParam == "R":
                if CRPARAMSHALO[selectKeyShort]['analysisType'] == "cgm":
                    ax.fill_betweenx([finalymin,finalymax],0,min(xData), color="tab:gray",alpha=opacityPercentiles)
                    custom_xlim = (0,max(xData)*1.05)
                else:
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

            if selectKey[-1] == "Stars":
                opslaan = (savePath+f"CR_{halo}_Stellar-{analysisParam}_Medians.pdf"
            )
            else:
                opslaan = (savePath+f"CR_{halo}_{analysisParam}_Medians.pdf"
                )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return


def mass_pdf_versus_by_radius_plot(
    dataDict,
    CRPARAMSHALO,
    halo,
    ylabel,
    xlimDict,
    snapRange,
    titleBool=False,
    densityBool = True,
    DPI=150,
    Nbins = 150,
    xsize = 6.0,
    ysize = 6.0,
    colourmapMain = "tab10",
    lineStyleDict = {"with_CRs": "solid", "no_CRs": "-."},
):
    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]


    savePath = f"./Plots/{halo}/{CRPARAMSHALO[selectKey0]['analysisType']}/PDFs/"
    tmp = "./"
    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    Nsnaps = float(len(snapRange))


    keys = list(CRPARAMSHALO.keys())

    plotParams = CRPARAMSHALO[selectKey0]['saveParams']

    fontsize = CRPARAMSHALO[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMSHALO[selectKey0]["fontsizeTitle"]
    for analysisParam in CRPARAMSHALO[selectKey0]['saveParams']:
        if (analysisParam != "mass")&(analysisParam != "R"):
            print("")
            print(f"Starting {analysisParam} plots!")

            if CRPARAMSHALO[selectKey0]['analysisType'] == 'cgm':
                xlimDict["R"]['xmin'] = dataDict[selectKey0]["maxDiskRadius"]
                xlimDict["R"]['xmax'] = CRPARAMSHALO[selectKey0]['Router']

            elif CRPARAMSHALO[selectKey0]['analysisType'] == 'disk':
                xlimDict["R"]['xmin'] = 0.0
                xlimDict["R"]['xmax'] = dataDict[selectKey0]["maxDiskRadius"]
            else:
                xlimDict["R"]['xmin'] = 0.0
                xlimDict["R"]['xmax'] =  CRPARAMSHALO[selectKey0]['Router']

            # xlimDict["R"]['xmin'] = dataDict[selectKey0]["maxDiskRadius"]
            Rrange = np.around(np.linspace(start=xlimDict["R"]["xmin"],stop=xlimDict["R"]["xmax"], num=CRPARAMSHALO[selectKey0]["nRbins"]),decimals=1)
            for rinner, router in zip(Rrange[:-1],Rrange[1:]):
                print(f"{rinner}<R<{router}!")
                fig, ax = plt.subplots(
                    nrows=1,
                    ncols=1,
                    sharex=True,
                    sharey=True,
                    figsize=(xsize, ysize),
                    dpi=DPI,
                )
                xminlist = []
                xmaxlist = []
                yminlist = []
                ymaxlist = []
                patchList = []
                labelList = []
                Nkeys = len(list(dataDict.items()))
                for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
                    if selectKey[-1] == "Stars":
                        continue
                    loadpath = CRPARAMSHALO[selectKey]['simfile']
                    if loadpath is not None :
                        print(f"{CRPARAMSHALO[selectKey]['resolution']}, @{CRPARAMSHALO[selectKey]['CR_indicator']}")
                        # Create a plot for each Temperature

                        try:
                            plotData = simDict[analysisParam].copy()
                            weightsData = simDict["mass"].copy()
                        except:
                            print(f"Variable {analysisParam} not found. Skipping plot...")
                            continue

                        lineStyle = lineStyleDict[CRPARAMSHALO[selectKey]['CR_indicator']]
                        whereInRadius = np.where((simDict["R"]>=rinner)&(simDict["R"]<router))[0]

                        plotData = plotData[whereInRadius].copy()
                        weightsData = weightsData[whereInRadius].copy()

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

                        hist, bin_edges = np.histogram(plotData,bins=xBins, weights = weightsData, density = densityBool)

                        hist = hist/Nsnaps
                        if densityBool is False:
                            hist = np.log10(hist)

                        yminlist.append(np.nanmin(hist[np.isfinite(hist)]))
                        ymaxlist.append(np.nanmax(hist[np.isfinite(hist)]))

                        xFromBins = np.array([(x1+x2)/2. for (x1,x2) in zip(bin_edges[:-1],bin_edges[1:])])

                        currentAx.plot(xFromBins,hist,label=f"{CRPARAMSHALO[selectKey]['resolution']}: {CRPARAMSHALO[selectKey]['CR_indicator']}", color=colour, linestyle= lineStyle)

                        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                        currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

                        if densityBool is False:
                            currentAx.set_ylabel(ylabel["mass"], fontsize=fontsize)
                        else:
                            currentAx.set_ylabel("PDF", fontsize=fontsize)

                        if titleBool is True:
                            fig.suptitle(
                                f"PDF of"+"\n"+f" mass vs {analysisParam}"+
                                +"\n"+f"{rinner}<R<{router} kpc",
                                fontsize=fontsizeTitle,
                            )

                    # Only give 1 x-axis a label, as they sharex


                ax.set_xlabel(ylabel[analysisParam], fontsize=fontsize)

                try:
                    finalxmin = min(np.nanmin(xminlist),xlimDict[analysisParam]['xmin'])
                    finalxmax = max(np.nanmax(xmaxlist),xlimDict[analysisParam]['xmax'])
                except:
                    finalxmin = np.nanmin(xminlist)
                    finalxmax = np.nanmax(xmaxlist)
                else:
                    pass

                if densityBool is False:
                    finalymin = np.nanmin(yminlist)
                    finalymax = np.nanmax(ymaxlist)
                else:
                    finalymin = 0.0
                    finalymax = np.nanmax(ymaxlist)

                if (
                    (np.isinf(finalxmin) == True)
                    or (np.isinf(finalxmax) == True)
                    or (np.isnan(finalxmin) == True)
                    or (np.isnan(finalxmax) == True)
                ):
                    print("Data All Inf/NaN! Skipping entry!")
                    continue

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

                opslaan = (savePath+f"CR_{halo}_{analysisParam}_{rinner:2.1f}R{router:2.1f}_PDF.pdf"
                )
                plt.savefig(opslaan, dpi=DPI, transparent=False)
                print(opslaan)
                plt.close()

    return

def cumulative_mass_versus_plot(
    dataDict,
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
    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]


    savePath = f"./Plots/{halo}/{CRPARAMSHALO[selectKey0]['analysisType']}/Mass_Summary/"

    tmp = "./"
    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    selectKeyOuter = keys[0]

    if selectKeyOuter[-1] == "Stars":
        selectKeyShort = selectKeyOuter[:-1]
    else:
        selectKeyShort = selectKeyOuter

    fontsize = CRPARAMSHALO[selectKeyOuter]["fontsize"]
    fontsizeTitle = CRPARAMSHALO[selectKeyOuter]["fontsizeTitle"]
    for analysisParam in ["mass"]:
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
            patchList = []
            labelList = []

            Nkeys = len(list(dataDict.items()))
            for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
                if selectKey[-1] == "Stars":
                    selectKeyShort = selectKey[:-1]
                else:
                    selectKeyShort = selectKey

                loadpath = CRPARAMSHALO[selectKeyShort]['simfile']
                if loadpath is not None :
                    print(f"{CRPARAMSHALO[selectKeyShort]['resolution']}, @{CRPARAMSHALO[selectKeyShort]['CR_indicator']}")
                    # Create a plot for each Temperature

                    plotData = simDict[analysisParam].copy()
                    xData = simDict[xParam].copy()

                    ind_sorted = np.argsort(xData)

                    # Sort the data
                    xData = xData[ind_sorted]
                    plotData = plotData[ind_sorted]
                    plotData = np.cumsum(plotData)

                    # xlimDict["R"]['xmin'] = simDict["maxDiskRadius"]
                    if CRPARAMSHALO[selectKeyShort]['analysisType'] == 'cgm':
                        xlimDict["R"]['xmin'] = simDict["maxDiskRadius"]
                        xlimDict["R"]['xmax'] = CRPARAMSHALO[selectKeyShort]['Router']

                    elif CRPARAMSHALO[selectKeyShort]['analysisType'] == 'disk':
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] = simDict["maxDiskRadius"]
                    else:
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] =  CRPARAMSHALO[selectKeyShort]['Router']

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if colourmapMain == "tab10":
                        colour = cmap(float(ii) / 10.)
                    else:
                        colour = cmap(float(ii) / float(Nkeys))

                    lineStyle = lineStyleDict[CRPARAMSHALO[selectKeyShort]['CR_indicator']]

                    if analysisParam in CRPARAMSHALO[selectKeyShort]['logParameters']:
                        plotData = np.log10(plotData)

                    try:
                        ymin = np.nanmin(plotData[np.isfinite(plotData)])
                        ymax = np.nanmax(plotData[np.isfinite(plotData)])
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

                    currentAx.plot(
                        xData,
                        plotData,
                        label=f"{CRPARAMSHALO[selectKeyShort]['resolution']}: {CRPARAMSHALO[selectKeyShort]['CR_indicator']}",
                        color=colour,
                        lineStyle=lineStyle,
                    )

                    currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

                    currentAx.set_ylabel(ylabel[analysisParam], fontsize=fontsize)


                    if titleBool is True:
                        if selectKey[-1] == "Stars":
                            fig.suptitle(
                                f"Cumulative Stellar-{analysisParam} vs {xParam}",
                                fontsize=fontsizeTitle,
                            )

                        else:
                            fig.suptitle(
                                f"Cumulative {analysisParam} vs {xParam}",
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

            custom_ylim = (finalymin, finalymax)

            xticks = [round_it(xx,2) for xx in np.linspace(min(xData),max(xData),5)]
            custom_xlim = (min(xData),max(xData)*1.05)
            if xParam == "R":
                if CRPARAMSHALO[selectKeyShort]['analysisType'] == "cgm":
                    ax.fill_betweenx([finalymin,finalymax],0,min(xData), color="tab:gray",alpha=opacityPercentiles)
                    custom_xlim = (0,max(xData)*1.05)
                else:
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


            if selectKey[-1] == "Stars":
                opslaan = (savePath+f"CR_{halo}_Cumulative-Stellar-{analysisParam}-vs-{xParam}.pdf"
            )
            else:
                opslaan = (savePath+f"CR_{halo}_Cumulative-{analysisParam}-vs-{xParam}.pdf"
            )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return
