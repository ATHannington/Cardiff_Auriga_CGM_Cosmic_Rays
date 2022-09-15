import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
import OtherConstants as oc
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
from CR_Subroutines import *
from CR_Plotting_Tools import *
import h5py
import json
import copy
import math
import os


def round_it(x, sig):
    if x != 0:
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
    else:
        return 0.0


def medians_versus_plot(
    statsDict,
    CRPARAMSHALO,
    halo,
    ylabel,
    xlimDict,
    yParam=None,
    xParam="R",
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    opacityPercentiles=0.25,
    lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
    colourmapMain="tab10",
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
        plotParams = CRPARAMSHALO[selectKey0]["saveParams"]
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

                loadpath = CRPARAMSHALO[selectKeyShort]["simfile"]
                if loadpath is not None:
                    print(
                        f"{CRPARAMSHALO[selectKeyShort]['resolution']}, @{CRPARAMSHALO[selectKeyShort]['CR_indicator']}"
                    )
                    # Create a plot for each Temperature

                    plotData = simDict.copy()
                    xData = np.array(simDict[xParam].copy())

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if colourmapMain == "tab10":
                        colour = cmap(float(ii) / 10.0)
                    else:
                        colour = cmap(float(ii) / float(Nkeys))

                    lineStyle = lineStyleDict[
                        CRPARAMSHALO[selectKeyShort]["CR_indicator"]
                    ]
                    try:
                        loadPercentilesTypes = [
                            analysisParam + "_" + str(percentile) + "%"
                            for percentile in CRPARAMSHALO[selectKeyShort]["percentiles"]
                        ]
                    except:
                        print(
                            f"Variable {analysisParam} not found. Skipping plot..."
                        )
                        continue
                    LO = (
                        analysisParam
                        + "_"
                        + str(min(CRPARAMSHALO[selectKeyShort]["percentiles"]))
                        + "%"
                    )
                    UP = (
                        analysisParam
                        + "_"
                        + str(max(CRPARAMSHALO[selectKeyShort]["percentiles"]))
                        + "%"
                    )
                    median = analysisParam + "_" + "50.00%"

                    if analysisParam in CRPARAMSHALO[selectKeyShort]["logParameters"]:
                        for k, v in plotData.items():
                            plotData.update({k: np.log10(v)})

                    try:
                        ymin = np.nanmin(
                            plotData[LO][np.isfinite(plotData[LO])])
                        ymax = np.nanmax(
                            plotData[UP][np.isfinite(plotData[UP])])
                    except:
                        print(
                            f"Variable {analysisParam} not found. Skipping plot...")
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
                        label=f"{CRPARAMSHALO[selectKeyShort]['resolution']}: {CRPARAMSHALO[selectKeyShort]['CR_indicator']}",
                        color=colour,
                        lineStyle=lineStyle,
                    )

                    currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                    currentAx.tick_params(
                        axis="both", which="both", labelsize=fontsize)

                    currentAx.set_ylabel(
                        ylabel[analysisParam], fontsize=fontsize)

                    if titleBool is True:
                        if selectKey[-1] == "Stars":
                            fig.suptitle(
                                f"Median and Percentiles of"
                                + "\n"
                                + f" Stellar-{analysisParam} vs {xParam}",
                                fontsize=fontsizeTitle,
                            )

                        else:
                            fig.suptitle(
                                f"Median and Percentiles of"
                                + "\n"
                                + f" {analysisParam} vs {xParam}",
                                fontsize=fontsizeTitle,
                            )

                # Only give 1 x-axis a label, as they sharex

            ax.set_xlabel(ylabel[xParam], fontsize=fontsize)

            # try:
            #     finalymin = min(np.nanmin(yminlist),xlimDict[analysisParam]['xmin'])
            #     finalymax = max(np.nanmax(ymaxlist),xlimDict[analysisParam]['xmax'])
            # except:
            #     finalymin = np.nanmin(yminlist)
            #     finalymax = np.nanmax(ymaxlist)
            # else:
            #     pass
            if (len(yminlist) == 0) | (len(ymaxlist) == 0):
                print(
                    f"Variable {analysisParam} not found. Skipping plot..."
                )
                continue

            finalymin = np.nanmin(yminlist)
            finalymax = np.nanmax(ymaxlist)

            if (
                (np.isinf(finalymin) == True)
                or (np.isinf(finalymax) == True)
                or (np.isnan(finalymin) == True)
                or (np.isnan(finalymax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue

            custom_ylim = (finalymin, finalymax)

            # xticks = [round_it(xx,2) for xx in np.linspace(min(xData),max(xData),5)]
            # custom_xlim = (min(xData),max(xData)*1.05)
            # if xParam == "R":
            #     if CRPARAMSHALO[selectKeyShort]['analysisType'] == "cgm":
            #         ax.fill_betweenx([finalymin,finalymax],0,min(xData), color="tab:gray",alpha=opacityPercentiles)
            #         custom_xlim = (0,max(xData)*1.05)
            #     else:
            #         custom_xlim = (0,max(xData)*1.05)
            # ax.set_xticks(xticks)
            ax.legend(loc="best", fontsize=fontsize)

            plt.setp(
                ax,
                ylim=custom_ylim
                # ,xlim=custom_xlim
            )
            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            if selectKey[-1] == "Stars":
                opslaan = savePath + \
                    f"CR_{halo}_Stellar-{analysisParam}_Medians.pdf"
            else:
                opslaan = savePath + f"CR_{halo}_{analysisParam}_Medians.pdf"
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
    densityBool=True,
    DPI=150,
    Nbins=150,
    xsize=6.0,
    ysize=6.0,
    colourmapMain="tab10",
    lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
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

    plotParams = CRPARAMSHALO[selectKey0]["saveParams"]

    fontsize = CRPARAMSHALO[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMSHALO[selectKey0]["fontsizeTitle"]

    Rrange = np.around(
        np.linspace(
            start=xlimDict["R"]["xmin"],
            stop=xlimDict["R"]["xmax"],
            num=CRPARAMSHALO[selectKey0]["nRbins"],
        ),
        decimals=1,
    )

    # massSumDict = {}
    # for rinner, router in zip(Rrange[:-1], Rrange[1:]):
    #     Nkeys = len(list(dataDict.items()))
    #     tmp = []
    #     for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
    #         if selectKey[-1] == "Stars":
    #             continue
    #         loadpath = CRPARAMSHALO[selectKey]["simfile"]
    #         if loadpath is not None:
    #             # Create a plot for each Temperature
    #
    #             try:
    #                 weightsData = simDict["mass"].copy()
    #             except:
    #                 print(f"Variable {'mass'} not found. Skipping plot...")
    #                 continue
    #
    #             whereInRadius = np.where(
    #                 (simDict["R"] >= rinner) & (simDict["R"] < router)
    #             )[0]
    #
    #             weightsData = weightsData[whereInRadius].copy()
    #             tmp.append(np.sum(weightsData))
    #     massSumDict.update({f"{rinner}R{router}": np.array(tmp)})

    for analysisParam in CRPARAMSHALO[selectKey0]["saveParams"]:
        if (analysisParam != "mass") & (analysisParam != "R"):
            print("")
            print(f"Starting {analysisParam} plots!")

            Rrange = np.around(
                np.linspace(
                    start=xlimDict["R"]["xmin"],
                    stop=xlimDict["R"]["xmax"],
                    num=CRPARAMSHALO[selectKey0]["nRbins"],
                ),
                decimals=1,
            )
            for rinner, router in zip(Rrange[:-1], Rrange[1:]):
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
                    loadpath = CRPARAMSHALO[selectKey]["simfile"]
                    if loadpath is not None:
                        print(
                            f"{CRPARAMSHALO[selectKey]['resolution']}, @{CRPARAMSHALO[selectKey]['CR_indicator']}"
                        )
                        # Create a plot for each Temperature
                        skipBool = False
                        try:
                            plotData = simDict[analysisParam].copy()
                            weightsData = simDict["mass"].copy()
                            skipBool = False
                        except:
                            print(
                                f"Variable {analysisParam} not found. Skipping plot..."
                            )
                            skipBool = True
                            continue

                        lineStyle = lineStyleDict[
                            CRPARAMSHALO[selectKey]["CR_indicator"]
                        ]
                        whereInRadius = np.where(
                            (simDict["R"] >= rinner) & (simDict["R"] < router)
                        )[0]

                        plotData = plotData[whereInRadius].copy()
                        weightsData = weightsData[whereInRadius].copy()

                        cmap = matplotlib.cm.get_cmap(colourmapMain)
                        if colourmapMain == "tab10":
                            colour = cmap(float(ii) / 10.0)
                        else:
                            colour = cmap(float(ii) / float(Nkeys))

                        if analysisParam in CRPARAMSHALO[selectKey]["logParameters"]:
                            plotData = np.log10(plotData)

                        try:
                            xmin = np.nanmin(plotData[np.isfinite(plotData)])
                            xmax = np.nanmax(plotData[np.isfinite(plotData)])
                            skipBool = False
                        except:
                            print(
                                f"Variable {analysisParam} not found. Skipping plot...")
                            skipBool = True
                            continue
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
                            skipBool = True
                            continue

                        try:
                            xBins = np.linspace(
                                start=xlimDict[analysisParam]["xmin"],
                                stop=xlimDict[analysisParam]["xmax"],
                                num=Nbins,
                            )
                        except:
                            xBins = np.linspace(
                                start=xmin, stop=xmax, num=Nbins)
                        else:
                            pass

                        currentAx = ax

                        hist, bin_edges = np.histogram(
                            plotData,
                            bins=xBins,
                            weights=weightsData,
                            density=densityBool,
                        )

                        hist = hist / Nsnaps
                        # massSum = massSumDict[f"{rinner}R{router}"]
                        # if densityBool is True:
                        #     hist = hist * massSum[ii]/np.nanmax(massSum)
                        # else:
                        #     pass

                        if np.all(np.isfinite(hist)) == False:
                            print("Hist All Inf/NaN! Skipping entry!")
                            continue

                        try:
                            yminlist.append(np.nanmin(hist[np.isfinite(hist)]))
                            ymaxlist.append(np.nanmax(hist[np.isfinite(hist)]))
                            skipBool = False
                        except:
                            print(
                                f"Variable {analysisParam} not found. Skipping plot...")
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
                            label=f"{CRPARAMSHALO[selectKey]['resolution']}: {CRPARAMSHALO[selectKey]['CR_indicator']}",
                            color=colour,
                            linestyle=lineStyle,
                        )

                        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                        currentAx.tick_params(
                            axis="both", which="both", labelsize=fontsize
                        )

                        if densityBool is False:
                            currentAx.set_ylabel(
                                ylabel["mass"], fontsize=fontsize)
                        else:
                            currentAx.set_ylabel("PDF", fontsize=fontsize)

                        if titleBool is True:
                            fig.suptitle(
                                f"PDF of"
                                + "\n"
                                + f" mass vs {analysisParam}"
                                + +"\n"
                                + f"{rinner}<R<{router} kpc",
                                fontsize=fontsizeTitle,
                            )

                    # Only give 1 x-axis a label, as they sharex

                ax.set_xlabel(ylabel[analysisParam], fontsize=fontsize)

                if (skipBool == True):
                    print(
                        f"Variable {analysisParam} not found. Skipping plot...")
                    continue

                try:
                    finalxmin = max(
                        np.nanmin(xminlist), xlimDict[analysisParam]["xmin"]
                    )
                    finalxmax = min(
                        np.nanmax(xmaxlist), xlimDict[analysisParam]["xmax"]
                    )
                except:
                    finalxmin = np.nanmin(xminlist)
                    finalxmax = np.nanmax(xmaxlist)
                else:
                    pass

                if (
                    (np.isinf(finalxmax) == True)
                    or (np.isinf(finalxmin) == True)
                    or (np.isnan(finalxmax) == True)
                    or (np.isnan(finalxmin) == True)
                ):
                    print("Data All Inf/NaN! Skipping entry!")
                    continue

                try:
                    if densityBool is False:
                        finalymin = np.nanmin(yminlist)
                        finalymax = np.nanmax(ymaxlist)
                    else:
                        finalymin = 0.0
                        finalymax = np.nanmax(ymaxlist)
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

                opslaan = (
                    savePath
                    + f"CR_{halo}_{analysisParam}_{rinner:2.1f}R{router:2.1f}_PDF.pdf"
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
    xParam="R",
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    opacityPercentiles=0.25,
    lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
    colourmapMain="tab10",
):
    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]

    analysisParamList = ["mass", "gz"]

    savePath = (
        f"./Plots/{halo}/{CRPARAMSHALO[selectKey0]['analysisType']}/Mass_Summary/"
    )

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
    for analysisParam in analysisParamList:
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
                    if analysisParam != "mass":
                        print(f"No {selectKey[-1]} {analysisParam}! Skipping!")
                        continue
                else:
                    selectKeyShort = selectKey

                loadpath = CRPARAMSHALO[selectKeyShort]["simfile"]
                if loadpath is not None:
                    print(
                        f"{CRPARAMSHALO[selectKeyShort]['resolution']}, @{CRPARAMSHALO[selectKeyShort]['CR_indicator']}"
                    )
                    # Create a plot for each Temperature
                    try:
                        plotData = simDict[analysisParam].copy()
                        xData = simDict[xParam].copy()
                    except:
                        print(
                            f"Variable {analysisParam} not found. Skipping plot..."
                        )
                        continue
                    ind_sorted = np.argsort(xData)

                    # Sort the data
                    xData = xData[ind_sorted]
                    plotData = plotData[ind_sorted]
                    plotData = np.cumsum(plotData)

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if colourmapMain == "tab10":
                        colour = cmap(float(ii) / 10.0)
                    else:
                        colour = cmap(float(ii) / float(Nkeys))

                    lineStyle = lineStyleDict[
                        CRPARAMSHALO[selectKeyShort]["CR_indicator"]
                    ]

                    if analysisParam in CRPARAMSHALO[selectKeyShort]["logParameters"]:
                        plotData = np.log10(plotData)

                    try:
                        ymin = np.nanmin(plotData[np.isfinite(plotData)])
                        ymax = np.nanmax(plotData[np.isfinite(plotData)])
                    except:
                        print(
                            f"Variable {analysisParam} not found. Skipping plot...")
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
                    currentAx.tick_params(
                        axis="both", which="both", labelsize=fontsize)

                    currentAx.set_ylabel(
                        "Cumulative " + ylabel[analysisParam], fontsize=fontsize)

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

            if (selectKey[-1] == "Stars") & (analysisParam != "mass"):
                continue

            ax.set_xlabel(ylabel[xParam], fontsize=fontsize)

            # try:
            #     finalymin = min(np.nanmin(yminlist),xlimDict[analysisParam]['xmin'])
            #     finalymax = max(np.nanmax(ymaxlist),xlimDict[analysisParam]['xmax'])
            # except:
            #     finalymin = np.nanmin(yminlist)
            #     finalymax = np.nanmax(ymaxlist)
            # else:
            #     pass
            if (len(yminlist) == 0) | (len(ymaxlist) == 0):
                print(
                    f"Variable {analysisParam} not found. Skipping plot..."
                )
                continue

            finalymin = np.nanmin(yminlist)
            finalymax = np.nanmax(ymaxlist)
            if (
                (np.isinf(finalymin) == True)
                or (np.isinf(finalymax) == True)
                or (np.isnan(finalymin) == True)
                or (np.isnan(finalymax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue

            custom_ylim = (math.floor(finalymin), math.ceil(finalymax))

            # xticks = [round_it(xx,2) for xx in np.linspace(min(xData),max(xData),5)]
            # custom_xlim = (min(xData),max(xData)*1.05)
            # if xParam == "R":
            #     if CRPARAMSHALO[selectKeyShort]['analysisType'] == "cgm":
            #         ax.fill_betweenx([finalymin,finalymax],0,min(xData), color="tab:gray",alpha=opacityPercentiles)
            #         custom_xlim = (0,max(xData)*1.05)
            #     else:
            #         custom_xlim = (0,max(xData)*1.05)
            # ax.set_xticks(xticks)
            ax.legend(loc="best", fontsize=fontsize)

            plt.setp(
                ax,
                ylim=custom_ylim
                # ,xlim=custom_xlim
            )
            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            if selectKey[-1] == "Stars":
                opslaan = (
                    savePath
                    + f"CR_{halo}_Cumulative-Stellar-{analysisParam}-vs-{xParam}.pdf"
                )
            else:
                opslaan = (
                    savePath +
                    f"CR_{halo}_Cumulative-{analysisParam}-vs-{xParam}.pdf"
                )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return


def phases_plot(
    dataDict,
    CRPARAMSHALO,
    halo,
    ylabel,
    xlimDict,
    weightKeys=["mass",
                "Pthermal_Pmagnetic",
                "PCR_Pthermal",
                "P_thermal",
                "P_CR",
                "PCR_Pthermal",
                "gz",
                "tcool_tff"
                ],
    titleBool=False,
    DPI=150,
    xsize=8.0,
    ysize=8.0,
    lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
    colourmapMain="plasma",
    Nbins=250,
):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    zlimDict = copy.deepcopy(xlimDict)

    zlimDict.update({"rho_rhomean": {"xmin": 0.25, "xmax": 6.5}})
    zlimDict.update({"T": {"xmin": 3.75, "xmax": 7.0}})
    zlimDict.update({"tcool_tff": {"xmin": -2.5, "xmax": 2.0}})
    zlimDict.update({"gz": {"xmin": -1.0, "xmax": 0.25}})
    zlimDict.update({"Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 10.0}})

    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]

    savePath = f"./Plots/{halo}/{CRPARAMSHALO[selectKey0]['analysisType']}/Phases/"
    tmp = "./"

    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    fontsize = CRPARAMSHALO[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMSHALO[selectKey0]["fontsizeTitle"]
    # ------------------------------------------------------------------------------#
    #               PLOTTING
    #
    # ------------------------------------------------------------------------------#
    for weightKey in weightKeys:
        print("\n" + f"Starting weightKey {weightKey}")

        zmin = zlimDict[weightKey]["xmin"]
        zmax = zlimDict[weightKey]["xmax"]

        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(xsize, ysize),
            dpi=DPI,
            sharey=True,
            sharex=True,
        )
        Nkeys = len(list(dataDict.items()))
        for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
            if selectKey[-1] == "Stars":
                selectKeyShort = selectKey[:-1]
                print("[@phases_plot]: WARNING! Stars not supported! Skipping!")
                continue
            else:
                selectKeyShort = selectKey

            loadpath = CRPARAMSHALO[selectKeyShort]["simfile"]
            if loadpath is not None:
                print(
                    f"{CRPARAMSHALO[selectKeyShort]['resolution']}, @{CRPARAMSHALO[selectKeyShort]['CR_indicator']}"
                )

                row, col = np.unravel_index(np.array([ii]), shape=(2, 2))

                currentAx = ax[row[0], col[0]]

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                #   Figure 1: Full Cells Data
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                xdataCells = np.log10(
                    simDict["rho_rhomean"]
                )
                ydataCells = np.log10(simDict["T"])
                massCells = simDict["mass"]
                try:
                    weightDataCells = (
                        simDict[weightKey] * massCells
                    )
                    skipBool = False
                except:
                    print(
                        f"Variable {weightKey} not found. Skipping plot..."
                    )
                    skipBool = True
                    continue

                if weightKey == "mass":
                    finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=massCells
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
                    if weightKey in CRPARAMSHALO[selectKeyShort]['logParameters']:
                        finalHistCells = np.log10(finalHistCells)
                except:
                    print(f"Variable {weightKey} not found. Skipping plot...")
                    skipBool = True
                    continue
                finalHistCells = finalHistCells.T

                xcells, ycells = np.meshgrid(xedgeCells, yedgeCells)

                img1 = currentAx.pcolormesh(
                    xcells,
                    ycells,
                    finalHistCells,
                    cmap=colourmapMain,
                    vmin=zmin,
                    vmax=zmax,
                    rasterized=True,
                )
                #
                # img1 = currentAx.imshow(finalHistCells,cmap=colourmapMain,vmin=xmin,vmax=xmax \
                # ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

                currentAx.set_xlabel(
                    r"Log10 Density ($ \rho / \langle \rho \rangle $)",
                    fontsize=fontsize,
                )
                currentAx.set_ylabel(
                    "Log10 Temperatures (K)", fontsize=fontsize)

                currentAx.set_ylim(
                    zlimDict["T"]["xmin"], zlimDict["T"]["xmax"])
                currentAx.set_xlim(
                    zlimDict["rho_rhomean"]["xmin"], zlimDict["rho_rhomean"]["xmax"])
                currentAx.tick_params(
                    axis="both", which="both", labelsize=fontsize)

                currentAx.set_title(
                    f"{halo}:" + "\n" +
                    f"{CRPARAMSHALO[selectKeyShort]['resolution']} resolution {CRPARAMSHALO[selectKeyShort]['CR_indicator']}",
                    fontsize=fontsizeTitle,
                )
                currentAx.set_aspect("auto")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        #   Figure: Finishing up
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        if skipBool == True:
            try:
                tmp = finalHistCells
            except:
                print(
                    f"Variable {weightKey} not found. Skipping plot..."
                )
                continue
            else:
                pass

            #left, bottom, width, height
            # x0,    y0,  delta x, delta y
        cax1 = fig.add_axes([0.925, 0.10, 0.05, 0.80])

        fig.colorbar(img1, cax=cax1, ax=ax[:, -1].ravel().tolist(), orientation="vertical", pad=0.05).set_label(
            label=ylabel[weightKey], size=fontsize
        )
        cax1.yaxis.set_ticks_position("left")
        cax1.yaxis.set_label_position("left")
        cax1.yaxis.label.set_color("black")
        cax1.tick_params(axis="y", colors="black", labelsize=fontsize)

        if titleBool is True:
            fig.suptitle(
                f"Temperature Density Diagram, weighted by {weightKey}",
                fontsize=fontsizeTitle,
            )

        if titleBool is True:
            plt.subplots_adjust(top=0.875, right=0.8, hspace=0.3, wspace=0.3)
        else:
            plt.subplots_adjust(right=0.8, hspace=0.3, wspace=0.3)

        opslaan = (
            savePath
            + f"CR_{halo}_{weightKey}-Phases-Plot.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)

    return


def sfr_pdf_versus_time_plot(
    dataDict,
    CRPARAMSHALO,
    halo,
    snapRange,
    ylabel,
    titleBool=False,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    colourmapMain="tab10",
    lineStyleDict={"with_CRs": "-.", "no_CRs": "solid"},
):
    keys = list(CRPARAMSHALO.keys())
    selectKey0 = keys[0]

    savePath = f"./Plots/{halo}/{CRPARAMSHALO[selectKey0]['analysisType']}/SFR/"
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

    plotParams = CRPARAMSHALO[selectKey0]["saveParams"]

    fontsize = CRPARAMSHALO[selectKey0]["fontsize"]
    fontsizeTitle = CRPARAMSHALO[selectKey0]["fontsizeTitle"]


    analysisParam = "gima"
    xParam = "age"
    Nkeys = len(list(dataDict.items()))

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

    for (ii, (selectKey, simDict)) in enumerate(dataDict.items()):
        if selectKey[-1] == "Stars":
            selectKeyShort = selectKey[:-2]
        else:
            selectKeyShort = selectKey[:-1]

        loadpath = CRPARAMSHALO[selectKeyShort]["simfile"]
        if loadpath is not None:
            print(
                f"{CRPARAMSHALO[selectKeyShort]['resolution']}, @{CRPARAMSHALO[selectKeyShort]['CR_indicator']}"
            )
            xBins = np.around(
                np.linspace(
                    start=np.nanmin(dataDict[selectKey][xParam]),
                    stop=np.nanmax(dataDict[selectKey][xParam]),
                    num=CRPARAMSHALO[selectKeyShort]["NxParamBins"],
                ),
                decimals=1,
            )

            delta = np.mean(np.diff(xBins))

            # Create a plot for each Temperature
            skipBool = False
            try:
                plotData = simDict[xParam].copy()
                weightsData = simDict[analysisParam].copy()
                skipBool = False
            except:
                print(
                    f"Variable {analysisParam} not found. Skipping plot..."
                )
                skipBool = True
                continue

            lineStyle = lineStyleDict[
                CRPARAMSHALO[selectKeyShort]["CR_indicator"]
            ]

            cmap = matplotlib.cm.get_cmap(colourmapMain)
            if colourmapMain == "tab10":
                colour = cmap(float(ii) / 10.0)
            else:
                colour = cmap(float(ii) / float(Nkeys))

            try:
                xmin = np.nanmin(plotData[np.isfinite(plotData)])
                xmax = np.nanmax(plotData[np.isfinite(plotData)])
                skipBool = False
            except:
                print(
                    f"Variable {analysisParam} not found. Skipping plot...")
                skipBool = True
                continue
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
                skipBool = True

                continue

            hist, bin_edges = np.histogram(
                plotData,
                bins=xBins,
                weights=weightsData,
            )

            hist = hist / delta

            if np.all(np.isfinite(hist)) == False:
                print("Hist All Inf/NaN! Skipping entry!")
                continue

            if analysisParam in CRPARAMSHALO[selectKeyShort]["logParameters"]:
                hist = np.log10(hist)
            try:
                yminlist.append(np.nanmin(hist[np.isfinite(hist)]))
                ymaxlist.append(np.nanmax(hist[np.isfinite(hist)]))
                skipBool = False
            except:
                print(
                    f"Variable {analysisParam} not found. Skipping plot...")
                skipBool = True
                continue
            xFromBins = np.array(
                [
                    (x1 + x2) / 2.0
                    for (x1, x2) in zip(bin_edges[:-1], bin_edges[1:])
                ]
            )

            ax.plot(
                xFromBins,
                hist,
                label=f"{CRPARAMSHALO[selectKeyShort]['resolution']}: {CRPARAMSHALO[selectKeyShort]['CR_indicator']}",
                color=colour,
                linestyle=lineStyle,
            )

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(
                axis="both", which="both", labelsize=fontsize
            )

            ax.set_ylabel(ylabel[analysisParam], fontsize=fontsize)

            if titleBool is True:
                fig.suptitle(
                    f" Star Formation Rate vs {xParam}",
                    fontsize=fontsizeTitle,
                )

# Only give 1 x-axis a label, as they sharex

    ax.set_xlabel("Lookback Time (Gyr)", fontsize=fontsize)

    if (skipBool == True):
        print(
            f"Variable {analysisParam} not found. Skipping plot...")
    else:

        try:
            finalxmin = max(
                np.nanmin(xminlist), xlimDict[analysisParam]["xmin"]
            )
            finalxmax = min(
                np.nanmax(xmaxlist), xlimDict[analysisParam]["xmax"]
            )
        except:
            finalxmin = np.nanmin(xminlist)
            finalxmax = np.nanmax(xmaxlist)
        else:
            pass

        if (
            (np.isinf(finalxmax) == True)
            or (np.isinf(finalxmin) == True)
            or (np.isnan(finalxmax) == True)
            or (np.isnan(finalxmin) == True)
        ):
            print("Data All Inf/NaN! Skipping entry!")

        else:

            finalymin = 0.0
            finalymax = np.nanmax(ymaxlist)

            custom_xlim = (np.around(finalxmax, decimals = 2), np.around(finalxmin, decimals = 2))
            custom_ylim = (finalymin, finalymax)
            print(custom_xlim)
            print(custom_ylim)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            ax.legend(loc="best", fontsize=fontsize)

            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            opslaan = (
                savePath
                + f"CR_{halo}_SFR-vs-time.pdf"
            )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return

def cr_plot_projections(
    quadPlotDict,
    CRPARAMS,
    Axes=[0, 1],
    zAxis=[2],
    boxsize=400.0,
    boxlos=20.0,
    pixres=0.2,
    pixreslos=0.2,
    fontsize = 13,
    fontsizeTitle = 14,
    DPI=200,
    CMAP=None,
    numThreads=8,
    savePathKeyword = "",
):
    print(f"Starting Projections Video Plots!")

    keys = list(CRPARAMS.keys())
    selectKey0 = keys[0]

    savePathBase = f"./Plots/{CRPARAMS['halo']}/{CRPARAMS['analysisType']}/{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}/Images/"

    tmp = "./"
    for savePathChunk in savePathBase.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if CMAP == None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = CMAP

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["x", "y", "z"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#

    proj_T = quadPlotDict["T"]

    proj_dens = quadPlotDict["dens"]

    proj_nH = quadPlotDict["n_H"]

    proj_B = quadPlotDict["B"]

    proj_gz = quadPlotDict["gz"]
    # ------------------------------------------------------------------------------#
    # PLOTTING TIME
    # Set plot figure sizes
    xsize = 10.0
    ysize = 10.0
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0
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
        nrows=2, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
    )

    # cmap = plt.get_cmap(CMAP)
    cmap.set_bad(color="grey")

    # -----------#
    # Plot Temperature #
    # -----------#
    # print("pcm1")
    ax1 = axes[0, 0]

    pcm1 = ax1.pcolormesh(
        proj_T["x"],
        proj_T["y"],
        np.transpose(proj_T["grid"] / proj_dens["grid"]),
        vmin=1e4,
        vmax=10 ** (6.5),
        norm=matplotlib.colors.LogNorm(),
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
    ax2 = axes[0, 1]

    pcm2 = ax2.pcolormesh(
        proj_nH["x"],
        proj_nH["y"],
        np.transpose(proj_nH["grid"]) / int(boxlos / pixreslos),
        vmin=1e-6,
        vmax=1e-1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax2.set_title(r"Hydrogen Number Density Projection", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"n$_H$ (cm$^{-3}$)", size=fontsize, weight="bold"
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
        proj_gz["x"],
        proj_gz["y"],
        np.transpose(proj_gz["grid"]) / int(boxlos / pixreslos),
        vmin=1e-2,
        vmax=1e1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax3.set_title(f"Metallicity Projection", y=-0.2, fontsize=fontsize)

    cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
        label=r"$Z/Z_{\odot}$", size=fontsize, weight="bold"
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
        proj_B["x"],
        proj_B["y"],
        np.transpose(proj_B["grid"]) / int(boxlos / pixreslos),
        vmin=1e-3,
        vmax=1e1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax4.set_title(r"Magnetic Field Strength Projection",
                  y=-0.2, fontsize=fontsize)

    cax4 = inset_axes(ax4, width="5%", height="95%", loc="right")
    fig.colorbar(pcm4, cax=cax4, orientation="vertical").set_label(
        label=r"B ($ \mu $G)", size=fontsize, weight="bold"
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
    fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.95)

    # fig.tight_layout()

    savePathKeyword = savePathKeyword.zfill(4)
    savePath = savePathBase + f"Quad_Plot_{savePathKeyword}.png"

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()

    print(f" ...done!")

    return
