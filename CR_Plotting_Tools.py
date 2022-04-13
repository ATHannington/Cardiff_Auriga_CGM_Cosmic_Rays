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


def medians_versus_plot(
    statsDict,
    CRPARAMSHALO,
    halo,
    ylabel,
    xlimDict,
    xParam = "R",
    DPI=150,
    xsize = 4.0,
    ysize = 4.0,
    opacityPercentiles = 0.25,
    lineStyleDict = {"with_CRs": "solid", "no_CRs": "-."},
    colourDict = {"with_CRs": "red", "no_CRs": "cyan"},
):

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(xsize, ysize),
        dpi=DPI,
    )
    for selectKey, simDict in statsDict.items():
        loadpath = CRPARAMSHALO[selectKey]['simfile']
        if loadpath is not None :
            print(f"{CRPARAMSHALO[selectKey]['resolution']}, @{CRPARAMSHALO[selectKey]['CR_indicator']}")
            for analysisParam in CRPARAMSHALO[selectKey]['saveParams']:
                if analysisParam != xParam:
                    print("")
                    print(f"Starting {analysisParam} plots!")

                    # Create a plot for each Temperature
                    yminlist = []
                    ymaxlist = []
                    patchList = []
                    labelList = []

                    plotData = simDict.copy()
                    xData = simDict[xParam].copy()

                    colour = colourDict[CR_indicator]
                    lineStyle = lineStyleDict[CR_indicator]

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

                    ymin = np.nanmin(plotData[LO])
                    ymax = np.nanmax(plotData[UP])
                    yminlist.append(ymin)
                    ymaxlist.append(ymax)

                    if (
                        (np.isinf(ymin) == True)
                        or (np.isinf(ymax) == True)
                        or (np.isnan(ymin) == True)
                        or (np.isnan(ymax) == True)
                    ):
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
                        lineStyle=lineStyleMedian,
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


                ax[-1].set_xlabel(ylabel[xParam], fontsize=fontsize)

                finalymin = min(np.nanmin(yminlist),xlimDict[analysisParam]['xmin'])
                finalymax = max(np.nanmax(ymaxlist),xlimDict[analysisParam]['xmax'])
                if (
                    (np.isinf(finalymin) == True)
                    or (np.isinf(finalymax) == True)
                    or (np.isnan(finalymin) == True)
                    or (np.isnan(finalymax) == True)
                ):
                    print("Data All Inf/NaN! Skipping entry!")
                    continue
                finalymin = numpy.round_(finalymin, decimals = 1)
                finalymax = numpy.round_(finalymax, decimals = 1)

                custom_ylim = (finalymin, finalymax)
                plt.setp(
                    ax,
                    ylim=custom_ylim,
                    xlim=(max(xData), min(xData)),
                )
                axis0.legend(loc="upper right",fontsize=fontsize)

                plt.tight_layout()
                if titleBool is True:
                    plt.subplots_adjust(top=0.875, hspace=0.1,left=0.15)
                else:
                    plt.subplots_adjust(hspace=0.1,left=0.15)

                opslaan = (f"./Plots/{halo}/{CRPARAMSHALO[selectKey]['resolution']}/{CRPARAMSHALO[selectKey]['CR_indicator']}"+f"CR_{halo}_{CRPARAMSHALO[selectKey]['resolution']}_{CRPARAMSHALO[selectKey]['CR_indicator']}_{analysisParam}_Medians.pdf"
                )
                plt.savefig(opslaan, dpi=DPI, transparent=False)
                print(opslaan)
                plt.close()

    return
