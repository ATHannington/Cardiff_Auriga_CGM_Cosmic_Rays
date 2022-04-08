import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

fontsize = 13
fontsizeTitle = 14

def medians_versus_plot(
    dataDict,
    CRPARAMS,
    saveParams,
    logParameters,
    ylabel,
    titleBool,
    DPI=150,
    xsize = 4.0,
    ysize = 8.0,
    opacityPercentiles = 0.25,
    lineStyleDict = {"with_CRs": "solid", "no_CRs": "-."},
    colourDict = {"with_CRs": "red", "no_CRs": "cyan"},
    DataSavepathSuffix=f".h5"
):

    xlimDict = {
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
    }


    for analysisParam in saveParams:
        if analysisParam != xParam:
            print("")
            print(f"Starting {analysisParam} Sub-plots!")

            print("")
            print("Loading Data!")
            # Create a plot for each Temperature
            fig, ax = plt.subplots(
                nrows=2,
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
            # for (ii,(resolution, pathsDict)) in enumerate(CRPARAMS['simfiles'].items()):
            #     print(f"{resolution}")
            #     for CR_indicator, loadpath in pathsDict.items():
            #         print(f"{CR_indicator}")
            #         if loadpath is not None :



                        xData = np.array(xData)
                        plotData = statsData

                        colour = colourDict[CR_indicator]
                        lineStyle = lineStyleDict[CR_indicator]

                        loadPercentilesTypes = [
                            analysisParam + "_" + str(percentile) + "%"
                            for percentile in CRPARAMS["percentiles"]
                        ]
                        LO = analysisParam + "_" + str(min(CRPARAMS["percentiles"])) + "%"
                        UP = analysisParam + "_" + str(max(CRPARAMS["percentiles"])) + "%"
                        median = analysisParam + "_" + "50.00%"

                        if analysisParam in logParameters:
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
                        print("")
                        print("Sub-Plot!")


                        currentAx = ax[ii]


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
                            label=f"{resolution}: {CR_indicator}",
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

                    opslaan = (f"./Plots/{resolution}/{CR_indicator}"+f"CR_{resolution}_{CR_indicator}"+f"_{analysisParam}_Medians.pdf"
                    )
                    plt.savefig(opslaan, dpi=DPI, transparent=False)
                    print(opslaan)
                    plt.close()

    return
