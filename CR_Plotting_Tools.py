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
#
# CRParamsPath = "CRParams.json"
# CRPARAMS = json.load(open(CRParamsPath, 'r'))
#
# if (CRPARAMS['sim']['with_CRs'] is True):
#     CRPARAMS['sim']['CR_indicator'] = "with_CRs"
# else:
#     CRPARAMS['sim']['CR_indicator'] = "no_CRs"
#
#
# DataSavepathBase = CRPARAMS['savepath']
#
# CRPARAMS['finalSnap'] = copy.copy(CRPARAMS['snapMax'])
#
#
# ylabel = {
#     "T": r"Temperature (K)",
#     "R": r"Radius (kpc)",
#     "n_H": r"n$_H$ (cm$^{-3}$)",
#     "B": r"|B| ($ \mu $G)",
#     "vrad": r"Radial Velocity (km s$^{-1}$)",
#     "gz": r"Average Metallicity Z/Z$_{\odot}$",
#     "L": r"Specific Angular Momentum" +"\n" + r"(kpc km s$^{-1}$)",
#     "P_thermal": r"P$_{Thermal}$ / k$_B$ (K cm$^{-3}$)",
#     "P_magnetic": r"P$_{Magnetic}$ / k$_B$ (K cm$^{-3}$)",
#     "P_kinetic": r"P$_{Kinetic}$ / k$_B$ (K cm$^{-3}$)",
#     "P_tot": r"P$_{tot}$ = (P$_{thermal}$ + P$_{magnetic}$)/ k$_B$" +"\n" + r"(K cm$^{-3}$)",
#     "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
#     "tcool": r"Cooling Time (Gyr)",
#     "theat": r"Heating Time (Gyr)",
#     "tcross": r"Sound Crossing Cell Time (Gyr)",
#     "tff": r"Free Fall Time (Gyr)",
#     "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
#     "csound": r"Sound Speed (km s$^{-1}$)",
#     "rho_rhomean": r"$\rho / \langle \rho \rangle$",
#     "dens": r"Density (g cm$^{-3}$)",
#     "ndens": r"Number density (cm$^{-3}$)",
#     "mass": r"Log10 Mass per pixel (M/M$_{\odot}$)",
# }
#
# for entry in CRPARAMS['logParameters']:
#     ylabel[entry] : r"$Log_{10}$" + ylabel[entry]



def medians_versus_plot(
    statsDict,
    ylabel,
    xParam = "R",
    DPI=150,
    xsize = 4.0,
    ysize = 4.0,
    opacityPercentiles = 0.25,
    lineStyleDict = {"with_CRs": "solid", "no_CRs": "-."},
    colourDict = {"with_CRs": "red", "no_CRs": "cyan"},
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


    for halo, allSimsDict in CRSELECTEDHALOES.items():
        print(f"Starting {halo}!")
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )
        for simDict in allSimsDict.values():
            loadpath = simDict['sim']['simfile']
            if loadpath is not None :
                print(f"{simDict['resolution']}, @{simDict['CR_indicator']}")

                selectKey = (f"{simDict['resolution']}",f"{simDict['CR_indicator']}")
                for analysisParam in saveParams:
                    if analysisParam != xParam:
                        print("")
                        print(f"Starting {analysisParam} plots!")

                        # Create a plot for each Temperature
                        yminlist = []
                        ymaxlist = []
                        patchList = []
                        labelList = []

                        plotData = statsDict[halo][selectKey].copy()
                        xData = statsDict[halo][selectKey][xParam].copy()

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
                            label=f"{simDict['resolution']}: {simDict['CR_indicator']}",
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

                    opslaan = (f"./Plots/{halo}/{simDict['resolution']}/{simDict['CR_indicator']}"+f"CR_{halo}_{simDict['resolution']}_{simDict['CR_indicator']}_{analysisParam}_Medians.pdf"
                    )
                    plt.savefig(opslaan, dpi=DPI, transparent=False)
                    print(opslaan)
                    plt.close()

    return
