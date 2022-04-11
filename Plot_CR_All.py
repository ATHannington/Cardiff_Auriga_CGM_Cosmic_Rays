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
import unittest

FullDataPathSuffix = f".h5"
xParam = "R"

CRPARAMSPATH = "CRParams.json"
CRPARAMSMASTERPATH = "CRParams.json"
CRPARAMSMASTER = load_cr_parameters(CRPARAMSMASTERPATH)

CRSELECTEDHALOESPATH = "CRSelectedHaloes.json"
CRSELECTEDHALOES = load_cr_parameters(CRSELECTEDHALOESPATH)

MustBeSameSettings = ["saveParams","saveEssentials","snapMin","snapMax"]

ylabel = {
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_H$ (cm$^{-3}$)",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"Radial Velocity (km s$^{-1}$)",
    "gz": r"Average Metallicity Z/Z$_{\odot}$",
    "L": r"Specific Angular Momentum" +"\n" + r"(kpc km s$^{-1}$)",
    "P_thermal": r"P$_{Thermal}$ / k$_B$ (K cm$^{-3}$)",
    "P_magnetic": r"P$_{Magnetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_kinetic": r"P$_{Kinetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_tot": r"P$_{tot}$ = (P$_{thermal}$ + P$_{magnetic}$)/ k$_B$" +"\n" + r"(K cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "tcool": r"Cooling Time (Gyr)",
    "theat": r"Heating Time (Gyr)",
    "tcross": r"Sound Crossing Cell Time (Gyr)",
    "tff": r"Free Fall Time (Gyr)",
    "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
    "csound": r"Sound Speed (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "mass": r"Log10 Mass per pixel (M/M$_{\odot}$)",
}

for entry in CRPARAMS['logParameters']:
    ylabel[entry] : r"$Log_{10}$" + ylabel[entry]

#==============================================================================#
#
#          Main Plotting...
#
#==============================================================================#

#------------------------------------------------------------------------------#
#       Load data...
#------------------------------------------------------------------------------#

dataDict = {}
for halo,allSimsDict in CRSELECTEDHALOES.items():
    dataDict.update({halo : None})
    for sim, simDict in allSimsDict.items():
        if simDict['simfile'] is not None:
            loadPath = CRPARAMSMASTER['savepath'] + CRPARAMSPATH
            TMPCRPARAMS= load_cr_parameters(loadPath)
            CRSELECTEDHALOES[halo][sim].update(TMPCRPARAMS)
            for key in MustBeSameSettings:
                unittest.assertCountEqual(TMPCRPARAMS[key],CRPARAMS[key], f"FAILURE! Loaded simulation {halo}: {sim} does not have the same base config as CRParams.json!")
            loadPath = CRPARAMSMASTER['savepath'] + f"{halo}/" + f"Data_CR__{TMPCRPARAMS['sim']['resolution']}_{TMPCRPARAMS['sim']['CR_indicator']}_CGM" + FullDataPathSuffix
            loadedData = hdf5_load(loadPath)
            dataDict[halo].update(loadedData)
#------------------------------------------------------------------------------#
#       Calculate statistics...
#------------------------------------------------------------------------------#
statsDict = {}
for halo ,allSimsDict in dataDict.items():
    statsDict.update({halo : None})
    for sim, simDict in allSimsDict.items():
        dat = cr_calculate_statistics(
            dataDict[halo],
            CRPARAMS[halo],
            xParam
        )
        selectKey = (f"{simDict['sim']['resolution']}",f"{simDict['sim']['CR_indicator']}")
        statsDict[halo].update({selectKey : dat})

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+#
#
#   Plots...
#
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+#

#------------------------------------------------------------------------------#
#       medians_versus_plot...
#------------------------------------------------------------------------------#
medians_versus_plot(
    statsDict,
    ylabel,
    xParam
)
