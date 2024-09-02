# coding=utf-8
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
import CR_Subroutines as cr
import Plotting_tools as apt
import h5py
import json
import copy
import os
import math

# =============================================================================#
#
#               USER DEFINED PARAMETERS
#
# ==============================================================================#
KnownAnalysisType = ["cgm", "ism", "all"]
singleValueKeys = ["Redshift", "Lookback", "Snap", "Rvir", "Rdisc"]

matplotlib.use("Agg")  # For suppressing plotting on clusters

DEBUG = False

keepPercentiles = []
# keepPercentiles = [
#     (f"high",f"no_CRs"),
#     (f"high",f"with_CRs"),
#     (f"high",f"with_CRs_no_Alfven"),
#     (f"high",f"with_CRs_CR29"),
#     (f"high",f"with_CRs_no_Alfven_CR29"),
# ]
    # (f"standard",f"no_CRs"),
    # (f"standard",f"with_CRs"),
    # (f"standard",f"with_CRs_no_Alfven"),

dataToInclude = [
    (f"high",f"no_CRs"),
    (f"high",f"with_CRs"),
    (f"high",f"with_CRs_no_Alfven"),
    (f"high",f"with_CRs_CR29"),
    (f"high",f"with_CRs_CR29_no_Alfven"),
]

ordering = [
    (f"high",f"no_CRs"),
    (f"high",f"with_CRs"),
    (f"high",f"with_CRs_no_Alfven"),
    (f"high",f"with_CRs_CR29"),
    (f"high",f"with_CRs_CR29_no_Alfven"),
    (f"standard",f"no_CRs"),
    (f"standard",f"with_CRs"),
    (f"standard",f"with_CRs_no_Alfven"),
    (f"standard",f"with_CRs_CR29"),
    (f"standard",f"with_CRs_CR29_no_Alfven"),
]
styleDictGroupingKeys = {skey : skey for skey in ordering}

# styleDictGroupingKeys = {
#     (f"high",f"no_CRs") : (f"high",f"no_CRs",f"CR28"),
#     (f"high",f"with_CRs"): (f"high",f"with_CRs",f"CR28"),
#     (f"high",f"with_CRs_no_Alfven"): (f"high",f"with_CRs_no_Alfven",f"CR28"),
#     (f"high",f"with_CRs_CR29"): (f"high",f"with_CRs",f"CR29"),
#     (f"high",f"with_CRs_CR29_no_Alfven"): (f"high",f"with_CRs_no_Alfven",f"CR29"),
#     (f"standard",f"no_CRs") : (f"standard",f"no_CRs",f"CR28"),
#     (f"standard",f"with_CRs"): (f"standard",f"with_CRs",f"CR28"),
#     (f"standard",f"with_CRs_no_Alfven"): (f"standard",f"with_CRs_no_Alfven",f"CR28"),
#     (f"standard",f"with_CRs_CR29"): (f"standard",f"with_CRs",f"CR29"),
#     (f"standard",f"with_CRs_CR29_no_Alfven"): (f"standard",f"with_CRs_no_Alfven",f"CR29"),
# }


customLegendLabels = {
            (f"high",f"no_CRs") : f"no CRs",
            (f"high",f"with_CRs") : f"CR28: with CRs",
            (f"high",f"with_CRs_no_Alfven") : f"CR28: with CRs \n no Alfven",
            (f"high",f"with_CRs_CR29") : f"CR29: with CRs",
            (f"high",f"with_CRs_CR29_no_Alfven"): f"CR29: with CRs \n no Alfven",
            (f"standard",f"no_CRs") : f"Std. Res. no CRs",
            (f"standard",f"with_CRs") : f"Std. Res. CR28: \n with CRs",
            (f"standard",f"with_CRs_no_Alfven") : f"Std. Res. CR28: \n with CRs  no Alfven",
            (f"standard",f"with_CRs_CR29") : f"Std. Res. CR29: \n with CRs",
            (f"standard",f"with_CRs_CR29_no_Alfven"): f"Std. Res. CR29: \n with CRs no Alfven",
            # (f"high",f"with_CRs_no_Alfven") : f"CR28: with CRs \n no Alfven",
            # (f"high",f"with_CRs_CR29") : f"CR29: with CRs",
            # (f"high",f"with_CRs_CR29_no_Alfven"): f"CR29: with CRs \n no Alfven",
            # (f"standard",f"no_CRs") : f"standard res. no CRs",
            # (f"standard",f"with_CRs") : f"standard res. CR28: \n with CRs",
            # (f"standard",f"with_CRs_no_Alfven") : f"standard res. CR28: \n with CRs  no Alfven",
            # (f"standard",f"with_CRs_CR29") : f"standard res. CR29: \n with CRs",
            # (f"standard",f"with_CRs_CR29_no_Alfven"): f"standard res. CR29: \n with CRs no Alfven",
 }
medianString = "50.00%"

stack = True
inplace = False
normalise = True
CRPARAMSPATHMASTER = "CRParams.json"
buckComparisonCMAP = "magma"


CRPARAMSMASTER = json.load(open(CRPARAMSPATHMASTER, "r"))

if CRPARAMSMASTER["ageWindow"] is not None:
    CRPARAMSMASTER["SFRBins"] = int(math.floor(CRPARAMSMASTER["ageWindow"]/CRPARAMSMASTER["windowBins"]))
else:
    CRPARAMSMASTER["SFRBins"]  = CRPARAMSMASTER["Nbins"] 

# File types for data save.
#   Full: full FullDict data
FullDataPathSuffix = f".h5"

FullDataPathSuffix = f".h5"

CRSELECTEDHALOESPATH = "CRSelectedHaloes.json"
CRSELECTEDHALOES = json.load(open(CRSELECTEDHALOESPATH, "r"))

ylabel = {
    "T": r"T (K)",
    "R": r"R/R$_{\mathrm{200c}}}$",
    "n_H": r"n$_{\mathrm{H}}$ (cm$^{-3}$)",
    "n_H_col": r"N$_{\mathrm{H}}$ (cm$^{-2}$)",
    "n_HI": r"n$_{\mathrm{HI}}$ (cm$^{-3}$)",
    "n_HI_col": r"N$_{\mathrm{HI}}$ (cm$^{-2}$)",
    "nh": r"Neutral Hydrogen Fraction",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"$v_{\mathrm{r}}$ (km s$^{-1}$)",
    "vrad_in": r"$v_{\mathrm{r}}$ (km s$^{-1}$)",
    "vrad_out": r"$v_{\mathrm{r}}$ (km s$^{-1}$)",
    "gz": r"Z/Z$_{\odot}$",
    "L": r"L" + "\n" + r"(kpc km s$^{-1}$)",
    "Pressure": r"P (erg cm$^{-3}$)",
    "P_thermal": r"P$_{\mathrm{Th}}$ (erg cm$^{-3}$)",
    "P_magnetic": r"P$_{\mathrm{B}}$ (erg cm$^{-3}$)",
    "P_kinetic": r"P$_{\mathrm{Kin}}$(erg cm$^{-3}$)",
    "P_tot": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "P_tot+k": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{\mathrm{Th}}$/P$_{\mathrm{B}}$",
    "P_CR": r"P$_{\mathrm{CR}}$ (erg cm$^{-3}$)",
    "PCR_Pmagnetic" : r"P$_{\mathrm{CR}}$/P$_{\mathrm{B}}$",
    "PCR_Pthermal": r"P$_{\mathrm{CR}}$/P$_{\mathrm{Th}}$",
    "gah": r"Alfvén Gas Heating (erg s$^{-1}$)",
    "bfld": r"$\mathbf{B}$ ($ \mu $G)",
    "Grad_T": r"||$\nabla$ T|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||$\nabla$ n$_{\mathrm{H}}$|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||$\nabla$ $\mathrm{B}$|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{\mathrm{CR}}$|| (erg kpc$^{-4}$)",
    "gima" : r"SFR (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfvén CR Cooling (erg s$^{-1}$)",
    "tcool": r"t$_{\mathrm{Cool}}$ (Gyr)",
    "theat": r"t$_{\mathrm{Heat}}$ (Gyr)",
    "tcross": r"t$_{\mathrm{Sound}}$ (Gyr)",
    "tff": r"t$_{\mathrm{FF}}$ (Gyr)",
    "tcool_tff": r"t$_{\mathrm{Cool}}$/t$_{\mathrm{FF}}$",
    "csound": r"c$_{\mathrm{s}}$ (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "rho": r"$\rho$ (M$_{\odot}$ kpc$^{-3}$)",
    "dens": r"$\rho$  (g cm$^{-3}$)",
    "ndens": r"n (cm$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
    "vol": r"Volume (kpc$^{3}$)",
    "age": "Lookback Time (Gyr)",
    "cool_length" : "Cooling Length (kpc)",
    "halo" : "FoF Halo",
    "subhalo" : "SubFind Halo",
    "x": r"x (kpc)",
    "y": r"y (kpc)",
    "z": r"z (kpc)",
    "count": r"Count per pixel",
    "e_CR": r"$\epsilon_{\mathrm{CR}}$ (eV cm$^{-3}$)",
}

colImagexlimDict ={
    "n_H_col": {"xmin": 19.0, "xmax": 21.5},
    "n_HI_col" : {"xmin": 14.0, "xmax": 21.5},
    "n_H": {"xmin": -5.5, "xmax": -2.5},
    }

imageCmapDict = {
    "Pressure": "tab10",
    "vrad": "seismic",
    "vrad_out": "Reds",
    "vrad_in": "Blues",
    "n_H": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
    "n_HI": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
    "n_H_col": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
    "n_HI_col": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
}
yaxisZeroLineDict = {
    "gz": True,
    "vrad": True,
    "Pthermal_Pmagnetic": True,
    "PCR_Pthermal": True,
    "PCR_Pmagnetic": True,
}

xlimDict = {
    "R": {}, #{"xmin": CRPARAMSMASTER["Rinner"], "xmax": CRPARAMSMASTER["Router"]},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 1.5, "xmax": 4.5},
    "T": {"xmin": 3.5, "xmax": 7.0},
    "n_H":  {"xmin": -6.0, "xmax": 1.0}, #{"xmin": -6.0, "xmax": -2.0},
    "n_HI" : {"xmin": -12.0, "xmax": -2.0},
    "n_H_col": {"xmin": 19.0, "xmax": 23.0},
    "n_HI_col" : {"xmin": 13.0, "xmax": 22.0},
    "B":{"xmin": -2.5, "xmax": 2.0},# {"xmin": -7.0, "xmax": 2.0},
    "vrad": {"xmin": -200.0, "xmax": 200.0},
    "vrad_in": {"xmin": -200.0, "xmax": 200.0},
    "vrad_out": {"xmin": -200.0, "xmax": 200.0},
    "gz": {"xmin": -2.0, "xmax": 1.0},
    "Pressure" : {"xmin": -16.0, "xmax": -10.0},
    "P_thermal": {"xmin": -16.0, "xmax": -12.0},
    "P_CR": {"xmin": -20.0, "xmax": -10.0},
    "PCR_Pthermal": {"xmin": -4.5, "xmax": 2.5},#{"xmin": -8.0, "xmax": 2.5},
    "PCR_Pmagnetic": {"xmin": -3.5, "xmax": 2.5},#{"xmin": -6.0, "xmax": 4.0},
    "Pthermal_Pmagnetic": {"xmin": -2.5, "xmax": 3.5},#{"xmin": -2.0, "xmax": 5.0},
    "P_magnetic": {"xmin": -23.0, "xmax": -13.0},
    "P_kinetic":  {"xmin": -16.0, "xmax": -10.0},
    "P_tot": {"xmin": -16.0, "xmax": -12.0},
    "P_tot+k": {"xmin": -15.5, "xmax": -11.5},
    "tcool": {"xmin": -4.0, "xmax": 4.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 5.5},
    "dens": {}, #{"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": -1.0},
    "rho" : {}, #{"xmin": 2.0, "xmax": 7.0},
    "vol": {},
    "cool_length" : {"xmin": -1.0, "xmax": 2.0},
    "csound" : {},
    "nh" : {"xmin": -7.0, "xmax": 1.0},
    "e_CR": {"xmin": -8.0, "xmax": 0.0},
}

for entry in CRPARAMSMASTER["logParameters"]:
    ylabel[entry] = r"$\mathrm{Log_{10}}$ " + ylabel[entry]
    ylabel[entry] = ylabel[entry].replace("(","[")
    ylabel[entry] = ylabel[entry].replace(")","]")

#   Perform forbidden log of Grad check
deleteParams = []
for entry in CRPARAMSMASTER["logParameters"]:
    entrySplit = entry.split("_")
    if (
        ("Grad" in entrySplit) &
        (np.any(np.isin(np.array(CRPARAMSMASTER["logParameters"]), np.array(
            "_".join(entrySplit[1:])))))
    ):
        deleteParams.append(entry)

for entry in deleteParams:
    CRPARAMSMASTER["logParameters"].remove(entry)


# ==============================================================================#
#
#          Main
#
# ==============================================================================#


def err_catcher(arg):
    raise Exception(f"Child Process died and gave error: {arg}")
    return


snapRange = [
    xx
    for xx in range(
        int(CRPARAMSMASTER["snapMin"]),
        int(CRPARAMSMASTER["snapMax"]) + 1,
        1,
    )
]

if __name__ == "__main__":

    styleKeys = []
    for halo, allSimsDict in CRSELECTEDHALOES.items():
        DataSavepathBase = CRPARAMSMASTER["savepathdata"]
        FigureSavepathBase = CRPARAMSMASTER["savepathfigures"]

        CRPARAMSHALO = {}
        selectKeysList = []
        for sim, simDict in allSimsDict.items():
            CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
            CRPARAMS.update({'halo': halo})
            selectKey = (f"{CRPARAMS['resolution']}", 
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")
            
            selectKeysList.append(selectKey)
            CRPARAMSHALO.update({selectKey: CRPARAMS})
            styleKeys.append(styleDictGroupingKeys[selectKey])

            if CRPARAMS['simfile'] is not None:
                analysisType = CRPARAMS["analysisType"]

                if analysisType not in KnownAnalysisType:
                    raise Exception(
                        f"ERROR! CRITICAL! Unknown analysis type: {analysisType}!"
                        + "\n"
                        + f"Availble analysis types: {KnownAnalysisType}"
                    )
            
            if CRPARAMS['analysisType'] == 'cgm':
                xlimDict["R"]['xmin'] = 0.0#CRPARAMS["Rinner"]
                xlimDict["R"]['xmax'] = CRPARAMS['Router']

            elif CRPARAMS['analysisType'] == 'ism':
                xlimDict["R"]['xmin'] = 0.0
                xlimDict["R"]['xmax'] = CRPARAMS['Rinner']
            else:
                xlimDict["R"]['xmin'] = 0.0
                xlimDict["R"]['xmax'] = CRPARAMS['Router']

            if ("R" in CRPARAMS["logParameters"]):
                xlimDict["R"]['xmax'] = np.log10(xlimDict["R"]['xmax'])
        # ----------------------------------------------------------------------#
        #  Plots...
        # ----------------------------------------------------------------------#
        
        tmpstyleDict = apt.get_linestyles_and_colours(styleKeys,colourmapMain="plasma",colourGroupBy=[f"no_CRs",f"with_CRs",f"with_CRs_no_Alfven",f"with_CRs_CR29",f"with_CRs_CR29_no_Alfven"],linestyleGroupBy=["high","standard"],lastColourOffset=0.0)

        styleDict = {}

        for selectKey,dd in zip(selectKeysList,tmpstyleDict.values()):
            styleDict.update({selectKey : dd})

        snapNumber=snapRange[-1]

        tmp = []
        for key in ordering:
            if key in keepPercentiles:
                tmp.append(key)

        keepPercentiles = copy.deepcopy(tmp)

        loadpath = CRPARAMS["savepathdata"]

        tmp = apt.cr_load_slice_plot_data(
            selectKeysList,
            CRPARAMSHALO,
            snapNumber,
            sliceParam = ["n_H","T","gz","vrad"],
            Axes = CRPARAMS["Axes"],
            projection=[False,False,False,False],
            loadPathBase = CRPARAMS["savepathdata"],
            loadPathSuffix = "",
            selectKeyLen=3,
            delimiter="-",
            stack = None,
            allowFindOtherAxesData = False,
            verbose = DEBUG,
            hush = not DEBUG,
            )

        orderedData = {}
        for key in ordering:
            orderedData.update({key : tmp[key]})

        tmpdict = apt.plot_slices(orderedData,
            ylabel=ylabel,
            xlimDict=xlimDict,
            logParameters = CRPARAMS["logParameters"],
            snapNumber=snapNumber,
            sliceParam = [["n_H","T","gz","vrad"]],
            Axes=CRPARAMS["Axes"],
            averageAcrossAxes = False,
            saveAllAxesImages = CRPARAMS["saveAllAxesImages"],
            xsize = CRPARAMS["xsizeImages"]*1.8,
            ysize = CRPARAMS["ysizeImages"]*1.8,
            fontsize = CRPARAMS["fontsize"]*1.2,
            colourmapMain = CRPARAMS["colourmapMain"],
            colourmapsUnique = imageCmapDict,
            boxsize = CRPARAMS["boxsize"],
            boxlos = CRPARAMS["boxlos"],
            pixreslos = CRPARAMS["pixreslos"],
            pixres = CRPARAMS["pixres"],
            projection = [[False,False,False,False]],
            DPI = CRPARAMS["DPI"],
            numthreads=CRPARAMS["numthreads"],
            savePathBase = CRPARAMS["savepathfigures"]+ f"/type-{analysisType}/{CRPARAMS['halo']}/",
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            saveFigure = CRPARAMS["SaveImages"],
            selectKeysList = dataToInclude,
            compareSelectKeysOn = CRPARAMS["compareSelectKeysOn"],
            subfigures = True,
            subfigureDatasetLabelsDict = customLegendLabels,
            subfigureDatasetLabelsBool = True,
            subfigureOffAlignmentAxisLabels = False,
            offAlignmentAxisLabels = None,
            inplace = False,
            replotFromData = True,
        )

        
        tmp = apt.cr_load_slice_plot_data(
            selectKeysList,
            CRPARAMSHALO,
            snapNumber,
            sliceParam = ["n_H_col","n_HI_col"],
            Axes = CRPARAMS["Axes"],
            projection=[True,True],
            loadPathBase = CRPARAMS["savepathdata"],
            loadPathSuffix = "",
            selectKeyLen=3,
            delimiter="-",
            stack = None,
            allowFindOtherAxesData = False,
            verbose = DEBUG,
            hush = not DEBUG,
            )
    

        orderedData = {}
        for key in ordering:
            orderedData.update({key : tmp[key]})

        variableAdjust = "2"

        tmp3 = {}
        for key in orderedData.keys():
            newkey = key[-1]
            inner = orderedData[key]
            for kk in inner.keys():
                dat = copy.deepcopy(inner[kk])
                if key[0] == "high":
                    newinnerkey = variableAdjust+kk
                else:
                    newinnerkey = kk  

                if newkey not in list(tmp3.keys()):
                    tmp3.update({newkey : {newinnerkey : dat}})
                else:
                    tmp3[newkey][newinnerkey] = dat


        for key in orderedData[list(orderedData.keys())[0]].keys():
            ylabel[variableAdjust+key] = ylabel[key]
            xlimDict[variableAdjust+key] = xlimDict[key]
            CRPARAMS["logParameters"] = CRPARAMS["logParameters"]+[variableAdjust+key]
            imageCmapDict[variableAdjust+key] = imageCmapDict[key]

        # # offAxisLabels = ["High Res." if xx[0]=="2" else "Standard Res." for xx in tmp3[list(tmp3.keys())[0]].keys()]
            
        plotOrder = ["n_H_col","2n_H_col","n_HI_col","2n_HI_col"]
        offAxisLabels = ["High Res." if xx[0]=="2" else "Standard Res." for xx in plotOrder]

        # Create variant of xlimDict specifically for images of col params
        tmpxlimDict = copy.deepcopy(xlimDict)

        # Add the col param specific limits to the xlimDict variant
        for key, value in colImagexlimDict.items():
            tmpxlimDict[key] = value

        dataToIncludeCol = [tuple(list(sKey)+["col"]) for sKey in dataToInclude]
        customLegendLabelsCol = {tuple(list(sKey)+["col"]):label for sKey,label in customLegendLabels.items()}

        tmpdict = apt.plot_slices(tmp3,
            ylabel=ylabel,
            xlimDict=tmpxlimDict,
            logParameters = CRPARAMS["logParameters"],
            snapNumber=snapNumber,
            sliceParam = [plotOrder],
            Axes=CRPARAMS["Axes"],
            averageAcrossAxes = CRPARAMS["averageAcrossAxes"],
            saveAllAxesImages = CRPARAMS["saveAllAxesImages"],
            xsize = CRPARAMS["xsizeImages"]*1.8,
            ysize = CRPARAMS["ysizeImages"]*1.8,
            fontsize = CRPARAMS["fontsize"]*1.2,
            colourmapMain = CRPARAMS["colourmapMain"],
            colourmapsUnique = imageCmapDict,
            boxsize = CRPARAMS["boxsize"],
            boxlos = CRPARAMS["boxlos"],
            pixreslos = CRPARAMS["pixreslos"],
            pixres = CRPARAMS["pixres"],
            projection = [[True,True,True,True]],
            DPI = CRPARAMS["DPI"],
            numthreads=CRPARAMS["numthreads"],
            savePathBase = CRPARAMS["savepathfigures"]+ f"/type-{analysisType}/{CRPARAMS['halo']}/Col-Projection-Mapped/",
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            saveFigure = CRPARAMS["SaveImages"],
            selectKeysList = None,
            compareSelectKeysOn = CRPARAMS["compareSelectKeysOn"],
            subfigures = True,
            subfigureDatasetLabelsDict = None,
            subfigureDatasetLabelsBool = True,
            subfigureOffAlignmentAxisLabels = True,
            offAlignmentAxisLabels = offAxisLabels,
            inplace = False,
            replotFromData = True,
        )

        snapNumber="Averaged"


        print(
            "\n" + f"[@{CRPARAMS['halo']}]: Time averaged Medians profile plots..."
        )
        matplotlib.rc_file_defaults()
        plt.close("all")     

        tmp = apt.cr_load_statistics_data(
            selectKeysList,
            CRPARAMSHALO,
            snapRange,
            loadPathBase = CRPARAMS["savepathdata"],
            loadFile = "statsDict",
            fileType = ".h5",
            stack = True,
            verbose = DEBUG,
            )

        statsOut = copy.deepcopy(tmp)    

        if (len(snapRange)>1)&(stack is True):
            for sKey, data in statsOut.items():
                dataCopy = copy.deepcopy(data)
                for key,dd in data.items():
                    for kk, value in dd.items():
                        dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})#np.nanpercentile(value,q=80.0,axis=-1,interpolation="lower")})
                statsOut[sKey].update(dataCopy)


        loadPercentilesTypes = [
                str(percentile) + "%"
            for percentile in CRPARAMS["percentiles"]
        ]

        fullStatsOut = copy.deepcopy(statsOut)
        for sKey, data in statsOut.items():
            if sKey not in keepPercentiles:
                dataCopy = copy.deepcopy(data)
                for key,dd in data.items():
                    if key not in keepPercentiles:
                        for kk, value in dd.items():
                            splitkk = kk.split("_")
                            perc = splitkk[-1]
                            if (medianString not in splitkk)&(perc in loadPercentilesTypes):
                                dataCopy[key].pop(kk)
                statsOut[sKey].update(dataCopy)


        # # for sKey, data in statsOut.items():
        # #     dataCopy = copy.deepcopy(data)
        # #     for key,dat in data.items():
        # #         tmpdat = copy.deepcopy(dat)
        # #         for pressureRatio in ["Pthermal_Pmagnetic","PCR_Pthermal","PCR_Pmagnetic"]:
        # #             pressures = pressureRatio.split("_")
        # #             pr1, pr2 = pressures
        # #             pr1proper = pr1[0]+"_"+pr1[1:]
        # #             pr2proper = pr2[0]+"_"+pr2[1:]
        # #             for percentile in CRPARAMS["percentiles"]:
        # #                 newPr = pressureRatio + "_" + f"{percentile:2.2f}" + "%"
        # #                 ps1 = pr1proper + "_" + f"{percentile:2.2f}" + "%"
        # #                 ps2 = pr2proper + "_" + f"{percentile:2.2f}" + "%"
        # #                 if newPr in list(dat.keys()):
        # #                     # print("updated")
        # #                     tmpdat[newPr] = copy.deepcopy(dat[ps1]/dat[ps2])
        # #         dataCopy[sKey] = copy.deepcopy(tmpdat)
        # #     statsOut[sKey] = copy.deepcopy(dataCopy)

        orderedData = {}
        for key in ordering:
            orderedData.update({key : statsOut[key]})

        apt.cr_medians_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel=ylabel,
            xlimDict=xlimDict,
            snapNumber=snapNumber,
            yParam=CRPARAMS["mediansParams"],
            xParam=CRPARAMS["xParam"],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            labels = customLegendLabels,
            yaxisZeroLine = yaxisZeroLineDict,
            DPI = CRPARAMS["DPI"],
            xsize = CRPARAMS["xsize"],
            ysize = CRPARAMS["ysize"],
            fontsize = CRPARAMS["fontsize"],
            fontsizeTitle = CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            opacityPercentiles = CRPARAMS["opacityPercentiles"],
            colourmapMain = CRPARAMS["colourmapMain"],
            colourmapsUnique = None,#imageCmapDict,
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            inplace = inplace,
            saveFigureData = False,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = dataToInclude,
            styleDict = styleDict,
            hush = not DEBUG,
            )

        apt.cr_medians_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel=ylabel,
            xlimDict=xlimDict,
            snapNumber=snapNumber,
            yParam=[["n_H"],["T"],["gz"],{"vrad": ["vrad_in","vrad_out"]},["B"]],
            xParam=CRPARAMS["xParam"],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            labels = customLegendLabels,
            yaxisZeroLine = yaxisZeroLineDict,
            DPI = CRPARAMS["DPI"],
            xsize = CRPARAMS["xsize"]*0.60*0.85*2.0,
            ysize = CRPARAMS["ysize"]*0.60*0.85,
            fontsize = CRPARAMS["fontsize"],
            fontsizeTitle = CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            opacityPercentiles = CRPARAMS["opacityPercentiles"],   
            colourmapMain = CRPARAMS["colourmapMain"],
            colourmapsUnique = None,#imageCmapDict,
            savePathBase = CRPARAMS["savepathfigures"]+"/Figure-A/",
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            subfigures = True,
            sharex = True,
            sharey = False,
            inplace = inplace,
            saveFigureData = False,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = dataToInclude,
            styleDict = styleDict,
            hush = not DEBUG,
            )

        apt.cr_medians_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel=ylabel,
            xlimDict=xlimDict,
            snapNumber=snapNumber,
            yParam=[["Pthermal_Pmagnetic"],["PCR_Pthermal"],["PCR_Pmagnetic"]],
            yaxisZeroLine = yaxisZeroLineDict,
            xParam=CRPARAMS["xParam"],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            labels = customLegendLabels,
            DPI = CRPARAMS["DPI"],
            xsize = CRPARAMS["xsize"]*0.60*0.85*2.0,
            ysize = CRPARAMS["ysize"]*0.60*0.85,
            fontsize = CRPARAMS["fontsize"],
            fontsizeTitle = CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            opacityPercentiles = CRPARAMS["opacityPercentiles"],   
            colourmapMain = CRPARAMS["colourmapMain"],
            colourmapsUnique = None,#imageCmapDict,
            savePathBase = CRPARAMS["savepathfigures"]+"/Figure-B/",
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            subfigures = True,
            sharex = True,
            sharey = False,
            inplace = inplace,
            saveFigureData = False,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = dataToInclude,
            styleDict = styleDict,
            hush = not DEBUG,
            )
        

        if len(CRPARAMS["colParams"])>0:
            print(
            "\n" + f"[@{CRPARAMS['halo']}]: Time averaged Column Density Medians profile plots..."
            )

            selectKeysListCol = [tuple(list(sKey)+["col"]) for sKey in selectKeysList]
            keepPercentilesCol = [tuple(list(sKey)+["col"]) for sKey in keepPercentiles]
            orderingCol = [tuple(list(sKey)+["col"]) for sKey in ordering]
            # # # Create variant of xlimDict specifically for images of col params
            # # tmpxlimDict = copy.deepcopy(xlimDict)

            # # # Add the col param specific limits to the xlimDict variant
            # # for key, value in colImagexlimDict.items():
            # #     tmpxlimDict[key] = value

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
            # # for key in additionalColParams:
            # #     tmpxlimDict[key] = {}

            cols = CRPARAMS["colParams"]+additionalColParams

            COLCRPARAMS= copy.deepcopy(CRPARAMS)
            COLCRPARAMS["saveParams"]=COLCRPARAMS["saveParams"]+cols

            COLCRPARAMSHALO = copy.deepcopy(CRPARAMSHALO)
            # # COLCRPARAMSHALO = {sKey: values for sKey,(_,values) in zip(selectKeysListCol,CRPARAMSHALO.items())}
            
            for kk in COLCRPARAMSHALO.keys():
                COLCRPARAMSHALO[kk]["saveParams"] = COLCRPARAMSHALO[kk]["saveParams"]+cols

            tmp = apt.cr_load_statistics_data(
                selectKeysListCol,
                COLCRPARAMSHALO,
                snapRange,
                loadPathBase = CRPARAMS["savepathdata"],
                loadFile = "colStatsDict",
                fileType = ".h5",
                stack = True,
                selectKeyLen=4,
                verbose = DEBUG,
                )

            statsOutCol = copy.deepcopy(tmp)    

            if (len(snapRange)>1)&(stack is True):
                for sKey, data in statsOutCol.items():
                    dataCopy = copy.deepcopy(data)
                    for key,dd in data.items():
                        for kk, value in dd.items():
                            dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                    statsOutCol[sKey].update(dataCopy)

            fullStatsOutCol = copy.deepcopy(statsOutCol)
            
            for sKey, data in statsOutCol.items():
                if sKey not in keepPercentilesCol:
                    dataCopy = copy.deepcopy(data)
                    for key,dd in data.items():
                        if key not in keepPercentilesCol:
                            for kk, value in dd.items():
                                splitkk = kk.split("_")
                                perc = splitkk[-1]
                                if (medianString not in splitkk)&(perc in loadPercentilesTypes):
                                    dataCopy[key].pop(kk)
                    statsOutCol[sKey].update(dataCopy)

            orderedData = {}
            for key in orderingCol:
                orderedData.update({key : statsOutCol[key]})

            apt.cr_medians_versus_plot(
                statsDict = orderedData,
                CRPARAMS = COLCRPARAMSHALO,
                ylabel=ylabel,
                xlimDict=xlimDict,
                snapNumber=snapNumber,
                yParam=COLCRPARAMS["colParams"],
                xParam=COLCRPARAMS["xParam"],
                titleBool=COLCRPARAMS["titleBool"],
                legendBool=COLCRPARAMS["legendBool"],
                separateLegend = True,
                labels = customLegendLabelsCol,
                DPI = COLCRPARAMS["DPI"],
                xsize = COLCRPARAMS["xsize"],
                ysize = COLCRPARAMS["ysize"],
                fontsize = COLCRPARAMS["fontsize"],
                fontsizeTitle = COLCRPARAMS["fontsizeTitle"],
                linewidth=COLCRPARAMS["linewidth"],
                opacityPercentiles = COLCRPARAMS["opacityPercentiles"],
                colourmapMain = COLCRPARAMS["colourmapMain"],
                colourmapsUnique = None,#imageCmapDict,
                savePathBase = COLCRPARAMS["savepathfigures"],
                savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                inplace = inplace,
                saveFigureData = False,
                replotFromData = True,
                combineMultipleOntoAxis = True,
                selectKeysList = selectKeysListCol,
                styleDict = styleDict,
                hush = not DEBUG,
            )

            apt.cr_medians_versus_plot(
                statsDict = orderedData,
                CRPARAMS = COLCRPARAMSHALO,
                ylabel=ylabel,
                xlimDict=xlimDict,
                snapNumber=snapNumber,
                yParam=[["n_H_col"],[ "n_HI_col" ]],
                xParam=COLCRPARAMS["xParam"],
                titleBool=COLCRPARAMS["titleBool"],
                legendBool= False,#COLCRPARAMS["legendBool"],
                separateLegend = False,
                labels = customLegendLabelsCol,
                DPI = COLCRPARAMS["DPI"],
                xsize = COLCRPARAMS["xsize"]*0.60*0.85*2.0,
                ysize = COLCRPARAMS["ysize"]*0.60*0.85,
                fontsize = COLCRPARAMS["fontsize"],
                fontsizeTitle = COLCRPARAMS["fontsizeTitle"],
                linewidth=COLCRPARAMS["linewidth"],
                opacityPercentiles = COLCRPARAMS["opacityPercentiles"],
                colourmapMain = COLCRPARAMS["colourmapMain"],
                colourmapsUnique = None,#imageCmapDict,
                savePathBase = COLCRPARAMS["savepathfigures"],
                savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                subfigures = True,
                sharex = True,
                sharey = False,
                inplace = inplace,
                saveFigureData = False,
                replotFromData = True,
                combineMultipleOntoAxis = True,
                selectKeysList = selectKeysListCol,
                styleDict = styleDict,
                hush = not DEBUG,
                )
            selectKey = (f"{CRPARAMS['resolution']}", 
                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

            print(
                "\n" + f"[@{CRPARAMS['halo']}]: Time averaged PDF of column Density gas plot"
            )
            matplotlib.rc_file_defaults()
            plt.close("all")     

            tmp = apt.cr_load_pdf_versus_plot_data(
                selectKeysListCol,
                COLCRPARAMSHALO,
                snapRange,
                weightKeys = COLCRPARAMS['nonMassWeightDict'],
                xParams = COLCRPARAMS["colParams"],
                cumulative = False,
                loadPathBase = COLCRPARAMS["savepathdata"],
                SFR = False,
                normalise = False,
                stack = True,
                selectKeyLen=4,
                verbose = DEBUG,
                hush = not DEBUG,
                )

            pdfOutCol = copy.deepcopy(tmp)    

            if (len(snapRange)>1)&(stack is True):
                for sKey, data in pdfOutCol.items():
                    dataCopy = copy.deepcopy(data)
                    for key,dd in data.items():
                        for kk, value in dd.items():
                            dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                    pdfOutCol[sKey].update(dataCopy)

            orderedData = {}
            for key in orderingCol:
                orderedData.update({key : pdfOutCol[key]})

            matplotlib.rc_file_defaults()
            plt.close("all")   
            apt.cr_pdf_versus_plot(
                orderedData,
                COLCRPARAMSHALO,
                ylabel,
                xlimDict,
                snapNumber = snapNumber,
                weightKeys = COLCRPARAMS['nonMassWeightDict'],
                xParams = COLCRPARAMS["colParams"],
                titleBool=COLCRPARAMS["titleBool"],
                legendBool=COLCRPARAMS["legendBool"],
                DPI=COLCRPARAMS["DPI"],
                xsize=COLCRPARAMS["xsize"]*0.75,
                ysize=COLCRPARAMS["ysize"]*0.75,
                fontsize=COLCRPARAMS["fontsize"],
                fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                linewidth=COLCRPARAMS["linewidth"],
                Nbins=COLCRPARAMS["Nbins"],
                ageWindow=None,
                cumulative = False,
                savePathBase = COLCRPARAMS["savepathfigures"],
                savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                saveFigureData = False,
                forceLogPDF = COLCRPARAMS["forceLogPDF"],
                SFR = False,
                normalise = normalise,
                verbose = DEBUG,
                inplace = inplace,
                replotFromData = True,
                combineMultipleOntoAxis = True,
                selectKeysList = None, #[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
                styleDict = styleDict,
                hush = not DEBUG,                                
            )
            matplotlib.rc_file_defaults()
            plt.close("all")   

            matplotlib.rc_file_defaults()
            plt.close("all")   
            apt.cr_pdf_versus_plot(
                orderedData,
                COLCRPARAMSHALO,
                ylabel,
                xlimDict,
                snapNumber = snapNumber,
                weightKeys = COLCRPARAMS['nonMassWeightDict'],
                xParams = COLCRPARAMS["colParams"],
                titleBool=COLCRPARAMS["titleBool"],
                legendBool=COLCRPARAMS["legendBool"],
                DPI=COLCRPARAMS["DPI"],
                xsize=COLCRPARAMS["xsize"]*0.75,
                ysize=COLCRPARAMS["ysize"]*0.75,
                fontsize=COLCRPARAMS["fontsize"],
                fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                linewidth=COLCRPARAMS["linewidth"],
                Nbins=COLCRPARAMS["Nbins"],
                ageWindow=None,
                cumulative = True,
                savePathBase = COLCRPARAMS["savepathfigures"],
                savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                saveFigureData = False,
                forceLogPDF = COLCRPARAMS["forceLogPDF"],
                SFR = False,
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                replotFromData = True,
                combineMultipleOntoAxis = True,
                selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
                styleDict = styleDict,
                hush = not DEBUG,                                
            )
            matplotlib.rc_file_defaults()
            plt.close("all")   


            if ((COLCRPARAMS["byType"] is True)&(len(COLCRPARAMS["pdfParams"])>0)):
                print(
                    "\n" + f"[@{COLCRPARAMS['halo']}]: Time averaged PDF of gas plot by particle type"
                )
                matplotlib.rc_file_defaults()
                plt.close("all")  
                possibleTypes = [0,1,2,3,4,5,6]

                for tp in possibleTypes:
                    print("Starting type load ",tp)          
                    tmp = apt.cr_load_pdf_versus_plot_data(
                        selectKeysListCol,
                        COLCRPARAMSHALO,
                        snapRange,
                        weightKeys = COLCRPARAMS['nonMassWeightDict'],
                        xParams = COLCRPARAMS["colParams"],
                        cumulative = False,
                        loadPathBase = COLCRPARAMS["savepathdata"],
                        loadPathSuffix = f"type{int(tp)}/",
                        SFR = False,
                        normalise = False,
                        stack = True,
                        selectKeyLen=4,
                        verbose = DEBUG,
                        hush = not DEBUG,
                    )
                    
                    loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
                    if loadedHasData == False:
                        print("\n" + f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
                        continue
            
                    binnedpdfOut = copy.deepcopy(tmp)    

                    if (len(snapRange)>1)&(stack is True):
                        for sKey, data in binnedpdfOut.items():
                            dataCopy = copy.deepcopy(data)
                            for key,dd in data.items():
                                for kk, value in dd.items():
                                    dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                            binnedpdfOut[sKey].update(dataCopy)

                    orderedData = {}
                    for key in orderingCol:
                        orderedData.update({key : binnedpdfOut[key]})


                    apt.cr_pdf_versus_plot(
                        orderedData,
                        COLCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber = snapNumber,
                        weightKeys = COLCRPARAMS['nonMassWeightDict'],
                        xParams = COLCRPARAMS["colParams"],
                        titleBool=COLCRPARAMS["titleBool"],
                        legendBool=COLCRPARAMS["legendBool"],
                        DPI=COLCRPARAMS["DPI"],
                        xsize=COLCRPARAMS["xsize"]*0.75,
                        ysize=COLCRPARAMS["ysize"]*0.75,
                        fontsize=COLCRPARAMS["fontsize"],
                        fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                        linewidth=COLCRPARAMS["linewidth"],
                        Nbins=COLCRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = False,
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        allSavePathsSuffix = f"type{int(tp)}/",
                        saveFigureData = False,
                        forceLogPDF = COLCRPARAMS["forceLogPDF"],
                        SFR = False,
                        normalise = normalise,
                        verbose = DEBUG,
                        inplace = inplace,
                        replotFromData = True,
                        combineMultipleOntoAxis = True,
                        selectKeysList = None,
                        styleDict = styleDict,
                        hush = not DEBUG,                                
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")

                    apt.cr_pdf_versus_plot(
                        orderedData,
                        COLCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber = snapNumber,
                        weightKeys = COLCRPARAMS['nonMassWeightDict'],
                        xParams = COLCRPARAMS["colParams"],
                        titleBool=COLCRPARAMS["titleBool"],
                        legendBool=COLCRPARAMS["legendBool"],
                        DPI=COLCRPARAMS["DPI"],
                        xsize=COLCRPARAMS["xsize"]*0.75,
                        ysize=COLCRPARAMS["ysize"]*0.75,
                        fontsize=COLCRPARAMS["fontsize"],
                        fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                        linewidth=COLCRPARAMS["linewidth"],
                        Nbins=COLCRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        allSavePathsSuffix = f"type{int(tp)}/",
                        saveFigureData = False,
                        forceLogPDF = COLCRPARAMS["forceLogPDF"],
                        SFR = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        replotFromData = True,
                        combineMultipleOntoAxis = True,
                        selectKeysList = None,
                        styleDict = styleDict,
                        hush = not DEBUG,                                
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")   
            
            if ((COLCRPARAMS["binByParam"] is True)&(len(COLCRPARAMS["pdfParams"])>0)):
                print(
                    "\n" + f"[@{COLCRPARAMS['halo']}]: Time averaged PDF of gas, binned by {COLCRPARAMS['xParam']} plot"
                )
                matplotlib.rc_file_defaults()
                plt.close("all")  
                tmpxlimDict = copy.deepcopy(xlimDict)

                if COLCRPARAMS['analysisType'] == 'cgm':
                    tmpxlimDict["R"]['xmin'] = 0.0#tmpCOLCRPARAMS["Rinner"]
                    tmpxlimDict["R"]['xmax'] = COLCRPARAMS['Router']

                elif COLCRPARAMS['analysisType'] == 'ism':
                    tmpxlimDict["R"]['xmin'] = 0.0
                    tmpxlimDict["R"]['xmax'] = COLCRPARAMS['Rinner']
                else:
                    tmpxlimDict["R"]['xmin'] = 0.0
                    tmpxlimDict["R"]['xmax'] = COLCRPARAMS['Router']

                
                binIndices = range(0,COLCRPARAMS["NParamBins"]+1,1)
                for ii,(lowerIndex,upperIndex) in enumerate(zip(binIndices[:-1],binIndices[1:])):
                    print("Starting Binned PDF plot load ",ii+1," of ",COLCRPARAMS["NParamBins"])
                    
                    bins = np.round(np.linspace(start=tmpxlimDict[COLCRPARAMS["xParam"]]["xmin"],stop=tmpxlimDict[COLCRPARAMS["xParam"]]["xmax"],num=COLCRPARAMS["NParamBins"]+1,endpoint=True),decimals=2)
                    
                    subdir = f"/{bins[lowerIndex]}-{COLCRPARAMS['xParam']}-{bins[upperIndex]}/"
                    tmp = apt.cr_load_pdf_versus_plot_data(
                        selectKeysListCol,
                        COLCRPARAMSHALO,
                        snapRange,
                        weightKeys = COLCRPARAMS['nonMassWeightDict'],
                        xParams = COLCRPARAMS["colParams"],
                        cumulative = False,
                        loadPathBase = COLCRPARAMS["savepathdata"],
                        loadPathSuffix = subdir,
                        SFR = False,
                        normalise = False,
                        stack = True,
                        selectKeyLen=4,
                        verbose = DEBUG,
                        hush = not DEBUG,
                    )
                    
                    loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
                    if loadedHasData == False:
                        print("\n" + f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
                        continue
            
                    binnedpdfOut = copy.deepcopy(tmp)    

                    if (len(snapRange)>1)&(stack is True):
                        for sKey, data in binnedpdfOut.items():
                            dataCopy = copy.deepcopy(data)
                            for key,dd in data.items():
                                for kk, value in dd.items():
                                    dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                            binnedpdfOut[sKey].update(dataCopy)
                            
                    orderedData = {}
                    for key in orderingCol:
                        orderedData.update({key : binnedpdfOut[key]})

                    apt.cr_pdf_versus_plot(
                        orderedData,
                        COLCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber = snapNumber,
                        weightKeys = COLCRPARAMS['nonMassWeightDict'],
                        xParams = COLCRPARAMS["colParams"],
                        titleBool=COLCRPARAMS["titleBool"],
                        legendBool=COLCRPARAMS["legendBool"],
                        DPI=COLCRPARAMS["DPI"],
                        xsize=COLCRPARAMS["xsize"]*0.75,
                        ysize=COLCRPARAMS["ysize"]*0.75,
                        fontsize=COLCRPARAMS["fontsize"],
                        fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                        linewidth=COLCRPARAMS["linewidth"],
                        Nbins=COLCRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = False,
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        allSavePathsSuffix = subdir,
                        saveFigureData = False,
                        forceLogPDF = COLCRPARAMS["forceLogPDF"],
                        SFR = False,
                        normalise = normalise,
                        verbose = DEBUG,
                        inplace = inplace,
                        replotFromData = True,
                        combineMultipleOntoAxis = True,
                        selectKeysList = None,
                        styleDict = styleDict,
                        hush = not DEBUG,                                
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")

                    apt.cr_pdf_versus_plot(
                        orderedData,
                        COLCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber = snapNumber,
                        weightKeys = COLCRPARAMS['nonMassWeightDict'],
                        xParams = COLCRPARAMS["colParams"],
                        titleBool=COLCRPARAMS["titleBool"],
                        legendBool=COLCRPARAMS["legendBool"],
                        DPI=COLCRPARAMS["DPI"],
                        xsize=COLCRPARAMS["xsize"]*0.75,
                        ysize=COLCRPARAMS["ysize"]*0.75,
                        fontsize=COLCRPARAMS["fontsize"],
                        fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                        linewidth=COLCRPARAMS["linewidth"],
                        Nbins=COLCRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        allSavePathsSuffix = subdir,
                        saveFigureData = False,
                        forceLogPDF = COLCRPARAMS["forceLogPDF"],
                        SFR = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        replotFromData = True,
                        combineMultipleOntoAxis = True,
                        selectKeysList = None,
                        styleDict = styleDict,
                        hush = not DEBUG,                                
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")   
            


        selectKey = (f"{CRPARAMS['resolution']}", 
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

        print(
            "\n" + f"[@{CRPARAMS['halo']}]: Time averaged PDF of gas plot"
        )
        matplotlib.rc_file_defaults()
        plt.close("all")     

        tmp = apt.cr_load_pdf_versus_plot_data(
            selectKeysList,
            CRPARAMSHALO,
            snapRange,
            weightKeys = CRPARAMS['nonMassWeightDict'],
            xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
            cumulative = False,
            loadPathBase = CRPARAMS["savepathdata"],
            SFR = False,
            normalise = False,
            stack = True,
            verbose = DEBUG,
            hush = not DEBUG,
            )

        pdfOut = copy.deepcopy(tmp)    

        if (len(snapRange)>1)&(stack is True):
            for sKey, data in pdfOut.items():
                dataCopy = copy.deepcopy(data)
                for key,dd in data.items():
                    for kk, value in dd.items():
                        dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                pdfOut[sKey].update(dataCopy)

        orderedData = {}
        for key in ordering:
            orderedData.update({key : pdfOut[key]})

        matplotlib.rc_file_defaults()
        plt.close("all")   
        apt.cr_pdf_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            weightKeys = CRPARAMS['nonMassWeightDict'],
            xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            DPI=CRPARAMS["DPI"],
            xsize=CRPARAMS["xsize"]*0.75,
            ysize=CRPARAMS["ysize"]*0.75,
            fontsize=CRPARAMS["fontsize"],
            fontsizeTitle=CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            Nbins=CRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = False,
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            forceLogPDF = CRPARAMS["forceLogPDF"],
            SFR = False,
            normalise = normalise,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            hush = not DEBUG,                                
        )
        matplotlib.rc_file_defaults()
        plt.close("all")   

        matplotlib.rc_file_defaults()
        plt.close("all")   
        apt.cr_pdf_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            weightKeys = CRPARAMS['nonMassWeightDict'],
            xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            DPI=CRPARAMS["DPI"],
            xsize=CRPARAMS["xsize"]*0.75,
            ysize=CRPARAMS["ysize"]*0.75,
            fontsize=CRPARAMS["fontsize"],
            fontsizeTitle=CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            Nbins=CRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = True,
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            forceLogPDF = CRPARAMS["forceLogPDF"],
            SFR = False,
            normalise = True,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            hush = not DEBUG,                                
        )
        matplotlib.rc_file_defaults()
        plt.close("all")   

        matplotlib.rc_file_defaults()
        plt.close("all")   
        apt.cr_pdf_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            weightKeys = CRPARAMS['nonMassWeightDict'],
            xParams = [CRPARAMS["xParam"], "n_H", "n_HI"],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            DPI=CRPARAMS["DPI"],
            xsize=CRPARAMS["xsize"]*0.75,
            ysize=CRPARAMS["ysize"]*0.75,
            fontsize=CRPARAMS["fontsize"],
            fontsizeTitle=CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            Nbins=CRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = False,
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            forceLogPDF = CRPARAMS["forceLogPDF"],
            SFR = False,
            normalise = normalise,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            hush = not DEBUG,                                
        )
        matplotlib.rc_file_defaults()
        plt.close("all")   

        matplotlib.rc_file_defaults()
        plt.close("all")   
        apt.cr_pdf_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            weightKeys = CRPARAMS['nonMassWeightDict'],
            xParams = [CRPARAMS["xParam"],"n_H", "n_HI"],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            DPI=CRPARAMS["DPI"],
            xsize=CRPARAMS["xsize"]*0.75,
            ysize=CRPARAMS["ysize"]*0.75,
            fontsize=CRPARAMS["fontsize"],
            fontsizeTitle=CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            Nbins=CRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = True,
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            forceLogPDF = CRPARAMS["forceLogPDF"],
            SFR = False,
            normalise = False,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            hush = not DEBUG,                                
        )
        matplotlib.rc_file_defaults()
        plt.close("all")  

        if ((CRPARAMS["byType"] is True)&(len(CRPARAMS["pdfParams"])>0)):
            print(
                "\n" + f"[@{CRPARAMS['halo']}]: Time averaged PDF of gas plot by particle type"
            )
            matplotlib.rc_file_defaults()
            plt.close("all")  
            possibleTypes = [0,1,2,3,4,5,6]

            for tp in possibleTypes:
                print("Starting type load ",tp)          
                tmp = apt.cr_load_pdf_versus_plot_data(
                    selectKeysList,
                    CRPARAMSHALO,
                    snapRange,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
                    cumulative = False,
                    loadPathBase = CRPARAMS["savepathdata"],
                    loadPathSuffix = f"type{int(tp)}/",
                    SFR = False,
                    normalise = False,
                    stack = True,
                    verbose = DEBUG,
                    hush = not DEBUG,
                )
                
                loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
                if loadedHasData == False:
                    print("\n" + f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
                    continue
        
                binnedpdfOut = copy.deepcopy(tmp)    

                if (len(snapRange)>1)&(stack is True):
                    for sKey, data in binnedpdfOut.items():
                        dataCopy = copy.deepcopy(data)
                        for key,dd in data.items():
                            for kk, value in dd.items():
                                dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                        binnedpdfOut[sKey].update(dataCopy)

                orderedData = {}
                for key in ordering:
                    orderedData.update({key : binnedpdfOut[key]})

                apt.cr_pdf_versus_plot(
                    orderedData,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
                    titleBool=CRPARAMS["titleBool"],
                    legendBool=CRPARAMS["legendBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"]*0.75,
                    ysize=CRPARAMS["ysize"]*0.75,
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    linewidth=CRPARAMS["linewidth"],
                    Nbins=CRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = False,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    allSavePathsSuffix = f"type{int(tp)}/",
                    saveFigureData = False,
                    forceLogPDF = CRPARAMS["forceLogPDF"],
                    SFR = False,
                    normalise = normalise,
                    verbose = DEBUG,
                    inplace = inplace,
                    replotFromData = True,
                    combineMultipleOntoAxis = True,
                    selectKeysList = None,
                    styleDict = styleDict,
                    hush = not DEBUG,                                
                )
                matplotlib.rc_file_defaults()
                plt.close("all")

                apt.cr_pdf_versus_plot(
                    orderedData,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
                    titleBool=CRPARAMS["titleBool"],
                    legendBool=CRPARAMS["legendBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"]*0.75,
                    ysize=CRPARAMS["ysize"]*0.75,
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    linewidth=CRPARAMS["linewidth"],
                    Nbins=CRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = True,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    allSavePathsSuffix = f"type{int(tp)}/",
                    saveFigureData = False,
                    forceLogPDF = CRPARAMS["forceLogPDF"],
                    SFR = False,
                    normalise = True,
                    verbose = DEBUG,
                    inplace = inplace,
                    replotFromData = True,
                    combineMultipleOntoAxis = True,
                    selectKeysList = None,
                    styleDict = styleDict,
                    hush = not DEBUG,                                
                )
                matplotlib.rc_file_defaults()
                plt.close("all")   

                apt.cr_pdf_versus_plot(
                    orderedData,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = [CRPARAMS["xParam"],"n_H", "n_HI"],
                    titleBool=CRPARAMS["titleBool"],
                    legendBool=CRPARAMS["legendBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"]*0.75,
                    ysize=CRPARAMS["ysize"]*0.75,
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    linewidth=CRPARAMS["linewidth"],
                    Nbins=CRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = True,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    allSavePathsSuffix = f"type{int(tp)}/",
                    saveFigureData = False,
                    forceLogPDF = CRPARAMS["forceLogPDF"],
                    SFR = False,
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    replotFromData = True,
                    combineMultipleOntoAxis = True,
                    selectKeysList = None,
                    styleDict = styleDict,
                    hush = not DEBUG,                                
                )
                matplotlib.rc_file_defaults()
                plt.close("all")

                apt.cr_pdf_versus_plot(
                    orderedData,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = [CRPARAMS["xParam"],"n_H", "n_HI"],
                    titleBool=CRPARAMS["titleBool"],
                    legendBool=CRPARAMS["legendBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"]*0.75,
                    ysize=CRPARAMS["ysize"]*0.75,
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    linewidth=CRPARAMS["linewidth"],
                    Nbins=CRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = True,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    allSavePathsSuffix = f"type{int(tp)}/",
                    saveFigureData = False,
                    forceLogPDF = CRPARAMS["forceLogPDF"],
                    SFR = False,
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    replotFromData = True,
                    combineMultipleOntoAxis = True,
                    selectKeysList = None,
                    styleDict = styleDict,
                    hush = not DEBUG,                                
                )
        if ((CRPARAMS["binByParam"] is True)&(len(CRPARAMS["pdfParams"])>0)):

            tmpxlimDict = copy.deepcopy(xlimDict)

            if CRPARAMS['analysisType'] == 'cgm':
                tmpxlimDict["R"]['xmin'] = 0.0#tmpCRPARAMS["Rinner"]
                tmpxlimDict["R"]['xmax'] = CRPARAMS['Router']

            elif CRPARAMS['analysisType'] == 'ism':
                tmpxlimDict["R"]['xmin'] = 0.0
                tmpxlimDict["R"]['xmax'] = CRPARAMS['Rinner']
            else:
                tmpxlimDict["R"]['xmin'] = 0.0
                tmpxlimDict["R"]['xmax'] = CRPARAMS['Router']

            

            print(
                "\n" + f"[@{CRPARAMS['halo']}]: Time averaged PDF of gas, binned by {CRPARAMS['xParam']} plot"
            )
            matplotlib.rc_file_defaults()
            plt.close("all")  

            binIndices = range(0,CRPARAMS["NParamBins"]+1,1)
            for ii,(lowerIndex,upperIndex) in enumerate(zip(binIndices[:-1],binIndices[1:])):
                print("Starting Binned PDF plot load ",ii+1," of ",CRPARAMS["NParamBins"])
                
                bins = np.round(np.linspace(start=tmpxlimDict[CRPARAMS["xParam"]]["xmin"],stop=tmpxlimDict[CRPARAMS["xParam"]]["xmax"],num=CRPARAMS["NParamBins"]+1,endpoint=True),decimals=2)
                
                subdir = f"/{bins[lowerIndex]}-{CRPARAMS['xParam']}-{bins[upperIndex]}/"
                tmp = apt.cr_load_pdf_versus_plot_data(
                    selectKeysList,
                    CRPARAMSHALO,
                    snapRange,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
                    cumulative = False,
                    loadPathBase = CRPARAMS["savepathdata"],
                    loadPathSuffix = subdir,
                    SFR = False,
                    normalise = False,
                    stack = True,
                    verbose = DEBUG,
                    hush = not DEBUG,
                )
                
                loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
                if loadedHasData == False:
                    print("\n" + f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
                    continue
        
                binnedpdfOut = copy.deepcopy(tmp)    

                if (len(snapRange)>1)&(stack is True):
                    for sKey, data in binnedpdfOut.items():
                        dataCopy = copy.deepcopy(data)
                        for key,dd in data.items():
                            for kk, value in dd.items():
                                dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                        binnedpdfOut[sKey].update(dataCopy)

                orderedData = {}
                for key in ordering:
                    orderedData.update({key : binnedpdfOut[key]})

                apt.cr_pdf_versus_plot(
                    orderedData,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
                    titleBool=CRPARAMS["titleBool"],
                    legendBool=CRPARAMS["legendBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"]*0.75,
                    ysize=CRPARAMS["ysize"]*0.75,
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    linewidth=CRPARAMS["linewidth"],
                    Nbins=CRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = False,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    allSavePathsSuffix = subdir,
                    saveFigureData = False,
                    forceLogPDF = CRPARAMS["forceLogPDF"],
                    SFR = False,
                    normalise = normalise,
                    verbose = DEBUG,
                    inplace = inplace,
                    replotFromData = True,
                    combineMultipleOntoAxis = True,
                    selectKeysList = None,
                    styleDict = styleDict,
                    hush = not DEBUG,                                
                )
                matplotlib.rc_file_defaults()
                plt.close("all") 

                apt.cr_pdf_versus_plot(
                    orderedData,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber = snapNumber,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = [CRPARAMS["xParam"],"n_H", "n_HI"],
                    titleBool=CRPARAMS["titleBool"],
                    legendBool=CRPARAMS["legendBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"]*0.75,
                    ysize=CRPARAMS["ysize"]*0.75,
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    linewidth=CRPARAMS["linewidth"],
                    Nbins=CRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = True,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    allSavePathsSuffix = subdir,
                    saveFigureData = False,
                    forceLogPDF = CRPARAMS["forceLogPDF"],
                    SFR = False,
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    replotFromData = True,
                    combineMultipleOntoAxis = True,
                    selectKeysList = None,
                    styleDict = styleDict,
                    hush = not DEBUG,                                
                )
                matplotlib.rc_file_defaults()
                plt.close("all")   



        selectKeyStars = (f"{CRPARAMS['resolution']}", 
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                "Stars")
        
        selectKeysListStars = [tuple(list(sKey)+["Stars"]) for sKey in selectKeysList]
        orderingStars = [tuple(list(sKey)+["Stars"]) for sKey in ordering]

        STARSCRPARAMS= copy.deepcopy(CRPARAMS)
        STARSCRPARAMSHALO = copy.deepcopy(CRPARAMSHALO)

        print(
            "\n" + f"[@{CRPARAMS['halo']}]: Time averaged PDF of stars plot"
        )
        matplotlib.rc_file_defaults()
        plt.close("all")     

        tmp = apt.cr_load_pdf_versus_plot_data(
            selectKeysListStars,
            STARSCRPARAMSHALO,
            snapRange,
            weightKeys = STARSCRPARAMS['nonMassWeightDict'],
            xParams = [STARSCRPARAMS["xParam"]],
            cumulative = False,
            loadPathBase = STARSCRPARAMS["savepathdata"],
            SFR = False,
            normalise = False,
            stack = True,
            selectKeyLen=4,
            verbose = DEBUG,
            hush = not DEBUG,
            )

        pdfOutStars = copy.deepcopy(tmp)    

        if (len(snapRange)>1)&(stack is True):
            for sKey, data in pdfOutStars.items():
                dataCopy = copy.deepcopy(data)
                for key,dd in data.items():
                    for kk, value in dd.items():
                        dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                pdfOutStars[sKey].update(dataCopy)

        orderedData = {}
        for key in orderingStars:
            orderedData.update({key : pdfOutStars[key]})

        matplotlib.rc_file_defaults()
        plt.close("all")   
        apt.cr_pdf_versus_plot(
            orderedData,
            STARSCRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            weightKeys = STARSCRPARAMS['nonMassWeightDict'],
            xParams = [STARSCRPARAMS["xParam"]],
            titleBool=STARSCRPARAMS["titleBool"],
            legendBool=STARSCRPARAMS["legendBool"],
            DPI=STARSCRPARAMS["DPI"],
            xsize=STARSCRPARAMS["xsize"]*0.75,
            ysize=STARSCRPARAMS["ysize"]*0.75,
            fontsize=STARSCRPARAMS["fontsize"],
            fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
            linewidth=STARSCRPARAMS["linewidth"],
            Nbins=STARSCRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = False,
            savePathBase = STARSCRPARAMS["savepathfigures"],
            savePathBaseFigureData = STARSCRPARAMS["savepathdata"],
            saveFigureData = False,
            forceLogPDF = CRPARAMS["forceLogPDF"],
            SFR = False,
            normalise = normalise,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            hush = not DEBUG,                                
        )
        matplotlib.rc_file_defaults()
        plt.close("all")   

        matplotlib.rc_file_defaults()
        plt.close("all")   
        apt.cr_pdf_versus_plot(
            orderedData,
            STARSCRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            weightKeys = STARSCRPARAMS['nonMassWeightDict'],
            xParams = [STARSCRPARAMS["xParam"]],
            titleBool=STARSCRPARAMS["titleBool"],
            legendBool=STARSCRPARAMS["legendBool"],
            DPI=STARSCRPARAMS["DPI"],
            xsize=STARSCRPARAMS["xsize"]*0.75,
            ysize=STARSCRPARAMS["ysize"]*0.75,
            fontsize=STARSCRPARAMS["fontsize"],
            fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
            linewidth=STARSCRPARAMS["linewidth"],
            Nbins=STARSCRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = True,
            savePathBase = STARSCRPARAMS["savepathfigures"],
            savePathBaseFigureData = STARSCRPARAMS["savepathdata"],
            saveFigureData = False,
            forceLogPDF = CRPARAMS["forceLogPDF"],
            SFR = False,
            normalise = False,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            hush = not DEBUG,                                
        )
        matplotlib.rc_file_defaults()
        plt.close("all")  

        selectKey = (f"{CRPARAMS['resolution']}", 
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

        print(
            "\n" + f"[@{CRPARAMS['halo']}]: Time averaged gas phases plots"
        )
        matplotlib.rc_file_defaults()
        plt.close("all")     

        tmp = apt.cr_load_phase_plot_data(
            selectKeysList,
            CRPARAMSHALO,
            snapRange,
            yParams = CRPARAMS["phasesyParams"],
            xParams = CRPARAMS["phasesxParams"],
            weightKeys = CRPARAMS["phasesColourbarParams"],
            loadPathBase = CRPARAMS["savepathdata"],
            stack = True,
            verbose = DEBUG,
            )

        phaseOut = copy.deepcopy(tmp)    

        if (len(snapRange)>1)&(stack is True):
            for sKey, data in phaseOut.items():
                dataCopy = copy.deepcopy(data)
                for key,dd in data.items():
                    for kk, value in dd.items():
                        dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
                phaseOut[sKey].update(dataCopy)

        matplotlib.rc_file_defaults()
        plt.close("all")
        apt.cr_phase_plot(
            phaseOut,
            CRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            yParams = CRPARAMS["phasesyParams"],
            xParams = CRPARAMS["phasesxParams"],
            colourBarKeys = CRPARAMS["phasesColourbarParams"],
            weightKeys = CRPARAMS["nonMassWeightDict"],
            titleBool=CRPARAMS["titleBool"],
            DPI=CRPARAMS["DPI"],
            xsize=CRPARAMS["xsize"],
            ysize=CRPARAMS["ysize"],
            fontsize=CRPARAMS["fontsize"],
            fontsizeTitle=CRPARAMS["fontsizeTitle"],
            colourmapMain= CRPARAMS["colourmapMain"],
            Nbins=CRPARAMS["Nbins"],
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = True,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            allowPlotsWithoutxlimits = False,
        )



        selectKey = (f"{CRPARAMS['resolution']}", 
                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                "Stars",
                f"{int(snapRange[-1])}"
            )

        selectKeysListStars = [tuple(list(sKey)+["Stars"]) for sKey in selectKeysList]
        orderingStars = [tuple(list(sKey)+["Stars"]) for sKey in ordering]

        print(
            "\n" + f"[@{CRPARAMS['halo']}]: SFR plot"
        )
        matplotlib.rc_file_defaults()
        plt.close("all")     

        tmp = apt.cr_load_pdf_versus_plot_data(
            selectKeysListStars,
            CRPARAMSHALO,
            [snapRange[-1]],
            xParams = ["age"],
            cumulative = False,
            loadPathBase = CRPARAMS["savepathdata"],
            SFR = True,
            normalise = False,
            stack = True,
            selectKeyLen=4,
            verbose = DEBUG,
            hush = not DEBUG,
            )

        pdfOutSFR = copy.deepcopy(tmp)    

        # if (len(snapRange)>1)&(stack is True):
        #     for sKey, data in pdfOut.items():
        #         dataCopy = copy.deepcopy(data)
        #         for key,dd in data.items():
        #             for kk, value in dd.items():
        #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #         pdfOut[sKey].update(dataCopy)

        orderedData = {}
        for key in orderingStars:
            orderedData.update({key : pdfOutSFR[key]})

        matplotlib.rc_file_defaults()
        plt.close("all")   
        apt.cr_pdf_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            xParams = ["age"],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            DPI=CRPARAMS["DPI"],
            xsize=CRPARAMS["xsize"]*0.75,
            ysize=CRPARAMS["ysize"]*0.75,
            fontsize=CRPARAMS["fontsize"],
            fontsizeTitle=CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            Nbins=CRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = False,
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            SFR = True,
            normalise = False,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            verbose = DEBUG,
            hush = not DEBUG,        
            )
        
        matplotlib.rc_file_defaults()
        plt.close("all")  

        apt.cr_pdf_versus_plot(
            orderedData,
            CRPARAMSHALO,
            ylabel,
            xlimDict,
            snapNumber = snapNumber,
            xParams = ["age"],
            titleBool=CRPARAMS["titleBool"],
            legendBool=CRPARAMS["legendBool"],
            DPI=CRPARAMS["DPI"],
            xsize=CRPARAMS["xsize"]*0.75,
            ysize=CRPARAMS["ysize"]*0.75,
            fontsize=CRPARAMS["fontsize"],
            fontsizeTitle=CRPARAMS["fontsizeTitle"],
            linewidth=CRPARAMS["linewidth"],
            Nbins=CRPARAMS["Nbins"],
            ageWindow=None,
            cumulative = True,
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            saveFigureData = False,
            SFR = True,
            normalise = False,
            inplace = inplace,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
            styleDict = styleDict,
            verbose = DEBUG,
            hush = not DEBUG,     
        )
        
        matplotlib.rc_file_defaults()
        plt.close("all")
        print(
        "\n" + f"Finished sim..."
        )
        matplotlib.rc_file_defaults()
        plt.close("all")
        print(
        "\n" + f"[@{halo}]: Finished halo..."
        )
    print(
        "\n"+
        "\n" + f"Finished completely! :)"
    )
