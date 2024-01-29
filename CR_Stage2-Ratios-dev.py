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
import re
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

matplotlib.use("Agg")  # For suppressing plotting on clusters

DEBUG = False
allowPlotsWithoutxlimits = True
comparisons = [["high","standard"],["with_CRs","no_CRs"],["with_CRs_no_Alfven","no_CRs"],["with_CRs_no_Alfven","with_CRs"]]

keepPercentiles = []
medianString = "50.00%"

stack = True
inplace = False
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


# To alter just units sections of labels...
# ctrl+f: regex: \s{1}\([\w\s\d$\*\-\^+_{}/\\]*\)
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
    "P_thermal": r"P$_{\mathrm{Th}}$ (erg cm$^{-3}$)",
    "P_magnetic": r"P$_{\mathrm{B}}$ (erg cm$^{-3}$)",
    "P_kinetic": r"P$_{\mathrm{Kin}}$(erg cm$^{-3}$)",
    "P_tot": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "P_tot+k": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{\mathrm{Th}}$/P$_{\mathrm{B}}$",
    "P_CR": r"P$_{\mathrm{CR}}$ (erg cm$^{-3}$)",
    "PCR_Pmagnetic" : r"P$_{\mathrm{CR}}$/P$_{\mathrm{B}}$",
    "PCR_Pthermal": r"P$_{\mathrm{CR}}$/P$_{\mathrm{Th}}$",
    "gah": r"Alfven Gas Heating (erg s$^{-1}$)",
    "bfld": r"$\mathbf{B}$ ($ \mu $G)",
    "Grad_T": r"||$\nabla$ T|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||$\nabla$ n$_{\mathrm{H}}$|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||$\nabla$ $\mathrm{B}$|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{\mathrm{CR}}$|| (erg kpc$^{-4}$)",
    "gima" : r"SFR (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfven CR Cooling (erg s$^{-1}$)",
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

xlimDict = {
    "R": {}, #{"xmin": CRPARAMSMASTER["Rinner"], "xmax": CRPARAMSMASTER["Router"]},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 1.5, "xmax": 4.5},
    "T": {"xmin": 3.5, "xmax": 7.0},
    "n_H": {"xmin": -6.0, "xmax": 1.0},
    "n_HI" : {"xmin": -13.0, "xmax": 0.0},
    "n_H_col": {"xmin": 19.0, "xmax": 21.5},
    "n_HI_col" : {"xmin": 12.0, "xmax": 21.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -200.0, "xmax": 200.0},
    "vrad_in": {"xmin": -200.0, "xmax": 200.0},
    "vrad_out": {"xmin": -200.0, "xmax": 200.0},
    "gz": {"xmin": -2.0, "xmax": 1.0},
    "P_thermal": {"xmin": -16.0, "xmax": -10.0},
    "P_CR": {"xmin": -19.5, "xmax": -10.0},
    "PCR_Pthermal": {"xmin": -4.5, "xmax": 2.5},
    "PCR_Pmagnetic": {"xmin": -3.5, "xmax": 2.5},
    "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 4.0},
    "P_magnetic": {"xmin": -19.5, "xmax": -10.0},
    "P_kinetic": {"xmin": -19.5, "xmax": -10.0},
    "P_tot": {"xmin": -19.5, "xmax": -10.0},
    "P_tot+k": {"xmin": -19.5, "xmax": -10.0},
    "tcool": {"xmin": -4.0, "xmax": 4.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 6.0},
    "dens": {"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": 2.0},
    "rho_rhomean": {"xmin": 0.25, "xmax": 6.5},
    "rho" : {"xmin": 2.0, "xmax": 7.0},
    "vol": {},
    "cool_length" : {"xmin": -1.0, "xmax": 2.0},
    "csound" : {},
    "nh" : {"xmin": -7.0, "xmax": 1.0},
    "e_CR": {"xmin": -8.0, "xmax": 0.0},
}

ylabelratio = {key: "of " + val + " ratio" for key, val in ylabel.items()}


for entry in CRPARAMSMASTER["logParameters"]:
    ylabel[entry] = r"$\mathrm{Log_{10}}$ " + ylabel[entry]
    ylabel[entry] = ylabel[entry].replace("(","[")
    ylabel[entry] = ylabel[entry].replace(")","]")

for entry in CRPARAMSMASTER["logParameters"]:
    ylabelratio[entry] = r"$\mathrm{Log_{10}}$ " + ylabelratio[entry]
    ylabelratio[entry] = ylabelratio[entry].replace("(","[")
    ylabelratio[entry] = ylabelratio[entry].replace(")","]")

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


ylabelratio = {key: re.sub(r"\s*[\(\[]+[\w\s\d$\*\-\^+_{}/\\]+[\)\]]+","",val) for key, val in ylabelratio.items()}

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

        
        # ----------------------------------------------------------------------#
        #  Plots...
        # ----------------------------------------------------------------------#

        

        print(
            f"[@{CRPARAMS['halo']}]: Time averaged Medians profile plots..."
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
                        dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
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

        comparisonsFeedbackModels = comparisons[1:]

        comparisonDictFeedback = cr.cr_stats_ratios(statsOut,comparisonsFeedbackModels,exclusions=[CRPARAMS["xParam"]],verbose = DEBUG)

        styleDict = apt.get_linestyles_and_colours(list(comparisonDictFeedback.keys()),colourmapMain=buckComparisonCMAP,colourGroupBy=[],linestyleGroupBy=["standard", "high"],lastColourOffset=0.0)

        # nolog10CRPARAMS = copy.deepcopy(CRPARAMS)
        # nolog10CRPARAMS["logParameters"] = []

        tmpCRPARAMSHALO = {key : CRPARAMS for key in comparisonDictFeedback.keys()}

        
        apt.cr_medians_versus_plot(
            comparisonDictFeedback,
            tmpCRPARAMSHALO,
            ylabel=ylabelratio,
            xlimDict=None,
            snapNumber="Feedback-Model-Ratios-of-Averaged",
            yParam=CRPARAMS["mediansParams"],
            xParam=CRPARAMS["xParam"],
            titleBool=CRPARAMS["titleBool"],
            DPI = CRPARAMS["DPI"],
            xsize = CRPARAMS["xsize"],
            ysize = CRPARAMS["ysize"],
            fontsize = CRPARAMS["fontsize"],
            fontsizeTitle = CRPARAMS["fontsizeTitle"],
            opacityPercentiles = CRPARAMS["opacityPercentiles"],
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            inplace = inplace,
            saveFigureData = False,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            allowPlotsWithoutxlimits = allowPlotsWithoutxlimits,
            selectKeysList = None,
            styleDict = styleDict,
            )
        
        comparisonDictRes = cr.cr_stats_ratios(statsOut,comparisons=[comparisons[0]],exclusions=[CRPARAMS["xParam"]],verbose = DEBUG)

        styleDict = apt.get_linestyles_and_colours(list(comparisonDictRes.keys()),colourmapMain=buckComparisonCMAP,colourGroupBy=[],linestyleGroupBy=["no_CRs", "with_CRs_no_Alfven", "with_CRs"],lastColourOffset=0.0)

        tmpCRPARAMSHALO = {key : CRPARAMS for key in comparisonDictRes.keys()}


        apt.cr_medians_versus_plot(
            comparisonDictRes,
            tmpCRPARAMSHALO,
            ylabel=ylabelratio,
            xlimDict=None,
            snapNumber="Resolution-Ratios-of-Averaged",
            yParam=CRPARAMS["mediansParams"],
            xParam=CRPARAMS["xParam"],
            titleBool=CRPARAMS["titleBool"],
            DPI = CRPARAMS["DPI"],
            xsize = CRPARAMS["xsize"],
            ysize = CRPARAMS["ysize"],
            fontsize = CRPARAMS["fontsize"],
            fontsizeTitle = CRPARAMS["fontsizeTitle"],
            opacityPercentiles = CRPARAMS["opacityPercentiles"],
            savePathBase = CRPARAMS["savepathfigures"],
            savePathBaseFigureData = CRPARAMS["savepathdata"],
            inplace = inplace,
            saveFigureData = False,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            allowPlotsWithoutxlimits = allowPlotsWithoutxlimits,
            selectKeysList = None,
            styleDict = styleDict,
            )
        
        comparisonDict = cr.cr_stats_ratios(fullStatsOut,comparisons=comparisons,exclusions=[CRPARAMS["xParam"]],verbose = DEBUG)
        cr.cr_save_to_excel(
            comparisonDict,
            CRPARAMSHALO,
            savePathBase = FigureSavepathBase,
            filename= "CR-Data_Full-Statistics_Ratios.xlsx"
        )
    # if ("Stars" in selectKey):
    #     simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}/Stars/"
    # elif ("col" in selectKey):
    #     simSavePath = f"type-{tmpCRPARAMS[selectKeyShortest]['analysisType']}/{tmpCRPARAMS[selectKeyShortest]['halo']}/Col-Projection-Mapped/"
    # else:
    # simSavePath = f"type-{CRPARAMS['analysisType']}/{CRPARAMS['halo']}/"


    # savePath = FigureSavepathBase + simSavePath
    # tmp = ""

    # for savePathChunk in savePath.split("/")[:-1]:
    #     tmp += savePathChunk + "/"
    #     try:
    #         os.mkdir(tmp)
    #     except:
    #         pass

    # excel = pd.ExcelWriter(path=savePath,mode="w")

    # comparisonDict = {}
    # for halo, stats in statsOutAllSims.items():
    #     comparisonDict.update({halo : {}})
    #     for comp in comparisons:
    #         numer = comp[0]
    #         denom = comp[1]
    #         comp = numer + "/" + denom

    #         # comparisonDict[halo].update({compKey : {}})
    #         for sKey in stats.keys():
    #             listed = list(sKey)
    #             if numer in listed:
    #                 denomKey = tuple([xx if xx != numer else denom for xx in listed])
    #                 compKey = tuple([xx if xx != numer else comp for xx in listed])
    #                 comparisonDict[halo].update({compKey : {}})
    #                 for key in stats[sKey][sKey].keys():
    #                     try:
    #                         val = stats[sKey][sKey][key]/stats[denomKey][denomKey][key]
    #                     except:
    #                         print(f"Variable {key} not found! Entering null data...")
    #                         val = np.full(shape=np.shape(stats[sKey][sKey][key]),fill_value=np.nan)
    #                     comparisonDict[halo][compKey].update({key : copy.deepcopy(val)})
                    

    # np.nanmedian(statsOutAllSims['halo_5'][('standard','with_CRs_no_Alfven')][('standard','with_CRs_no_Alfven')]['P_thermal_50.00%']/statsOutAllSims['halo_5'][('standard','no_CRs')][('standard','no_CRs')['P_thermal_50.00%'])
        # for sKey, data in statsOut.items():
        #     if sKey not in keepPercentiles:
        #         dataCopy = copy.deepcopy(data)
        #         for key,dd in data.items():
        #             if key not in keepPercentiles:
        #                 for kk, value in dd.items():
        #                     splitkk = kk.split("_")
        #                     perc = splitkk[-1]
        #                     if (medianString not in splitkk)&(perc in loadPercentilesTypes):
        #                         dataCopy[key].pop(kk)
        #         statsOut[sKey].update(dataCopy)

        # apt.cr_medians_versus_plot(
        #     statsOut,
        #     CRPARAMSHALO,
        #     ylabel=ylabel,
        #     xlimDict=xlimDict,
        #     snapNumber=snapNumber,
        #     yParam=CRPARAMS["mediansParams"],
        #     xParam=CRPARAMS["xParam"],
        #     titleBool=CRPARAMS["titleBool"],
        #     DPI = CRPARAMS["DPI"],
        #     xsize = CRPARAMS["xsize"],
        #     ysize = CRPARAMS["ysize"],
        #     fontsize = CRPARAMS["fontsize"],
        #     fontsizeTitle = CRPARAMS["fontsizeTitle"],
        #     opacityPercentiles = CRPARAMS["opacityPercentiles"],
        #     savePathBase = CRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = CRPARAMS["savepathdata"],
        #     inplace = inplace,
        #     saveFigureData = False,
        #     replotFromData = True,
        #     combineMultipleOntoAxis = True,
        #     selectKeysList = None,
        #     styleDict = styleDict,
        #     )
        
        # if len(CRPARAMS["colParams"])>0:
        #     print(
        #     f"[@{CRPARAMS['halo']}]: Time averaged Column Density Medians profile plots..."
        #     )

        #     selectKeysListCol = [tuple(list(sKey)+["col"]) for sKey in selectKeysList]
        #     keepPercentilesCol = [tuple(list(sKey)+["col"]) for sKey in keepPercentiles]
        #     # # # Create variant of xlimDict specifically for images of col params
        #     # # tmpxlimDict = copy.deepcopy(xlimDict)

        #     # # # Add the col param specific limits to the xlimDict variant
        #     # # for key, value in colImagexlimDict.items():
        #     # #     tmpxlimDict[key] = value

        #     #---------------#
        #     # Check for any none-position-based parameters we need to track for col params:
        #     #       Start with mass (always needed!) and xParam:
        #     additionalColParams = ["mass"]
        #     if np.any(np.isin(np.asarray([CRPARAMS["xParam"]]),np.array(["R","x","y","z","pos"]))) == False:
        #         additionalColParams.append(CRPARAMS["xParam"])

        #     #       Now add in anything we needed to track for weights of col params in statistics
        #     cols = CRPARAMS["colParams"]
        #     for param in cols:
        #         additionalParam = CRPARAMS["nonMassWeightDict"][param]
        #         if (np.any(np.isin(np.asarray([additionalParam]),np.asarray(additionalColParams))) == False) \
        #         & (additionalParam is not None) & (additionalParam != "count"):
        #             additionalColParams.append(additionalParam)
        #     #---------------#

        #     # If there are other params to be tracked for col params, we need to create a projection
        #     # of them so as to be able to map these projection values back to the col param maps.
        #     # A side effect of this is that we will create "images" of any of these additional params.
        #     # Thus, we want to provide empty limits for the colourbars of these images as they will almost
        #     # certainly require different limits to those provided for the PDF plots, for example. 
        #     # In particular, params like mass will need very different limits to those used in the
        #     # PDF plots. We have left this side effect in this code version as it provides a useful
        #     # way of testing whether making a projection of unusual params to image (e.g. mass, or volume)
        #     # provide sensible, physical results.
        #     # # for key in additionalColParams:
        #     # #     tmpxlimDict[key] = {}

        #     cols = CRPARAMS["colParams"]+additionalColParams

        #     COLCRPARAMS= copy.deepcopy(CRPARAMS)
        #     COLCRPARAMS["saveParams"]=COLCRPARAMS["saveParams"]+cols

        #     COLCRPARAMSHALO = copy.deepcopy(CRPARAMSHALO)
        #     # # COLCRPARAMSHALO = {sKey: values for sKey,(_,values) in zip(selectKeysListCol,CRPARAMSHALO.items())}
            
        #     for kk in COLCRPARAMSHALO.keys():
        #         COLCRPARAMSHALO[kk]["saveParams"] = COLCRPARAMSHALO[kk]["saveParams"]+cols

        #     tmp = apt.cr_load_statistics_data(
        #         selectKeysListCol,
        #         COLCRPARAMSHALO,
        #         snapRange,
        #         loadPathBase = CRPARAMS["savepathdata"],
        #         loadFile = "colStatsDict",
        #         fileType = ".h5",
        #         stack = True,
        #         selectKeyLen=4,
        #         verbose = DEBUG,
        #         )

        #     statsOutCol = copy.deepcopy(tmp)    

        #     if (len(snapRange)>1)&(stack is True):
        #         for sKey, data in statsOutCol.items():
        #             dataCopy = copy.deepcopy(data)
        #             for key,dd in data.items():
        #                 for kk, value in dd.items():
        #                     dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #             statsOutCol[sKey].update(dataCopy)

        #     fullStatsOutCol = copy.deepcopy(statsOutCol)
            
        #     for sKey, data in statsOutCol.items():
        #         if sKey not in keepPercentilesCol:
        #             dataCopy = copy.deepcopy(data)
        #             for key,dd in data.items():
        #                 if key not in keepPercentilesCol:
        #                     for kk, value in dd.items():
        #                         splitkk = kk.split("_")
        #                         perc = splitkk[-1]
        #                         if (medianString not in splitkk)&(perc in loadPercentilesTypes):
        #                             dataCopy[key].pop(kk)
        #             statsOutCol[sKey].update(dataCopy)

        #     apt.cr_medians_versus_plot(
        #         statsDict = statsOutCol,
        #         CRPARAMS = COLCRPARAMSHALO,
        #         ylabel=ylabel,
        #         xlimDict=xlimDict,
        #         snapNumber=snapNumber,
        #         yParam=COLCRPARAMS["colParams"],
        #         xParam=COLCRPARAMS["xParam"],
        #         titleBool=COLCRPARAMS["titleBool"],
        #         DPI = COLCRPARAMS["DPI"],
        #         xsize = COLCRPARAMS["xsize"],
        #         ysize = COLCRPARAMS["ysize"],
        #         fontsize = COLCRPARAMS["fontsize"],
        #         fontsizeTitle = COLCRPARAMS["fontsizeTitle"],
        #         opacityPercentiles = COLCRPARAMS["opacityPercentiles"],
        #         colourmapMain = "tab10",
        #         savePathBase = COLCRPARAMS["savepathfigures"],
        #         savePathBaseFigureData = COLCRPARAMS["savepathdata"],
        #         inplace = inplace,
        #         saveFigureData = False,
        #         replotFromData = True,
        #         combineMultipleOntoAxis = True,
        #         selectKeysList = None,
        #         styleDict = styleDict,
        #     )

        #     selectKey = (f"{CRPARAMS['resolution']}", 
        #             f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

        #     print(
        #         f"[@{CRPARAMS['halo']}]: Time averaged PDF of column Density gas plot"
        #     )
        #     matplotlib.rc_file_defaults()
        #     plt.close("all")     

        #     tmp = apt.cr_load_pdf_versus_plot_data(
        #         selectKeysListCol,
        #         COLCRPARAMSHALO,
        #         snapRange,
        #         weightKeys = ['mass'],
        #         xParams = COLCRPARAMS["colParams"],
        #         cumulative = False,
        #         loadPathBase = COLCRPARAMS["savepathdata"],
        #         SFR = False,
        #         normalise = False,
        #         stack = True,
        #         selectKeyLen=4,
        #         verbose = DEBUG,
        #         )

        #     pdfOutCol = copy.deepcopy(tmp)    

        #     if (len(snapRange)>1)&(stack is True):
        #         for sKey, data in pdfOutCol.items():
        #             dataCopy = copy.deepcopy(data)
        #             for key,dd in data.items():
        #                 for kk, value in dd.items():
        #                     dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #             pdfOutCol[sKey].update(dataCopy)

        #     matplotlib.rc_file_defaults()
        #     plt.close("all")   
        #     apt.cr_pdf_versus_plot(
        #         pdfOutCol,
        #         COLCRPARAMSHALO,
        #         ylabel,
        #         xlimDict,
        #         snapNumber = None,
        #         weightKeys = ['mass'],
        #         xParams = COLCRPARAMS["colParams"],
        #         titleBool=CRPARAMS["titleBool"],
        #         DPI=COLCRPARAMS["DPI"],
        #         xsize=COLCRPARAMS["xsize"],
        #         ysize=COLCRPARAMS["ysize"],
        #         fontsize=COLCRPARAMS["fontsize"],
        #         fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
        #         Nbins=COLCRPARAMS["Nbins"],
        #         ageWindow=None,
        #         cumulative = False,
        #         savePathBase = COLCRPARAMS["savepathfigures"],
        #         savePathBaseFigureData = COLCRPARAMS["savepathdata"],
        #         saveFigureData = False,
        #         SFR = False,
        #         normalise = True,
        #         verbose = DEBUG,
        #         inplace = inplace,
        #         replotFromData = True,
        #         combineMultipleOntoAxis = True,
        #         selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #         styleDict = styleDict,                                
        #     )
        #     matplotlib.rc_file_defaults()
        #     plt.close("all")   

        #     matplotlib.rc_file_defaults()
        #     plt.close("all")   
        #     apt.cr_pdf_versus_plot(
        #         pdfOutCol,
        #         COLCRPARAMSHALO,
        #         ylabel,
        #         xlimDict,
        #         snapNumber = None,
        #         weightKeys = ['mass'],
        #         xParams = COLCRPARAMS["colParams"],
        #         titleBool=COLCRPARAMS["titleBool"],
        #         DPI=COLCRPARAMS["DPI"],
        #         xsize=COLCRPARAMS["xsize"],
        #         ysize=COLCRPARAMS["ysize"],
        #         fontsize=COLCRPARAMS["fontsize"],
        #         fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
        #         Nbins=COLCRPARAMS["Nbins"],
        #         ageWindow=None,
        #         cumulative = True,
        #         savePathBase = COLCRPARAMS["savepathfigures"],
        #         savePathBaseFigureData = COLCRPARAMS["savepathdata"],
        #         saveFigureData = False,
        #         SFR = False,
        #         normalise = True,
        #         verbose = DEBUG,
        #         inplace = inplace,
        #         replotFromData = True,
        #         combineMultipleOntoAxis = True,
        #         selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #         styleDict = styleDict,                                
        #     )
        #     matplotlib.rc_file_defaults()
        #     plt.close("all")   


        #     if ((COLCRPARAMS["byType"] is True)&(len(COLCRPARAMS["pdfParams"])>0)):
        #         print(
        #             f"[@{COLCRPARAMS['halo']}]: Time averaged PDF of gas plot by particle type"
        #         )
        #         matplotlib.rc_file_defaults()
        #         plt.close("all")  
        #         possibleTypes = [0,1,2,3,4,5,6]

        #         for tp in possibleTypes:
        #             print("Starting type load ",tp)          
        #             tmp = apt.cr_load_pdf_versus_plot_data(
        #                 selectKeysListCol,
        #                 COLCRPARAMSHALO,
        #                 snapRange,
        #                 weightKeys = ['mass'],
        #                 xParams = COLCRPARAMS["colParams"],
        #                 cumulative = False,
        #                 loadPathBase = COLCRPARAMS["savepathdata"],
        #                 loadPathSuffix = f"type{int(tp)}/",
        #                 SFR = False,
        #                 normalise = False,
        #                 stack = True,
        #                 selectKeyLen=4,
        #                 verbose = DEBUG,
        #             )
                    
        #             loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
        #             if loadedHasData == False:
        #                 print(f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
        #                 continue
            
        #             binnedpdfOut = copy.deepcopy(tmp)    

        #             if (len(snapRange)>1)&(stack is True):
        #                 for sKey, data in binnedpdfOut.items():
        #                     dataCopy = copy.deepcopy(data)
        #                     for key,dd in data.items():
        #                         for kk, value in dd.items():
        #                             dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #                     binnedpdfOut[sKey].update(dataCopy)

        #             apt.cr_pdf_versus_plot(
        #                 binnedpdfOut,
        #                 COLCRPARAMSHALO,
        #                 ylabel,
        #                 xlimDict,
        #                 snapNumber = None,
        #                 weightKeys = ['mass'],
        #                 xParams = COLCRPARAMS["colParams"],
        #                 titleBool=COLCRPARAMS["titleBool"],
        #                 DPI=COLCRPARAMS["DPI"],
        #                 xsize=COLCRPARAMS["xsize"],
        #                 ysize=COLCRPARAMS["ysize"],
        #                 fontsize=COLCRPARAMS["fontsize"],
        #                 fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
        #                 Nbins=COLCRPARAMS["Nbins"],
        #                 ageWindow=None,
        #                 cumulative = False,
        #                 savePathBase = COLCRPARAMS["savepathfigures"],
        #                 savePathBaseFigureData = COLCRPARAMS["savepathdata"],
        #                 allSavePathsSuffix = f"type{int(tp)}/",
        #                 saveFigureData = False,
        #                 SFR = False,
        #                 normalise = True,
        #                 verbose = DEBUG,
        #                 inplace = inplace,
        #                 replotFromData = True,
        #                 combineMultipleOntoAxis = True,
        #                 selectKeysList = None,
        #                 styleDict = styleDict,                                
        #             )
        #             matplotlib.rc_file_defaults()
        #             plt.close("all")

        #             apt.cr_pdf_versus_plot(
        #                 binnedpdfOut,
        #                 COLCRPARAMSHALO,
        #                 ylabel,
        #                 xlimDict,
        #                 snapNumber = None,
        #                 weightKeys = ['mass'],
        #                 xParams = COLCRPARAMS["colParams"],
        #                 titleBool=COLCRPARAMS["titleBool"],
        #                 DPI=COLCRPARAMS["DPI"],
        #                 xsize=COLCRPARAMS["xsize"],
        #                 ysize=COLCRPARAMS["ysize"],
        #                 fontsize=COLCRPARAMS["fontsize"],
        #                 fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
        #                 Nbins=COLCRPARAMS["Nbins"],
        #                 ageWindow=None,
        #                 cumulative = True,
        #                 savePathBase = COLCRPARAMS["savepathfigures"],
        #                 savePathBaseFigureData = COLCRPARAMS["savepathdata"],
        #                 allSavePathsSuffix = f"type{int(tp)}/",
        #                 saveFigureData = False,
        #                 SFR = False,
        #                 normalise = True,
        #                 verbose = DEBUG,
        #                 inplace = inplace,
        #                 replotFromData = True,
        #                 combineMultipleOntoAxis = True,
        #                 selectKeysList = None,
        #                 styleDict = styleDict,                                
        #             )
        #             matplotlib.rc_file_defaults()
        #             plt.close("all")   
            
        #     if ((COLCRPARAMS["binByParam"] is True)&(len(COLCRPARAMS["pdfParams"])>0)):
        #         print(
        #             f"[@{COLCRPARAMS['halo']}]: Time averaged PDF of gas, binned by {COLCRPARAMS['xParam']} plot"
        #         )
        #         matplotlib.rc_file_defaults()
        #         plt.close("all")  

        #         binIndices = range(0,COLCRPARAMS["NParamBins"]+1,1)
        #         for ii,(lowerIndex,upperIndex) in enumerate(zip(binIndices[:-1],binIndices[1:])):
        #             print("Starting Binned PDF plot load ",ii+1," of ",COLCRPARAMS["NParamBins"])
                    
        #             bins = np.round(np.linspace(start=xlimDict[COLCRPARAMS["xParam"]]["xmin"],stop=xlimDict[COLCRPARAMS["xParam"]]["xmax"],num=COLCRPARAMS["NParamBins"]+1,endpoint=True),decimals=2)
                    
        #             subdir = f"/{bins[lowerIndex]}-{COLCRPARAMS['xParam']}-{bins[upperIndex]}/"
        #             tmp = apt.cr_load_pdf_versus_plot_data(
        #                 selectKeysListCol,
        #                 COLCRPARAMSHALO,
        #                 snapRange,
        #                 weightKeys = ['mass'],
        #                 xParams = COLCRPARAMS["colParams"],
        #                 cumulative = False,
        #                 loadPathBase = COLCRPARAMS["savepathdata"],
        #                 loadPathSuffix = subdir,
        #                 SFR = False,
        #                 normalise = False,
        #                 stack = True,
        #                 selectKeyLen=4,
        #                 verbose = DEBUG,
        #             )
                    
        #             loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
        #             if loadedHasData == False:
        #                 print(f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
        #                 continue
            
        #             binnedpdfOut = copy.deepcopy(tmp)    

        #             if (len(snapRange)>1)&(stack is True):
        #                 for sKey, data in binnedpdfOut.items():
        #                     dataCopy = copy.deepcopy(data)
        #                     for key,dd in data.items():
        #                         for kk, value in dd.items():
        #                             dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #                     binnedpdfOut[sKey].update(dataCopy)

        #             apt.cr_pdf_versus_plot(
        #                 binnedpdfOut,
        #                 COLCRPARAMSHALO,
        #                 ylabel,
        #                 xlimDict,
        #                 snapNumber = None,
        #                 weightKeys = ['mass'],
        #                 xParams = COLCRPARAMS["colParams"],
        #                 titleBool=COLCRPARAMS["titleBool"],
        #                 DPI=COLCRPARAMS["DPI"],
        #                 xsize=COLCRPARAMS["xsize"],
        #                 ysize=COLCRPARAMS["ysize"],
        #                 fontsize=COLCRPARAMS["fontsize"],
        #                 fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
        #                 Nbins=COLCRPARAMS["Nbins"],
        #                 ageWindow=None,
        #                 cumulative = False,
        #                 savePathBase = COLCRPARAMS["savepathfigures"],
        #                 savePathBaseFigureData = COLCRPARAMS["savepathdata"],
        #                 allSavePathsSuffix = subdir,
        #                 saveFigureData = False,
        #                 SFR = False,
        #                 normalise = True,
        #                 verbose = DEBUG,
        #                 inplace = inplace,
        #                 replotFromData = True,
        #                 combineMultipleOntoAxis = True,
        #                 selectKeysList = None,
        #                 styleDict = styleDict,                                
        #             )
        #             matplotlib.rc_file_defaults()
        #             plt.close("all")

        #             apt.cr_pdf_versus_plot(
        #                 binnedpdfOut,
        #                 COLCRPARAMSHALO,
        #                 ylabel,
        #                 xlimDict,
        #                 snapNumber = None,
        #                 weightKeys = ['mass'],
        #                 xParams = COLCRPARAMS["colParams"],
        #                 titleBool=COLCRPARAMS["titleBool"],
        #                 DPI=COLCRPARAMS["DPI"],
        #                 xsize=COLCRPARAMS["xsize"],
        #                 ysize=COLCRPARAMS["ysize"],
        #                 fontsize=COLCRPARAMS["fontsize"],
        #                 fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
        #                 Nbins=COLCRPARAMS["Nbins"],
        #                 ageWindow=None,
        #                 cumulative = True,
        #                 savePathBase = COLCRPARAMS["savepathfigures"],
        #                 savePathBaseFigureData = COLCRPARAMS["savepathdata"],
        #                 allSavePathsSuffix = subdir,
        #                 saveFigureData = False,
        #                 SFR = False,
        #                 normalise = True,
        #                 verbose = DEBUG,
        #                 inplace = inplace,
        #                 replotFromData = True,
        #                 combineMultipleOntoAxis = True,
        #                 selectKeysList = None,
        #                 styleDict = styleDict,                                
        #             )
        #             matplotlib.rc_file_defaults()
        #             plt.close("all")   
            


        # selectKey = (f"{CRPARAMS['resolution']}", 
        #         f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

        # print(
        #     f"[@{CRPARAMS['halo']}]: Time averaged PDF of gas plot"
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")     

        # tmp = apt.cr_load_pdf_versus_plot_data(
        #     selectKeysList,
        #     CRPARAMSHALO,
        #     snapRange,
        #     weightKeys = ['mass'],
        #     xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #     cumulative = False,
        #     loadPathBase = CRPARAMS["savepathdata"],
        #     SFR = False,
        #     normalise = False,
        #     stack = True,
        #     verbose = DEBUG,
        #     )

        # pdfOut = copy.deepcopy(tmp)    

        # if (len(snapRange)>1)&(stack is True):
        #     for sKey, data in pdfOut.items():
        #         dataCopy = copy.deepcopy(data)
        #         for key,dd in data.items():
        #             for kk, value in dd.items():
        #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #         pdfOut[sKey].update(dataCopy)

        # matplotlib.rc_file_defaults()
        # plt.close("all")   
        # apt.cr_pdf_versus_plot(
        #     pdfOut,
        #     CRPARAMSHALO,
        #     ylabel,
        #     xlimDict,
        #     snapNumber = None,
        #     weightKeys = ['mass'],
        #     xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #     titleBool=CRPARAMS["titleBool"],
        #     DPI=CRPARAMS["DPI"],
        #     xsize=CRPARAMS["xsize"],
        #     ysize=CRPARAMS["ysize"],
        #     fontsize=CRPARAMS["fontsize"],
        #     fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #     Nbins=CRPARAMS["Nbins"],
        #     ageWindow=None,
        #     cumulative = False,
        #     savePathBase = CRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = CRPARAMS["savepathdata"],
        #     saveFigureData = False,
        #     SFR = False,
        #     normalise = True,
        #     verbose = DEBUG,
        #     inplace = inplace,
        #     replotFromData = True,
        #     combineMultipleOntoAxis = True,
        #     selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #     styleDict = styleDict,                                
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")   

        # matplotlib.rc_file_defaults()
        # plt.close("all")   
        # apt.cr_pdf_versus_plot(
        #     pdfOut,
        #     CRPARAMSHALO,
        #     ylabel,
        #     xlimDict,
        #     snapNumber = None,
        #     weightKeys = ['mass'],
        #     xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #     titleBool=CRPARAMS["titleBool"],
        #     DPI=CRPARAMS["DPI"],
        #     xsize=CRPARAMS["xsize"],
        #     ysize=CRPARAMS["ysize"],
        #     fontsize=CRPARAMS["fontsize"],
        #     fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #     Nbins=CRPARAMS["Nbins"],
        #     ageWindow=None,
        #     cumulative = True,
        #     savePathBase = CRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = CRPARAMS["savepathdata"],
        #     saveFigureData = False,
        #     SFR = False,
        #     normalise = True,
        #     verbose = DEBUG,
        #     inplace = inplace,
        #     replotFromData = True,
        #     combineMultipleOntoAxis = True,
        #     selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #     styleDict = styleDict,                                
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")   


        # if ((CRPARAMS["byType"] is True)&(len(CRPARAMS["pdfParams"])>0)):
        #     print(
        #         f"[@{CRPARAMS['halo']}]: Time averaged PDF of gas plot by particle type"
        #     )
        #     matplotlib.rc_file_defaults()
        #     plt.close("all")  
        #     possibleTypes = [0,1,2,3,4,5,6]

        #     for tp in possibleTypes:
        #         print("Starting type load ",tp)          
        #         tmp = apt.cr_load_pdf_versus_plot_data(
        #             selectKeysList,
        #             CRPARAMSHALO,
        #             snapRange,
        #             weightKeys = ['mass'],
        #             xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #             cumulative = False,
        #             loadPathBase = CRPARAMS["savepathdata"],
        #             loadPathSuffix = f"type{int(tp)}/",
        #             SFR = False,
        #             normalise = False,
        #             stack = True,
        #             verbose = DEBUG,
        #         )
                
        #         loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
        #         if loadedHasData == False:
        #             print(f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
        #             continue
        
        #         binnedpdfOut = copy.deepcopy(tmp)    

        #         if (len(snapRange)>1)&(stack is True):
        #             for sKey, data in binnedpdfOut.items():
        #                 dataCopy = copy.deepcopy(data)
        #                 for key,dd in data.items():
        #                     for kk, value in dd.items():
        #                         dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #                 binnedpdfOut[sKey].update(dataCopy)

        #         apt.cr_pdf_versus_plot(
        #             binnedpdfOut,
        #             CRPARAMSHALO,
        #             ylabel,
        #             xlimDict,
        #             snapNumber = None,
        #             weightKeys = ['mass'],
        #             xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #             titleBool=CRPARAMS["titleBool"],
        #             DPI=CRPARAMS["DPI"],
        #             xsize=CRPARAMS["xsize"],
        #             ysize=CRPARAMS["ysize"],
        #             fontsize=CRPARAMS["fontsize"],
        #             fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #             Nbins=CRPARAMS["Nbins"],
        #             ageWindow=None,
        #             cumulative = False,
        #             savePathBase = CRPARAMS["savepathfigures"],
        #             savePathBaseFigureData = CRPARAMS["savepathdata"],
        #             allSavePathsSuffix = f"type{int(tp)}/",
        #             saveFigureData = False,
        #             SFR = False,
        #             normalise = True,
        #             verbose = DEBUG,
        #             inplace = inplace,
        #             replotFromData = True,
        #             combineMultipleOntoAxis = True,
        #             selectKeysList = None,
        #             styleDict = styleDict,                                
        #         )
        #         matplotlib.rc_file_defaults()
        #         plt.close("all")

        #         apt.cr_pdf_versus_plot(
        #             binnedpdfOut,
        #             CRPARAMSHALO,
        #             ylabel,
        #             xlimDict,
        #             snapNumber = None,
        #             weightKeys = ['mass'],
        #             xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #             titleBool=CRPARAMS["titleBool"],
        #             DPI=CRPARAMS["DPI"],
        #             xsize=CRPARAMS["xsize"],
        #             ysize=CRPARAMS["ysize"],
        #             fontsize=CRPARAMS["fontsize"],
        #             fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #             Nbins=CRPARAMS["Nbins"],
        #             ageWindow=None,
        #             cumulative = True,
        #             savePathBase = CRPARAMS["savepathfigures"],
        #             savePathBaseFigureData = CRPARAMS["savepathdata"],
        #             allSavePathsSuffix = f"type{int(tp)}/",
        #             saveFigureData = False,
        #             SFR = False,
        #             normalise = True,
        #             verbose = DEBUG,
        #             inplace = inplace,
        #             replotFromData = True,
        #             combineMultipleOntoAxis = True,
        #             selectKeysList = None,
        #             styleDict = styleDict,                                
        #         )
        #         matplotlib.rc_file_defaults()
        #         plt.close("all")   
        
        # if ((CRPARAMS["binByParam"] is True)&(len(CRPARAMS["pdfParams"])>0)):
        #     print(
        #         f"[@{CRPARAMS['halo']}]: Time averaged PDF of gas, binned by {CRPARAMS['xParam']} plot"
        #     )
        #     matplotlib.rc_file_defaults()
        #     plt.close("all")  

        #     binIndices = range(0,CRPARAMS["NParamBins"]+1,1)
        #     for ii,(lowerIndex,upperIndex) in enumerate(zip(binIndices[:-1],binIndices[1:])):
        #         print("Starting Binned PDF plot load ",ii+1," of ",CRPARAMS["NParamBins"])
                
        #         bins = np.round(np.linspace(start=xlimDict[CRPARAMS["xParam"]]["xmin"],stop=xlimDict[CRPARAMS["xParam"]]["xmax"],num=CRPARAMS["NParamBins"]+1,endpoint=True),decimals=2)
                
        #         subdir = f"/{bins[lowerIndex]}-{CRPARAMS['xParam']}-{bins[upperIndex]}/"
        #         tmp = apt.cr_load_pdf_versus_plot_data(
        #             selectKeysList,
        #             CRPARAMSHALO,
        #             snapRange,
        #             weightKeys = ['mass'],
        #             xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #             cumulative = False,
        #             loadPathBase = CRPARAMS["savepathdata"],
        #             loadPathSuffix = subdir,
        #             SFR = False,
        #             normalise = False,
        #             stack = True,
        #             verbose = DEBUG,
        #         )
                
        #         loadedHasData = np.all(np.asarray([bool(val) for key,val in tmp.items()]))
        #         if loadedHasData == False:
        #             print(f"[@load_pdf_versus_plot_data]: Loaded Dictionary is empty! Skipping...")
        #             continue
        
        #         binnedpdfOut = copy.deepcopy(tmp)    

        #         if (len(snapRange)>1)&(stack is True):
        #             for sKey, data in binnedpdfOut.items():
        #                 dataCopy = copy.deepcopy(data)
        #                 for key,dd in data.items():
        #                     for kk, value in dd.items():
        #                         dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #                 binnedpdfOut[sKey].update(dataCopy)

        #         apt.cr_pdf_versus_plot(
        #             binnedpdfOut,
        #             CRPARAMSHALO,
        #             ylabel,
        #             xlimDict,
        #             snapNumber = None,
        #             weightKeys = ['mass'],
        #             xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #             titleBool=CRPARAMS["titleBool"],
        #             DPI=CRPARAMS["DPI"],
        #             xsize=CRPARAMS["xsize"],
        #             ysize=CRPARAMS["ysize"],
        #             fontsize=CRPARAMS["fontsize"],
        #             fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #             Nbins=CRPARAMS["Nbins"],
        #             ageWindow=None,
        #             cumulative = False,
        #             savePathBase = CRPARAMS["savepathfigures"],
        #             savePathBaseFigureData = CRPARAMS["savepathdata"],
        #             allSavePathsSuffix = subdir,
        #             saveFigureData = False,
        #             SFR = False,
        #             normalise = True,
        #             verbose = DEBUG,
        #             inplace = inplace,
        #             replotFromData = True,
        #             combineMultipleOntoAxis = True,
        #             selectKeysList = None,
        #             styleDict = styleDict,                                
        #         )
        #         matplotlib.rc_file_defaults()
        #         plt.close("all")

        #         apt.cr_pdf_versus_plot(
        #             binnedpdfOut,
        #             CRPARAMSHALO,
        #             ylabel,
        #             xlimDict,
        #             snapNumber = None,
        #             weightKeys = ['mass'],
        #             xParams = CRPARAMS["pdfParams"] + [CRPARAMS["xParam"]],
        #             titleBool=CRPARAMS["titleBool"],
        #             DPI=CRPARAMS["DPI"],
        #             xsize=CRPARAMS["xsize"],
        #             ysize=CRPARAMS["ysize"],
        #             fontsize=CRPARAMS["fontsize"],
        #             fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #             Nbins=CRPARAMS["Nbins"],
        #             ageWindow=None,
        #             cumulative = True,
        #             savePathBase = CRPARAMS["savepathfigures"],
        #             savePathBaseFigureData = CRPARAMS["savepathdata"],
        #             allSavePathsSuffix = subdir,
        #             saveFigureData = False,
        #             SFR = False,
        #             normalise = True,
        #             verbose = DEBUG,
        #             inplace = inplace,
        #             replotFromData = True,
        #             combineMultipleOntoAxis = True,
        #             selectKeysList = None,
        #             styleDict = styleDict,                                
        #         )
        #         matplotlib.rc_file_defaults()
        #         plt.close("all")   
        


        # selectKeyStars = (f"{CRPARAMS['resolution']}", 
        #         f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
        #         "Stars")
        
        # selectKeysListStars = [tuple(list(sKey)+["Stars"]) for sKey in selectKeysList]

        # STARSCRPARAMS= copy.deepcopy(CRPARAMS)
        # STARSCRPARAMSHALO = copy.deepcopy(CRPARAMSHALO)

        # print(
        #     f"[@{CRPARAMS['halo']}]: Time averaged PDF of stars plot"
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")     

        # tmp = apt.cr_load_pdf_versus_plot_data(
        #     selectKeysListStars,
        #     STARSCRPARAMSHALO,
        #     snapRange,
        #     weightKeys = ['mass'],
        #     xParams = [STARSCRPARAMS["xParam"]],
        #     cumulative = False,
        #     loadPathBase = STARSCRPARAMS["savepathdata"],
        #     SFR = False,
        #     normalise = False,
        #     stack = True,
        #     verbose = DEBUG,
        #     )

        # pdfOutStars = copy.deepcopy(tmp)    

        # if (len(snapRange)>1)&(stack is True):
        #     for sKey, data in pdfOutStars.items():
        #         dataCopy = copy.deepcopy(data)
        #         for key,dd in data.items():
        #             for kk, value in dd.items():
        #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #         pdfOutStars[sKey].update(dataCopy)

        # matplotlib.rc_file_defaults()
        # plt.close("all")   
        # apt.cr_pdf_versus_plot(
        #     pdfOutStars,
        #     STARSCRPARAMSHALO,
        #     ylabel,
        #     xlimDict,
        #     snapNumber = None,
        #     weightKeys = ['mass'],
        #     xParams = [STARSCRPARAMS["xParam"]],
        #     titleBool=STARSCRPARAMS["titleBool"],
        #     DPI=STARSCRPARAMS["DPI"],
        #     xsize=STARSCRPARAMS["xsize"],
        #     ysize=STARSCRPARAMS["ysize"],
        #     fontsize=STARSCRPARAMS["fontsize"],
        #     fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
        #     Nbins=STARSCRPARAMS["Nbins"],
        #     ageWindow=None,
        #     cumulative = False,
        #     savePathBase = STARSCRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = STARSCRPARAMS["savepathdata"],
        #     saveFigureData = False,
        #     SFR = False,
        #     normalise = True,
        #     verbose = DEBUG,
        #     inplace = inplace,
        #     replotFromData = True,
        #     combineMultipleOntoAxis = True,
        #     selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #     styleDict = styleDict,                                
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")   

        # matplotlib.rc_file_defaults()
        # plt.close("all")   
        # apt.cr_pdf_versus_plot(
        #     pdfOutStars,
        #     STARSCRPARAMSHALO,
        #     ylabel,
        #     xlimDict,
        #     snapNumber = None,
        #     weightKeys = ['mass'],
        #     xParams = [STARSCRPARAMS["xParam"]],
        #     titleBool=STARSCRPARAMS["titleBool"],
        #     DPI=STARSCRPARAMS["DPI"],
        #     xsize=STARSCRPARAMS["xsize"],
        #     ysize=STARSCRPARAMS["ysize"],
        #     fontsize=STARSCRPARAMS["fontsize"],
        #     fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
        #     Nbins=STARSCRPARAMS["Nbins"],
        #     ageWindow=None,
        #     cumulative = True,
        #     savePathBase = STARSCRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = STARSCRPARAMS["savepathdata"],
        #     saveFigureData = False,
        #     SFR = False,
        #     normalise = True,
        #     verbose = DEBUG,
        #     inplace = inplace,
        #     replotFromData = True,
        #     combineMultipleOntoAxis = True,
        #     selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #     styleDict = styleDict,                                
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")  

        # selectKey = (f"{CRPARAMS['resolution']}", 
        #         f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

        # print(
        #     f"[@{CRPARAMS['halo']}]: Time averaged gas phases plots"
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")     

        # tmp = apt.cr_load_phase_plot_data(
        #     selectKeysList,
        #     CRPARAMSHALO,
        #     snapRange,
        #     yParams = CRPARAMS["phasesyParams"],
        #     xParams = CRPARAMS["phasesxParams"],
        #     weightKeys = CRPARAMS["phasesWeightParams"],
        #     loadPathBase = CRPARAMS["savepathdata"],
        #     stack = True,
        #     verbose = DEBUG,
        #     )

        # phaseOut = copy.deepcopy(tmp)    

        # if (len(snapRange)>1)&(stack is True):
        #     for sKey, data in phaseOut.items():
        #         dataCopy = copy.deepcopy(data)
        #         for key,dd in data.items():
        #             for kk, value in dd.items():
        #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        #         phaseOut[sKey].update(dataCopy)

        # matplotlib.rc_file_defaults()
        # plt.close("all")
        # apt.cr_phase_plot(
        #     phaseOut,
        #     CRPARAMSHALO,
        #     ylabel,
        #     xlimDict,
        #     snapNumber = None,
        #     yParams = CRPARAMS["phasesyParams"],
        #     xParams = CRPARAMS["phasesxParams"],
        #     colourBarKeys = CRPARAMS["phasesColourbarParams"],
        #     weightKeys = CRPARAMS["nonMassWeightDict"],
        #     titleBool=CRPARAMS["titleBool"],
        #     DPI=CRPARAMS["DPI"],
        #     xsize=CRPARAMS["xsize"],
        #     ysize=CRPARAMS["ysize"],
        #     fontsize=CRPARAMS["fontsize"],
        #     fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #     colourmapMain= CRPARAMS["colourmapMain"],
        #     Nbins=CRPARAMS["Nbins"],
        #     savePathBase = CRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = CRPARAMS["savepathdata"],
        #     saveFigureData = True,
        #     verbose = DEBUG,
        #     inplace = inplace,
        #     replotFromData = True,
        #     allowPlotsWithoutxlimits = False,
        # )



        # selectKey = (f"{CRPARAMS['resolution']}", 
        #         f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
        #         "Stars",
        #         f"{int(snapRange[-1])}"
        #     )

        # selectKeysListStars = [tuple(list(sKey)+["Stars"]) for sKey in selectKeysList]


        # print(
        #     f"[@{CRPARAMS['halo']}]: SFR plot"
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")     

        # tmp = apt.cr_load_pdf_versus_plot_data(
        #     selectKeysListStars,
        #     CRPARAMSHALO,
        #     [snapRange[-1]],
        #     weightKeys = ['gima'],
        #     xParams = ["age"],
        #     cumulative = False,
        #     loadPathBase = CRPARAMS["savepathdata"],
        #     SFR = True,
        #     normalise = False,
        #     stack = True,
        #     verbose = DEBUG,
        #     )

        # pdfOutSFR = copy.deepcopy(tmp)    

        # # if (len(snapRange)>1)&(stack is True):
        # #     for sKey, data in pdfOut.items():
        # #         dataCopy = copy.deepcopy(data)
        # #         for key,dd in data.items():
        # #             for kk, value in dd.items():
        # #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
        # #         pdfOut[sKey].update(dataCopy)

        # matplotlib.rc_file_defaults()
        # plt.close("all")   
        # apt.cr_pdf_versus_plot(
        #     pdfOutSFR,
        #     CRPARAMSHALO,
        #     ylabel,
        #     xlimDict,
        #     snapNumber = None,
        #     weightKeys = ['gima'],
        #     xParams = ["age"],
        #     titleBool=CRPARAMS["titleBool"],
        #     DPI=CRPARAMS["DPI"],
        #     xsize=CRPARAMS["xsize"],
        #     ysize=CRPARAMS["ysize"],
        #     fontsize=CRPARAMS["fontsize"],
        #     fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #     Nbins=CRPARAMS["Nbins"],
        #     ageWindow=None,
        #     cumulative = False,
        #     savePathBase = CRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = CRPARAMS["savepathdata"],
        #     saveFigureData = False,
        #     SFR = True,
        #     normalise = False,
        #     verbose = DEBUG,
        #     inplace = inplace,
        #     replotFromData = True,
        #     combineMultipleOntoAxis = True,
        #     selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #     styleDict = styleDict,                                
        # )
        # matplotlib.rc_file_defaults()
        # plt.close("all")  

        # apt.cr_pdf_versus_plot(
        #     pdfOutSFR,
        #     CRPARAMSHALO,
        #     ylabel,
        #     xlimDict,
        #     snapNumber = None,
        #     weightKeys = ['gima'],
        #     xParams = ["age"],
        #     titleBool=CRPARAMS["titleBool"],
        #     DPI=CRPARAMS["DPI"],
        #     xsize=CRPARAMS["xsize"],
        #     ysize=CRPARAMS["ysize"],
        #     fontsize=CRPARAMS["fontsize"],
        #     fontsizeTitle=CRPARAMS["fontsizeTitle"],
        #     Nbins=CRPARAMS["Nbins"],
        #     ageWindow=None,
        #     cumulative = True,
        #     savePathBase = CRPARAMS["savepathfigures"],
        #     savePathBaseFigureData = CRPARAMS["savepathdata"],
        #     saveFigureData = False,
        #     SFR = True,
        #     normalise = False,
        #     verbose = DEBUG,
        #     inplace = inplace,
        #     replotFromData = True,
        #     combineMultipleOntoAxis = True,
        #     selectKeysList = None,#[('standard', 'with_CRs'),('standard', 'with_CRs_no_Alfven'),('standard', 'no_CRs')],
        #     styleDict = styleDict,                                
        # )

    #     matplotlib.rc_file_defaults()
    #     plt.close("all")
    #     print(
    #     f"Finished sim..."
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all")
    #     print(
    #     f"[@{halo}]: Finished halo..."
    #     )
    # print(
    #     "\n"+
    #     f"Finished completely! :)"
    # )
