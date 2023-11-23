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

matplotlib.use("Agg")  # For suppressing plotting on clusters

determineXlimits = False     #Intended for use when first configuring xlimits in xlimDict. Use this and "param" : {} in xlimDict for each param to explore axis limits needed for time averaging
figureDataSavePathModifier = "" #Intended for use when debugging and you don't want to overwrite the full/working figure dataset

DEBUG = False
inplace = False
CRPARAMSPATHMASTER = "CRParams.json"

singleValueKeys = ["Redshift", "Lookback", "Snap", "Rvir"]


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

    if determineXlimits is True: 
        print(
            "!!!!!"
            +"\n"
            +"WARNING! determineXlimits set to True!"
            +"\n"
            +"This feature is intended for exploratory purposes to determine desired axis limits for xlimDict."
            +"\n"
            +"Time averaging will NOT work with this feature enabled!"
            +"\n"
            +"!!!!!"
        )

    for halo, allSimsDict in CRSELECTEDHALOES.items():
        statsDict = {}
        colStatsDict = {}
        for snapNumber in snapRange:
            runAnalysisBool = True
            DataSavepathBase = CRPARAMSMASTER["savepathdata"]
            FigureSavepathBase = CRPARAMSMASTER["savepathfigures"]

            lastSnapStarsDict = {}
            colDict = {}
            starsDict = {}
            dataDict = {}
            # statsDict = {}
            # colStatsDict = {}

            if CRPARAMSMASTER["restartFlag"] is True:
                try:
                    print("Restart Flag True! Will try to recover previous analysis data products.")
                    print("Attempting to load data products...")
                    CRPARAMSHALO = {}
                    for sim, simDict in allSimsDict.items():
                        CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
                        CRPARAMS.update({'halo': halo})
                        selectKey = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            f"{int(snapNumber)}")
                                
                        CRPARAMSHALO.update({selectKey: CRPARAMS})

                        if CRPARAMS['simfile'] is not None:
                            analysisType = CRPARAMS["analysisType"]
                            print(
                                f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Loading data..."
                            )
                            if analysisType not in KnownAnalysisType:
                                raise Exception(
                                    f"ERROR! CRITICAL! Unknown analysis type: {analysisType}!"
                                    + "\n"
                                    + f"Availble analysis types: {KnownAnalysisType}"
                                )
                            
                            saveDir = (
                                DataSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                            )
                            saveDirFigures = ( 
                                FigureSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                            )

                            # Generate halo directory
                            tmp = ""
                            for savePathChunk in saveDir.split("/")[:-1]:
                                tmp += savePathChunk + "/"
                                try:
                                    os.mkdir(tmp)
                                except:
                                    pass
                                else:
                                    pass

                            tmp = ""
                            for savePathChunk in saveDirFigures.split("/")[:-1]:
                                tmp += savePathChunk + "/"
                                try:
                                    os.mkdir(tmp)
                                except:
                                    pass
                                else:
                                    pass

                            DataSavepath = (
                                saveDir + f"CR-Data_{int(snapNumber)}_"
                            )

                            # # # loadPath = DataSavepath + "statsDict.h5"
                            # # # statsDict.update(tr.hdf5_load(loadPath))

                            # # # if len(CRPARAMS["colParams"])>0:
                            # # #     loadPath = DataSavepath + "colStatsDict.h5"
                            # # #     colStatsDict.update(tr.hdf5_load(loadPath))           

                            if (snapNumber == snapRange[-1]):
                                loadPath = DataSavepath + "starsDict.h5"
                                lastSnapStarsDict.update(tr.hdf5_load(loadPath,selectKeyLen=4,delimiter="-"))
                            
                            if len(CRPARAMS["colParams"])>0:
                                loadPath = DataSavepath + "colDict.h5"
                                colDict.update(tr.hdf5_load(loadPath,selectKeyLen=4,delimiter="-"))

                            loadPath = DataSavepath + "starsDict.h5"
                            starsDict.update(tr.hdf5_load(loadPath,selectKeyLen=4,delimiter="-"))

                            loadPath = DataSavepath + "dataDict.h5"
                            dataDict.update(tr.hdf5_load(loadPath,selectKeyLen=3,delimiter="-"))



                    print("...done!")
                    runAnalysisBool = False
                    
                    if CRPARAMS['analysisType'] == 'cgm':
                        xlimDict["R"]['xmin'] = 0.0#CRPARAMS["Rinner"]
                        xlimDict["R"]['xmax'] = CRPARAMS['Router']

                    elif CRPARAMS['analysisType'] == 'ism':
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] = CRPARAMS['Rinner']
                    else:
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] = CRPARAMS['Router']

                except Exception as e:

                    print("Restart Failed! \n" + f"exception: {e}" + "\n Re-running Analysis!")
                    runAnalysisBool = True
            else:
                print("Restart Flag False! Re-running Analysis!")
                runAnalysisBool = True

            if runAnalysisBool is True:
                print("\n" + f"Starting ~New~ Analysis!")
                CRPARAMSHALO = {}
                print("\n"+f"Starting {halo} ...")
                # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
                #   MAIN ANALYSIS
                # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                for sim, simDict in allSimsDict.items():
                    CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
                    CRPARAMS.update({'halo': halo})
                    selectKey = (f"{CRPARAMS['resolution']}", 
                                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

                    analysisType = CRPARAMS["analysisType"]

                    if analysisType not in KnownAnalysisType:
                        raise Exception(
                            f"ERROR! CRITICAL! Unknown analysis type: {analysisType}!"
                            + "\n"
                            + f"Availble analysis types: {KnownAnalysisType}"
                        )

                    # innerStatsDict = {}
                    # innerColStatsDict = {}

                    if CRPARAMS['simfile'] is not None:

                        selectKey = (f"{CRPARAMS['resolution']}",
                                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

                        saveDir = ( DataSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                            )
                        saveDirFigures = ( FigureSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                            )

                        # Generate halo directory
                        tmp = ""
                        for savePathChunk in saveDir.split("/")[:-1]:
                            tmp += savePathChunk + "/"
                            try:
                                os.mkdir(tmp)
                            except Exception as e:
                                if DEBUG == True: print(f"{str(e)}")
                                pass
                            else:
                                pass

                        tmp = ""
                        for savePathChunk in saveDirFigures.split("/")[:-1]:
                            tmp += savePathChunk + "/"
                            try:
                                os.mkdir(tmp)
                            except Exception as e:
                                if DEBUG == True: print(f"{str(e)}")
                                pass
                            else:
                                pass

                        if (CRPARAMS["loadRotationMatrix"] == True)  & (CRPARAMS["constantRotationMatrix"] == True):
                            rotationloadpath = saveDir + f"rotation_matrix_{int(snapNumber)}.h5"
                            tmp = tr.hdf5_load(rotationloadpath)
                            rotation_matrix = tmp[selectKey]["rotation_matrix"]              
                        else:
                            rotation_matrix = None

                        selectKey = (f"{CRPARAMS['resolution']}", 
                                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                                    f"{int(snapNumber)}")
                        
                        CRPARAMSHALO.update({selectKey: CRPARAMS})

                        innerDataDict, innerStarsDict, innerColDict, _, rotation_matrix = cr.cr_analysis_radial(
                            snapNumber=snapNumber,
                            CRPARAMS=CRPARAMS,
                            ylabel=ylabel,
                            xlimDict=xlimDict,
                            colImagexlimDict = colImagexlimDict,
                            DataSavepathBase = DataSavepathBase,
                            FigureSavepathBase = FigureSavepathBase,
                            FullDataPathSuffix=FullDataPathSuffix,
                            rotation_matrix=rotation_matrix,
                            verbose = DEBUG,
                        )

                        tmpDataDict = {}

                        for key, val in innerStarsDict.items():
                            if (int(key[-1]) == int(snapRange[-1])):
                                lastSnapStarsDict.update({key: copy.deepcopy(val)})

                        # # # #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
                        # # # #----------------------------------------------------------------------#
                        # # # #      Calculate statistics...
                        # # # #----------------------------------------------------------------------#

                        # # # print("")
                        # # # print("Calculate Statistics!")
                        # # # print(f"{halo}",f"{sim}")

                        # # # print("Gas...")
                        # # # selectKey = (f"{CRPARAMS['resolution']}",
                        # # #             f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                        # # #             f"{int(snapNumber)}")

                        # # # tmpCRPARAMS = copy.deepcopy(CRPARAMS)
                        # # # tmpCRPARAMS['saveParams'] = tmpCRPARAMS['saveParams'] + ["mass"]

                        # # # if tmpCRPARAMS['analysisType'] == 'cgm':
                        # # #     xlimDict["R"]['xmin'] = 0.0#tmpCRPARAMS["Rinner"]
                        # # #     xlimDict["R"]['xmax'] = tmpCRPARAMS['Router']

                        # # # elif tmpCRPARAMS['analysisType'] == 'ism':
                        # # #     xlimDict["R"]['xmin'] = 0.0
                        # # #     xlimDict["R"]['xmax'] = tmpCRPARAMS['Rinner']
                        # # # else:
                        # # #     xlimDict["R"]['xmin'] = 0.0
                        # # #     xlimDict["R"]['xmax'] = tmpCRPARAMS['Router']

                        # # # statsWeightkeys = ["mass"] + np.unique(np.asarray(list(CRPARAMSMASTER["nonMassWeightDict"].values()))).tolist()
                        # # # exclusions = [] 
                        
                        # # # for param in CRPARAMSMASTER["saveEssentials"]:
                        # # #     if param not in statsWeightkeys:
                        # # #         exclusions.append(param)

                        # # # print(tmpCRPARAMS['analysisType'], xlimDict["R"]['xmin'],
                        # # #     xlimDict["R"]['xmax'])
                        # # # dat = cr.cr_calculate_statistics(
                        # # #     dataDict=innerDataDict[selectKey],
                        # # #     CRPARAMS=tmpCRPARAMS,
                        # # #     xParam=CRPARAMSMASTER["xParam"],
                        # # #     Nbins=CRPARAMSMASTER["NStatsBins"],
                        # # #     xlimDict=xlimDict,
                        # # #     exclusions=exclusions,
                        # # # )

                        # # # innerStatsDict = {selectKey: dat}

                        # # # if len(CRPARAMS["colParams"])>0:

                        # # #     # Create variant of xlimDict specifically for images of col params
                        # # #     tmpxlimDict = copy.deepcopy(xlimDict)

                        # # #     # Add the col param specific limits to the xlimDict variant
                        # # #     for key, value in colImagexlimDict.items():
                        # # #         tmpxlimDict[key] = value

                        # # #     #---------------#
                        # # #     # Check for any none-position-based parameters we need to track for col params:
                        # # #     #       Start with mass (always needed!) and xParam:
                        # # #     additionalColParams = ["mass"]
                        # # #     if np.any(np.isin(np.asarray([CRPARAMS["xParam"]]),np.array(["R","x","y","z","pos"]))) == False:
                        # # #         additionalColParams.append(CRPARAMS["xParam"])

                        # # #     #       Now add in anything we needed to track for weights of col params in statistics
                        # # #     cols = CRPARAMS["colParams"]
                        # # #     for param in cols:
                        # # #         additionalParam = CRPARAMS["nonMassWeightDict"][param]
                        # # #         if (np.any(np.isin(np.asarray([additionalParam]),np.asarray(additionalColParams))) == False) \
                        # # #         & (additionalParam is not None) & (additionalParam != "count"):
                        # # #             additionalColParams.append(additionalParam)
                        # # #     #---------------#

                        # # #     # If there are other params to be tracked for col params, we need to create a projection
                        # # #     # of them so as to be able to map these projection values back to the col param maps.
                        # # #     # A side effect of this is that we will create "images" of any of these additional params.
                        # # #     # Thus, we want to provide empty limits for the colourbars of these images as they will almost
                        # # #     # certainly require different limits to those provided for the PDF plots, for example. 
                        # # #     # In particular, params like mass will need very different limits to those used in the
                        # # #     # PDF plots. We have left this side effect in this code version as it provides a useful
                        # # #     # way of testing whether making a projection of unusual params to image (e.g. mass, or volume)
                        # # #     # provide sensible, physical results.
                        # # #     for key in additionalColParams:
                        # # #         tmpxlimDict[key] = {}


                        # # #     print("Col Dens Gas...")
                        # # #     selectKey = (f"{CRPARAMS['resolution']}",
                        # # #                 f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                        # # #                 f"{int(snapNumber)}")
                        # # #     selectKeyCol = (f"{CRPARAMS['resolution']}",
                        # # #                 f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                        # # #                 "col",
                        # # #                 f"{int(snapNumber)}")
                        # # #     COLCRPARAMS= copy.deepcopy(tmpCRPARAMS)
                        # # #     COLCRPARAMS['saveParams']=COLCRPARAMS['saveParams']+COLCRPARAMS['colParams']
                            
                        # # #     dat = cr.cr_calculate_statistics(
                        # # #         dataDict=innerColDict[selectKeyCol],
                        # # #         CRPARAMS=COLCRPARAMS,
                        # # #         xParam=COLCRPARAMS["xParam"],
                        # # #         Nbins=COLCRPARAMS["NStatsBins"],
                        # # #         xlimDict=tmpxlimDict,
                        # # #         exclusions=exclusions,
                        # # #         weightedStatsBool = False,
                        # # #     )

                        # # #     innerColStatsDict = {selectKeyCol: dat}

                        # ----------------------------------------------------------------------#
                        # Save output ...
                        # ----------------------------------------------------------------------#
                        print("")
                        print("***")
                        print("Saving data products...")

                        DataSavepath = (
                            saveDir + f"CR-Data_{int(snapNumber)}_"
                        )

                        # Generate halo directory
                        tmp = ""
                        for savePathChunk in saveDir.split("/")[:-1]:
                            tmp += savePathChunk + "/"
                            try:
                                os.mkdir(tmp)
                            except:
                                pass
                            else:
                                pass

                        tmp = ""
                        for savePathChunk in saveDirFigures.split("/")[:-1]:
                            tmp += savePathChunk + "/"
                            try:
                                os.mkdir(tmp)
                            except:
                                pass
                            else:
                                pass

                        savePath = DataSavepath + "dataDict.h5"
                        tr.hdf5_save(savePath,innerDataDict)

                        savePath = DataSavepath + "starsDict.h5"
                        tr.hdf5_save(savePath,innerStarsDict)

                        if len(CRPARAMS["colParams"])>0:
                            savePath = DataSavepath + "colDict.h5"
                            tr.hdf5_save(savePath,innerColDict)

                        # # # savePath = DataSavepath + "statsDict.h5"
                        # # # tr.hdf5_save(savePath,innerStatsDict)

                        # # # if len(CRPARAMS["colParams"])>0:
                        # # #     savePath = DataSavepath + "colStatsDict.h5"
                        # # #     tr.hdf5_save(savePath,innerColStatsDict)

                        dataDict.update(innerDataDict)
                        starsDict.update(innerStarsDict)
                        if len(CRPARAMS["colParams"])>0:
                            colDict.update(innerColDict)
                        # statsDict.update(innerStatsDict)
                        # if len(CRPARAMS["colParams"])>0:
                        #     colStatsDict.update(innerColStatsDict)
                        
                        print("...done!")
                        print("***")
                        print("")
            
            # ----------------------------------------------------------------------#
            #  Plots...
            # ----------------------------------------------------------------------#
            for sim, simDict in allSimsDict.items():
                CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
                CRPARAMS.update({'halo': halo})
                selectKey = (f"{CRPARAMS['resolution']}", 
                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                    f"{int(snapNumber)}")
                
                saveDir = (
                    DataSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                )
                saveDirFigures = ( 
                    FigureSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                )
                DataSavepath = (
                    saveDir + f"CR-Data_{int(snapNumber)}_"
                )

                #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
                #----------------------------------------------------------------------#
                #      Calculate statistics...
                #----------------------------------------------------------------------#
                # print("commented out ln 613")
                print("")
                print("Calculate Statistics!")
                print(f"{halo}",f"{sim}")

                print("Gas...")
                selectKey = (f"{CRPARAMS['resolution']}",
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            f"{int(snapNumber)}")

                tmpCRPARAMS = copy.deepcopy(CRPARAMS)
                tmpCRPARAMS['saveParams'] = tmpCRPARAMS['saveParams'] + ["mass"]

                if tmpCRPARAMS['analysisType'] == 'cgm':
                    xlimDict["R"]['xmin'] = 0.0#tmpCRPARAMS["Rinner"]
                    xlimDict["R"]['xmax'] = tmpCRPARAMS['Router']

                elif tmpCRPARAMS['analysisType'] == 'ism':
                    xlimDict["R"]['xmin'] = 0.0
                    xlimDict["R"]['xmax'] = tmpCRPARAMS['Rinner']
                else:
                    xlimDict["R"]['xmin'] = 0.0
                    xlimDict["R"]['xmax'] = tmpCRPARAMS['Router']

                statsWeightkeys = ["mass"] + np.unique(np.asarray(list(CRPARAMS["nonMassWeightDict"].values()))).tolist()
                exclusions = [] 
                
                for param in CRPARAMS["saveEssentials"]:
                    if param not in statsWeightkeys:
                        exclusions.append(param)

                print(tmpCRPARAMS['analysisType'], xlimDict["R"]['xmin'],
                    xlimDict["R"]['xmax'])
                dat = cr.cr_calculate_statistics(
                    dataDict=dataDict[selectKey],
                    CRPARAMS=tmpCRPARAMS,
                    xParam=CRPARAMS["xParam"],
                    Nbins=CRPARAMS["NStatsBins"],
                    xlimDict=xlimDict,
                    exclusions=exclusions,
                )

                innerStatsDict = {selectKey: dat}
                statsDict.update(innerStatsDict)
                savePath = DataSavepath + "statsDict.h5"
                tr.hdf5_save(savePath,innerStatsDict)

                if len(CRPARAMS["colParams"])>0:

                    # Create variant of xlimDict specifically for images of col params
                    tmpxlimDict = copy.deepcopy(xlimDict)

                    # Add the col param specific limits to the xlimDict variant
                    for key, value in colImagexlimDict.items():
                        tmpxlimDict[key] = value

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
                    for key in additionalColParams:
                        tmpxlimDict[key] = {}


                    print("Col Dens Gas...")
                    selectKey = (f"{CRPARAMS['resolution']}",
                                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                                f"{int(snapNumber)}")
                    selectKeyCol = (f"{CRPARAMS['resolution']}",
                                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                                "col",
                                f"{int(snapNumber)}")
                    COLCRPARAMS= copy.deepcopy(tmpCRPARAMS)
                    COLCRPARAMS['saveParams']=COLCRPARAMS['saveParams']+COLCRPARAMS['colParams']
                    
                    dat = cr.cr_calculate_statistics(
                        dataDict=colDict[selectKeyCol],
                        CRPARAMS=COLCRPARAMS,
                        xParam=COLCRPARAMS["xParam"],
                        Nbins=COLCRPARAMS["NStatsBins"],
                        xlimDict=tmpxlimDict,
                        exclusions=exclusions,
                        weightedStatsBool = False,
                    )

                    innerColStatsDict = {selectKeyCol: dat}
                    colStatsDict.update(innerColStatsDict)
                    savePath = DataSavepath + "colStatsDict.h5"
                    tr.hdf5_save(savePath,innerColStatsDict)
                

                tmpStatsDict = {selectKey : copy.deepcopy(innerStatsDict[selectKey])}
                print(
                f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Medians profile plots..."
                )

                apt.cr_medians_versus_plot(
                    tmpStatsDict,
                    CRPARAMSHALO,
                    ylabel=ylabel,
                    xlimDict=xlimDict,
                    snapNumber=snapNumber,
                    yParam=CRPARAMS["mediansParams"],
                    xParam=CRPARAMS["xParam"],
                    titleBool=CRPARAMS["titleBool"],
                    DPI = CRPARAMS["DPI"],
                    xsize = CRPARAMS["xsize"],
                    ysize = CRPARAMS["ysize"],
                    fontsize = CRPARAMS["fontsize"],
                    fontsizeTitle = CRPARAMS["fontsizeTitle"],
                    opacityPercentiles = CRPARAMS["opacityPercentiles"],
                    colourmapMain = "tab10",
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"] + figureDataSavePathModifier,
                    inplace = inplace,
                    saveFigureData = True,
                    allowPlotsWithoutxlimits = determineXlimits,
                    )

                if len(CRPARAMS["colParams"])>0:
                    print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Column Density Medians profile plots..."
                    )
                    selectKeyCol = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            "col",
                            f"{int(snapNumber)}")
                    
                    tmpColStatsDict = {selectKeyCol : copy.deepcopy(innerColStatsDict[selectKeyCol])}

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

                    innerColout = {}
                    cols = CRPARAMS["colParams"]+additionalColParams


                    COLCRPARAMS= copy.deepcopy(CRPARAMS)
                    COLCRPARAMS['saveParams']=COLCRPARAMS['saveParams']+cols
                    COLCRPARAMSHALO = {selectKey: COLCRPARAMS}

                    apt.cr_medians_versus_plot(
                        statsDict = tmpColStatsDict,
                        CRPARAMS = COLCRPARAMSHALO,
                        ylabel=ylabel,
                        xlimDict=xlimDict,
                        snapNumber=snapNumber,
                        yParam=COLCRPARAMS["colParams"],
                        xParam=COLCRPARAMS["xParam"],
                        titleBool=COLCRPARAMS["titleBool"],
                        DPI = COLCRPARAMS["DPI"],
                        xsize = COLCRPARAMS["xsize"],
                        ysize = COLCRPARAMS["ysize"],
                        fontsize = COLCRPARAMS["fontsize"],
                        fontsizeTitle = COLCRPARAMS["fontsizeTitle"],
                        opacityPercentiles = COLCRPARAMS["opacityPercentiles"],
                        colourmapMain = "tab10",
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                        inplace = inplace,
                        saveFigureData = True,
                        allowPlotsWithoutxlimits = determineXlimits,
                    )

                    tmpColDict = {selectKeyCol : copy.deepcopy(colDict[selectKeyCol])}

                selectKey = (f"{CRPARAMS['resolution']}", 
                        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                        f"{int(snapNumber)}")

                tmpDataDict = {selectKey : copy.deepcopy(dataDict[selectKey])}


                statsWeightkeys = ["mass"] + np.unique(np.asarray(list(CRPARAMS["nonMassWeightDict"].values()))).tolist()
                exclusions = [] 

                labels = ["x", "y", "z"]

                for label, kk, in zip(labels,range(0,3,1)):
                    tmpDataDict[selectKey].update({label: tmpDataDict[selectKey]["pos"][:,kk]})

                for excl in singleValueKeys:
                    if excl in list(tmpDataDict[selectKey].keys()):
                        tmpDataDict[selectKey].pop(excl)

                if ((CRPARAMS["byType"] is True)&(len(CRPARAMS["pdfParams"])>0)):
                    uniqueTypes = np.unique(tmpDataDict[selectKey]["type"])
                    for tp in uniqueTypes:
                        print("Starting type ",tp)
                        whereNotType = tmpDataDict[selectKey]["type"] != tp

                        tpData = cr.remove_selection(
                            copy.deepcopy(tmpDataDict[selectKey]),
                            removalConditionMask = whereNotType,
                            errorString = "byType PDF whereNotType",
                            verbose = DEBUG,
                        )

                        print(
                            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of gas plot binned by Type"
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")
                        apt.cr_pdf_versus_plot(
                            {selectKey : tpData},
                            CRPARAMSHALO,
                            ylabel,
                            xlimDict,
                            snapNumber,
                            weightKeys = CRPARAMS['nonMassWeightDict'],
                            xParams = CRPARAMS["pdfParams"],
                            titleBool=CRPARAMS["titleBool"],
                            DPI=CRPARAMS["DPI"],
                            xsize=CRPARAMS["xsize"],
                            ysize=CRPARAMS["ysize"],
                            fontsize=CRPARAMS["fontsize"],
                            fontsizeTitle=CRPARAMS["fontsizeTitle"],
                            Nbins=CRPARAMS["Nbins"],
                            ageWindow=None,
                            cumulative = False,
                            savePathBase = CRPARAMS["savepathfigures"],
                            savePathBaseFigureData = CRPARAMS["savepathdata"] + figureDataSavePathModifier,
                            allSavePathsSuffix = f"/type-{str(tp)}/",
                            saveFigureData = True,
                            SFR = False,
                            normalise = False,
                            verbose = DEBUG,
                            inplace = inplace,
                            allowPlotsWithoutxlimits = determineXlimits,                 
                        )

                if ((CRPARAMS["binByParam"] is True)&(len(CRPARAMS["pdfParams"])>0)):
                    binIndices = range(0,CRPARAMS["NParamBins"]+1,1)
                    for ii,(lowerIndex,upperIndex) in enumerate(zip(binIndices[:-1],binIndices[1:])):
                        print("Starting Binned PDF plot ",ii+1," of ",CRPARAMS["NParamBins"])
                        
                        bins = np.round(np.linspace(start=xlimDict[CRPARAMS["binParam"]]["xmin"],stop=xlimDict[CRPARAMS["binParam"]]["xmax"],num=CRPARAMS["NParamBins"]+1,endpoint=True),decimals=2)
                        
                        whereNotInBin = ((tmpDataDict[selectKey][CRPARAMS["binParam"]]>=bins[lowerIndex])&(tmpDataDict[selectKey][CRPARAMS["binParam"]]<=bins[upperIndex]))==False

                        binnedData = cr.remove_selection(
                            copy.deepcopy(tmpDataDict[selectKey]),
                            removalConditionMask = whereNotInBin,
                            errorString = "binByParam PDF whereNotInBin",
                            verbose = DEBUG,
                            )

                        subdir = f"/{bins[lowerIndex]}-{CRPARAMS['binParam']}-{bins[upperIndex]}/"

                        print(
                            f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of gas plot binned by {CRPARAMS['binParam']}"
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")
                        apt.cr_pdf_versus_plot(
                            {selectKey: binnedData},
                            CRPARAMSHALO,
                            ylabel,
                            xlimDict,
                            snapNumber,
                            weightKeys = CRPARAMS['nonMassWeightDict'],
                            xParams = CRPARAMS["pdfParams"],
                            titleBool=CRPARAMS["titleBool"],
                            DPI=CRPARAMS["DPI"],
                            xsize=CRPARAMS["xsize"],
                            ysize=CRPARAMS["ysize"],
                            fontsize=CRPARAMS["fontsize"],
                            fontsizeTitle=CRPARAMS["fontsizeTitle"],
                            Nbins=CRPARAMS["Nbins"],
                            ageWindow=None,
                            cumulative = False,
                            savePathBase = CRPARAMS["savepathfigures"],
                            savePathBaseFigureData = CRPARAMS["savepathdata"] + figureDataSavePathModifier,
                            allSavePathsSuffix = subdir,
                            saveFigureData = True,
                            SFR = False,
                            normalise = False,
                            verbose = DEBUG,
                            inplace = inplace,
                            allowPlotsWithoutxlimits = determineXlimits,                 
                        )

                print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of gas plot"
                )
                matplotlib.rc_file_defaults()
                plt.close("all")

                apt.cr_pdf_versus_plot(
                    tmpDataDict,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber,
                    weightKeys = CRPARAMS['nonMassWeightDict'],
                    xParams = CRPARAMS["pdfParams"],
                    titleBool=CRPARAMS["titleBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"],
                    ysize=CRPARAMS["ysize"],
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    Nbins=CRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = False,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"] + figureDataSavePathModifier,
                    saveFigureData = True,
                    SFR = False,
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    allowPlotsWithoutxlimits = determineXlimits,                 
                )
                
                matplotlib.rc_file_defaults()
                plt.close("all")     

                if len(CRPARAMS["colParams"])>0:

                    # Create variant of xlimDict specifically for images of col params
                    tmpxlimDict = copy.deepcopy(xlimDict)

                    # Add the col param specific limits to the xlimDict variant
                    for key, value in colImagexlimDict.items():
                        tmpxlimDict[key] = value

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
                    for key in additionalColParams:
                        tmpxlimDict[key] = {}

                    selectKeyCol = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            "col",
                            f"{int(snapNumber)}")
                    
                    tmpColDict = {selectKeyCol : copy.deepcopy(colDict[selectKeyCol])}

                    for excl in singleValueKeys:
                        if excl in list(tmpColDict[selectKeyCol].keys()):
                            tmpColDict[selectKeyCol].pop(excl)

                    print(
                        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of column density properties"
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     


                    COLCRPARAMS= copy.deepcopy(CRPARAMS)
                    COLCRPARAMS['saveParams']=COLCRPARAMS['saveParams']+COLCRPARAMS['colParams']
                    COLCRPARAMSHALO = {selectKey: COLCRPARAMS}


                    if ((COLCRPARAMS["byType"] is True)&(len(COLCRPARAMS["pdfParams"])>0)):
                        uniqueTypes = np.unique(tmpColDict[selectKeyCol]["type"])
                        for tp in uniqueTypes:
                            print("Starting type ",tp)
                            whereNotType = tmpColDict[selectKeyCol]["type"] != tp

                            tpData = cr.remove_selection(
                                copy.deepcopy(tmpColDict[selectKeyCol]),
                                removalConditionMask = whereNotType,
                                errorString = "byType PDF whereNotType",
                                verbose = DEBUG,
                            )

                            print(
                                f"[@{COLCRPARAMS['halo']}, @{COLCRPARAMS['resolution']}, @{COLCRPARAMS['CR_indicator']}{COLCRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of column density gas plot binned by Type"
                            )
                            matplotlib.rc_file_defaults()
                            plt.close("all")
                            apt.cr_pdf_versus_plot(
                                {selectKeyCol : tpData},
                                COLCRPARAMSHALO,
                                ylabel,
                                tmpxlimDict,
                                snapNumber,
                                weightKeys = COLCRPARAMS['nonMassWeightDict'],
                                xParams = COLCRPARAMS["colParams"],
                                titleBool=COLCRPARAMS["titleBool"],
                                DPI=COLCRPARAMS["DPI"],
                                xsize=COLCRPARAMS["xsize"],
                                ysize=COLCRPARAMS["ysize"],
                                fontsize=COLCRPARAMS["fontsize"],
                                fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                                Nbins=COLCRPARAMS["Nbins"],
                                ageWindow=None,
                                cumulative = False,
                                savePathBase = COLCRPARAMS["savepathfigures"],
                                savePathBaseFigureData = COLCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                                allSavePathsSuffix = f"/type-{str(tp)}/",
                                saveFigureData = True,
                                SFR = False,
                                normalise = False,
                                verbose = DEBUG,
                                inplace = inplace,
                                allowPlotsWithoutxlimits = determineXlimits,                 
                            )

                    if ((COLCRPARAMS["binByParam"] is True)&(len(COLCRPARAMS["pdfParams"])>0)):
                        binIndices = range(0,COLCRPARAMS["NParamBins"]+1,1)
                        for ii,(lowerIndex,upperIndex) in enumerate(zip(binIndices[:-1],binIndices[1:])):
                            print("Starting Binned PDF plot ",ii+1," of ",COLCRPARAMS["NParamBins"])
                            
                            bins = np.round(np.linspace(start=xlimDict[COLCRPARAMS["binParam"]]["xmin"],stop=xlimDict[COLCRPARAMS["binParam"]]["xmax"],num=COLCRPARAMS["NParamBins"]+1,endpoint=True),decimals=2)
                            
                            whereNotInBin = ((tmpColDict[selectKeyCol][COLCRPARAMS["binParam"]]>=bins[lowerIndex])&(tmpColDict[selectKeyCol][COLCRPARAMS["binParam"]]<=bins[upperIndex]))==False

                            binnedData = cr.remove_selection(
                                copy.deepcopy(tmpColDict[selectKeyCol]),
                                removalConditionMask = whereNotInBin,
                                errorString = "binByParam PDF whereNotInBin",
                                verbose = DEBUG,
                                )

                            subdir = f"/{bins[lowerIndex]}-{COLCRPARAMS['binParam']}-{bins[upperIndex]}/"

                            print(
                                f"[@{COLCRPARAMS['halo']}, @{COLCRPARAMS['resolution']}, @{COLCRPARAMS['CR_indicator']}{COLCRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of column density gas plot binned by {COLCRPARAMS['binParam']}"
                            )
                            matplotlib.rc_file_defaults()
                            plt.close("all")
                            apt.cr_pdf_versus_plot(
                                {selectKeyCol: binnedData},
                                COLCRPARAMSHALO,
                                ylabel,
                                tmpxlimDict,
                                snapNumber,
                                weightKeys = COLCRPARAMS['nonMassWeightDict'],
                                xParams = COLCRPARAMS["colParams"],
                                titleBool=COLCRPARAMS["titleBool"],
                                DPI=COLCRPARAMS["DPI"],
                                xsize=COLCRPARAMS["xsize"],
                                ysize=COLCRPARAMS["ysize"],
                                fontsize=COLCRPARAMS["fontsize"],
                                fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                                Nbins=COLCRPARAMS["Nbins"],
                                ageWindow=None,
                                cumulative = False,
                                savePathBase = COLCRPARAMS["savepathfigures"],
                                savePathBaseFigureData = COLCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                                allSavePathsSuffix = subdir,
                                saveFigureData = True,
                                SFR = False,
                                normalise = False,
                                verbose = DEBUG,
                                inplace = inplace,
                                allowPlotsWithoutxlimits = determineXlimits,                 
                            )

                    matplotlib.rc_file_defaults()
                    plt.close("all")     
                    print(
                        f"[@{COLCRPARAMS['halo']}, @{COLCRPARAMS['resolution']}, @{COLCRPARAMS['CR_indicator']}{COLCRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of column density gas plot"
                    )
                    apt.cr_pdf_versus_plot(
                        tmpColDict,
                        COLCRPARAMSHALO,
                        ylabel,
                        tmpxlimDict,
                        snapNumber,
                        weightKeys = COLCRPARAMS['nonMassWeightDict'],
                        xParams = COLCRPARAMS["colParams"],
                        titleBool=COLCRPARAMS["titleBool"],
                        DPI=COLCRPARAMS["DPI"],
                        xsize=COLCRPARAMS["xsize"],
                        ysize=COLCRPARAMS["ysize"],
                        fontsize=COLCRPARAMS["fontsize"],
                        fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                        Nbins=COLCRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = False,
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                        saveFigureData = True,
                        SFR = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        allowPlotsWithoutxlimits = determineXlimits,                 
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                selectKeyStars = (f"{CRPARAMS['resolution']}", 
                        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                        "Stars",
                        f"{int(snapNumber)}")
                
                STARSCRPARAMS= copy.deepcopy(CRPARAMS)
                STARSCRPARAMSHALO = {selectKey: STARSCRPARAMS}
                tmpStarsDict = {selectKeyStars : copy.deepcopy(starsDict[selectKeyStars])}

                for excl in singleValueKeys:
                    if excl in list(tmpStarsDict[selectKeyStars].keys()):
                        tmpStarsDict[selectKeyStars].pop(excl)

                print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: PDF of stars plot"
                )
                matplotlib.rc_file_defaults()
                plt.close("all")     

                apt.cr_pdf_versus_plot(
                    tmpStarsDict,
                    STARSCRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber,
                    weightKeys = STARSCRPARAMS['nonMassWeightDict'],
                    xParams = [STARSCRPARAMS["xParam"]],
                    titleBool=STARSCRPARAMS["titleBool"],
                    DPI=STARSCRPARAMS["DPI"],
                    xsize=STARSCRPARAMS["xsize"],
                    ysize=STARSCRPARAMS["ysize"],
                    fontsize=STARSCRPARAMS["fontsize"],
                    fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
                    Nbins=STARSCRPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = False,
                    savePathBase = STARSCRPARAMS["savepathfigures"],
                    savePathBaseFigureData = STARSCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                    saveFigureData = True,
                    SFR = False,
                    
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    allowPlotsWithoutxlimits = determineXlimits,
                                    
                )

                print(
                f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Phases Plot..."
                )
                matplotlib.rc_file_defaults()
                plt.close("all")
                apt.cr_phase_plot(
                    tmpDataDict,
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
                    savePathBaseFigureData = CRPARAMS["savepathdata"] + figureDataSavePathModifier,
                    saveFigureData = True,
                    verbose = DEBUG,
                    inplace = inplace,
                    allowPlotsWithoutxlimits = determineXlimits,    
                )
                matplotlib.rc_file_defaults()
                plt.close("all")

                if (snapNumber == snapRange[-1]):

                    selectKeyLast = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            "Stars",
                            f"{int(snapRange[-1])}"
                            )
                                        
                    STARSCRPARAMS= copy.deepcopy(CRPARAMS)
                    STARSCRPARAMSHALO = {selectKey: STARSCRPARAMS}

                    tmpLastSnapStarsDict = {selectKeyLast : copy.deepcopy(lastSnapStarsDict[selectKeyLast])}

                    for excl in singleValueKeys:
                        if excl in list(tmpLastSnapStarsDict[selectKeyLast].keys()):
                            tmpLastSnapStarsDict[selectKeyLast].pop(excl)

                    print("")


                    print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: SFR plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    apt.cr_pdf_versus_plot(
                        tmpLastSnapStarsDict,
                        STARSCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        xParams = ["age"],
                        titleBool=STARSCRPARAMS["titleBool"],
                        DPI=STARSCRPARAMS["DPI"],
                        xsize=STARSCRPARAMS["xsize"],
                        ysize=STARSCRPARAMS["ysize"],
                        fontsize=STARSCRPARAMS["fontsize"],
                        fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
                        Nbins = STARSCRPARAMS["SFRBins"],
                        ageWindow = STARSCRPARAMS["ageWindow"],
                        cumulative = False,
                        savePathBase = STARSCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = STARSCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                        SFR = True,
                        
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        allowPlotsWithoutxlimits = determineXlimits,
                                        
                    )
                
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Cumulative SFR plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     
                    apt.cr_pdf_versus_plot(
                        tmpLastSnapStarsDict,
                        STARSCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        xParams = ["age"],
                        titleBool=STARSCRPARAMS["titleBool"],
                        DPI=STARSCRPARAMS["DPI"],
                        xsize=STARSCRPARAMS["xsize"],
                        ysize=STARSCRPARAMS["ysize"],
                        fontsize=STARSCRPARAMS["fontsize"],
                        fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
                        Nbins = STARSCRPARAMS["SFRBins"],
                        ageWindow = STARSCRPARAMS["ageWindow"],
                        cumulative = True,
                        savePathBase = STARSCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = STARSCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                        saveFigureData = True,
                        SFR = True,
                        
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        allowPlotsWithoutxlimits = determineXlimits,
                                        
                    )

                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Normalised Cumulative SFR plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     
                    apt.cr_pdf_versus_plot(
                        tmpLastSnapStarsDict,
                        STARSCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        xParams = ["age"],
                        titleBool=STARSCRPARAMS["titleBool"],
                        DPI=STARSCRPARAMS["DPI"],
                        xsize=STARSCRPARAMS["xsize"],
                        ysize=STARSCRPARAMS["ysize"],
                        fontsize=STARSCRPARAMS["fontsize"],
                        fontsizeTitle=STARSCRPARAMS["fontsizeTitle"],
                        Nbins = STARSCRPARAMS["SFRBins"],
                        ageWindow = STARSCRPARAMS["ageWindow"],
                        cumulative = True,
                        savePathBase = STARSCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = STARSCRPARAMS["savepathdata"] + figureDataSavePathModifier,
                        saveFigureData = True,
                        SFR = True,
                        
                        normalise = True,
                        verbose = DEBUG,
                        inplace = inplace,
                        allowPlotsWithoutxlimits = determineXlimits,
                    )

                matplotlib.rc_file_defaults()
                plt.close("all")                        

            statsDict.update(innerStatsDict)
            if len(CRPARAMS["colParams"])>0:
                colStatsDict.update(innerColStatsDict)
            print(
            f"[@{int(snapNumber)}]: Finished snapshot..."
            )

        matplotlib.rc_file_defaults()
        plt.close("all")
        print(
        f"[@{halo}]: Finished halo..."
        )
    print(
        "\n"+
        f"Finished completely! :)"
    )
