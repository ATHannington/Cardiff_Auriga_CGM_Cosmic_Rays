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

determineXlimits = True     #Intended for use when first configuring xlimits in xlimDict. Use this and "param" : {} in xlimDict for each param to explore axis limits needed for time averaging
DEBUG = False
inplace = False
CRPARAMSPATHMASTER = "CRParams.json"

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
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_H$ (cm$^{-3}$)",
    "n_H_col": r"n$_H$ (cm$^{-2}$)",
    "n_HI": r"n$_{HI}$ (cm$^{-3}$)",
    "n_HI_col": r"n$_{HI}$ (cm$^{-2}$)",
    "nh": r"Neutral Hydrogen Fraction",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"Radial Velocity (km s$^{-1}$)",
    "gz": r"Metallicity Z$_{\odot}$",
    "L": r"Specific Angular Momentum" + "\n" + r"(kpc km s$^{-1}$)",
    "P_thermal": r"P$_{Thermal}$ (erg cm$^{-3}$)",
    "P_magnetic": r"P$_{Magnetic}$ (erg cm$^{-3}$)",
    "P_kinetic": r"P$_{Kinetic}$ (erg cm$^{-3}$)",
    "P_tot": r"P$_{tot}$ (erg cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "P_CR": r"P$_{CR}$ (erg cm$^{-3}$)",
    "PCR_Pmagnetic" : r"P$_{CR}$/P$_{magnetic}$",
    "PCR_Pthermal": r"(X$_{CR}$ = P$_{CR}$/P$_{Thermal}$)",
    "gah": r"Alfven Gas Heating (erg s$^{-1}$)",
    "bfld": r"||B-Field|| ($ \mu $G)",
    "Grad_T": r"||Temperature Gradient|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||n$_H$ Gradient|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||B-Field Gradient|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{CR}$ Gradient|| (erg kpc$^{-4}$)",
    "gima" : r"Star Formation Rate (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfven CR Cooling (erg s$^{-1}$)",
    "tcool": r"Cooling Time (Gyr)",
    "theat": r"Heating Time (Gyr)",
    "tcross": r"Sound Crossing Cell Time (Gyr)",
    "tff": r"Free Fall Time (Gyr)",
    "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
    "csound": r"Sound Speed (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "rho": r"Density (M$_{\odot}$ kpc$^{-3}$)",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
    "vol": r"Volume (kpc$^{3}$)",
    "age": "Lookback Time (Gyr)",
    "cool_length" : "Cooling Length (kpc)",
    "halo" : "FoF Halo",
    "subhalo" : "SubFind Halo",
    "x": r"x (kpc)",
    "y": r"y (kpc)",
    "z": r"z (kpc)",
    "count": r"Number of data points per pixel",
    "e_CR": r"Cosmic Ray Energy Density (eV cm$^{-3}$)",
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
    "vrad": {"xmin": -100.0, "xmax": 100.0},
    "gz": {"xmin": -2.0, "xmax": 1.0},
    "P_thermal": {"xmin": -19.5, "xmax": -10.0},
    "P_CR": {"xmin": -19.5, "xmax": -10.0},
    "PCR_Pthermal": {"xmin": -4.0, "xmax": 1.0},
    "PCR_Pmagnetic": {},
    "P_magnetic": {"xmin": -19.5, "xmax": -10.0},
    "P_kinetic": {"xmin": -19.5, "xmax": -10.0},
    "P_tot": {"xmin": -19.5, "xmax": -10.0},
    "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 4.0},
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
    ylabel[entry] = r"$Log_{10}$" + ylabel[entry]

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
        runAnalysisBool = True
        DataSavepathBase = CRPARAMSMASTER["savepathdata"]
        FigureSavepathBase = CRPARAMSMASTER["savepathfigures"]

        lastSnapStarsDict = {}
        colDict = {}
        starsDict = {}
        dataDict = {}
        statsDict = {}
        colStatsDict = {}

        if CRPARAMSMASTER["restartFlag"] is True:
            CRPARAMSHALO = {}
            for sim, simDict in allSimsDict.items():
                for snapNumber in snapRange:
                    CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
                    CRPARAMS.update({'halo': halo})
                    selectKey = (f"{CRPARAMS['resolution']}", 
                        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                        f"{int(snapNumber)}")
                            
                    CRPARAMSHALO.update({selectKey: CRPARAMS})
                
            try:
                print("Restart Flag True! Will try to recover previous analysis data products.")
                print("Attempting to load data products...")
                for sim, CRPARAMS in CRPARAMSHALO.items():
                    if CRPARAMS['simfile'] is not None:
                        for snapNumber in snapRange:
                            analysisType = CRPARAMS["analysisType"]

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
                            
                            DataSavepath = (
                                saveDir + f"{CRPARAMS['snapMin']}-{CRPARAMS['snapMax']}_{CRPARAMS['Rinner']}R{CRPARAMS['Router']}_{int(snapNumber)}_"
                            )

                            loadPath = DataSavepath + "statsDict.h5"
                            statsDict.update(tr.hdf5_load(loadPath))

                            loadPath = DataSavepath + "colStatsDict.h5"
                            colStatsDict.update(tr.hdf5_load(loadPath))           

                            loadPath = DataSavepath + "lastSnapStarsDict.h5"
                            lastSnapStarsDict.update(tr.hdf5_load(loadPath))

                            loadPath = DataSavepath + "colDict.h5"
                            colDict.update(tr.hdf5_load(loadPath))

                            loadPath = DataSavepath + "starsDict.h5"
                            starsDict.update(tr.hdf5_load(loadPath))

                            loadPath = DataSavepath + "dataDict.h5"
                            dataDict.update(tr.hdf5_load(loadPath))



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
            print("\n" + f"Starting SERIAL type Analysis!")
            CRPARAMSHALO = {}
            print("\n"+f"Starting {halo} Analysis!")
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

                if CRPARAMS['simfile'] is not None:

                    selectKey = (f"{CRPARAMS['resolution']}",
                                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

                    if (CRPARAMS["loadRotationMatrix"] == True)  & (CRPARAMS["constantRotationMatrix"] == True):
                        rotationloadpath = DataSavepathBase + "rotation_matrix.h5"
                        tmp = tr.hdf5_load(rotationloadpath)
                        rotation_matrix = tmp[selectKey]["rotation_matrix"]              
                    else:
                        rotation_matrix = None

                    saveDir = ( DataSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                        )
                    saveDirFigures = ( FigureSavepathBase + f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}/"
                        )

                    for snapNumber in snapRange:
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

                        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
                        #----------------------------------------------------------------------#
                        #      Calculate statistics...
                        #----------------------------------------------------------------------#

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

                        statsWeightkeys = ["mass"] + np.unique(np.asarray(list(CRPARAMSMASTER["nonMassWeightDict"].values()))).tolist()
                        exclusions = [] 
                        
                        for param in CRPARAMSMASTER["saveEssentials"]:
                            if param not in statsWeightkeys:
                                exclusions.append(param)

                        print(tmpCRPARAMS['analysisType'], xlimDict["R"]['xmin'],
                            xlimDict["R"]['xmax'])
                        dat = cr.cr_calculate_statistics(
                            dataDict=innerDataDict[selectKey],
                            CRPARAMS=tmpCRPARAMS,
                            xParam=CRPARAMSMASTER["xParam"],
                            Nbins=CRPARAMSMASTER["NxParamBins"],
                            xlimDict=xlimDict,
                            exclusions=exclusions,
                        )

                        innerStatsDict = {selectKey: dat}

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
                                & (additionalParam is not None):
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
                                        "col",
                                        f"{int(snapNumber)}")
                            COLCRPARAMS= copy.deepcopy(tmpCRPARAMS)
                            COLCRPARAMS['saveParams']+=COLCRPARAMS['colParams']
                            
                            dat = cr.cr_calculate_statistics(
                                dataDict=innerColDict[selectKey],
                                CRPARAMS=COLCRPARAMS,
                                xParam=COLCRPARAMS["xParam"],
                                Nbins=COLCRPARAMS["NxParamBins"],
                                xlimDict=tmpxlimDict,
                                exclusions=exclusions,
                                weightedStatsBool = False,
                            )

                            innerColStatsDict = {selectKey: dat}

                        # ----------------------------------------------------------------------#
                        # Save output ...
                        # ----------------------------------------------------------------------#
                        print("")
                        print("***")
                        print("Saving data products...")

                        DataSavepath = (
                            saveDir + f"{CRPARAMS['snapMin']}-{CRPARAMS['snapMax']}_{CRPARAMS['Rinner']}R{CRPARAMS['Router']}_{int(snapNumber)}_"
                        )

                        # Generate halo directory
                        tmp = ""
                        for savePathChunk in saveDir.split("/")[1:-1]:
                            tmp += savePathChunk + "/"
                            try:
                                os.mkdir(tmp)
                            except:
                                pass
                            else:
                                pass

                        tmp = ""
                        for savePathChunk in saveDirFigures.split("/")[1:-1]:
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

                        savePath = DataSavepath + "colDict.h5"
                        tr.hdf5_save(savePath,innerColDict)

                        if (snapNumber == snapRange[-1]):
                            selectKeyLast = (f"{CRPARAMS['resolution']}", 
                                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                                    f"{int(snapRange[-1])}"
                                    )
                            savePath = DataSavepath + "lastSnapStarsDict.h5"
                            tr.hdf5_save(savePath,lastSnapStarsDict)

                        savePath = DataSavepath + "statsDict.h5"
                        tr.hdf5_save(savePath,innerStatsDict)


                        dataDict.update(innerDataDict)
                        starsDict.update(innerStarsDict)
                        colDict.update(innerColDict)
                        statsDict.update(innerStatsDict)
                        colStatsDict.update(innerColStatsDict)
                        
                        print("...done!")
                        print("***")
                        print("")
        # ----------------------------------------------------------------------#
        #  Plots...
        # ----------------------------------------------------------------------#
        for sim, simDict in allSimsDict.items():
            CRPARAMSHALO = {}
            for snapNumber in snapRange:
                CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
                CRPARAMS.update({'halo': halo})
                selectKey = (f"{CRPARAMS['resolution']}", 
                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                    f"{int(snapNumber)}")
                        
                CRPARAMSHALO.update({selectKey: CRPARAMS})
                for snapNumber in snapRange:
                    analysisType = CRPARAMS["analysisType"]
                    selectKey = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            f"{int(snapNumber)}")
                    
                    tmpStatsDict = {selectKey : copy.deepcopy(statsDict[selectKey])}
                    print(
                    f"[@{int(snapNumber)}]: Medians profile plots..."
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
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        inplace = inplace,
                        saveFigureData = True,
                        forcePlotsWithoutxlimits = determineXlimits,
                        )

                    print(
                    f"[@{int(snapNumber)}]: Columnd Density Medians profile plots..."
                    )
                    selectKey = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            "col",
                            f"{int(snapNumber)}")
                    
                    tmpColStatsDict = {selectKey : copy.deepcopy(colStatsDict[selectKey])}

                    COLCRPARAMS= copy.deepcopy(CRPARAMS)
                    COLCRPARAMS['saveParams']+=COLCRPARAMS['colParams']
                    COLCRPARAMSHALO = {selectKey: COLCRPARAMS}

                    apt.cr_medians_versus_plot(
                        tmpColStatsDict,
                        COLCRPARAMSHALO,
                        ylabel=ylabel,
                        xlimDict=xlimDict,
                        snapNumber=snapNumber,
                        yParam=COLCRPARAMS["mediansParams"],
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
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        inplace = inplace,
                        saveFigureData = True,
                        forcePlotsWithoutxlimits = determineXlimits,
                        )

                    tmpColDict = {selectKey : copy.deepcopy(colDict[selectKey])}


                    print(
                    f"[@{int(snapNumber)}]: PDF of Col mass vs R plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")
                    apt.cr_pdf_versus_plot(
                        tmpColDict,
                        COLCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [COLCRPARAMS["xParam"]],
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
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,
                                        
                    )
                    
                    matplotlib.rc_file_defaults()
                    plt.close("all")

                    print(
                    f"[@{int(snapNumber)}]: Cumulative PDF of Col mass vs R plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")
                    apt.cr_pdf_versus_plot(
                        tmpColDict,
                        COLCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [COLCRPARAMS["xParam"]],
                        titleBool=COLCRPARAMS["titleBool"],
                        DPI=COLCRPARAMS["DPI"],
                        xsize=COLCRPARAMS["xsize"],
                        ysize=COLCRPARAMS["ysize"],
                        fontsize=COLCRPARAMS["fontsize"],
                        fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                        Nbins=COLCRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits, 
                                        
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")
                    print(
                    f"[@{int(snapNumber)}]: Normalised Cumulative PDF of Col mass vs R plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")

                    apt.cr_pdf_versus_plot(
                        tmpColDict,
                        COLCRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [COLCRPARAMS["xParam"]],
                        titleBool=COLCRPARAMS["titleBool"],
                        DPI=COLCRPARAMS["DPI"],
                        xsize=COLCRPARAMS["xsize"],
                        ysize=COLCRPARAMS["ysize"],
                        fontsize=COLCRPARAMS["fontsize"],
                        fontsizeTitle=COLCRPARAMS["fontsizeTitle"],
                        Nbins=COLCRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = COLCRPARAMS["savepathfigures"],
                        savePathBaseFigureData = COLCRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = True,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,
                                        
                    )


                    selectKey = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            f"{int(snapNumber)}")

                    tmpDataDict = {selectKey : copy.deepcopy(dataDict[selectKey])}

                    print(
                    f"[@{int(snapNumber)}]: PDF of mass vs R plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")
                    apt.cr_pdf_versus_plot(
                        tmpDataDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [CRPARAMS["xParam"]],
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
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,
                                        
                    )
                    
                    matplotlib.rc_file_defaults()
                    plt.close("all")

                    print(
                    f"[@{int(snapNumber)}]: Cumulative PDF of mass vs R plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")
                    apt.cr_pdf_versus_plot(
                        tmpDataDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [CRPARAMS["xParam"]],
                        titleBool=CRPARAMS["titleBool"],
                        DPI=CRPARAMS["DPI"],
                        xsize=CRPARAMS["xsize"],
                        ysize=CRPARAMS["ysize"],
                        fontsize=CRPARAMS["fontsize"],
                        fontsizeTitle=CRPARAMS["fontsizeTitle"],
                        Nbins=CRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = CRPARAMS["savepathfigures"],
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits, 
                                        
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")
                    print(
                    f"[@{int(snapNumber)}]: Normalised Cumulative PDF of mass vs R plot..."
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")

                    apt.cr_pdf_versus_plot(
                        tmpDataDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [CRPARAMS["xParam"]],
                        titleBool=CRPARAMS["titleBool"],
                        DPI=CRPARAMS["DPI"],
                        xsize=CRPARAMS["xsize"],
                        ysize=CRPARAMS["ysize"],
                        fontsize=CRPARAMS["fontsize"],
                        fontsizeTitle=CRPARAMS["fontsizeTitle"],
                        Nbins=CRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = CRPARAMS["savepathfigures"],
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = True,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,
                                        
                    )

                    print(
                        f"[@{int(snapNumber)}]: PDF of gas plot"
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    apt.cr_pdf_versus_plot(
                        tmpDataDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
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
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,   
                                        
                    )
                    
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    print(
                        f"[@{int(snapNumber)}]: Cumulative PDF of gas plot"
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    apt.cr_pdf_versus_plot(
                        tmpDataDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = CRPARAMS["pdfParams"],
                        titleBool=CRPARAMS["titleBool"],
                        DPI=CRPARAMS["DPI"],
                        xsize=CRPARAMS["xsize"],
                        ysize=CRPARAMS["ysize"],
                        fontsize=CRPARAMS["fontsize"],
                        fontsizeTitle=CRPARAMS["fontsizeTitle"],
                        Nbins=CRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = CRPARAMS["savepathfigures"],
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,  
                                                
                    )
                    
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    print(
                        f"[@{int(snapNumber)}]: Normalised Cumulative PDF of gas plot"
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    apt.cr_pdf_versus_plot(
                        tmpDataDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = CRPARAMS["pdfParams"],
                        titleBool=CRPARAMS["titleBool"],
                        DPI=CRPARAMS["DPI"],
                        xsize=CRPARAMS["xsize"],
                        ysize=CRPARAMS["ysize"],
                        fontsize=CRPARAMS["fontsize"],
                        fontsizeTitle=CRPARAMS["fontsizeTitle"],
                        Nbins=CRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = CRPARAMS["savepathfigures"],
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = True,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,    
                                        
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
                            & (additionalParam is not None):
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

                        print(
                            f"[@{int(snapNumber)}]: PDF of column density properties"
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")     

                        apt.cr_pdf_versus_plot(
                            tmpColDict,
                            CRPARAMSHALO,
                            ylabel,
                            tmpxlimDict,
                            snapNumber,
                            weightKeys = ['mass'],
                            xParams = CRPARAMS["colParams"],
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
                            savePathBaseFigureData = CRPARAMS["savepathdata"],
                            saveFigureData = True,
                            SFR = False,
                            byType = False,
                            normalise = False,
                            verbose = DEBUG,
                            inplace = inplace,
                            forcePlotsWithoutxlimits = determineXlimits,
                                            
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")     

                        print(
                            f"[@{int(snapNumber)}]: Cumulative PDF of column density properties"
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")     
                        apt.cr_pdf_versus_plot(
                            tmpColDict,
                            CRPARAMSHALO,
                            ylabel,
                            tmpxlimDict,
                            snapNumber,
                            weightKeys = ['mass'],
                            xParams = CRPARAMS["colParams"],
                            titleBool=CRPARAMS["titleBool"],
                            DPI=CRPARAMS["DPI"],
                            xsize=CRPARAMS["xsize"],
                            ysize=CRPARAMS["ysize"],
                            fontsize=CRPARAMS["fontsize"],
                            fontsizeTitle=CRPARAMS["fontsizeTitle"],
                            Nbins=CRPARAMS["Nbins"],
                            ageWindow=None,
                            cumulative = True,
                            savePathBase = CRPARAMS["savepathfigures"],
                            savePathBaseFigureData = CRPARAMS["savepathdata"],
                            saveFigureData = True,
                            SFR = False,
                            byType = False,
                            normalise = False,
                            verbose = DEBUG,
                            inplace = inplace,
                            forcePlotsWithoutxlimits = determineXlimits,  
                                                                        
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")     

                        print(
                            f"[@{int(snapNumber)}]: Normalised Cumulative PDF of column density properties"
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")     
                        apt.cr_pdf_versus_plot(
                            tmpColDict,
                            CRPARAMSHALO,
                            ylabel,
                            tmpxlimDict,
                            snapNumber,
                            weightKeys = ['mass'],
                            xParams = CRPARAMS["colParams"],
                            titleBool=CRPARAMS["titleBool"],
                            DPI=CRPARAMS["DPI"],
                            xsize=CRPARAMS["xsize"],
                            ysize=CRPARAMS["ysize"],
                            fontsize=CRPARAMS["fontsize"],
                            fontsizeTitle=CRPARAMS["fontsizeTitle"],
                            Nbins=CRPARAMS["Nbins"],
                            ageWindow=None,
                            cumulative = True,
                            savePathBase = CRPARAMS["savepathfigures"],
                            savePathBaseFigureData = CRPARAMS["savepathdata"],
                            saveFigureData = True,
                            SFR = False,
                            byType = False,
                            normalise = True,
                            verbose = DEBUG,
                            inplace = inplace,
                            forcePlotsWithoutxlimits = determineXlimits,
                                            
                        )
                        matplotlib.rc_file_defaults()
                        plt.close("all")     

                    selectKeyStars = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            "Stars",
                            f"{int(snapNumber)}")
                    
                    tmpStarsDict = {selectKeyStars : copy.deepcopy(starsDict[selectKeyStars])}
                    print(
                        f"[@{int(snapNumber)}]: PDF of stars plot"
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    apt.cr_pdf_versus_plot(
                        tmpStarsDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [CRPARAMS["xParam"]],
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
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,
                                        
                    )
                    
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    print(
                        f"[@{int(snapNumber)}]: Cumulative PDF of stars plot"
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    apt.cr_pdf_versus_plot(
                        tmpStarsDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [CRPARAMS["xParam"]],
                        titleBool=CRPARAMS["titleBool"],
                        DPI=CRPARAMS["DPI"],
                        xsize=CRPARAMS["xsize"],
                        ysize=CRPARAMS["ysize"],
                        fontsize=CRPARAMS["fontsize"],
                        fontsizeTitle=CRPARAMS["fontsizeTitle"],
                        Nbins=CRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = CRPARAMS["savepathfigures"],
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = False,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,
                                        
                    )
                    
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    print(
                        f"[@{int(snapNumber)}]: Normalised Cumulative PDF of stars plot"
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")     

                    apt.cr_pdf_versus_plot(
                        tmpStarsDict,
                        CRPARAMSHALO,
                        ylabel,
                        xlimDict,
                        snapNumber,
                        weightKeys = ['mass'],
                        xParams = [CRPARAMS["xParam"]],
                        titleBool=CRPARAMS["titleBool"],
                        DPI=CRPARAMS["DPI"],
                        xsize=CRPARAMS["xsize"],
                        ysize=CRPARAMS["ysize"],
                        fontsize=CRPARAMS["fontsize"],
                        fontsizeTitle=CRPARAMS["fontsizeTitle"],
                        Nbins=CRPARAMS["Nbins"],
                        ageWindow=None,
                        cumulative = True,
                        savePathBase = CRPARAMS["savepathfigures"],
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        SFR = False,
                        byType = False,
                        normalise = True,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,
                                        
                    )

                    print(
                    f"[@{int(snapNumber)}]: Phases Plot..."
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
                        weightKeys = CRPARAMS["phasesWeightParams"],
                        titleBool=CRPARAMS["titleBool"],
                        DPI=CRPARAMS["DPI"],
                        xsize=CRPARAMS["xsize"],
                        ysize=CRPARAMS["ysize"],
                        fontsize=CRPARAMS["fontsize"],
                        fontsizeTitle=CRPARAMS["fontsizeTitle"],
                        Nbins=CRPARAMS["Nbins"],
                        savePathBase = CRPARAMS["savepathfigures"],
                        savePathBaseFigureData = CRPARAMS["savepathdata"],
                        saveFigureData = True,
                        verbose = DEBUG,
                        inplace = inplace,
                        forcePlotsWithoutxlimits = determineXlimits,    
                    )
                    matplotlib.rc_file_defaults()
                    plt.close("all")
                    print(
                    f"[@{int(snapNumber)}]: Finished snapshot..."
                    )


                selectKeyLast = (f"{CRPARAMS['resolution']}", 
                        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                        "Stars",
                        f"{int(snapRange[-1])}"
                        )
                
                tmpLastSnapStarsDict = {selectKeyLast : copy.deepcopy(lastSnapStarsDict[selectKeyLast])}
                print("")


                print(
                f"[@{int(snapNumber)}]: SFR plot..."
                )
                matplotlib.rc_file_defaults()
                plt.close("all")     

                apt.cr_pdf_versus_plot(
                    tmpLastSnapStarsDict,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber,
                    weightKeys = ['gima'],
                    xParams = ["age"],
                    titleBool=CRPARAMS["titleBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"],
                    ysize=CRPARAMS["ysize"],
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    Nbins = CRPARAMS["SFRBins"],
                    ageWindow = CRPARAMS["ageWindow"],
                    cumulative = False,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    saveFigureData = True,
                    SFR = True,
                    byType = False,
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    forcePlotsWithoutxlimits = determineXlimits,
                                    
                )

                matplotlib.rc_file_defaults()
                plt.close("all")     

                print(
                f"[@{int(snapNumber)}]: Cumulative SFR plot..."
                )
                matplotlib.rc_file_defaults()
                plt.close("all")     
                apt.cr_pdf_versus_plot(
                    tmpLastSnapStarsDict,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber,
                    weightKeys = ['gima'],
                    xParams = ["age"],
                    titleBool=CRPARAMS["titleBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"],
                    ysize=CRPARAMS["ysize"],
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    Nbins = CRPARAMS["SFRBins"],
                    ageWindow = CRPARAMS["ageWindow"],
                    cumulative = True,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    saveFigureData = True,
                    SFR = True,
                    byType = False,
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    forcePlotsWithoutxlimits = determineXlimits,
                                    
                )

                matplotlib.rc_file_defaults()
                plt.close("all")     

                print(
                f"[@{int(snapNumber)}]: Normalised Cumulative SFR plot..."
                )
                matplotlib.rc_file_defaults()
                plt.close("all")     
                apt.cr_pdf_versus_plot(
                    tmpLastSnapStarsDict,
                    CRPARAMSHALO,
                    ylabel,
                    xlimDict,
                    snapNumber,
                    weightKeys = ['gima'],
                    xParams = ["age"],
                    titleBool=CRPARAMS["titleBool"],
                    DPI=CRPARAMS["DPI"],
                    xsize=CRPARAMS["xsize"],
                    ysize=CRPARAMS["ysize"],
                    fontsize=CRPARAMS["fontsize"],
                    fontsizeTitle=CRPARAMS["fontsizeTitle"],
                    Nbins = CRPARAMS["SFRBins"],
                    ageWindow = CRPARAMS["ageWindow"],
                    cumulative = True,
                    savePathBase = CRPARAMS["savepathfigures"],
                    savePathBaseFigureData = CRPARAMS["savepathdata"],
                    saveFigureData = True,
                    SFR = True,
                    byType = False,
                    normalise = True,
                    verbose = DEBUG,
                    inplace = inplace,
                    forcePlotsWithoutxlimits = determineXlimits,
                                    
                )
                matplotlib.rc_file_defaults()
                plt.close("all")
            print(
            f"Finished sim..."

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
