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

matplotlib.use("Agg")  # For suppressing plotting on clusters

DEBUG = False
inplace = False
CRPARAMSPATHMASTER = "CRParams_Quantitative-Summary.json"

singleValueKeys = ["Redshift", "Lookback", "Snap", "Rvir", "Rdisc"]


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
    "vrad": "seismic",
    "vrad_out": "Reds",
    "vrad_in": "Blues",
    "n_H": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
    "n_HI": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
    "n_H_col": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
    "n_HI_col": (CRPARAMSMASTER["colourmapMain"].split("_"))[0],
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

    for halo, allSimsDict in CRSELECTEDHALOES.items():
        finalOut = {}
        CRPARAMSHALO = {}
        for sim, simDict in allSimsDict.items():    
            runAnalysisBool = True
            DataSavepathBase = CRPARAMSMASTER["savepathdata"]
            FigureSavepathBase = CRPARAMSMASTER["savepathfigures"]
            lastSnapStarsDict = {}
            colDict = {}
            starsDict = {}
            dataDict = {}
            fullDataDict = {}
            for snapNumber in snapRange:
                CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
                CRPARAMS.update({'halo': halo})
                selectKey = (f"{CRPARAMS['resolution']}", 
                    f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                    f"{int(snapNumber)}")
                        
                CRPARAMSHALO.update({selectKey: CRPARAMS})

                if CRPARAMS['simfile'] is not None:
                    analysisType = CRPARAMS["analysisType"]
                    print(
                        "\n"
                        +"\n"
                        +f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Beginning analysis..."
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

                    innerDataDict, innerStarsDict, innerColDict, _, innerFullDataDict, rotation_matrix = cr.cr_analysis_radial(
                        snapNumber=snapNumber,
                        CRPARAMS=CRPARAMS,
                        ylabel=ylabel,
                        xlimDict=xlimDict,
                        types = [0,1,4,5],
                        colImagexlimDict = colImagexlimDict,
                        imageCmapDict = imageCmapDict,
                        DataSavepathBase = DataSavepathBase,
                        FigureSavepathBase = FigureSavepathBase,
                        FullDataPathSuffix=FullDataPathSuffix,
                        rotation_matrix=rotation_matrix,
                        verbose = DEBUG,
                    )

                    for key, val in innerStarsDict.items():
                        if (int(key[-1]) == int(snapRange[-1])):
                            lastSnapStarsDict.update({key: copy.deepcopy(val)})

                    typesCombos ={
                        "M200c" : [0,1,4,5],
                        "Mstars" : [4],
                        "MBH" : [5]
                        }

                    for label in KnownAnalysisType:
                        typesCombos.update({label : [0]})

                    print(
                        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Halo masses by type..."
                    )

                    flattened = {}
                    for sKey, snapDat in innerFullDataDict.items():
                        toCombine = {}
                        for label, typeCombo in typesCombos.items():
                            # dataDict starsDict fullDataDict colDict 
                            tmpDat = copy.deepcopy(snapDat)

                            whereNotType = np.isin(tmpDat["type"],np.array(typeCombo))==False
                            tmpDat = cr.remove_selection(
                                tmpDat,
                                removalConditionMask = whereNotType,
                                errorString = f"Remove types from {label} summary data",
                                hush = not DEBUG,
                                verbose = DEBUG,
                            )

                            if label == "ism":
                                whereNotCGM = tmpDat["R"] > CRPARAMS["Router"]
                                tmpDat = cr.remove_selection(
                                    tmpDat,
                                    removalConditionMask = whereNotCGM,
                                    errorString = f"Remove whereNotCGM from {label} summary data",
                                    hush = not DEBUG,
                                    verbose = DEBUG,
                                )

                                whereBelowCritDens = (tmpDat["ndens"] < 1.1e-1)

                                tmpDat = cr.remove_selection(
                                    tmpDat,
                                    removalConditionMask = whereBelowCritDens,
                                    errorString = f"Remove whereBelowCritDens from {label} summary data",
                                    verbose = DEBUG,
                                    )
                                label = "M_"+label
                            elif label == "cgm":

                                whereNotCGM = tmpDat["R"] > CRPARAMS["Router"]
                                tmpDat = cr.remove_selection(
                                    tmpDat,
                                    removalConditionMask = whereNotCGM,
                                    errorString = f"Remove whereNotCGM from {label} summary data",
                                    hush = not DEBUG,
                                    verbose = DEBUG,
                                )

                                whereAboveCritDens = (tmpDat["ndens"] >= 1.1e-1)

                                tmpDat = cr.remove_selection(
                                    tmpDat,
                                    removalConditionMask = whereAboveCritDens,
                                    errorString = f"Remove whereAboveCritDens from {label} summary data",
                                    verbose = DEBUG,
                                    )
                                label = "M_"+label
                            elif label == "all":
                                label = "M_"+label


                            toCombine.update({label : np.sum(tmpDat["mass"],axis=0)})
                        
                        # innerFlattened = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = DEBUG, hush = not DEBUG)
                        flattened.update({sKey: copy.deepcopy(toCombine)})

                    # unaveragedFlattenedMasses = copy.deepcopy(flattened)
                    # if (len(snapRange)>1):
                    #     for label, data in flattened.items():
                    #         dataCopy = copy.deepcopy(data)
                    #         for key,value in data.items():
                    #             dataCopy.update({key: np.nanmedian(value,axis=-1)})
                    #         flattened[label].update(dataCopy)

                    print(
                        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Hydrogen species masses..."
                    )
                    masses = copy.deepcopy(flattened)
                    flattened = {}
                    for sKey, snapDat in innerFullDataDict.items():
                        toCombine = {}
                        for label in ["H","HI"]:
                            # dataDict starsDict fullDataDict colDict 
                            tmpDat = copy.deepcopy(snapDat)

                            whereNotGas = tmpDat["type"]!=0
                            tmpDat = cr.remove_selection(
                                tmpDat,
                                removalConditionMask = whereNotGas,
                                errorString = f"Remove whereNotGas from {label} summary data",
                                hush = not DEBUG,
                                verbose = DEBUG,
                            )

                            whereNotCGM = tmpDat["R"] > CRPARAMS["Router"]
                            tmpDat = cr.remove_selection(
                                tmpDat,
                                removalConditionMask = whereNotCGM,
                                errorString = f"Remove whereNotCGM from {label} summary data",
                                hush = not DEBUG,
                                verbose = DEBUG,
                            )

                            whereAboveCritDens = (tmpDat["ndens"] >= 1.1e-1)

                            tmpDat = cr.remove_selection(
                                tmpDat,
                                removalConditionMask = whereAboveCritDens,
                                errorString = f"Remove whereAboveCritDens from {label} summary data",
                                verbose = DEBUG,
                                )


                            mass = (tmpDat["n_"+label]*((c.parsec * 1e3) ** 3)*tmpDat["vol"]*c.amu*tmpDat["gmet"][:,0]/(c.msol))
                            toCombine.update({"M_"+label+";CGM" : np.sum(mass,axis=0)})
                        
                        # innerFlattened = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = DEBUG, hush = not DEBUG)
                        flattened.update({sKey: copy.deepcopy(toCombine)})

                    # unaveragedFlattenedHydrogenMasses = copy.deepcopy(flattened)
                    # if (len(snapRange)>1):
                    #     for label, data in flattened.items():
                    #         dataCopy = copy.deepcopy(data)
                    #         for key,value in data.items():
                    #             dataCopy.update({key: np.nanmedian(value,axis=-1)})
                    #         flattened[label].update(dataCopy)

                    hydrogenMasses = copy.deepcopy(flattened)

                    print(
                        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Radii..."
                    )
                    
                    flattened = {}
                    for sKey, snapDat in innerFullDataDict.items():      
                        toCombine = {}
                        for label in ["Rdisc","Rvir"]:  
                            tmpDat = copy.deepcopy(snapDat)
                            toCombine.update({label : tmpDat[label][0]})
                        # innerFlattened = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = DEBUG, hush = not DEBUG)
                        flattened.update({sKey: copy.deepcopy(toCombine)})

                    # unaveragedFlattenedRadii = copy.deepcopy(flattened)
                    # if (len(snapRange)>1):
                    #     for label, data in flattened.items():
                    #         dataCopy = copy.deepcopy(data)
                    #         for key,value in data.items():
                    #             dataCopy.update({key: np.nanmedian(value,axis=-1)})
                    #         flattened[label].update(dataCopy)

                    radii = copy.deepcopy(flattened)

                    print(
                        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}, @{int(snapNumber)}]: Combine data..."
                    )
                    

                    inner = {}
                    for dat in [masses,hydrogenMasses,radii]:
                        for key, dd in dat.items():
                            for kk, vv in dd.items():
                                inner[kk] = copy.deepcopy(np.asarray(vv))

                    out = {selectKey : copy.deepcopy(inner)}

                    dataDict.update(out)
                    
                    starsDict.update(innerStarsDict)
                    # fullDataDict.update(innerFullDataDict)
                    # if len(CRPARAMS["colParams"])>0:
                        # colDict.update(innerColDict)
                    print(
                        f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}]: Snap calculation complete!"
                    )

            print("\n"+f"Starting {halo} quantitative time-averaged summary ...")
            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
            #   Quantitative Time-Averaged Summary
            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
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
                            f"{CRPARAMS['CR_indicator']}"
                            +f"{CRPARAMS['no-alfven_indicator']}",
                            f"{int(snapNumber)}")
                

                flattened = cr.cr_flatten_wrt_time(dataDict, stack = True, verbose = DEBUG, hush = not DEBUG)
                if (len(snapRange)>1):
                    for sKey, data in flattened.items():
                        dataCopy = copy.deepcopy(data)
                        for label,value in data.items():
                            dataCopy.update({label: np.asarray([np.nanmedian(value,axis=-1)])})
                        flattened[sKey].update(dataCopy)

                selectKey = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"
                            +f"{CRPARAMS['no-alfven_indicator']}"
                )

                simOut = copy.deepcopy(flattened)
                # for key, dat in flattened.items():
                #     inner = copy.deepcopy(dat)
                #     res, crAlfven = key

                #     alfvenindicator = "" 
                #     crindicator = ""
                #     crAlfvenSplit = crAlfven.split("_")
                #     if crAlfvenSplit[-1] == "Alfven":
                #         alfvenindicator = "_".join(crAlfvenSplit[-2:])
                #     else:
                #         alfvenindicator = ""
                #     crindicator = "_".join(crAlfvenSplit[:2])

                if CRPARAMS["SFR"] is True:
                    selectKeyLast = (f"{CRPARAMS['resolution']}", 
                            f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                            "Stars",
                            f"{int(snapRange[-1])}"
                            )
    
                    if CRPARAMS["ageWindow"] is None:
                        selectKeyFirst = (f"{CRPARAMS['resolution']}", 
                                f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                                "Stars",
                                f"{int(snapRange[0])}"
                                )
                        CRPARAMS["ageWindow"] = starsDict[selectKeyFirst]["Lookback"][0]
                                                                    

                    sfrData = copy.deepcopy(lastSnapStarsDict[selectKeyLast])

                    analysisParam = "age"
                    weightKey = "gima"

                    for excl in singleValueKeys:
                        if excl in list(sfrData.keys()):
                            sfrData.pop(excl)

                    plotData = sfrData[analysisParam]

                    whereAgeBelowLimit = np.full(shape=np.shape(plotData),fill_value=True)
                    if CRPARAMS["ageWindow"] is not None:
                        print("Minimum age detected = ", np.nanmin(plotData), "Gyr")
                        # minAge = np.nanmin(tmpPlot) + ((np.nanmax(tmpPlot) - np.nanmin(tmpPlot))*ageWindow)
                        maxAge = np.nanmin(plotData)+CRPARAMS["ageWindow"]
                        print("Maximum age for plotting = ", maxAge, "Gyr")

                        whereAgeBelowLimit = plotData<=maxAge
                        print("Number of data points meeting age = ",np.shape(np.where(whereAgeBelowLimit==True)[0])[0])
                        whereAgeBeyondLimit = plotData>maxAge
                        sfrData = cr.remove_selection(
                            sfrData,
                            removalConditionMask = whereAgeBeyondLimit,
                            errorString = "Remove stars formed beyond age limit",
                            hush = not DEBUG,
                            verbose = DEBUG,
                        )

                    plotData = sfrData[analysisParam]
                    if analysisParam in CRPARAMS["logParameters"]:
                        plotData = np.log10(plotData).copy()
                    else:
                        plotData = plotData.copy()
                    print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}]: SFR Calculation..."
                    )

                    # Calculate SFR
                    cumulativeStellarMass = np.sum(sfrData[weightKey],axis=0)
                    delta = CRPARAMS["ageWindow"]
                    sfrval = cumulativeStellarMass/(delta*1e9) # SFR [per yr]


                    selectKey = (
                        f"{CRPARAMS['resolution']}",
                        f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}"
                        )


                    simOut[selectKey]["SFR"] = np.asarray([copy.deepcopy(sfrval)])

                finalOut.update({selectKey : copy.deepcopy(simOut)})

                print(
                    f"[@{CRPARAMS['halo']}, @{CRPARAMS['resolution']}, @{CRPARAMS['CR_indicator']}{CRPARAMS['no-alfven_indicator']}]: Sim calculations complete!"
                )


        cr.cr_save_to_excel(
        finalOut,
        CRPARAMSHALO,
        savePathBase = "./",
        filename = f"CR-Data_{CRPARAMS['halo']}.xlsx",
        replacements = [["high","hi"],["standard","std"],["no_CRs","MHD"],["with_CRs","CRs"],["_no_Alfven","-NA"]]
        )
        print(
            f"[@{CRPARAMS['halo']}]: Finished halo..."
        )
    print(
        "\n"+
        f"Finished completely! :)"
    )
