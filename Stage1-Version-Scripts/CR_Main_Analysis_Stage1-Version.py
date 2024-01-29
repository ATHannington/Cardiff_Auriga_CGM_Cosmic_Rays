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

KnownAnalysisType = ["cgm", "ism", "all"]

matplotlib.use("Agg")  # For suppressing plotting on clusters

DEBUG = False
CRPARAMSPATHMASTER = "CRParams.json"

CRPARAMSMASTER = json.load(open(CRPARAMSPATHMASTER, "r"))

if CRPARAMSMASTER["ageWindow"] is not None:
    CRPARAMSMASTER["SFRBins"] = int(math.floor(CRPARAMSMASTER["ageWindow"]/CRPARAMSMASTER["windowBins"]))
else:
    CRPARAMSMASTER["SFRBins"]  = CRPARAMSMASTER["Nbins"] 
# =============================================================================#
#
#               USER DEFINED PARAMETERS
#
# ==============================================================================#
# File types for data save.
#   Full: full FullDict data
FullDataPathSuffix = f".h5"

FullDataPathSuffix = f".h5"

CRSELECTEDHALOESPATH = "CRSelectedHaloes.json"
CRSELECTEDHALOES = json.load(open(CRSELECTEDHALOESPATH, "r"))

ylabel = {
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_{\mathrm{H}}$ (cm$^{-3}$)",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"Radial Velocity (km s$^{-1}$)",
    "gz": r"Metallicity Z$_{\odot}$",
    "L": r"Specific Angular Momentum" + "\n" + r"(kpc km s$^{-1}$)",
    "P_thermal": r"P$_{Thermal}$ / k$_B$ (K cm$^{-3}$)",
    "P_magnetic": r"P$_{Magnetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_kinetic": r"P$_{Kinetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_tot": r"P$_{\mathrm{Tot}}$ = (P$_{thermal}$ + P$_{magnetic}$)/ k$_B$"
    + "\n"
    + r"(K cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "P_CR": r"P$_{\mathrm{CR}}$ (K cm$^{-3}$)",
    "PCR_Pthermal": r"(X$_{\mathrm{CR}}$ = P$_{\mathrm{CR}}$/P$_{Thermal}$)",
    "gah": r"Alfven Gas Heating (erg s$^{-1}$)",
    "bfld": r"||B-Field|| ($ \mu $G)",
    "Grad_T": r"||Temperature Gradient|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||n$_{\mathrm{H}}$ Gradient|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||B-Field Gradient|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{\mathrm{CR}}$ Gradient|| (K kpc$^{-4}$)",
    "gima" : r"Star Formation Rate (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfven CR Cooling (erg s$^{-1}$)",
    "tcool": r"Cooling Time (Gyr)",
    "theat": r"Heating Time (Gyr)",
    "tcross": r"Sound Crossing Cell Time (Gyr)",
    "tff": r"Free Fall Time (Gyr)",
    "tcool_tff": r"t$_{\mathrm{Cool}}$/t$_{FreeFall}$",
    "csound": r"Sound Speed (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "rho": r"Density (M$_{\odot}$ kpc$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
    "vol": r"Volume (kpc$^{3}$)",
    "age": "Lookback Time (Gyr)",
    "cool_length" : "Cooling Length (kpc)",
    "e_CR": r"Cosmic Ray Energy Density (erg cm$^{-3}$)",
}

xlimDict = {
    "R": {"xmin": 0.0, "xmax": CRPARAMSMASTER["Router"]},
    # "mass": {"xmin": 5.0, "xmax": 9.0},
    "L": {"xmin": 3.0, "xmax": 4.5},
    "T": {"xmin": 3.75, "xmax": 7.0},
    "n_H": {"xmin": -5.5, "xmax": -0.5},
    "B": {"xmin": -2.5, "xmax": 2.5},
    "vrad": {"xmin": -150.0, "xmax": 150.0},
    "gz": {"xmin": -1.5, "xmax": 1.5},
    "P_thermal": {"xmin": 0.5, "xmax": 3.5},
    "P_CR": {"xmin": -1.5, "xmax": 5.5},
    "PCR_Pthermal": {"xmin": -2.0, "xmax": 2.0},
    "P_magnetic": {"xmin": -2.0, "xmax": 4.5},
    "P_kinetic": {"xmin": 0.0, "xmax": 6.0},
    "P_tot": {"xmin": -1.0, "xmax": 7.0},
    "Pthermal_Pmagnetic": {"xmin": -1.5, "xmax": 3.0},
    "tcool": {"xmin": -3.5, "xmax": 2.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 6.0},
    "dens": {"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": 1.0},
    "rho": {"xmin": 2.0, "xmax": 8.0},
    "e_CR": {},
    }

for entry in CRPARAMSMASTER["logParameters"]:
    ylabel[entry] = r"$\mathrm{Log_{10}}$" + ylabel[entry]

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
        runAnalysisBool = True
        DataSavepathBase = CRPARAMSMASTER['savepath']
        if CRPARAMSMASTER["restartFlag"] is True:
            CRPARAMSHALO = {}
            for sim, simDict in allSimsDict.items():
                CRPARAMS = cr.cr_parameters(CRPARAMSMASTER, simDict)
                CRPARAMS.update({'halo': halo})
                selectKey = (f"{CRPARAMS['resolution']}",
                             f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")
                CRPARAMSHALO.update({selectKey: CRPARAMS})
            try:
                print("Restart Flag True! Will try to recover previous analysis data products.")
                print("Attempting to load data products...")
                for sim, CRPARAMS in CRPARAMSHALO.items():
                    if CRPARAMS['simfile'] is not None:
                        analysisType = CRPARAMS["analysisType"]

                        if analysisType not in KnownAnalysisType:
                            raise Exception(
                                f"ERROR! CRITICAL! Unknown analysis type: {analysisType}!"
                                + "\n"
                                + f"Availble analysis types: {KnownAnalysisType}"
                            )

                        saveDir = ( DataSavepathBase+f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}/"
                        )
                        DataSavepath = (
                            saveDir + "CR_" + f"{CRPARAMS['snapMin']}-{CRPARAMS['snapMax']}_{CRPARAMS['Rinner']}R{CRPARAMS['Router']}_"
                        )

                        loadPath = DataSavepath + "lastSnapDict.h5"
                        lastSnapDict = tr.hdf5_load(loadPath)

                        loadPath = DataSavepath + "statsDict.h5"
                        statsDict = tr.hdf5_load(loadPath)

                        loadPath = DataSavepath + "statsDictStars.h5"
                        statsDictStars = tr.hdf5_load(loadPath)

                        loadPath = DataSavepath + "starsDict.h5"
                        starsDict = tr.hdf5_load(loadPath)

                        loadPath = DataSavepath + "dataDict.h5"
                        dataDict = tr.hdf5_load(loadPath)



                print("...done!")
                runAnalysisBool = False
            except Exception as e:

                print("Restart Failed! \n" + f"exception: {e}" + "\n Re-running Analysis!")
                runAnalysisBool = True
        else:
            print("Restart Flag False! Re-running Analysis!")
            runAnalysisBool = True


        if runAnalysisBool is True:
            print("\n" + f"Starting SERIAL type Analysis!")
            dataDict = {}
            starsDict = {}
            lastSnapDict = {}
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
                CRPARAMSHALO.update({selectKey: CRPARAMS})
                if CRPARAMS['simfile'] is not None:
                    out = {}
                    quadPlotDict = {}
                    rotation_matrix = None
                    for snapNumber in snapRange:
                        tmpOut, rotation_matrix, tmpquadPlotDict = cr.cr_analysis_radial(
                            snapNumber=snapNumber,
                            CRPARAMS=CRPARAMS,
                            ylabel=ylabel,
                            xlimDict=xlimDict,
                            DataSavepathBase=DataSavepathBase,
                            FullDataPathSuffix=FullDataPathSuffix,
                            rotation_matrix=rotation_matrix,
                        )
                        out.update(tmpOut)
                        quadPlotDict.update(tmpquadPlotDict)

                    del tmpOut, tmpquadPlotDict
                    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

                    for key, val in out.items():
                        if key[-1] == "Stars":
                            if key[-2] == f"{int(snapRange[-1])}":
                                lastSnapDict.update({key : copy.deepcopy(val)})

                    flatDict = cr.cr_flatten_wrt_time(out, CRPARAMS, snapRange)

                    for key, val in flatDict.items():
                        if key[-1] == "Stars":
                            starsDict.update({key: copy.deepcopy(val)})
                        else:
                            dataDict.update({key: copy.deepcopy(val)})

                    del out, flatDict

                    quadPlotDictAveraged = cr.cr_quad_plot_averaging(
                        quadPlotDict,
                        CRPARAMS,
                        snapRange,
                    )

                    apt.cr_plot_projections(
                        quadPlotDictAveraged,
                        CRPARAMS,        
                        ylabel,
                        xlimDict,
                        xsize = CRPARAMS["xsizeImages"],
                        ysize = CRPARAMS["ysizeImages"],
                        projection=CRPARAMS["projection"],
                        Axes=CRPARAMS["Axes"],
                        boxsize=CRPARAMS["boxsize"],
                        boxlos=CRPARAMS["boxlos"],
                        pixres=CRPARAMS["pixres"],
                        pixreslos=CRPARAMS["pixreslos"],
                        fontsize = CRPARAMS["fontsize"],
                        DPI=CRPARAMS["DPI"],
                        numthreads=CRPARAMS["numthreads"],
                        savePathKeyword = "Averaged",
                        DEBUG = DEBUG,
                    )
            # # #----------------------------------------------------------------------#
            # # #      Calculate Radius xmin
            # # #----------------------------------------------------------------------#
            #
            # # for sim, CRPARAMS in CRPARAMSHALO.items():
            # #     if CRPARAMS['simfile'] is not None:
            # #         print(f"{sim}")
            # #         print("Calculate Radius xmin...")
            # #         selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}")
            # #
            # #         dataDict[selectKey]['maxDiskRadius'] = np.nanmedian(dataDict[selectKey]['maxDiskRadius'])
            # #
            # #         selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}","Stars")
            # # #         starsDict[selectKey]['maxDiskRadius'] = np.nanmedian(starsDict[selectKey]['maxDiskRadius'])
            #----------------------------------------------------------------------#
            #      Calculate statistics...
            #----------------------------------------------------------------------#

            print("")
            print("Calculate Statistics!")
            print(f"{halo}")
            statsDict = {}
            statsDictStars = {}
            for sim, CRPARAMS in CRPARAMSHALO.items():
                if CRPARAMS['simfile'] is not None:

                    print(f"{sim}")
                    print("Calculate Statistics...")
                    print("Gas...")
                    selectKey = (f"{CRPARAMS['resolution']}",
                                 f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}")

                    tmpCRPARAMS = copy.deepcopy(CRPARAMS)
                    tmpCRPARAMS['saveParams'] = tmpCRPARAMS['saveParams'] + ["mass"]

                    if tmpCRPARAMS['analysisType'] == 'cgm':
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] = tmpCRPARAMS['Router']

                    elif tmpCRPARAMS['analysisType'] == 'ism':
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] = tmpCRPARAMS['Rinner']
                    else:
                        xlimDict["R"]['xmin'] = 0.0
                        xlimDict["R"]['xmax'] = tmpCRPARAMS['Router']

                    print(tmpCRPARAMS['analysisType'], xlimDict["R"]['xmin'],
                          xlimDict["R"]['xmax'])
                    dat = cr.cr_calculate_statistics(
                        dataDict=dataDict[selectKey],
                        CRPARAMS=tmpCRPARAMS,
                        xParam=CRPARAMSMASTER["xParam"],
                        Nbins=CRPARAMSMASTER["NxParamBins"],
                        xlimDict=xlimDict
                    )

                    statsDict.update({selectKey: dat})

                    print("Stars...")
                    selectKey = (f"{CRPARAMS['resolution']}",
                                 f"{CRPARAMS['CR_indicator']}"+f"{CRPARAMS['no-alfven_indicator']}",
                                 "Stars")

                    dat = cr.cr_calculate_statistics(
                        dataDict=starsDict[selectKey],
                        CRPARAMS=tmpCRPARAMS,
                        xParam=CRPARAMSMASTER["xParam"],
                        Nbins=CRPARAMSMASTER["NxParamBins"],
                        xlimDict=xlimDict
                    )

                    statsDictStars.update({selectKey: dat})
            # ----------------------------------------------------------------------#
            # Save output ...
            # ----------------------------------------------------------------------#
            print("")
            print("***")
            print("Saving data products...")
            for sim, CRPARAMS in CRPARAMSHALO.items():
                if CRPARAMS['simfile'] is not None:
                    analysisType = CRPARAMS["analysisType"]

                    if analysisType not in KnownAnalysisType:
                        raise Exception(
                            f"ERROR! CRITICAL! Unknown analysis type: {analysisType}!"
                            + "\n"
                            + f"Availble analysis types: {KnownAnalysisType}"
                        )
                    saveDir = ( DataSavepathBase+f"type-{analysisType}/{CRPARAMS['halo']}/"+f"{CRPARAMS['resolution']}/{CRPARAMS['CR_indicator']}/"
                    )
                    DataSavepath = (
                        saveDir + "CR_" + f"{CRPARAMS['snapMin']}-{CRPARAMS['snapMax']}_{CRPARAMS['Rinner']}R{CRPARAMS['Router']}_"
                    )

                    savePath = DataSavepath + "dataDict.h5"
                    tr.hdf5_save(savePath,dataDict)

                    savePath = DataSavepath + "starsDict.h5"
                    tr.hdf5_save(savePath,starsDict)

                    savePath = DataSavepath + "lastSnapDict.h5"
                    tr.hdf5_save(savePath,lastSnapDict)

                    savePath = DataSavepath + "statsDict.h5"
                    tr.hdf5_save(savePath,statsDict)

                    savePath = DataSavepath + "statsDictStars.h5"
                    tr.hdf5_save(savePath,statsDictStars)
            print("...done!")
            print("***")
            print("")
        # ----------------------------------------------------------------------#
        #  Plots...
        # ----------------------------------------------------------------------#

        # ----------------------------------------------------------------------#
        #      medians_versus_plot...
        # ----------------------------------------------------------------------#

        print("")
        print(f"Medians vs {CRPARAMSMASTER['xParam']} Plot!")
        matplotlib.rc_file_defaults()
        plt.close("all")
        apt.cr_medians_versus_plot(
            statsDict=statsDict,
            CRPARAMSHALO=CRPARAMSHALO,
            ylabel=ylabel,
            yParam=CRPARAMSMASTER['mediansParams'],
            xParam=CRPARAMSMASTER["xParam"],
            xlimDict=xlimDict,
            titleBool=CRPARAMSMASTER["titleBool"],
            DPI=CRPARAMSMASTER["DPI"],
            xsize=CRPARAMSMASTER["xsize"],
            ysize=CRPARAMSMASTER["ysize"],
            fontsize=CRPARAMSMASTER["fontsize"],
            fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
            opacityPercentiles=0.25,
            snapNumber = "Averaged",
            colourmapMain=CRPARAMSMASTER['colourmapMain'],
        )
        matplotlib.rc_file_defaults()
        plt.close("all")

        #print("")
        #print(f"Mass PDF Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_pdf_versus_plot(
        #    dataDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = CRPARAMSMASTER["pdfParams"],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    Nbins=CRPARAMSMASTER["Nbins"],
        #    ageWindow=None,
        #    cumulative = False,
        #    saveCurve = True,
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #)
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Combined Mass PDF Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_combined_pdf_versus_plot(
        #    dataDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = CRPARAMSMASTER["pdfParams"],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    cumulative = False,
        #    savePathBase = "./",
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #    )
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Cumulative Mass PDF Plot!")

        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_pdf_versus_plot(
        #    dataDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = CRPARAMSMASTER["pdfParams"],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    Nbins=CRPARAMSMASTER["Nbins"],
        #    ageWindow=None,
        #    cumulative = True,
        #    saveCurve = True,
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #)
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Combined Cumulative Mass PDF Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_combined_pdf_versus_plot(
        #    dataDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = CRPARAMSMASTER["pdfParams"],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    cumulative = True,
        #    savePathBase = "./",
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #    )
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"PDF Plot Stars!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_pdf_versus_plot(
        #    starsDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = CRPARAMSMASTER["pdfParams"],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    Nbins=CRPARAMSMASTER["Nbins"],
        #    ageWindow=None,
        #    cumulative = False,
        #    saveCurve = True,
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #)
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        
        #print("")
        #print(f"Combined PDF Plot Stars!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_combined_pdf_versus_plot(
        #    starsDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = [CRPARAMSMASTER["xParam"]],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    cumulative = False,
        #    savePathBase = "./",
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #    )
        #matplotlib.rc_file_defaults()
        #plt.close("all")


        #print("")
        #print(f"Cumulative PDF Plot Stars!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_pdf_versus_plot(
        #    starsDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = [CRPARAMSMASTER["xParam"]],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    Nbins=CRPARAMSMASTER["Nbins"],
        #    ageWindow=None,
        #    cumulative = True,
        #    saveCurve = True,
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #)
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Combined Cumulative PDF Plot Stars!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_combined_pdf_versus_plot(
        #    starsDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    weightKeys = ['mass'],
        #    xParams = [CRPARAMSMASTER["xParam"]],
        #    titleBool=False,
        #    DPI=150,
        #    xsize=6.0,
        #    ysize=6.0,
        #    fontsize=13,
        #    fontsizeTitle=14,
        #    cumulative = True,
        #    savePathBase = "./",
        #    SFR = False,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #    )
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"SFR Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_pdf_versus_plot(
        #    dataDict=lastSnapDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    Nbins=CRPARAMSMASTER["SFRBins"],
        #    ageWindow=None,
        #    cumulative = False,
        #    saveCurve = True,
        #    SFR = True,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #)
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Combined SFR Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_combined_pdf_versus_plot(
        #    dataDict=lastSnapDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    cumulative = False,
        #    savePathBase = "./",
        #    SFR = True,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #    )
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Cumulative SFR Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_pdf_versus_plot(
        #    dataDict=lastSnapDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    Nbins=CRPARAMSMASTER["SFRBins"],
        #    ageWindow=None,
        #    cumulative = True,
        #    saveCurve = True,
        #    SFR = True,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #)
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Combined Cumulative SFR Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_combined_pdf_versus_plot(
        #    dataDict=lastSnapDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    cumulative = True,
        #    savePathBase = "./",
        #    SFR = True,
        #    byType = False,
        #    forceLogPDF = False,
        #    normalise = False,
        #    DEBUG = DEBUG,
        #    )
        #matplotlib.rc_file_defaults()
        #plt.close("all")

        #print("")
        #print(f"Phases Plot!")
        #matplotlib.rc_file_defaults()
        #plt.close("all")
        #apt.cr_hist_plot_xyz(
        #    dataDict=dataDict,
        #    CRPARAMSHALO=CRPARAMSHALO,
        #    ylabel=ylabel,
        #    xlimDict=xlimDict,
        #    colourmapMain=CRPARAMSMASTER["colourmapMain"],
        #    yParams = CRPARAMSMASTER["phasesyParams"],
        #    xParams = CRPARAMSMASTER["phasesxParams"],
        #    weightKeys = CRPARAMSMASTER["phasesWeightParams"],
        #    titleBool=CRPARAMSMASTER["titleBool"],
        #    DPI=CRPARAMSMASTER["DPI"],
        #    xsize=CRPARAMSMASTER["xsize"],
        #    ysize=CRPARAMSMASTER["ysize"],
        #    fontsize=CRPARAMSMASTER["fontsize"],
        #    fontsizeTitle=CRPARAMSMASTER["fontsizeTitle"],
        #    Nbins=CRPARAMSMASTER["Nbins"],
        #    saveCurve = True,
        #    savePathBase = "./",
        #    DEBUG = DEBUG,
        #)

        #matplotlib.rc_file_defaults()
        #plt.close("all")


