import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
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
import math
import os

plt.rcParams.update(matplotlib.rcParamsDefault)


# We want to utilise inplace operations to keep memory within RAM limits...
inplace = False

stack = True
DEBUG = False

HYPARAMSPATH = "HYParams.json"
HYPARAMS = json.load(open(HYPARAMSPATH, "r"))


#if "mass" not in HYPARAMS["colParams"]:
#    HYPARAMS["colParams"]+=["mass"]

if HYPARAMS["ageWindow"] is not None:
    HYPARAMS["SFRBins"] = int(math.floor(HYPARAMS["ageWindow"]/HYPARAMS["windowBins"]))
else:
    HYPARAMS["SFRBins"]  = HYPARAMS["Nbins"] 

loadPathBase = ""
loadDirectories = [


    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_1kpc/",
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/surge/level4_cgm/h5_500pc/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/",
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/",
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/",
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_standard/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_standard/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc-hy-1kpc/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc-hy-500pc/",
    ]

styleDictGroupingKeys = {
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_1kpc/" : ("surge","1kpc","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/surge/level4_cgm/h5_500pc/" : ("surge","500pc","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/" : ("hy","500pc","final","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/" : ("hy","1kpc","final","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/" : ("hy","1kpc","l3-mass","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/" : ("hy","1kpc","hard","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/" : ("std","L3"), 
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_standard/" : ("std","L4"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_standard/" : ("std","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc/" : ("surge","2kpc","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc/" : ("surge","1kpc","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc-hy-1kpc/" : ("hy","2kpc","final","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc-hy-500pc/" : ("hy","1kpc","final","L5"),
}


simulations = []
savePaths = []
savePathsData = []

for dir in loadDirectories:
    loadpath = loadPathBase+dir+"/output/"
    simulations.append(loadpath)
    savepath = HYPARAMS["savepathfigures"] + dir + "/"
    savepathdata = HYPARAMS["savepathdata"] + dir + "/"
    savePaths.append(savepath)
    savePathsData.append(savepathdata)



snapRange = [
        xx
        for xx in range(
            int(HYPARAMS["snapMin"]),
            int(HYPARAMS["snapMax"]) + 1,
            1,
        )
    ]

ylabel = {
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_{\mathrm{H}}$ (cm$^{-3}$)",
    "n_H_col": r"N$_{\mathrm{H}}$ (cm$^{-2}$)",
    "n_HI": r"n$_{\mathrm{HI}}$ (cm$^{-3}$)",
    "n_HI_col": r"N$_{\mathrm{HI}}$ (cm$^{-2}$)",
    "nh": r"Neutral Hydrogen Fraction",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"Radial Velocity (km s$^{-1}$)",
    "vrad_in": r"Inflow Velocity (km s$^{-1}$)",
    "vrad_out": r"Outflow Velocity (km s$^{-1}$)",
    "gz": r"Metallicity Z$_{\odot}$",
    "L": r"Specific Angular Momentum" + "\n" + r"(kpc km s$^{-1}$)",
    "P_thermal": r"P$_{Thermal}$ (erg cm$^{-3}$)",
    "P_magnetic": r"P$_{Magnetic}$ (erg cm$^{-3}$)",
    "P_kinetic": r"P$_{Kinetic}$ (erg cm$^{-3}$)",
    "P_tot": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "P_tot+k": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "P_CR": r"P$_{\mathrm{CR}}$ (erg cm$^{-3}$)",
    "PCR_Pmagnetic" : r"P$_{\mathrm{CR}}$/P$_{magnetic}$",
    "PCR_Pthermal": r"(X$_{\mathrm{CR}}$ = P$_{\mathrm{CR}}$/P$_{Thermal}$)",
    "gah": r"Alfven Gas Heating (erg s$^{-1}$)",
    "bfld": r"||B-Field|| ($ \mu $G)",
    "Grad_T": r"||Temperature Gradient|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||n$_{\mathrm{H}}$ Gradient|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||B-Field Gradient|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{\mathrm{CR}}$ Gradient|| (erg kpc$^{-4}$)",
    "gima" : r"Star Formation Rate (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfven CR Cooling (erg s$^{-1}$)",
    "tcool": r"Cooling Time (Gyr)",
    "theat": r"Heating Time (Gyr)",
    "tcross": r"Sound Crossing Cell Time (Gyr)",
    "tff": r"Free Fall Time (Gyr)",
    "tcool_tff": r"t$_{\mathrm{Cool}}$/t$_{FreeFall}$",
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
    "R": {"xmin": 0.0, "xmax": 200.0},
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
    "PCR_Pmagnetic": {"xmin": -3.0, "xmax": 3.0},
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
    "vol": {"xmin": -2.0, "xmax" : 3.0},
    "cool_length" : {"xmin": -1.0, "xmax": 2.0},
    "csound" : {},
    "nh" : {"xmin": -7.0, "xmax": 1.0},
    "e_CR": {"xmin": -8.0, "xmax": 0.0},
}



# ==============================================================================#
#
#          Main
#
# ==============================================================================#



for entry in HYPARAMS["logParameters"]:
    ylabel[entry] = r"$\mathrm{Log_{10}}$ " + ylabel[entry]
    ylabel[entry] = ylabel[entry].replace("(","[")
    ylabel[entry] = ylabel[entry].replace(")","]")

#   Perform forbidden log of Grad check
deleteParams = []
for entry in HYPARAMS["logParameters"]:
    entrySplit = entry.split("_")
    if (
        ("Grad" in entrySplit) &
        (np.any(np.isin(np.array(HYPARAMS["logParameters"]), np.array(
            "_".join(entrySplit[1:])))))
    ):
        deleteParams.append(entry)

for entry in deleteParams:
    HYPARAMS["logParameters"].remove(entry)




if __name__ == "__main__":


    # # # loadDirectories = [
    # # # "/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/",
    # # # "/spxfv/surge/level4_cgm/h5_500pc/",
    # # # "/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/",
    # # # "/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/",
    # # # "/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/",
    # # # "/c1838736/Auriga/level3_cgm_almost/h5_standard/",
    # # # "/spxfv/Auriga/level4_cgm/h5_standard/",
    # # # "/spxfv/Auriga/level4_cgm/h5_1kpc/",
    # # # ]

    selectKeysList = []
    HYPARAMSHALO = {}
    styleKeys = []
    for (loadpath,savePathBase,savePathBaseFigureData) in zip(loadDirectories,savePaths,savePathsData):
        print(loadpath)
        # we need to nest the
        # statistics dictionaries in an outer dicitionary with some simulation descriptors, such as resolution and
        # Auriga halo number.
        splitList = loadpath.split("/")
        baseResLevel, haloLabel = splitList[-3:-1]
        baseAdjusted = "L"+(baseResLevel.split("_"))[0][-1]
        tmp = haloLabel.split("_")
        haloSplitList = []
        for xx in tmp:
            splitxx = xx.split("-")
            haloSplitList += splitxx
        haloLabelKeySaveable = "_".join(haloSplitList)
        auHalo, resLabel = haloSplitList[0], "_".join([ll for ll in haloSplitList[1:] if (ll!="transition")&(ll!="res")])

        selectKey = (baseAdjusted, resLabel)
        selectKeysList.append(selectKey)
        HYPARAMSHALO.update({selectKey: HYPARAMS})
        styleKeys.append(styleDictGroupingKeys[loadpath])

    # ----------------------------------------------------------------------#
    #  Plots...
    # ----------------------------------------------------------------------#
    


    tmpstyleDict = apt.get_linestyles_and_colours(styleKeys,colourmapMain="plasma",colourGroupBy=[],linestyleGroupBy=["std","hy","surge"],lastColourOffset = 0.0)
    styleDict = {}
    for selectKey,dd in zip(selectKeysList,tmpstyleDict.values()):
        styleDict.update({selectKey : dd})

    snapNumber="Averaged"

    print(
        f"Time averaged Medians profile plots..."
    )
    matplotlib.rc_file_defaults()
    plt.close("all")     

    tmp = apt.hy_load_statistics_data(
        selectKeysList,
        loadDirectories,
        snapRange,
        loadPathBase = HYPARAMS["savepathdata"],
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


    apt.medians_versus_plot(
        statsOut,
        HYPARAMSHALO,
        ylabel=ylabel,
        xlimDict=xlimDict,
        snapNumber=snapNumber,
        yParam=HYPARAMS["mediansParams"],
        xParam=HYPARAMS["xParam"],
        titleBool=HYPARAMS["titleBool"],
        DPI = HYPARAMS["DPI"],
        xsize = HYPARAMS["xsize"],
        ysize = HYPARAMS["ysize"],
        fontsize = HYPARAMS["fontsize"],
        fontsizeTitle = HYPARAMS["fontsizeTitle"],
        opacityPercentiles = HYPARAMS["opacityPercentiles"],
        savePathBase = HYPARAMS["savepathfigures"],
        savePathBaseFigureData = HYPARAMS["savepathdata"],
        inplace = inplace,
        saveFigureData = False,
        replotFromData = True,
        combineMultipleOntoAxis = True,
        selectKeysList = None,
        styleDict = styleDict,
        )


    print(
        f"Time averaged Gas PDF plots..."
    )
    matplotlib.rc_file_defaults()
    plt.close("all")     

    tmp = apt.hy_load_pdf_versus_plot_data(
        selectKeysList,
        loadDirectories,
        snapRange,
        weightKeys = HYPARAMS['nonMassWeightDict'],
        xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
        cumulative = False,
        loadPathBase = HYPARAMS["savepathdata"],
        loadPathSuffix = "",
        SFR = False,
        normalise = False,
        stack = True,
        verbose = DEBUG,
        )

    pdfOut = copy.deepcopy(tmp)    

    if (len(snapRange)>1)&(stack is True):
        for sKey, data in pdfOut.items():
            dataCopy = copy.deepcopy(data)
            for key,dd in data.items():
                for kk, value in dd.items():
                    dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
            pdfOut[sKey].update(dataCopy)

    print(
        f"[@{int(snapNumber)}]: PDF of gas plot"
    )

    apt.pdf_versus_plot(
        pdfOut,
        ylabel,
        xlimDict,
        HYPARAMS["logParameters"],
        snapNumber,
        weightKeys = ['mass'], #<<<< Need to rerun these with vol weights
        xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
        titleBool=HYPARAMS["titleBool"],
        DPI=HYPARAMS["DPI"],
        xsize=HYPARAMS["xsize"],
        ysize=HYPARAMS["ysize"],
        fontsize=HYPARAMS["fontsize"],
        fontsizeTitle=HYPARAMS["fontsizeTitle"],
        Nbins=HYPARAMS["Nbins"],
        ageWindow=None,
        cumulative = False,
        savePathBase = savePathBase,
        savePathBaseFigureData = savePathBaseFigureData,
        saveFigureData = True,
        SFR = False,
        forceYAxisLog = HYPARAMS["forceYAxisLog"],
        normalise = False,
        verbose = DEBUG,
        inplace = inplace,
    )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
               f"[@{int(snapNumber)}]: PDF of gas, binned by {HYPARAMS['xParam']} plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                
                
                
                ageWindow=None,
                cumulative = False,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})
            print(
               f"[@{int(snapNumber)}]: Cumulative PDF of gas plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = True,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})
            print(
               f"[@{int(snapNumber)}]: Normalised Cumulative PDF of gas plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = True,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = True,
                verbose = DEBUG,
                inplace = inplace,
                
            )


            # # # # # -----------------------------------------------#
            # # # # #           
            # # # # #                     by Type
            # # # # #
            # # # # # -----------------------------------------------#



            # # # # # By type plots do work, but this caused memory issues when trying to load in all particle types
            # # # # # for our highest resolution simulations so we have commented these calls out for now. Useful diagnostic
            # # # # # for contamination of the subhalo by different dark matter resolution types,
            # # # # # and for ensuring any changes made to Arepo haven't broken the dark matter in an unintended way.
            # # # # print(
            # # # #    f"[@{int(snapNumber)}]: By Type PDF of mass vs R plot..."
            # # # # )

            # # # # apt.pdf_versus_plot(
            # # # #    out,
            # # # #    ylabel,
            # # # #    xlimDict,
            # # # #    HYPARAMS["logParameters"],
            # # # #    snapNumber,
            # # # #    weightKeys = ['mass'],
            # # # #    xParams = ["R"],
            # # # #    savePathBase = savePathBase,
            # # # #    savePathBaseFigureData = savePathBaseFigureData,
            # # # #    saveFigureData = True,
            # # # #    
            # # # #    forceYAxisLog = HYPARAMS["forceYAxisLog"],
            # # # # )

            # # # # print(
            # # # #    f"[@{int(snapNumber)}]: By Type Cumulative PDF of mass vs R plot..."
            # # # # )

            # # # # apt.pdf_versus_plot(
            # # # #    out,
            # # # #    ylabel,
            # # # #    xlimDict,
            # # # #    HYPARAMS["logParameters"],
            # # # #    snapNumber,
            # # # #    weightKeys = ['mass'],
            # # # #    xParams = ["R"],
            # # # #    cumulative = True,
            # # # #    savePathBase = savePathBase,
            # # # #    savePathBaseFigureData = savePathBaseFigureData,
            # # # #    saveFigureData = True,
            # # # #    
            # # # #    forceYAxisLog = HYPARAMS["forceYAxisLog"],
            # # # # )

            # # # # print(
            # # # #    f"[@{int(snapNumber)}]: By Type Normalised Cumulative PDF of mass vs R plot..."
            # # # # )

            # # # # apt.pdf_versus_plot(
            # # # #    out,
            # # # #    ylabel,
            # # # #    xlimDict,
            # # # #    HYPARAMS["logParameters"],
            # # # #    snapNumber,
            # # # #    weightKeys = ['mass'],
            # # # #    xParams = ["R"],
            # # # #    cumulative = True,
            # # # #    normalise = True,
            # # # #    savePathBase = savePathBase,
            # # # #    savePathBaseFigureData = savePathBaseFigureData,
            # # # #    saveFigureData = True,
            # # # #    
            # # # #    forceYAxisLog = HYPARAMS["forceYAxisLog"],
            # # # # )







            #-----------------------------------------------#
            #           
            #                     SFR
            #
            #-----------------------------------------------#


            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})


            print(
               f"[@{int(snapNumber)}]: SFR plot..."
            )
            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["SFRBins"],
                ageWindow=HYPARAMS["ageWindow"],
                cumulative = False,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = True,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
               f"[@{int(snapNumber)}]: Cumulative SFR plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["SFRBins"],
                ageWindow=HYPARAMS["ageWindow"],
                cumulative = True,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = True,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
               f"[@{int(snapNumber)}]: Normalised Cumulative SFR plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["SFRBins"],
                ageWindow=HYPARAMS["ageWindow"],
                cumulative = True,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = True,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = True,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
                f"[@{int(snapNumber)}]: Remove stars..."
            )
            whereStars = snap.data["type"] == 4
            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereStars,
                errorString = "Remove Stars from Gas",
                verbose = DEBUG
                )






            #-----------------------------------------------#
            #           
            #                     CGM
            #
            #-----------------------------------------------#



            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove <30 Kpc..."
            )

            whereInnerRadius = out["R"]<=30.0

            out = cr.remove_selection(
                out,
                removalConditionMask = whereInnerRadius,
                errorString = "Remove <30 Kpc",
                verbose = DEBUG,
                )

            print(
               f"[@{int(snapNumber)}]: CGM PDF of gas plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = False,
                savePathBase = savePathBase + "CGM_only/",
                savePathBaseFigureData = savePathBaseFigureData + "CGM_only/",
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove <30 Kpc..."
            )

            whereInnerRadius = out["R"]<=30.0

            out = cr.remove_selection(
                out,
                removalConditionMask = whereInnerRadius,
                errorString = "Remove <30 Kpc",
                verbose = DEBUG,
                )

            print(
               f"[@{int(snapNumber)}]: Cumulative CGM PDF of gas plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = True,
                savePathBase = savePathBase + "CGM_only/",
                savePathBaseFigureData = savePathBaseFigureData + "CGM_only/",
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove <30 Kpc..."
            )

            whereInnerRadius = out["R"]<=30.0

            out = cr.remove_selection(
                out,
                removalConditionMask = whereInnerRadius,
                errorString = "Remove <30 Kpc",
                verbose = DEBUG,
                )

            print(
               f"[@{int(snapNumber)}]: Normalised Cumulative CGM PDF of gas plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = True,
                savePathBase = savePathBase + "CGM_only/",
                savePathBaseFigureData = savePathBaseFigureData + "CGM_only/",
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = True,
                verbose = DEBUG,
                inplace = inplace,
                
            )


            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})




    if len(HYPARAMS["colParams"])>0:
        print(
        "Time averaged Column Density Medians profile plots..."
        )

        selectKeysListCol = [tuple(list(sKey)+["col"]) for sKey in selectKeysList]

        # # # Create variant of xlimDict specifically for images of col params
        # # tmpxlimDict = copy.deepcopy(xlimDict)

        # # # Add the col param specific limits to the xlimDict variant
        # # for key, value in colImagexlimDict.items():
        # #     tmpxlimDict[key] = value

        #---------------#
        # Check for any none-position-based parameters we need to track for col params:
        #       Start with mass (always needed!) and xParam:
        additionalColParams = ["mass"]
        if np.any(np.isin(np.asarray([HYPARAMS["xParam"]]),np.array(["R","x","y","z","pos"]))) == False:
            additionalColParams.append(HYPARAMS["xParam"])

        #       Now add in anything we needed to track for weights of col params in statistics
        cols = HYPARAMS["colParams"]
        for param in cols:
            additionalParam = HYPARAMS["nonMassWeightDict"][param]
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

        cols = HYPARAMS["colParams"]+additionalColParams

        COLHYPARAMS= copy.deepcopy(HYPARAMS)
        COLHYPARAMS["saveParams"]=COLHYPARAMS["saveParams"]+cols

        COLHYPARAMSHALO = copy.deepcopy(HYPARAMSHALO)
        # # COLHYPARAMSHALO = {sKey: values for sKey,(_,values) in zip(selectKeysListCol,HYPARAMSHALO.items())}
        
        for kk in COLHYPARAMSHALO.keys():
            COLHYPARAMSHALO[kk]["saveParams"] = COLHYPARAMSHALO[kk]["saveParams"]+cols

        matplotlib.rc_file_defaults()
        plt.close("all")     

        selectKeysListCol = [tuple(list(sKey)+["col"]) for sKey in selectKeysList]

        tmp = apt.hy_load_statistics_data(
            selectKeysListCol,
            loadDirectories,
            snapRange,
            loadPathBase = HYPARAMS["savepathdata"],
            loadFile = "colStatsDict",
            fileType = ".h5",
            stack = True,
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


        apt.medians_versus_plot(
            statsOutCol,
            COLHYPARAMSHALO,
            ylabel=ylabel,
            xlimDict=xlimDict,
            snapNumber=snapNumber,
            yParam=COLHYPARAMS["colParams"],
            xParam=HYPARAMS["xParam"],
            titleBool=HYPARAMS["titleBool"],
            DPI = HYPARAMS["DPI"],
            xsize = HYPARAMS["xsize"],
            ysize = HYPARAMS["ysize"],
            fontsize = HYPARAMS["fontsize"],
            fontsizeTitle = HYPARAMS["fontsizeTitle"],
            opacityPercentiles = HYPARAMS["opacityPercentiles"],
            savePathBase = HYPARAMS["savepathfigures"],
            savePathBaseFigureData = HYPARAMS["savepathdata"],
            inplace = inplace,
            saveFigureData = False,
            replotFromData = True,
            combineMultipleOntoAxis = True,
            selectKeysList = None,
            styleDict = styleDict,
            )

            # -----------------------------------------------#
            #           
            #             column density PDFs
            #
            # -----------------------------------------------#

            print(
               f"[@{int(snapNumber)}]: PDF of col dens gas plot"
            )

            apt.pdf_versus_plot(
                copy.deepcopy(colout),
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["colParams"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = False,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
               f"[@{int(snapNumber)}]: PDF of col dens gas, binned by {HYPARAMS['xParam']} plot"
            )


            apt.pdf_versus_plot(
                copy.deepcopy(colout),
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["colParams"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                
                
                
                ageWindow=None,
                cumulative = False,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
               f"[@{int(snapNumber)}]: Cumulative PDF of col dens gas plot"
            )

            apt.pdf_versus_plot(
                copy.deepcopy(colout),
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["colParams"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = True,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                
            )

            print(
               f"[@{int(snapNumber)}]: Normalised Cumulative PDF of col dens gas plot"
            )


            apt.pdf_versus_plot(
                copy.deepcopy(colout),
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["colParams"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                ageWindow=None,
                cumulative = True,
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                SFR = False,
                
                forceYAxisLog = HYPARAMS["forceYAxisLog"],
                normalise = True,
                verbose = DEBUG,
                inplace = inplace,
                
            )


            # -----------------------------------------------#
            #           
            #           CGM column density PDFs
            #
            # -----------------------------------------------#

            if bool(colout) is True:

                print(
                    f"[@{int(snapNumber)}]: Remove <30 Kpc..."
                )

                whereInnerRadius = colout["R"]<=30.0
                cgmcolout = copy.deepcopy(colout)
                cgmcolout = cr.remove_selection(
                    cgmcolout,
                    removalConditionMask = whereInnerRadius,
                    errorString = "Remove <30 Kpc",
                    verbose = DEBUG,
                    )
    

                # -----------------------------------------------#
                #           
                #             column density PDFs
                #
                # -----------------------------------------------#

                print(
                f"[@{int(snapNumber)}]: PDF of col dens gas plot"
                )

                apt.pdf_versus_plot(
                    copy.deepcopy(cgmcolout),
                    ylabel,
                    xlimDict,
                    HYPARAMS["logParameters"],
                    snapNumber,
                    weightKeys = ['mass'],
                    xParams = HYPARAMS["colParams"],
                    titleBool=HYPARAMS["titleBool"],
                    DPI=HYPARAMS["DPI"],
                    xsize=HYPARAMS["xsize"],
                    ysize=HYPARAMS["ysize"],
                    fontsize=HYPARAMS["fontsize"],
                    fontsizeTitle=HYPARAMS["fontsizeTitle"],
                    Nbins=HYPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = False,
                    savePathBase = savePathBase+ "CGM_only/",
                    savePathBaseFigureData = savePathBaseFigureData+ "CGM_only/",
                    saveFigureData = True,
                    SFR = False,
                    
                    forceYAxisLog = HYPARAMS["forceYAxisLog"],
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    
                )

                print(
                f"[@{int(snapNumber)}]: PDF of col dens gas, binned by {HYPARAMS['xParam']} plot"
                )


                apt.pdf_versus_plot(
                    copy.deepcopy(cgmcolout),
                    ylabel,
                    xlimDict,
                    HYPARAMS["logParameters"],
                    snapNumber,
                    weightKeys = ['mass'],
                    xParams = HYPARAMS["colParams"],
                    titleBool=HYPARAMS["titleBool"],
                    DPI=HYPARAMS["DPI"],
                    xsize=HYPARAMS["xsize"],
                    ysize=HYPARAMS["ysize"],
                    fontsize=HYPARAMS["fontsize"],
                    fontsizeTitle=HYPARAMS["fontsizeTitle"],
                    Nbins=HYPARAMS["Nbins"],
                    
                    
                    
                    ageWindow=None,
                    cumulative = False,
                    savePathBase = savePathBase+ "CGM_only/",
                    savePathBaseFigureData = savePathBaseFigureData+ "CGM_only/",
                    saveFigureData = True,
                    SFR = False,
                    
                    forceYAxisLog = HYPARAMS["forceYAxisLog"],
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    
                )

                print(
                f"[@{int(snapNumber)}]: Cumulative PDF of col dens gas plot"
                )

                apt.pdf_versus_plot(
                    copy.deepcopy(cgmcolout),
                    ylabel,
                    xlimDict,
                    HYPARAMS["logParameters"],
                    snapNumber,
                    weightKeys = ['mass'],
                    xParams = HYPARAMS["colParams"],
                    titleBool=HYPARAMS["titleBool"],
                    DPI=HYPARAMS["DPI"],
                    xsize=HYPARAMS["xsize"],
                    ysize=HYPARAMS["ysize"],
                    fontsize=HYPARAMS["fontsize"],
                    fontsizeTitle=HYPARAMS["fontsizeTitle"],
                    Nbins=HYPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = True,
                    savePathBase = savePathBase+ "CGM_only/",
                    savePathBaseFigureData = savePathBaseFigureData+ "CGM_only/",
                    saveFigureData = True,
                    SFR = False,
                    
                    forceYAxisLog = HYPARAMS["forceYAxisLog"],
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    
                )

                print(
                f"[@{int(snapNumber)}]: Normalised Cumulative PDF of col dens gas plot"
                )


                apt.pdf_versus_plot(
                    copy.deepcopy(cgmcolout),
                    ylabel,
                    xlimDict,
                    HYPARAMS["logParameters"],
                    snapNumber,
                    weightKeys = ['mass'],
                    xParams = HYPARAMS["colParams"],
                    titleBool=HYPARAMS["titleBool"],
                    DPI=HYPARAMS["DPI"],
                    xsize=HYPARAMS["xsize"],
                    ysize=HYPARAMS["ysize"],
                    fontsize=HYPARAMS["fontsize"],
                    fontsizeTitle=HYPARAMS["fontsizeTitle"],
                    Nbins=HYPARAMS["Nbins"],
                    ageWindow=None,
                    cumulative = True,
                    savePathBase = savePathBase+ "CGM_only/",
                    savePathBaseFigureData = savePathBaseFigureData+ "CGM_only/",
                    saveFigureData = True,
                    SFR = False,
                    
                    forceYAxisLog = HYPARAMS["forceYAxisLog"],
                    normalise = True,
                    verbose = DEBUG,
                    inplace = inplace,
                    
                )






            # -----------------------------------------------#
            #           
            #                   Medians
            #
            # -----------------------------------------------#

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})


            # print(
            #     "\n"+f"[@{int(snapNumber)}]: Calculate full statistics..."
            # )

            statsWeightkeys = ["mass"] + np.unique(np.asarray(list(HYPARAMS["nonMassWeightDict"].values()))).tolist()
            exclusions = [] 
            
            for param in HYPARAMS["saveEssentials"]:
                if param not in statsWeightkeys:
                    exclusions.append(param)

            statsDict = cr.cr_calculate_statistics(
                out,
                HYPARAMS = HYPARAMS,
                xParam=HYPARAMS["xParam"],
                Nbins=HYPARAMS["NStatsBins"],
                xlimDict=xlimDict,
                printpercent=2.5,
                exclusions = exclusions,
                weightedStatsBool = True,
            )

            # In order to label the plots correctly for the medians versus plots (and to allow for cross-compatibility
            # when plotting multiple variants of the same simulation, such as in the CR analysis) we need to nest the
            # statistics dictionaries in an outer dicitionary with some simulation descriptors, such as resolution and
            # Auriga halo number.

            ## Empty data checks ## 
            if bool(statsDict) is False:
                print("\n"
                    +f"[@{int(snapNumber)}]: WARNING! statsDict is empty! Skipping save ..."
                    +"\n"
                )
            else:
                tmp = copy.copy(statsDict)
                statsDict = {(baseResLevel, haloLabel): tmp}
                opslaanData = savePathBaseFigureData + f"Data_{int(snapNumber)}_" + "statsDict.h5"
                tr.hdf5_save(opslaanData,statsDict)

            tmpHYPARAMS = {((baseResLevel, haloLabel)): copy.copy(HYPARAMS)}

            print(
                "\n"+f"[@{int(snapNumber)}]: Calculate column density statistics..."
            )

            COLHYPARAMS = copy.deepcopy(HYPARAMS)
            COLHYPARAMS['saveParams']+=COLHYPARAMS['colParams']
            #COLHYPARAMS["NStatsBins"] = int(HYPARAMS["NStatsBins"]**(2/3))
            colstatsDict = cr.cr_calculate_statistics(
                colout,
                HYPARAMS = COLHYPARAMS,
                xParam=COLHYPARAMS["xParam"],
                Nbins=COLHYPARAMS["NStatsBins"],
                xlimDict=xlimDict,
                printpercent=2.5,
                exclusions = exclusions,
                weightedStatsBool = False,
            )
            ## Empty data checks ## 
            if bool(colstatsDict) is False:
                print("\n"
                    +f"[@{int(snapNumber)}]: WARNING! colstatsDict is empty! Skipping save ..."
                    +"\n"
                )
            else:
                tmp = copy.copy(colstatsDict)
                colstatsDict = {(baseResLevel, haloLabel): tmp}
                opslaanData = savePathBaseFigureData  + f"Data_{int(snapNumber)}_" + "colStatsDict.h5"
                tr.hdf5_save(opslaanData,colstatsDict)

            tmpCOLHYPARAMS = {((baseResLevel, haloLabel)): copy.copy(COLHYPARAMS)}
            


            print(
                "\n"+f"[@{int(snapNumber)}]: Plot column density medians versus {HYPARAMS['xParam']}..."
            )

            apt.medians_versus_plot(
                colstatsDict,
                tmpCOLHYPARAMS,
                ylabel=ylabel,
                xlimDict=xlimDict,
                snapNumber=snapNumber,
                yParam=COLHYPARAMS["colParams"],
                xParam=COLHYPARAMS["xParam"],
                titleBool=COLHYPARAMS["titleBool"],
                DPI = COLHYPARAMS["DPI"],
                xsize = COLHYPARAMS["xsize"],
                ysize = COLHYPARAMS["ysize"]*0.75,
                fontsize = COLHYPARAMS["fontsize"],
                fontsizeTitle = COLHYPARAMS["fontsizeTitle"],
                opacityPercentiles = COLHYPARAMS["opacityPercentiles"],
                colourmapMain = "tab10",
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                inplace = inplace,
                
            )

            print(
                "\n"+f"[@{int(snapNumber)}]: Plot full statistics medians versus {HYPARAMS['xParam']}..."
            )

            apt.medians_versus_plot(
                statsDict,
                tmpHYPARAMS,
                ylabel=ylabel,
                xlimDict=xlimDict,
                snapNumber=snapNumber,
                yParam=HYPARAMS["mediansParams"],
                xParam=HYPARAMS["xParam"],
                titleBool=HYPARAMS["titleBool"],
                DPI = HYPARAMS["DPI"],
                xsize = HYPARAMS["xsize"],
                ysize = HYPARAMS["ysize"],
                fontsize = HYPARAMS["fontsize"],
                fontsizeTitle = HYPARAMS["fontsizeTitle"],
                opacityPercentiles = HYPARAMS["opacityPercentiles"],
                colourmapMain = "tab10",
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                inplace = inplace,
                
            )


            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            apt.phase_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber = snapNumber,
                yParams = HYPARAMS["phasesyParams"],
                xParams = HYPARAMS["phasesxParams"],
                colourBarKeys = HYPARAMS["phasesColourbarParams"],
                weightKeys = HYPARAMS["nonMassWeightDict"],
                titleBool=HYPARAMS["titleBool"],
                DPI=HYPARAMS["DPI"],
                xsize=HYPARAMS["xsize"],
                ysize=HYPARAMS["ysize"],
                fontsize=HYPARAMS["fontsize"],
                fontsizeTitle=HYPARAMS["fontsizeTitle"],
                Nbins=HYPARAMS["Nbins"],
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                verbose = DEBUG,
                inplace = inplace,
                
            )
     


    print("Finished fully! :)")