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
import os
import math

# =============================================================================#
#
#               USER DEFINED PARAMETERS
#
# ==============================================================================#

plt.rcParams.update(matplotlib.rcParamsDefault)

# We want to utilise inplace operations to keep memory within RAM limits...
inplace = True
DEBUG = False
determineXlimits = False     #Intended for use when first configuring xlimits in xlimDict. Use this and "param" : {} in xlimDict for each param to explore axis limits needed for time averaging
allowPlotsWithoutxlimits = determineXlimits,

HYPARAMSPATH = "HYParams.json"
HYPARAMS = json.load(open(HYPARAMSPATH, "r"))

#if "mass" not in HYPARAMS["colParams"]:
#    HYPARAMS["colParams"]+=["mass"]

if HYPARAMS["ageWindow"] is not None:
    HYPARAMS["SFRBins"] = int(math.floor(HYPARAMS["ageWindow"]/HYPARAMS["windowBins"]))
else:
    HYPARAMS["SFRBins"]  = HYPARAMS["Nbins"] 

loadPathBase = "/home/cosmos/"
loadDirectories = [
    "spxfv/Auriga/level4_cgm/h5_standard",
    "c1838736/Auriga/level3_cgm_almost/h5_standard",
    "spxfv/Auriga/level4_cgm/h5_1kpc",
    "c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc",
    # # "/home/tango/""spxfv/surge/level4_cgm/h5_500pc",
    "spxfv/surge/level4_cgm/h5_500pc",
    "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc",
    "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition",
    "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition",
    # "c1838736/Auriga/level5_cgm/h5_standard",
    # "c1838736/Auriga/level5_cgm/h5_2kpc",
    # "c1838736/Auriga/level5_cgm/h5_1kpc",
    # "c1838736/Auriga/level5_cgm/h5_2kpc-hy-1kpc",
    # "c1838736/Auriga/level5_cgm/h5_1kpc-hy-500pc",
    # "c1838736/Auriga/level5_cgm/h5_hy-v2",
    

        
    

    ]

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
    "n_H": (HYPARAMS["colourmapMain"].split("_"))[0],
    "n_HI": (HYPARAMS["colourmapMain"].split("_"))[0],
    "n_H_col": (HYPARAMS["colourmapMain"].split("_"))[0],
    "n_HI_col": (HYPARAMS["colourmapMain"].split("_"))[0],
}

xlimDict = {
    "R": {"xmin": HYPARAMS["Rinner"], "xmax": HYPARAMS["Router"]},
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
    "vol": {"xmin": -2.0, "xmax": 0.5},
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

    xlimDict.update({"xmin": HYPARAMS["Rinner"]})
    xlimDict.update({"xmax": HYPARAMS["Router"]})

    for (loadpath,savePathBase,savePathBaseFigureData) in zip(simulations,savePaths,savePathsData):
        print(loadpath)
        # we need to nest the
        # statistics dictionaries in an outer dicitionary with some simulation descriptors, such as resolution and
        # Auriga halo number.
        splitList = loadpath.split("/")
        baseResLevel, haloLabel = splitList[-4:-2]
        tmp = haloLabel.split("_")
        haloSplitList = []
        for xx in tmp:
            splitxx = xx.split("-")
            haloSplitList += splitxx
        haloLabelKeySaveable = "_".join(haloSplitList)
        auHalo, resLabel = haloSplitList[0], "_".join(haloSplitList[1:])


        tmp = ""
        for savePathChunk in savePathBaseFigureData.split("/")[:-1]:
            tmp += savePathChunk + "/"
            try:
                os.mkdir(tmp)
            except:
                pass
            else:
                pass


        tmp = ""
        for savePathChunk in savePathBase.split("/")[:-1]:
            tmp += savePathChunk + "/"
            try:
                os.mkdir(tmp)
            except:
                pass
            else:
                pass

        if (HYPARAMS["loadRotationMatrix"] == True) & (HYPARAMS["constantRotationMatrix"] == True):
            rotationloadpath = savePathBaseFigureData + f"rotation_matrix_{int(snapNumber)}.h5"
            tmp = tr.hdf5_load(rotationloadpath)
            rotation_matrix = tmp[(baseResLevel, haloLabelKeySaveable)]["rotation_matrix"]
            print(
                "\n" + f"Loaded rotation_matrxix : "+
                "\n" + f"{(baseResLevel, haloLabelKeySaveable)} : 'rotation_matrix': ..."+
                "\n" + f"from {rotationloadpath}"
            )      
        else:
            rotation_matrix = None

        for snapNumber in snapRange:
            # rotation_matrix = None
            # snapNumber = 100
            # loadPathBase = "/home/cosmos/c1838736/Auriga/level5_cgm/"
            # simulation = "h5_2kpc"
            # loadpath = loadPathBase+simulation+"/output/"
            print(f"[@{int(snapNumber)}]: Load subfind")
            # load in the subfind group files
            snap_subfind = load_subfind(snapNumber, dir=loadpath)

            print(f"[@{int(snapNumber)}]: Load snapshot")
            snap = gadget_readsnap(
                snapNumber,
                loadpath,
                hdf5=True,
                loadonlytype=[0,1,4],#[0, 1, 2, 3, 4, 5],
                lazy_load=True,
                subfind=snap_subfind,
                #loadonlyhalo=int(HYPARAMS["HaloID"]),
            )


            print(f"[@{int(snapNumber)}]: Rotate and centre snapshot")
            snap.calc_sf_indizes(snap_subfind)
            if rotation_matrix is None:
                print(f"[@{int(snapNumber)}]: New rotation of snapshots")
                rotation_matrix = snap.select_halo(snap_subfind, do_rotation=True)
                rotationsavepath = savePathBaseFigureData + f"rotation_matrix_{int(snapNumber)}.h5"
                tr.hdf5_save(rotationsavepath,{(baseResLevel, haloLabelKeySaveable) : {"rotation_matrix": rotation_matrix}})
                print(
                    "\n" + f"[@{int(snapNumber)}]: Saved rotation_matrxix as"+
                    "\n" + f"{(baseResLevel, haloLabelKeySaveable)} : 'rotation_matrix': ..."+
                    "\n" + f"at {rotationsavepath}"
                )
                ## If we don't want to use the same rotation matrix for all snapshots, set rotation_matrix back to None
                if (HYPARAMS["constantRotationMatrix"] == False):
                    rotation_matrix = None
            else:
                print(f"[@{int(snapNumber)}]: Existing rotation of snapshots")
                snap.select_halo(snap_subfind, do_rotation=False)
                snap.rotateto(
                    rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
                )


            print(
                f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}"
            )
            
            print(
                f"[@{int(snapNumber)}]: Clean SnapShot parameters..."
            )

            snap = cr.clean_snap_params(
                snap,
                paramsOfInterest = HYPARAMS["saveParams"] + HYPARAMS["saveEssentials"]
            )

            # --------------------------#
            ##    Units Conversion    ##
            # --------------------------#

            # Convert Units
            ## Make this a seperate function at some point??
            snap.pos *= 1e3  # [kpc]
            snap.vol *= 1e9  # [kpc^3]
            snap.mass *= 1e10  # [Msol]
            snap.hrgm *= 1e10  # [Msol]
            snap.gima *= 1e10  # [Msol]

            snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)
            rvir = (snap_subfind.data["frc2"] * 1e3)[int(0)]
            boxmax = max([HYPARAMS['boxsize'],HYPARAMS['boxlos'],HYPARAMS['coldenslos']])

            print(
                f"[@{int(snapNumber)}]: Remove beyond {boxmax:2.2f} kpc..."
            )

            whereOutsideBox = np.abs(snap.data["pos"]) > boxmax

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereOutsideBox,
                errorString = "Remove Outside Box",
                verbose = DEBUG,
                )

            print(
                f"[@{int(snapNumber)}]: Select stars..."
            )

            whereWind = snap.data["age"] < 0.0

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereWind,
                errorString = "Remove Wind from Gas",
                verbose = DEBUG,
                )

            box = [boxmax, boxmax, boxmax]

            # Calculate New Parameters and Load into memory others we want to track
            snap = tr.calculate_tracked_parameters(
                snap,
                oc.elements,
                oc.elements_Z,
                oc.elements_mass,
                oc.elements_solar,
                oc.Zsolar,
                oc.omegabaryon0,
                snapNumber,
                logParameters=HYPARAMS["logParameters"],
                paramsOfInterest=HYPARAMS["saveParams"],
                mappingBool=True,
                box=box,
                numthreads=HYPARAMS['numthreads'],
                DataSavepath=savePathBaseFigureData,
                verbose = DEBUG,
            )

            snap.data["R"] = snap.data["R"]/rvir

            print(
                f"[@{int(snapNumber)}]: Ages: get_lookback_time_from_a() ..."
            )

            ages = snap.cosmology_get_lookback_time_from_a(snap.data["age"],is_flat=True)

            snap.data["age"] = ages

            
            # -----------------------------------------------#
            #           
            #              column density images
            #
            # -----------------------------------------------#

            savePathFigureData = savePathBaseFigureData + "Plots/Slices/"

            if snapNumber is not None:
                if type(snapNumber) == int:
                    SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                else:
                    SaveSnapNumber = "_" + str(snapNumber)
            else:
                SaveSnapNumber = ""

            # Axes Labels to allow for adaptive axis selection
            AxesLabels = ["z","x","y"]
            colout = {}
            if len(HYPARAMS['colParams'])>0:

                # Create variant of xlimDict specifically for images of col params
                tmpxlimDict = copy.deepcopy(xlimDict)

                # Add the col param specific limits to the xlimDict variant
                for key, value in colImagexlimDict.items():
                    tmpxlimDict[key] = value

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
                    & (additionalParam is not None) & (additionalParam != "count") & (additionalParam != "count"):
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

                cols = HYPARAMS["colParams"]+additionalColParams

                for param in cols:
                    if HYPARAMS["restartFlag"] == True:

                        paramSplitList = param.split("_")

                        if paramSplitList[-1] == "col":
                            ## If _col variant is called we want to calculate a projection of the non-col parameter
                            ## Thus, we force projection to now be true, and incorporate a dummy variable tmpsliceParam
                            ## to force plots to generate non-col variants but save output as column density version

                            tmpsliceParam = "_".join(paramSplitList[:-1])
                            projection = True
                        else:
                            tmpsliceParam = param
                            projection = False

                        if HYPARAMS["averageAcrossAxes"] is True:

                            if projection is False:
                                loadPathFigureData = savePathFigureData + f"Slice_Plot_AxAv_{param}{SaveSnapNumber}"
                            else:
                                loadPathFigureData = savePathFigureData + f"Projection_Plot_AxAv_{param}{SaveSnapNumber}" 

                        else:
                            if projection is False:
                                loadPathFigureData = savePathFigureData + f"Slice_Plot_{AxesLabels[HYPARAMS['Axes'][0]]}-{AxesLabels[HYPARAMS['Axes'][1]]}_{param}{SaveSnapNumber}"
                            else:
                                loadPathFigureData = savePathFigureData + f"Projection_Plot_{AxesLabels[HYPARAMS['Axes'][0]]}-{AxesLabels[HYPARAMS['Axes'][1]]}_{param}{SaveSnapNumber}" 

                        print("\n"+f"[@{int(snapNumber)}]: Loading {loadPathFigureData}")

                        try:
                            tmpdict = apt.hy_load_individual_slice_plot_data(
                                HYPARAMS,
                                snapNumber,
                                sliceParam = param,
                                Axes = HYPARAMS["Axes"],
                                averageAcrossAxes = HYPARAMS["averageAcrossAxes"],
                                projection=False,
                                loadPathBase = savePathBaseFigureData,
                                loadPathSuffix = "",
                                selectKeyLen=1,
                                delimiter="-",
                                stack = None,
                                allowFindOtherAxesData = False,
                                verbose = True,#DEBUG
                                hush = False
                            )
                            colout.update(tmpdict)
                        except Exception as e:
                            print(str(e))
                            print(
                                "\n"+f"[@{int(snapNumber)}]: File not found! Re-plotting {param} map..."
                            )
                            
                            # By default, we set projection here to False. This ensures any weighting maps are
                            # slices (projection versions were found to produce unphysical and unreliable results).
                            # However, any _col parameters are forced into Projection=True inside apt.plot_slices().
                            tmpdict = apt.plot_slices(snap,
                                ylabel=ylabel,
                                xlimDict=tmpxlimDict,
                                logParameters = HYPARAMS["logParameters"],
                                snapNumber=snapNumber,
                                sliceParam = param,
                                Axes=HYPARAMS["Axes"],
                                averageAcrossAxes = HYPARAMS["averageAcrossAxes"],
                                saveAllAxesImages = HYPARAMS["saveAllAxesImages"],
                                xsize = HYPARAMS["xsizeImages"],
                                ysize = HYPARAMS["ysizeImages"],
                                colourmapMain=HYPARAMS["colourmapMain"],
                                colourmapsUnique = imageCmapDict,
                                boxsize=HYPARAMS["boxsize"],
                                boxlos=HYPARAMS["coldenslos"],
                                pixreslos=HYPARAMS["pixreslos"],
                                pixres=HYPARAMS["pixres"],
                                projection = projection,
                                DPI = HYPARAMS["DPIimages"],
                                numthreads=HYPARAMS["numthreads"],
                                savePathBase = savePathBase,
                                savePathBaseFigureData = savePathBaseFigureData,
                                saveFigureData = True,
                                saveFigure = True,
                                inplace = inplace,
                                replotFromData = True,
                            )


                        colout.update({param: (copy.deepcopy(tmpdict[param]["grid"])).reshape(-1)})
                        
                        # !!
                        # You !! MUST !! provide type data for data that is no longer snapshot associated and thus
                        # no longer has type data associated with it. This is to ensure any future selections made from
                        # the dataset do not break the type length associated logic which is engrained in all of these
                        # tools, primarily via the ' cr.remove_selection() ' function.
                        # 
                        # You may choose if you wish to discard/mask
                        # a subset of your data from future figures by setting it to a non-zero integer value for type
                        # but beware, this is an untested use-case and (especially for pre-existing types between 0-6)
                        # the tools provided here may exhibit unexpected behaviours!
                        # !!
                        newShape = np.shape(colout[param])
                        colout.update({"type": np.full(shape=newShape, fill_value=0)})

                        if (HYPARAMS["xParam"] == "R") & (HYPARAMS["xParam"] not in list(colout.keys())):
                            xx = (copy.deepcopy(tmpdict[param]["x"])).reshape(-1)
                            xx = np.array(
                                [
                                    (x1 + x2) / 2.0
                                    for (x1, x2) in zip(xx[:-1], xx[1:])
                                ]
                            )
                            yy = (copy.deepcopy(tmpdict[param]["y"])).reshape(-1)
                            yy = np.array(
                                [
                                    (x1 + x2) / 2.0
                                    for (x1, x2) in zip(yy[:-1], yy[1:])
                                ]
                            )
                            values = np.linalg.norm(np.asarray(np.meshgrid(xx,yy)), axis=0).reshape(-1)
                            colout.update({"R": copy.deepcopy(values/rvir)})
                    else:
                        print(
                            "\n"+f"[@{int(snapNumber)}]: Calculate {param} map..."
                        )

                        # By default, we set projection here to False. This ensures any weighting maps are
                        # slices (projection versions were found to produce unphysical and unreliable results).
                        # However, any _col parameters are forced into Projection=True inside apt.plot_slices().
                        tmpdict = apt.plot_slices(snap,
                            ylabel=ylabel,
                            xlimDict=tmpxlimDict,
                            logParameters = HYPARAMS["logParameters"],
                            snapNumber=snapNumber,
                            sliceParam = param,
                            Axes=HYPARAMS["Axes"],
                            averageAcrossAxes = HYPARAMS["averageAcrossAxes"],
                            saveAllAxesImages = HYPARAMS["saveAllAxesImages"],
                            xsize = HYPARAMS["xsizeImages"],
                            ysize = HYPARAMS["ysizeImages"],
                            colourmapMain=HYPARAMS["colourmapMain"],
                            colourmapsUnique = imageCmapDict,
                            boxsize=HYPARAMS["boxsize"],
                            boxlos=HYPARAMS["coldenslos"],
                            pixreslos=HYPARAMS["pixreslos"],
                            pixres=HYPARAMS["pixres"],
                            projection = projection,
                            DPI = HYPARAMS["DPIimages"],
                            numthreads=HYPARAMS["numthreads"],
                            savePathBase = savePathBase,
                            savePathBaseFigureData = savePathBaseFigureData,
                            saveFigureData = True,
                            saveFigure = True,
                            inplace = inplace,
                            replotFromData = True,
                        )

                        colout.update({param: (copy.deepcopy(tmpdict[param]["grid"])).reshape(-1)})
                        
                        # !!
                        # You !! MUST !! provide type data for data that is no longer snapshot associated and thus
                        # no longer has type data associated with it. This is to ensure any future selections made from
                        # the dataset do not break the type length associated logic which is engrained in all of these
                        # tools, primarily via the ' cr.remove_selection() ' function.
                        # 
                        # You may choose if you wish to discard/mask
                        # a subset of your data from future figures by setting it to a non-zero integer value for type
                        # but beware, this is an untested use-case and (especially for pre-existing types between 0-6)
                        # the tools provided here may exhibit unexpected behaviours!
                        # !!
                        newShape = np.shape(colout[param])
                        colout.update({"type": np.full(shape=newShape, fill_value=0)})

                        if (HYPARAMS["xParam"] == "R") & (HYPARAMS["xParam"] not in list(colout.keys())):
                            xx = (copy.deepcopy(tmpdict[param]["x"])).reshape(-1)
                            xx = np.array(
                                [
                                    (x1 + x2) / 2.0
                                    for (x1, x2) in zip(xx[:-1], xx[1:])
                                ]
                            )
                            yy = (copy.deepcopy(tmpdict[param]["y"])).reshape(-1)
                            yy = np.array(
                                [
                                    (x1 + x2) / 2.0
                                    for (x1, x2) in zip(yy[:-1], yy[1:])
                                ]
                            )
                            values = np.linalg.norm(np.asarray(np.meshgrid(xx,yy)), axis=0).reshape(-1)
                            colout.update({"R": copy.deepcopy(values/rvir)})



            # -----------------------------------------------#
            #           
            #              images
            #
            # -----------------------------------------------#

            if HYPARAMS["restartFlag"] == False:

                for param in HYPARAMS["imageParams"]+["Tdens", "rho_rhomean"]:
                    try:
                        tmp = snap.data[param]
                    except:
                        snap = tr.calculate_tracked_parameters(
                            snap,
                            oc.elements,
                            oc.elements_Z,
                            oc.elements_mass,
                            oc.elements_solar,
                            oc.Zsolar,
                            oc.omegabaryon0,
                            snapNumber,
                            logParameters = HYPARAMS['logParameters'],
                            paramsOfInterest=[param],
                            mappingBool=True,
                            box=box,
                            numthreads=HYPARAMS['numthreads'],
                            DataSavepath=savePathBaseFigureData,
                            verbose = DEBUG,
                        )

                for param in HYPARAMS["imageParams"]:      
                    _ = apt.plot_slices(
                        snap,
                        ylabel=ylabel,
                        xlimDict=tmpxlimDict,
                        logParameters = HYPARAMS["logParameters"],
                        snapNumber=snapNumber,
                        sliceParam = param,
                        Axes=HYPARAMS["Axes"],
                        xsize = HYPARAMS["xsizeImages"],
                        ysize = HYPARAMS["ysizeImages"],
                        colourmapMain = HYPARAMS["colourmapMain"],
                        colourmapsUnique = imageCmapDict,
                        boxsize=HYPARAMS["boxsize"],
                        boxlos=HYPARAMS["boxlos"],
                        pixreslos=HYPARAMS["pixreslos"],
                        pixres=HYPARAMS["pixres"],
                        projection = HYPARAMS["projections"],
                        DPI = HYPARAMS["DPIimages"],
                        numthreads=HYPARAMS["numthreads"],
                        savePathBase = savePathBase ,
                        savePathBaseFigureData = savePathBaseFigureData ,
                        saveFigureData = True,
                        saveFigure = True,
                        inplace = inplace,
                    )
            else:
                print(
                    "\n"
                    +f"[@{int(snapNumber)}]: RestartFlag is True! Skipping image/slice plots not needed for quantative analysis..."
                    +"\n"
                )
                pass


            whereOthers = np.isin(snap.data["type"],np.array([1,2,3,5,6]))

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereOthers,
                errorString = "Remove all types other than Gas and Stars",
                verbose = DEBUG
                )

            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(snap.data["subhalo"],np.array([-1,int(HYPARAMS["HaloID"]),np.nan]))==False

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                verbose = DEBUG,
                )

            ## Check that radii are still being stored in units of Rvir...
            if np.all(snap.data["R"][np.where(np.linalg.norm(snap.data["pos"],axis=1)<=HYPARAMS["Router"]*rvir)[0]]<=HYPARAMS["Router"]): 
                pass
            else:
                ## if radii are not in units of rvir, set that now...
                snap.data["R"] = snap.data["R"]/rvir

            print(
                f"[@{int(snapNumber)}]: Remove beyond {HYPARAMS['Router']:2.2f} x Rvir..."
            )

            whereBeyondVirial = snap.data["R"] > float(HYPARAMS['Router'])

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereBeyondVirial,
                errorString = f"Remove Beyond {HYPARAMS['Router']:2.2f} x Rvir",
                verbose = DEBUG,
                )




            # -----------------------------------------------#
            #           
            #                     gas PDFs
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

            print(
               f"[@{int(snapNumber)}]: PDF of gas plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = HYPARAMS['nonMassWeightDict'], #<<<< Need to rerun these with vol weights
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
                forceLogPDF = HYPARAMS["forceLogPDF"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                allowPlotsWithoutxlimits = determineXlimits,
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
            # # # #    weightKeys = HYPARAMS['nonMassWeightDict'], #<<<< Need to rerun these with vol weights
            # # # #    xParams = ["R"],
            # # # #    savePathBase = savePathBase,
            # # # #    savePathBaseFigureData = savePathBaseFigureData,
            # # # #    saveFigureData = True,
            # # # #    
            # # # #    forceLogPDF = HYPARAMS["forceLogPDF"],
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
            # # # #    weightKeys = HYPARAMS['nonMassWeightDict'], #<<<< Need to rerun these with vol weights
            # # # #    xParams = ["R"],
            # # # #    cumulative = True,
            # # # #    savePathBase = savePathBase,
            # # # #    savePathBaseFigureData = savePathBaseFigureData,
            # # # #    saveFigureData = True,
            # # # #    
            # # # #    forceLogPDF = HYPARAMS["forceLogPDF"],
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
            # # # #    weightKeys = HYPARAMS['nonMassWeightDict'], #<<<< Need to rerun these with vol weights
            # # # #    xParams = ["R"],
            # # # #    cumulative = True,
            # # # #    normalise = True,
            # # # #    savePathBase = savePathBase,
            # # # #    savePathBaseFigureData = savePathBaseFigureData,
            # # # #    saveFigureData = True,
            # # # #    
            # # # #    forceLogPDF = HYPARAMS["forceLogPDF"],
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
                forceLogPDF = HYPARAMS["forceLogPDF"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                allowPlotsWithoutxlimits = determineXlimits,
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

            whereInnerRadius = out["R"]*rvir<=30.0

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
                weightKeys = HYPARAMS['nonMassWeightDict'], #<<<< Need to rerun these with vol weights
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
                forceLogPDF = HYPARAMS["forceLogPDF"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                allowPlotsWithoutxlimits = determineXlimits,
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
                weightKeys = HYPARAMS['nonMassWeightDict'], #<<<< Need to rerun these with vol weights
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
                
                forceLogPDF = HYPARAMS["forceLogPDF"],
                normalise = False,
                verbose = DEBUG,
                inplace = inplace,
                allowPlotsWithoutxlimits = determineXlimits,
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

                whereInnerRadius = colout["R"]*rvir<=30.0
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
                    weightKeys = HYPARAMS['nonMassWeightDict'], #<<<< Need to rerun these with vol weights
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
                    
                    forceLogPDF = HYPARAMS["forceLogPDF"],
                    normalise = False,
                    verbose = DEBUG,
                    inplace = inplace,
                    allowPlotsWithoutxlimits = determineXlimits,
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

            tmp = np.asarray(list(HYPARAMS["nonMassWeightDict"].values()))
            whereNone = np.where(tmp==None)[0]
            whereNOTNone = np.where(tmp!=None)[0]

            statsWeightkeys = ["mass"] + np.unique(tmp[whereNOTNone]).tolist()
            exclusions = [] 
            
            for param in HYPARAMS["saveEssentials"]:
                if param not in statsWeightkeys:
                    exclusions.append(param)


            statsDict = cr.cr_calculate_statistics(
                out,
                CRPARAMS = HYPARAMS,
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
                CRPARAMS = COLHYPARAMS,
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
                ysize = COLHYPARAMS["ysize"],
                fontsize = COLHYPARAMS["fontsize"],
                fontsizeTitle = COLHYPARAMS["fontsizeTitle"],
                opacityPercentiles = COLHYPARAMS["opacityPercentiles"],
                colourmapMain = "tab10",
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                inplace = inplace,
                allowPlotsWithoutxlimits = determineXlimits,
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
                allowPlotsWithoutxlimits = determineXlimits,
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
                colourmapMain= HYPARAMS["colourmapMain"],
                Nbins=HYPARAMS["Nbins"],
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData,
                saveFigureData = True,
                verbose = DEBUG,
                inplace = inplace,
                allowPlotsWithoutxlimits = determineXlimits,
            )

            # # # apt.phase_plot(
            # # #     out,
            # # #     ylabel,
            # # #     xlimDict,
            # # #     HYPARAMS["logParameters"],
            # # #     snapNumber = "subfig",
            # # #     yParams = [["T","ndens"],["T"]],
            # # #     xParams = [["R","R"],["ndens"]],
            # # #     colourBarKeys = [["mass","mass"],["mass"]],
            # # #     weightKeys = HYPARAMS["nonMassWeightDict"],
            # # #     titleBool=HYPARAMS["titleBool"],
            # # #     DPI=HYPARAMS["DPI"],
            # # #     xsize=HYPARAMS["xsize"]*2.0,
            # # #     ysize=HYPARAMS["ysize"]*2.0,
            # # #     fontsize=HYPARAMS["fontsize"],
            # # #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
            # # #     Nbins=HYPARAMS["Nbins"],
            # # #     savePathBase = savePathBase,
            # # #     savePathBaseFigureData = savePathBaseFigureData,
            # # #     saveFigureData = True,
            # # #     subfigures = True,
            # # #     verbose = DEBUG,
            # # #     inplace = inplace,
            # # #     allowPlotsWithoutxlimits = determineXlimits,
            # # # )
            
            print(
                f"[@{int(snapNumber)}]: Done"
            )
            plt.close("all")
        print("finished sim:", loadpath)
    print("Finished fully! :)")
