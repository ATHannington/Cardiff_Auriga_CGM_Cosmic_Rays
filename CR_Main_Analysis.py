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
from CR_Plotting_Tools import *
import copy
import h5py
import json
import math
from random import sample
import multiprocessing as mp
import sys
import logging
import copy

CRPARAMSPATHMASTER = "CRParams.json"
CRPARAMSMASTER = json.load(open(CRPARAMSPATHMASTER, 'r'))
# =============================================================================#
#
#               USER DEFINED PARAMETERS
#
#==============================================================================#
# File types for data save.
#   Full: full FullDict data
FullDataPathSuffix = f".h5"

lazyLoadBool = True

# Number of cores to run on:
n_processes = 2

FullDataPathSuffix = f".h5"

CRSELECTEDHALOESPATH = "CRSelectedHaloes.json"
CRSELECTEDHALOES = json.load(open(CRSELECTEDHALOESPATH, 'r'))

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
    "P_CR": r"P$_{CR}$ (K cm$^{-3}$)",
    "PCR_Pthermal": r"(X$_{CR}$ = P$_{CR}$/P$_{Thermal}$)",
    "gah" : r"Alfven Gas Heating (erg s$^{-1}$)",
    "Grad_T" : r"||Temperature Gradient|| (K cm$^{-1}$)",
    "Grad_n_H" : r"||n$_H$ Gradient|| (cm$^{-4}$)",
    "Grad_bfld" : r"||B-Field Gradient|| ($ \mu $G cm$^{-1}$)",
    "Grad_P_CR" : r"||P$_{CR}$ Gradient|| (K cm$^{-4}$)",
    # "crac" : r"Alfven CR Cooling (erg s$^{-1}$)",
    "tcool": r"Cooling Time (Gyr)",
    "theat": r"Heating Time (Gyr)",
    "tcross": r"Sound Crossing Cell Time (Gyr)",
    "tff": r"Free Fall Time (Gyr)",
    "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
    "csound": r"Sound Speed (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "mass": r"Mass (M/M$_{\odot}$)",
}

xlimDict = {
    "R": {"xmin": 0.0, "xmax": CRPARAMSMASTER['Router']},
    # "mass": {"xmin": 5.0, "xmax": 9.0},
    "L": {"xmin": 3.0, "xmax": 4.5},
    "T": {"xmin": 3.75, "xmax": 6.5},
    "n_H": {"xmin": -5.5, "xmax": -0.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -150.0, "xmax": 150.0},
    "gz": {"xmin": -1.5, "xmax": 0.5},
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
    "ndens": {"xmin": -6.0, "xmax": 2.0}
}


for entry in CRPARAMSMASTER['logParameters']:
    ylabel[entry] = r"$Log_{10}$" + ylabel[entry]


#==============================================================================#
#
#          Main
#
#==============================================================================#

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
    for halo,allSimsDict in CRSELECTEDHALOES.items():
        dataDict = {}
        starsDict ={}
        CRPARAMSHALO = {}
        DataSavepathBase = CRPARAMSMASTER['savepath'] + f"{halo}/"
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        #   MAIN ANALYSIS
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        for sim, simDict in allSimsDict.items():
            CRPARAMS = cr_parameters(CRPARAMSMASTER, simDict)
            selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}")
            CRPARAMSHALO.update({selectKey : CRPARAMS})
            if CRPARAMS['simfile'] is not None:
        # #         # # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        # #         # # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        # #         #
        # #         # print("\n" + f"Starting MULTIPROCESSING type Analysis!")
        # #         # # Setup arguments combinations for parallel processing pool
        # #         # print("\n" + f"Sorting multi-core arguments!")
        # #         # manager = mp.Manager()
        # #         # args_list = manager.list()
        # #         # args_default =  [
        # #         #     CRPARAMS,
        # #         #     DataSavepathBase,
        # #         #     FullDataPathSuffix,
        # #         #     lazyLoadBool
        # #         #
        # #         # ]
        # #         #
        # #         # args_list = manager.list([[snapNumber] + args_default for snapNumber in snapRange])
        # #         #
        # #         # # Open multiprocesssing pool
        # #         #
        # #         # print("\n" + f"Opening {n_processes} core Pool!")
        # #         # pool = mp.Pool(processes=n_processes)
        # #         #
        # #         # # C ompute Snap analysis
        # #         # output_list = [
        # #         #     pool.apply_async(cr_analysis, args=args, error_callback=err_catcher)
        # #         #     for args in args_list
        # #         # ]
        # #         #
        # #         # pool.close()
        # #         # pool.join()
        # #         # # Close multiprocesssing pool
        # #         # print(f"Closing core Pool!")
        # #         # print(f"Error checks")
        # #         # success = [result.successful() for result in output_list]
        # #         # assert all(success) == True, "WARNING: CRITICAL: Child Process Returned Error!"
        # #         # print("No Errors!")
        # #         #
        # #         # print("Gather the multiprocess outputs")
        # #         # out = {}
        # #         # for output in output_list:
        # #         #
        # #         #     tmpOut = output.get()
        # #         #
        # #         #     # as function gives out dictionary extract what want (or just save dict)
        # #         #     out.update(tmpOut)
        # #         #
        # #         # del output_list, pool
        # #         #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

                print("\n" + f"Starting SERIAL type Analysis!")
                out = {}
                for snapNumber in snapRange:
                    tmpOut = cr_analysis_radial(
                        snapNumber,
                        CRPARAMS,
                        DataSavepathBase,
                        FullDataPathSuffix,
                        lazyLoadBool
                        )
                    out.update(tmpOut)

                del tmpOut
                #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

                flatDict = flatten_wrt_time(out, CRPARAMS, snapRange)

                del out
                for key, dict in flatDict.items():
                    flatDict[key]["dmdt"]
                for key, dict in flatDict.items():
                    if key[-1] == "Stars":
                        starsDict.update({key : dict})
                    else:
                        dataDict.update({key : dict})

        # #----------------------------------------------------------------------#
        # #      Calculate Radius xmin
        # #----------------------------------------------------------------------#

        # for sim, CRPARAMS in CRPARAMSHALO.items():
        #     if CRPARAMS['simfile'] is not None:
        #         print(f"{sim}")
        #         print("Calculate Radius xmin...")
        #         selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}")
        #
        #         dataDict[selectKey]['maxDiskRadius'] = np.nanmedian(dataDict[selectKey]['maxDiskRadius'])
        #
        #         selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}","Stars")
        # #         starsDict[selectKey]['maxDiskRadius'] = np.nanmedian(starsDict[selectKey]['maxDiskRadius'])
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
                selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}")

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
                    xlimDict["R"]['xmax'] =  tmpCRPARAMS['Router']

                dat = cr_calculate_statistics(
                    dataDict = dataDict[selectKey],
                    CRPARAMS = tmpCRPARAMS,
                    xParam = CRPARAMSMASTER["xParam"],
                    Nbins = CRPARAMSMASTER["NxParamBins"],
                    xlimDict = xlimDict
                )

                statsDict.update({selectKey: dat})

                print("Stars...")
                selectKey = (f"{CRPARAMS['resolution']}",f"{CRPARAMS['CR_indicator']}","Stars")

                dat = cr_calculate_statistics(
                    dataDict = starsDict[selectKey],
                    CRPARAMS = tmpCRPARAMS,
                    xParam = CRPARAMSMASTER["xParam"],
                    Nbins = CRPARAMSMASTER["NxParamBins"],
                    xlimDict = xlimDict
                )

                statsDictStars.update({selectKey: dat})
        #----------------------------------------------------------------------#
        #  Plots...
        #----------------------------------------------------------------------#

        #----------------------------------------------------------------------#
        #      medians_versus_plot...
        #----------------------------------------------------------------------#

        print("")
        print(f"Medians vs {xParam} Plot!")
        matplotlib.rc_file_defaults()
        plt.close("all")
        medians_versus_plot(
            statsDict = statsDict,
            CRPARAMSHALO =  CRPARAMSHALO,
            halo = halo,
            ylabel = ylabel,
            xParam = CRPARAMSMASTER["xParam"],
            xlimDict = xlimDict
        )
        matplotlib.rc_file_defaults()
        plt.close("all")

        print("")
        print(f"Mass PDF Plot!")
        matplotlib.rc_file_defaults()
        plt.close("all")
        mass_pdf_versus_by_radius_plot(
            dataDict = dataDict,
            CRPARAMSHALO = CRPARAMSHALO,
            halo = halo,
            ylabel = ylabel,
            xlimDict = xlimDict,
            snapRange = snapRange,
            densityBool = True
        )
        matplotlib.rc_file_defaults()
        plt.close("all")

        print("")
        print(f"Mass vs Plot Gas!")

        matplotlib.rc_file_defaults()
        plt.close("all")
        cumulative_mass_versus_plot(
            dataDict = dataDict,
            CRPARAMSHALO =  CRPARAMSHALO,
            halo = halo,
            ylabel = ylabel,
            xParam = CRPARAMSMASTER["xParam"],
            xlimDict = xlimDict,
        )
        matplotlib.rc_file_defaults()
        plt.close("all")

        print("")
        print(f"Mass vs Plot Stars!")
        matplotlib.rc_file_defaults()
        plt.close("all")
        cumulative_mass_versus_plot(
            dataDict = starsDict,
            CRPARAMSHALO =  CRPARAMSHALO,
            halo = halo,
            ylabel = ylabel,
            xParam = CRPARAMSMASTER["xParam"],
            xlimDict = xlimDict
        )
        matplotlib.rc_file_defaults()
        plt.close("all")
