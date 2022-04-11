"""
Author: A. T. Hannington
Created: 31/03/2022
Known Bugs:
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
import OtherConstants as oc
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
from CR_Subroutines import *
import copy
import h5py
import json
import multiprocessing as mp
import sys
import logging

CRPARAMSPATH = "CRParams.json"
CRPARAMS = load_cr_parameters(CRPARAMSPATH)
DataSavepathBase = CRPARAMS['savepath']


# ==============================================================================#
#       USER DEFINED PARAMETERS
# ==============================================================================#
# File types for data save.
#   Full: full FullDict data
FullDataPathSuffix = f".h5"

lazyLoadBool = True

# Number of cores to run on:
n_processes = 4
# ==============================================================================#
#       MAIN PROGRAM
# ==============================================================================#
def err_catcher(arg):
    raise Exception(f"Child Process died and gave error: {arg}")
    return


if __name__ == "__main__":

    snapRange = [
        xx
        for xx in range(
            int(CRPARAMS["snapMin"]),
            min(int(CRPARAMS["snapMax"]) + 1, int(CRPARAMS["finalSnap"]) + 1),
            1,
        )
    ]


# # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# #   MAIN ANALYSIS
# # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
    print("\n" + f"Starting MULTIPROCESSING type Analysis!")
    # Setup arguments combinations for parallel processing pool
    print("\n" + f"Sorting multi-core arguments!")

    args_default =  [
        CRPARAMS,
        DataSavepathBase,
        FullDataPathSuffix,
        lazyLoadBool
    ]

    args_list = [[snapNumber] + args_default for snapNumber in snapRange]

    # Open multiprocesssing pool

    print("\n" + f"Opening {n_processes} core Pool!")
    pool = mp.Pool(processes=n_processes)

    # C ompute Snap analysis
    output_list = [
        pool.apply_async(cr_cgm_analysis, args=args, error_callback=err_catcher)
        for args in args_list
    ]

    pool.close()
    pool.join()
    # Close multiprocesssing pool
    print(f"Closing core Pool!")
    print(f"Error checks")
    success = [result.successful() for result in output_list]
    assert all(success) == True, "WARNING: CRITICAL: Child Process Returned Error!"

    print("No Errors!")

    print("Gather the multiprocess outputs")
    out = {}
    for output in output_list:

        tmpOut = output.get()

        # as function gives out dictionary extract what want (or just save dict)
        out.update(tmpOut)

    del output_list, pool

    dataDict = flatten_wrt_time(out, CRPARAMS, snapRange)

    del out

    savePath = DataSavepathBase + f"Data_CR__{CRPARAMS['sim']['resolution']}_{CRPARAMS['sim']['CR_indicator']}_CGM" + FullDataPathSuffix
    print("Saving data as ", savePath)
    hdf5_save(savePath, dataDict)


  ###-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
    #
    # print("\n" + f"Starting SERIAL type Analysis!")
    # out = {}
    # for snapNumber in snapRange:
    #     tmpOut = cr_cgm_analysis(
    #         snapNumber,
    #         CRPARAMS,
    #         DataSavepathBase,
    #         FullDataPathSuffix,
    #         lazyLoadBool
    #         )
    #     out.update(tmpOut)
    #
    # del tmpOut
    #
    # dataDict = flatten_wrt_time(out, CRPARAMS, snapRange)
    #
    # del out
    #
    # savePath = DataSavepathBase + f"Data_CR__{CRPARAMS['sim']['resolution']}_{CRPARAMS['sim']['CR_indicator']}_CGM" + FullDataPathSuffix
    # print("Saving data as ", savePath)
    # hdf5_save(savePath, dataDict)

    ##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


    print("Done! End of Analysis :)")
