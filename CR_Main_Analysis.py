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

CRParamsPath = "CRParams.json"
CRPARAMS = json.load(open(CRParamsPath, 'r'))

DataSavepathBase = CRPARAMS['savepath']

CRPARAMS['finalSnap'] = copy.copy(CRPARAMS['snapMax'])

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


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#   MAIN ANALYSIS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
    print("\n" + f"Starting MULTIPROCESSING type Analysis!")
    # Setup arguments combinations for parallel processing pool
    print("\n" + f"Sorting multi-core arguments!")

    args_default =  [
        CRPARAMS,
        DataSavepathBase,
        FullDataPathSuffix,
        lazyLoadBool
    ]

    args_list = []
    for resolution, pathsDict in CRPARAMS['simfiles'].items():
        print(f"{resolution}")
        for CR_indicator, loadpath in pathsDict.items():
            print(f"{CR_indicator}")
            if loadpath is not None :
                for snapNumber in snapRange:
                    args_list.append([snapNumber,resolution,CR_indicator,loadpath] + args_default)

    # Open multiprocesssing pool

    print("\n" + f"Opening {n_processes} core Pool!")
    pool = mp.Pool(processes=n_processes)

    # C ompute Snap analysis
    output_list = [
        pool.apply_async(cr_analysis, args=args, error_callback=err_catcher)
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

    out = {}
    for output in output_list:

        tmpOut = output.get()

        # as function gives out dictionary extract what want (or just save dict)
        out.update(tmpOut)
  ###-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

    # print("\n" + f"Starting SERIAL type Analysis!")
    # out = {}
    # for resolution, pathsDict in CRPARAMS['simfiles'].items():
    #     print(f"{resolution}")
    #     for CR_indicator, loadpath in pathsDict.items():
    #         print(f"{CR_indicator}")
    #         if loadpath is not None :
    #             for snapNumber in snapRange:
    #                 tmpOut = cr_analysis(
    #                     snapNumber,
    #                     resolution,
    #                     CR_indicator,
    #                     loadpath,
    #                     CRPARAMS,
    #                     DataSavepathBase,
    #                     FullDataPathSuffix,
    #                     lazyLoadBool
    #                     )
    #                 out.update(tmpOut)
    #         else:
    #             pass
    ##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


    print("Done! End of Analysis :)")
