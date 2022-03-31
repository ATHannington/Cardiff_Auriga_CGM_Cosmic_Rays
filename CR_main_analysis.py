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
import h5py
import multiprocessing as mp
import sys
import logging

# ==============================================================================#
#       USER DEFINED PARAMETERS
# ==============================================================================#
# File types for data save.
#   Full: full FullDict data
FullDataPathSuffix = f".h5"

# Lazy Load switch. Set to False to save all data (warning, pickle file may explode)
lazyLoadBool = True

# Number of cores to run on:
n_processes = 2
# ==============================================================================#
#       MAIN PROGRAM
# ==============================================================================#
def err_catcher(arg):
    raise Exception(f"Child Process died and gave error: {arg}")
    return


# if __name__ == "__main__":
#     TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC, _, _ = tracer_selection_snap_analysis(
#         TRACERSPARAMS,
#         HaloID,
#         elements,
#         elements_Z,
#         elements_mass,
#         elements_solar,
#         Zsolar,
#         omegabaryon0,
#         saveParams,
#         saveTracersOnly,
#         DataSavepath,
#         FullDataPathSuffix,
#         MiniDataPathSuffix,
#         lazyLoadBool,
#         SUBSET=None,
#     )
#
#     snapRange = [
#         zz
#         for zz in range(
#             int(TRACERSPARAMS["snapMin"]),
#             min(int(TRACERSPARAMS["finalSnap"]) + 1, int(TRACERSPARAMS["snapMax"]) + 1),
#             1,
#         )
#     ]
#
#     # Loop over snaps from snapMin to snapmax, taking the finalSnap (the final snap) as the endpoint if snapMax is greater
#
#     # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#     #   MAIN ANALYSIS
#     # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#     print("\n" + f"Starting MULTIPROCESSING type Analysis!")
#     # Setup arguments combinations for parallel processing pool
#     print("\n" + f"Sorting multi-core arguments!")
#
#     args_default = [
#         TRACERSPARAMS,
#         HaloID,
#         TracersTFC,
#         elements,
#         elements_Z,
#         elements_mass,
#         elements_solar,
#         Zsolar,
#         omegabaryon0,
#         saveParams,
#         saveTracersOnly,
#         DataSavepath,
#         FullDataPathSuffix,
#         MiniDataPathSuffix,
#         lazyLoadBool,
#     ]
#
#     args_list = [[snap] + args_default for snap in snapRange]
#
#     # Open multiprocesssing pool
#
#     print("\n" + f"Opening {n_processes} core Pool!")
#     pool = mp.Pool(processes=n_processes)
#
#     # Compute Snap analysis
#     output_list = [
#         pool.apply_async(snap_analysis, args=args, error_callback=err_catcher)
#         for args in args_list
#     ]
#
#     pool.close()
#     pool.join()
#     # Close multiprocesssing pool
#     print(f"Closing core Pool!")
#     print(f"Final Error checks")
#     success = [result.successful() for result in output_list]
#     assert all(success) == True, "WARNING: CRITICAL: Child Process Returned Error!"
#     print("Done! End of Analysis :)")
#     #
#     # print("\n" + f"Starting SERIAL type Analysis!")
#     # for snap in snapRange:
#     #     out = snap_analysis(snap,TRACERSPARAMS,HaloID,TracersTFC,\
#     #     elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
#     #     saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool)
#     # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
