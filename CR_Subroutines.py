"""
Author: A. T. Hannington
Created: 31/03/2022
"""

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
from Tracers_Subroutines import *
import h5py

def cr_analysis(
    snapRange,
    resolution,
    CR_indicator,
    loadpath,
    CRPARAMS,
    DataSavepathBase,
    FullDataPathSuffix=".h5",
    lazyLoadBool = True,
):
    out = {}
    snapList = []
    DataSavepath = DataSavepathBase + f"Data_{resolution}_{CR_indicator}"
    for snapNumber in snapRange:

        print("")
        print(f"[@{resolution}, @{CR_indicator}, @{int(snapNumber)}]: Starting Snap {snapNumber}")

        # load in the subfind group files
        snap_subfind = load_subfind(snapNumber, dir=loadpath)

        # load in the gas particles mass and position only for HaloID 0.
        #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
        #       gas and stars (type 0 and 4) MUST be loaded first!!
        snapGas = gadget_readsnap(
            snapNumber,
            loadpath,
            hdf5=True,
            loadonlytype=[0, 4, 1],
            lazy_load=lazyLoadBool,
            subfind=snap_subfind,
        )

        # Load Cell IDs - avoids having to turn lazy_load off...
        # But ensures 'id' is loaded into memory before halo_only_gas_select is called
        #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
        #   Be in memory so taking the subset would be skipped.

        tmp = snapGas.data["id"]
        tmp = snapGas.data["age"]
        tmp = snapGas.data["hrgm"]
        tmp = snapGas.data["mass"]
        tmp = snapGas.data["pos"]
        tmp = snapGas.data["vol"]
        del tmp

        print(
            f"[@{resolution}, @{CR_indicator}, @{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
        )

        # Centre the simulation on HaloID 0
        # snapGas = set_centre(
        #     snap=snapGas, snap_subfind=snap_subfind, HaloID=CRPARAMS['HaloID'], snapNumber=snapNumber
        # )

        snapGas.calc_sf_indizes(snap_subfind, halolist=[CRPARAMS['HaloID']])
        snapGas.select_halo(snap_subfind, do_rotation=True)
        # --------------------------#
        ##    Units Conversion    ##
        # --------------------------#

        # Convert Units
        ## Make this a seperate function at some point??
        snapGas.pos *= 1e3  # [kpc]
        snapGas.vol *= 1e9  # [kpc^3]
        snapGas.mass *= 1e10  # [Msol]
        snapGas.hrgm *= 1e10  # [Msol]

        # Calculate New Parameters and Load into memory others we want to track
        snapGas = calculate_tracked_parameters(
            snapGas,
            oc.elements,
            oc.elements_Z,
            oc.elements_mass,
            oc.elements_solar,
            oc.Zsolar,
            oc.omegabaryon0,
            snapNumber,
        )

        # Pad stars and gas data with Nones so that all keys have values of same first dimension shape
        snapGas = pad_non_entries(snapGas, snapNumber)

        # Select only gas in High Res Zoom Region
        snapGas = high_res_only_gas_select(snapGas, snapNumber)

        # Find Halo=HaloID data for only selection snapshot.

        # Assign SubHaloID and FoFHaloIDs
        snapGas = halo_id_finder(snapGas, snap_subfind, snapNumber)

        # if snapNumber == int(CRPARAMS["selectSnap"]):
        snapGas = halo_only_gas_select(snapGas, snap_subfind, CRPARAMS['HaloID'], snapNumber)

        # Pad stars and gas data with Nones so that all keys have values of same first dimension shape
        snapGas = pad_non_entries(snapGas, snapNumber)

        # Redshift
        redshift = snapGas.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # Get lookback time in Gyrs
        # [0] to remove from numpy array for purposes of plot title
        lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
            0
        ]  # [Gyrs]

        snapGas.data["Lookback"] = np.array([lookback])
        snapGas.data["Snap"] = np.array([snapNumber])

        snapList.append(snapGas)


    snapGas = combine_snapshots(snapList, flatConcatBool = True)

    if (
        (CRPARAMS["QuadPlotBool"] == True)
        # & (targetT == int(CRPARAMS["targetTLst"][0]))
        # & (rin == CRPARAMS["Rinner"][0])
    ):
        plot_projections(
            snapGas,
            snapNumber=0,
            targetT=None,
            rin=None,
            rout=None,
            TRACERSPARAMS=CRPARAMS,
            DataSavepath=DataSavepath,
            FullDataPathSuffix=None,
            titleBool = False,
            Axes=CRPARAMS["Axes"],
            zAxis=CRPARAMS["zAxis"],
            boxsize=CRPARAMS["boxsize"],
            boxlos=CRPARAMS["boxlos"],
            pixres=CRPARAMS["pixres"],
            pixreslos=CRPARAMS["pixreslos"],
        )

    # Trim snapshot...
    keys = list(snapGas.data.keys())
    for key in keys:
        if key not in CRPARAMS['saveParams']+CRPARAMS['saveEssentials']:
            del snapGas.data[key]

    # Make normal dictionary form of snapGas
    for key, value in snapGas.items():
        out.update({key : value})

    print(f"[@{resolution}, @{CR_indicator}, @{int(snapNumber)}]: Finishing process...")
    return {(f"{resolution}",f"{CR_indicator}") : out}

def combine_snapshots(snapList, flatConcatBool = True):

    if len(snapList)==0:
        raise Exception("[@combine_snapshots]: WARNING:CRITICAL! Empty snapList received! Cannot combine!")
    elif (len(snapList)==1):
        print("[@combine_snapshots]: WARNING: SINGULAR snapList received! Cannot combine!")
        return snapList[0]
    else:
        outSnap = snapList[-1]

    # Combine all snapshots into one output snap shot, concatenating data as needed
    for snap in snapList[-2::-1]:
        for key, values in snap.data.items():
            outValues = outSnap.data[key]

            # If we don't want a temporal axis...
            if flatConcatBool is True:
                if len(outValues)==0:
                    updatedValues = values
                else:
                    updatedValues = np.concatenate((outValues,values),axis=0)

            # If we DO want a temporal axis...
            else:
                outValues = outValues[np.newaxis]
                if np.shape(outValues)[1]==0:
                    updatedValues = values[np.newaxis]
                else:
                    updatedValues = np.concatenate((outValues,values[np.newaxis]),axis=0)
            outSnap.data.update({key : updatedValues})

    return outSnap
