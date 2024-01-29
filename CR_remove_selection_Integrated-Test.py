import numpy as np
import pandas as pd
import const as c
import OtherConstants as oc
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
from CR_Subroutines import *
import copy
import h5py
import json
import math
import logging
import copy

import pytest


def _load_snap(loadTypes,loadonlyhalo=0):
    # load in the subfind group files
    snap_subfind = load_subfind(127, dir="/home/cosmos/spxfv/Auriga/level4_cgm/h5_standard/output/")

    # load in the gas particles mass and position only for HaloID 0.
    #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    #       gas and stars (type 0 and 4) MUST be loaded first!!
    snap = gadget_readsnap(
        127,
        "/home/cosmos/spxfv/Auriga/level4_cgm/h5_standard/output/",
        hdf5=True,
        loadonlytype=loadTypes,
        loadonlyhalo=loadonlyhalo,
        lazy_load=False,
        subfind=snap_subfind
    )
    return snap


@pytest.mark.parametrize("loadTypes,loadonlyhalo",[([0],0),([1],0),([2],-1),([3],-1),([4],0),([5],0)])
def test_single_type_sensible_remove(loadTypes,loadonlyhalo):
    snap = _load_snap(loadTypes,loadonlyhalo=loadonlyhalo)
    snap, _, _ = map_params_to_types(snap)
    whereToRemove = np.isin(snap.data["type"],np.array(loadTypes))
    whereToRemove[127] = False
    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_single_type_sensible_remove types {loadTypes}",
        DEBUG = True,
        )
    whereRemaining = np.isin(snap.data["type"],np.array(loadTypes))
    assert len(whereRemaining) == 1, f"[test_single_type_sensible_remove]: Expected single return value! Len values returned {len(whereRemaining)}"


@pytest.mark.parametrize("loadTypes",[([0,1]),([0,1,4]),([0,4]),([1,0,4])])
def test_multi_type_sensible_remove(loadTypes):
    snap = _load_snap(loadTypes)
    snap, _, _ = map_params_to_types(snap)
    whereToRemove = np.isin(snap.data["type"],np.array(loadTypes))
    whereToRemove[583] = False
    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_sensible_remove types {loadTypes}",
        DEBUG = True,
        )
    whereRemaining = np.isin(snap.data["type"],np.array(loadTypes))
    assert len(whereRemaining) == 1, f"[test_multi_type_sensible_remove]: Expected single return value! Len values returned {len(whereRemaining)}"

@pytest.mark.xfail(reason="Should fail if degeneracy is detected within remove_selection(). map_params_to_types() must be called first in this case (and as good practice generally...)", raises=Exception)
@pytest.mark.parametrize("loadTypes",[([0,1]),([0,1,4]),([0,4]),([1,0,4])])
def test_multi_type_degeneracy_test_xfail(loadTypes):
    snap = _load_snap(loadTypes)
    ## NO CALL TO map_params_to_types()
    whereToRemove = np.isin(snap.data["type"],np.array(loadTypes))
    types = pd.unique(snap.data["type"])
    for tp in types:
        whereType = np.where(snap.data["type"]==tp)[0]
        secondIndexOfType = whereType[1]
        whereToRemove[secondIndexOfType] = False
        print(whereType)
        print(secondIndexOfType)
        print(whereToRemove[secondIndexOfType])
    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_degeneracy_test_xfail (before degeneracy) types {loadTypes}",
        DEBUG = True,
        )

    ## NO CALL TO map_params_to_types()
    whereToRemove = np.isin(snap.data["type"],np.array(loadTypes))
    types = pd.unique(snap.data["type"])
    for tp in types:
        whereType = np.where(snap.data["type"]==tp)[0]
        secondIndexOfType = whereType[1]
        whereToRemove[secondIndexOfType] = False
        print(whereType)
        print(secondIndexOfType)
        print(whereToRemove[secondIndexOfType])

    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_degeneracy_test_xfail (after degeneracy) types {loadTypes}",
        DEBUG = True,
        )

    assert True,"[test_multi_type_degeneracy_test_xfail]: we shouldn't get here..."


@pytest.mark.parametrize("loadTypes",[([0,1]),([0,1,4]),([0,4]),([1,0,4])])
def test_multi_type_degeneracy_test_pass(loadTypes):
    snap = _load_snap(loadTypes)
    snap, _, _ = map_params_to_types(snap)
    whereToRemove = np.isin(snap.data["type"],np.array(loadTypes))
    types = pd.unique(snap.data["type"])

    for tp in types:
        whereType = np.where(snap.data["type"]==tp)[0]
        secondIndexOfType = whereType[1]
        whereToRemove[secondIndexOfType] = False

    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_degeneracy_test_pass (before degeneracy) types {loadTypes}",
        DEBUG = True,
        )

    snap, _, _ = map_params_to_types(snap)
    whereToRemove = np.isin(snap.data["type"],np.array(loadTypes))
    types = pd.unique(snap.data["type"])

    for tp in types:
        whereType = np.where(snap.data["type"]==tp)[0]
        secondIndexOfType = whereType[1]
        whereToRemove[secondIndexOfType] = False

    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_degeneracy_test_pass (after degeneracy) types {loadTypes}",
        DEBUG = True,
        )

    assert True,"[test_multi_type_degeneracy_test_pass]: all is good"

@pytest.mark.parametrize("loadTypes,exclusionArg",[([0,4],"age"),([4,0],"age"),([0,1],"rho"),([1,0],"rho")])
def test_multi_type_partial_remove(loadTypes,exclusionArg):
    snap = _load_snap(loadTypes)
    snap, _, _ = map_params_to_types(snap)
    whereToRemove = np.isfinite(snap.data[exclusionArg])
    removeShape = np.shape(np.where(whereToRemove)[0])[0]
    shapeBefore = np.shape(snap.data["type"])[0]
    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_partial_remove types {loadTypes}",
        DEBUG = True,
        )
    shapeAfter = np.shape(snap.data["type"])[0]
    assert shapeAfter == shapeBefore - removeShape, f"[test_multi_type_partial_remove]: Expected partial removal from one type, but not other! Incorrect removal of entire type detected! {shapeAfter} != {shapeBefore} - {removeShape} ({shapeBefore-removeShape})"

@pytest.mark.parametrize("loadTypes,exclusionArg",[([0,4],"age"),([4,0],"age"),([0,1],"rho"),([1,0],"rho"),([0,1,4],"gz"),([0,4,1],"gz"),([4,0,1],"gz"),([4,1,0],"gz"),([4,1,0,5],"gz")])
def test_multi_type_sub_partial_remove(loadTypes,exclusionArg):
    snap = _load_snap(loadTypes)
    snap, _, _ = map_params_to_types(snap)
    whereToRemove = (snap.data[exclusionArg] >= np.nanmean(snap.data[exclusionArg]))
    removeShape = np.shape(np.where(whereToRemove)[0])[0]
    shapeBefore = np.shape(snap.data["type"])[0]
    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_sub_partial_remove types {loadTypes}",
        DEBUG = True,
        )
    shapeAfter = np.shape(snap.data["type"])[0]
    assert shapeAfter == shapeBefore - removeShape, f"[test_multi_type_sub_partial_remove]: Expected removal from one type, but not other! Incorrect removal of entire type detected! {shapeAfter} != {shapeBefore} - {removeShape} ({shapeBefore-removeShape})"


@pytest.mark.parametrize("loadTypes",[([0,1]),([0,1,4]),([0,4]),([1,0,4])])
def test_multi_type_no_remove(loadTypes):
    snap = _load_snap(loadTypes)
    snap, _, _ = map_params_to_types(snap)
    whereToRemove = (snap.data["type"]==999)
    shapeBefore = np.shape(snap.data["type"])[0]
    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_multi_type_no_remove types {loadTypes}",
        DEBUG = True,
        )
    shapeAfter = np.shape(snap.data["type"])[0]
    assert shapeAfter == shapeBefore, f"[test_multi_type_no_remove]: Expected removalof no data! Shapes should be the same! {shapeAfter} != {shapeBefore}"

@pytest.mark.xfail(reason="Should fail if we return a snapshot with empty data",raises=Exception)
@pytest.mark.parametrize("loadTypes",[([0]),([0,1]),([0,1,4]),([0,4]),([1,0,4])])
def test_mixed_type_full_remove(loadTypes):
    snap = _load_snap(loadTypes)
    snap, _, _ = map_params_to_types(snap)
    whereToRemove = np.isin(snap.data["type"],np.array(loadTypes))
    snap = remove_selection(
        snap=snap,
        removalConditionMask=whereToRemove,
        errorString = f"test_mixed_type_full_remove types {loadTypes}",
        DEBUG = True,
        )
