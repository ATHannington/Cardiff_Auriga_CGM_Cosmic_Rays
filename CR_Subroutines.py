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
from gadget import *
from gadget_subfind import *
import CRParams as param
from Tracers_Subroutines import *
import h5py
