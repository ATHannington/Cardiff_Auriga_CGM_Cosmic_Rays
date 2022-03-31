# Set path to simulation data without Cosmic Rays
simfile_NO_CRs="/home/universe/spxfv/Auriga/level4_cgm/h12_standard/output/"
# Set path to simulation data with Cosmic Rays
simfile_CRs="/home/universe/spxfv/Auriga/level4_cgm/h12_standard_CRs/output/"
# Set path to save data
savepath = "/home/universe/c1838736/Cosmic_Rays/V0-0/"
# Set saved parameters
saveParams = ["T","R","n_H","B","vrad","gz","L","P_thermal","P_magnetic","P_kinetic","P_tot","Pthermal_Pmagnetic","tcool","theat","tff","tcool_tff"]
# Set other saved parameters that should NOT be changed! These are required to be tracked in order for the analysis to work
saveEssentials = ["FoFHaloID","SubHaloID","Lookback","Snap","type","mass","pos"]
# Set minimum snap (largest lookback time) to look at
snapMin = 100
# Set Max snapshot to look at tracers from snapnum
snapMax = 127
# Set Percentiles of parameter of interest
percentiles =  [0.13, 2.28 , 15.87, 50.0, 84.13, 97.72, 99.87]
# Set target temperatures of form: [10^{target +- delta}]
targetTlst = [4.0, 5.0, 6.0]
# Set delta dex for Temperature as given in above form
deltaT = 0.25
# Inner Radius of selection kpc
Rinner = [25.0, 75.0, 125.0]
# Outer Radius of selection kpc
Router = [75.0, 125.0, 175.0]
# Axis for Plots 0 = x, 1 = y, 2 = z
Axes = [2,0]
# line of sight axis
zAxis = [1]
# Boolean for whether to plot QUAD plot
QuadPlotBool = True
# Plots box size (+/- boxsize/2 kpc on each edge)
boxsize = 400.0
# Plots Line of sight box size (+/- boxlos/2 kpc centred on galaxy centre)
boxlos = 50.0
# Pixel Resolution in plane of Plot
pixres = 0.2
# Pixel Resolution in LOS of plot
pixreslos = 4.0
# Select SubHalo to select. Unbound in relevant FoF halo will also be selected
haloID = 0
