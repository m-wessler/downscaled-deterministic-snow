import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Directory Configuration
chpcdir = ('/uufs/chpc.utah.edu/common/home/u1070830/' +
           'code/downscaled-deterministic-snow/')
           
datadir = '/scratch/general/lustre/u1070830/gfs_prism/archive/'
tmpdir = '/scratch/general/lustre/u1070830/gfs_prism/temp/'

prism_dir = chpcdir + 'prism/'
terrainfile = prism_dir + 'usterrain.nc'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PRISM Configuration

grid_filter = 'box' #['gaussian', 'box']
sigma_override = round(27.75/.8, 0) #0.25deg

# if sigma override is None, calculate sigma from below
res_prism = 0.8
res_model = {'NAMDS': 12., 'GFSDS': 25.}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Configuration

minlon, minlat = -130., 30.
maxlon, maxlat = -100., 50.

# Give the NAM a 3 hour delay, the GFS a 4 hour delay
run_avail = {
    'NAMDS':{
        '06':[9, 10, 11, 12, 13, 14],
        '12':[15, 16, 17, 18, 19, 20],
        '18':[21, 22, 23, 0, 1, 2],
        '00':[3, 4, 5, 6, 7, 8]},
    'GFSDS':{
        '06':[10, 11, 12, 13, 14, 15],
        '12':[16, 17, 18, 19, 20, 21],
        '18':[22, 23, 0, 1, 2, 3],
        '00':[4, 5, 6, 7, 8, 9]}}

# NAM is 1 hourly to 36, then 3 hourly to 84
fhrs_nam = np.arange(0, 84.1, 3).astype(int)
fhrs_gfs = np.arange(0, 84.1, 3).astype(int)

# Upper vertical limit (mb)
model_upper = 500

# WBZ determination
# Level of the 'wet bulb zero height' degC
# +0.5 re: WRH Methodology White Paper
wbzparam = 0.50

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MPI Configuration
mem_need = 7.45e9
mpi_limit = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Plot Configuration

plotvars = [
#     'wbzh_agl', 
#     'wbzh', 
#     'slr',
    'qpf', 
    'dqpf',
    'dqsf',]

webvars = {
    'qpf':'QP',
    'qsf':'QS',
    'dqpf':'DQ',
    'dqsf':'DS',
    'slr':'SR',
    'wbzh':'ZS',
    'wbzh_agl':'ZG'}

# Region Boundaries (minlat, maxlat, minlon, maxlon)
map_regions = {
    'UT':(-114.7, -108.5, 36.7, 42.5),
#     'WM':(-117, -108.5, 43, 49),
    'CO':(-110, -104, 36, 41.9),
#     'SN':(-123.5, -116.0, 33.5, 41),
#     'WE':(-125, -102.5, 31.0, 49.2),
#     'NW':(-125, -116.5, 42.0, 49.1),
    'NU':(-113.4, -110.7, 39.5, 41.9),
    }
