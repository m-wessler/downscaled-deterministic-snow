#!/bin/bash
#
#   getnam12ds.sh
#
#   This script obtains the NAM218 12km grids from NCEP
#   A subset for the western US and a limited number of fields are downloaded
#
#   Script requires the model to be called. Options are NAMDS, GFSDS
#   Call these from the cron entry
#            $1
#
#   If nothing is entered on the command line, script automatically 
#   determines time to retrieve.  Otherwise:
#            $2=yr  $3=mn  $4=dy  $5=hr
#

module load anaconda3/2018.12

SCRIPTDIR='/uufs/chpc.utah.edu/common/home/u1070830/code/downscaled-deterministic-snow/scripts'

if [ $1 ]; then
    python -W ignore "$SCRIPTDIR/downscaledqsf.py" "$1" "$2$3$4$5"
else
    python -W ignore "$SCRIPTDIR/downscaledqsf.py" "$1"
fi
