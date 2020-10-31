import os
import gc
import sys
import wrf

import numpy as np
import xarray as xr
import pandas as pd
import metpy.calc as mpc

from glob import glob
from subprocess import call
from metpy.units import units as mpu
from datetime import datetime, timedelta

import psutil
from functools import partial
from multiprocessing import Pool, cpu_count, get_context

import warnings
warnings.filterwarnings('ignore')

from ds_funcs import *
from ds_config import *

os.environ['OMP_NUM_THREADS'] = str("1")

if __name__ == '__main__':
    
    input('filter type: %s\nsigma override:%s'%(grid_filter, sigma_override))

    # Determine which model run to grab
    model = sys.argv[1] 
    fhrs = fhrs_nam if 'NAM' in model else fhrs_gfs
    utchour = datetime.utcnow().hour
    run_hour = [run for run in run_avail[model].keys() if utchour in run_avail[model][run]][0]
    date = datetime.utcnow() - timedelta(days=1) if ((run_hour == '18')&(utchour < 3)) else datetime.utcnow()
    init_req = datetime.strftime(date, '%Y%m%d') + run_hour
    
    # Override the automatic run choice if one is provided at command line
    if len(sys.argv) >= 3:
        if sys.argv[2] == '':
            pass
        else:
            print('init overridden')
            init_req = sys.argv[2]
    else:
        pass

    # Determine if we are making grids or only making plots
    if len(sys.argv) == 4:
        if sys.argv[3] == 'plots':
            plotonly = True
        else:
            plotonly = False
    else:
        plotonly = False

    # Create temp dir
    print(init_req)
    temp = mkdir_p(tmpdir + '%s/%s/%s/'%(model.lower(), init_req[:-2], init_req))

    if plotonly:
            print('Skipping processing, making plots...')

    else:
        # Get the model files
        print('Gathering model files...')
        get_model = get_NAM if 'NAM' in model else get_GFS
        surface, isobaric = get_model(init_req)

        if 'GFS' in model:
            # Some lat-lon grid manipulation to be consistent with the NAM 2d Coordinates
            lon, lat = np.meshgrid(surface.lon, surface.lat)
            surface = surface.rename({'lat':'y', 'lon':'x'})
            surface = surface.drop(['x', 'y'])
            surface['lon'] = xr.DataArray(lon).rename({'dim_0':'y', 'dim_1':'x'})
            surface['lat'] = xr.DataArray(lat).rename({'dim_0':'y', 'dim_1':'x'})
            surface = surface.set_coords(['lat', 'lon'])

            isobaric = isobaric.rename({'lat':'y', 'lon':'x'})
            isobaric = isobaric.drop(['x', 'y'])
            isobaric['lon'] = xr.DataArray(lon).rename({'dim_0':'y', 'dim_1':'x'})
            isobaric['lat'] = xr.DataArray(lat).rename({'dim_0':'y', 'dim_1':'x'})
            isobaric = isobaric.set_coords(['lat', 'lon'])

        # We can chagne the upper limit of the model levels later if desired
        isobaric = isobaric.isel(level=np.where(isobaric.level >= model_upper)[0])

        # Establish PRISM ratios and output grid mesh
        print('\nGetting PRISM ratios...')
        prism = downscale_prism(datetime.strptime(init_req, '%Y%m%d%H%M'), model)

        # For each model time step get the appropriate PRISM field
        _prism = []
        for ti in surface.time.values:

            # Use the PRISM julian date format for indexing
            tj = int(datetime.strftime(pd.to_datetime(ti), '%j'))

            try:
                _prism.append(prism.sel(time=tj))
            except:
                _prism.append(prism.isel(time=-1))

        prism = xr.concat(_prism, dim='time')
        prism['time'] = surface.time.values

        # Get elevation data from the DEM and clip to the PRISM grid
        print('\nReading elevation data...')
        elev = get_elev(prism)
        # elev.values[np.where(elev < 0)] = 0

        # Repack the high-resolution data into a neat dataset
        hires = xr.Dataset({'prism':prism, 'elev':elev})

        # There is no need for a hi resolution wet bulb, the difference
        # in order of calculate, downscale is negligible!
        print('Calculating Tw...')

        # Broadcast pressure levels to the dimensions of the data
        p = isobaric.level
        _p = np.ones(isobaric.t.shape)
        _p = np.array([_p[:, i, :, :]*p[i].values 
            for i in range(p.size)]).transpose(1, 0, 2, 3)
        p = isobaric.t.copy().rename('p')
        p.values = _p

        # Calculate the mixing ratio
        qv = isobaric.t.copy().rename('qv')
                
        qv.values = np.array(
            mpc.mixing_ratio_from_relative_humidity(
                isobaric.r.values*mpu.percent, 
                (isobaric.t.values-273.15)*mpu.degC,
                p.values*mpu.millibar))
                
        # Repair the dimensions after metpy messes with them
        qv['time'] = isobaric.time
        qv['level'] = isobaric.level
        qv['lat'] = isobaric.lat
        qv['lon'] = isobaric.lon

        # Finally, we can calculate the wet bulb
        tw = wrf.wetbulb(p*100, isobaric.t, qv, units='degC')
        # Repair the dimensions after wrf messes with them
        tw['time'] = isobaric.time
        tw['level'] = isobaric.level
        tw['lat'] = isobaric.lat
        tw['lon'] = isobaric.lon

        # There is no need for a hi resolution wbzh, the difference
        # in order of calculate, downscale is negligible!
        print('Calculating WBZ height ASL')
        wbz = calcwbz(tw, isobaric.gh, surface.orog)

        # Package the low res data
        lores = isobaric.copy().drop([k for k in isobaric.keys()])#.transpose('time', 'level', 'y', 'x')
        lores['t'] = isobaric.t - 273.15
        lores['gh'] = isobaric.gh
        lores['wbz'] = wbz
        lores['qpf'] = surface.qpf

        # Cache files to disk
        lrxy = (lores.lon.values.flatten(), lores.lat.values.flatten())
        hrxy = (hires.lon.values, hires.lat.values)
        np.save(temp + 'xy.npy', [lrxy, hrxy])

        # Cache the data on disk
        hires.to_netcdf(temp + 'hires_cache.nc')
        [lores.sel(time=t).to_netcdf(temp + '%s_cache.nc'%t.values) for t in lores.time]
        indexer = [('%s'%t, fhr) for t, fhr in zip(lores.time.values, fhrs)]

        # del lores, hires, lrxy, hrxy, wbz, isobaric, surface, p
        gc.collect()

        # Figure out how many pool workers we can get away with
        mem_avail = psutil.virtual_memory().available
        # Using ceil, if causing memory problems in the future, use floor
        memlim = np.floor(mem_avail/mem_need).astype(int)

        # Friendly, readable printouts
        hmem_avail = bytes2human(mem_avail) 
        hmem_need = bytes2human(mem_need)

        # Set the limits - CORES is memory limited automatically!
        cores = (memlim if memlim <= len(indexer) 
            else len(indexer))
        mpi_limit = mpi_limit if mpi_limit is not None else 9999
        cores = cores if cores <= mpi_limit else mpi_limit

        print(("\nMem Avail: {}\nWorkers Needed: {}\nMem Needed Each Worker: {}\n" +
                "Cores Available: {}\nCores to use: {}\n").format(
                hmem_avail, len(indexer), hmem_need,
                memlim, cores))
        
        downscale_calc_grids_mpi = partial(
            downscale_calc_grids, 
            model=model, init_req=init_req, temp=temp)

        with get_context("spawn").Pool(cores) as p:
            fo_list = p.map(downscale_calc_grids_mpi, indexer, chunksize=1)
            p.close()
            p.join()

        print('Jobs complete, workers terminated')

        os.remove(temp + 'hires_cache.nc')
        os.remove(temp + 'xy.npy')

    # Make the plots
    if plotonly:
        try:
            archive = datadir + '%s/models/%s/%s/'%(init_req[:-2], model.lower(), init_req)
            flist = glob(archive + '*.nc')
            flist = np.array(flist)[np.argsort(flist)]
            print('Opening dataset')
            alldata = xr.open_mfdataset(flist)

        except:
            archive = temp
            print(temp)
            flist = glob(temp + '*.nc')
            flist = np.array(flist)[np.argsort(flist)]
            print('Opening dataset')
            alldata = xr.open_mfdataset(flist)

    else:
        flist = glob(temp + '*.nc')
        flist = np.array(flist)[np.argsort(flist)]
        print('Opening dataset')
        alldata = xr.open_mfdataset(flist)

    print('Dataset Opened')
    lons, lats = alldata.lon.values, alldata.lat.values
    elev = xr.open_dataset(terrainfile)
    elats = elev.latitude.values
    elons = elev.longitude.values
    elev = elev.elevation.sel(
        latitude=elats[np.searchsorted(elats, lats)], 
        longitude=elons[np.searchsorted(elons, lons)]).values

    alldata['wbzh_agl'] = alldata['wbzh'] - elev

    init = datetime.strftime(datetime.strptime(init_req, '%Y%m%d%H'), '%Y-%m-%d %H:%M UTC')

    fhrs3h = np.arange(3, 84.1, 3, dtype=int)

    for domain in map_regions.keys():

        fhrdata = []
    
        plotstep = 24
        for var in plotvars:
            for fhr in fhrs3h:
                i0 = 0 if fhr <= plotstep else np.where(fhrs == fhr-plotstep)[0][0]
                i = np.where(fhrs == fhr)[0][0] 
                data = alldata[var].isel(time=slice(i0+1, i+1))
                fhrdata.append([[init, init_req, fhr, plotstep], data])

        plotstep = 6
        for var in plotvars:
            if var in ['qpf', 'qsf', 'dqpf', 'dqsf']:
                for fhr in fhrs3h:
                    i0 = 0 if fhr <= plotstep else np.where(fhrs == fhr-plotstep)[0][0]
                    i = np.where(fhrs == fhr)[0][0] 
                    data = alldata[var].isel(time=slice(i0+1, i+1))
                    fhrdata.append([[init, init_req, fhr, plotstep], data])

        imgdir  = mkdir_p(datadir + '%s/images/models/%s/'%(init_req[:-2], model.lower()))
        make_plots_mpi = partial(make_plots, 
            model=model, imgdir=imgdir, domain=domain)

        print('Spawning new plotting pool')
        with get_context('spawn').Pool(40) as p:
            p.map(make_plots_mpi, fhrdata, chunksize=1)
            p.close()
            p.join()

        print('Pool Complete')
    
    del alldata
    gc.collect()
    print('Plotting done...')

    if plotonly:
        pass
    else:
        print('Cleaning up temp, moving grids to archive...')
        # Compress the output files
        os.chdir(temp)
        mkdir_p(temp + 'compressed/')
        call('parallel ncks -O -L 9 --no_tmp_fl {} ./compressed\/\{} ::: %s'%'*.nc', shell=True)
        [os.remove(f) for f in flist]
        
        # Move the compressed files to archive
        cf_in = temp + 'compressed/'
        cf_out = mkdir_p(datadir + '%s/models/%s/%s/'%(init_req[:-2], model.lower(), init_req))
        call('mv %s*.nc %s'%(cf_in, cf_out), shell=True)
        
        # Clean up directories
        # For some reason this works sometimes, not others
        # This is just temp on lustre littered with empty dirs so not a big deal
        # Will be auto-cleaned after 60 days
        try:
            os.chdir(tmpdir)
            os.rmdir(cf_in)
            os.rmdir(temp)
        except:
            pass

        # Don't save the NAM218 or GFS025 grids. They are already in archive
        temp = temp[:-11]
        arch = mkdir_p(datadir + '%s/models/%s/'%(init_req[:-2], model.lower()))
        call('mv %s* %s'%(temp, arch), shell=True)
        call('rm -fv %s*.idx %s*.grib2'%(temp, temp), shell=True)
        try:
            os.rmdir(temp)
        except:
            pass

    print('Script complete')
