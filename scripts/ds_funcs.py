import gc
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from colortables import *
from ds_config import *

def mkdir_p(ipath):
    from os import makedirs, path
    import errno
    
    try:
        makedirs(ipath)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and path.isdir(ipath):
            pass
        else:
            raise
        
    return ipath

def bytes2human(n):
    ''' http://code.activestate.com/recipes/578019 '''

    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n

def download(url, init, mod, minsize=1e5, timeout=10, wait=120):
    import urllib.request
    from subprocess import call
    from os import remove, stat, path
    from os.path import isfile
    from time import sleep
        
    _file = url.split('file=')[1].split('&')[0].split('.')
    _file = '{}.{}.{}'.format(_file[0], init, '.'.join(_file[1:]))
    f = mkdir_p(tmpdir + '%s/%s/'%(mod.lower(), init)) + _file

    # Download the grib to disk
    while not isfile(f):
        print('Downloading %s'%_file)

        try:
            urllib.request.urlretrieve(url, f)

        except OSError:
            # Sometimes urllib struggles. Before totally giving up, try this
            # the old fashioned way first...
            curlcommand = 'curl -s -m {} -o {} {}'.format(timeout, f, url)
            call(curlcommand, shell=True)

        try:
            fsize = stat(f).st_size
        except:
            print('FILE NOT FOUND Data not yet available. Waiting', 
                wait, 'seconds...')
        else:
            if (fsize > minsize):
                pass
            else:
                print('FSIZE ERROR JUNK FILE Data not yet available. Waiting', 
                    wait, 'seconds...')
                remove(f)
                sleep(wait)
                
                now = datetime.utcnow()
                if ((now-start_time).days >= 1 
                    or (now-start_time).seconds > killtime * 3600):
            
                    exit()
                
    return f

def get_GFS(_inittime):
    from multiprocessing import cpu_count, Pool
    from multiprocessing.dummy import Pool as ThreadPool
    from functools import partial
    import pygrib

    _model = 'GFSDS'
    init_date = _inittime[:-2]
    init_hour = int(_inittime[-2:])
    
    base = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?'
    ext = '&dir=%2Fgfs.{}%2F{:02d}'.format(init_date, init_hour)
    
    levs = ('&lev_1000_mb=on&lev_10_m_above_ground=on&lev_200_mb=on&lev_250_mb=on' + 
                '&lev_2_m_above_ground=on&lev_300_mb=on&lev_350_mb=on&lev_400_mb=on' +
                '&lev_450_mb=on&lev_500_mb=on&lev_550_mb=on&lev_600_mb=on&lev_650_mb=on' + 
                '&lev_700_mb=on&lev_750_mb=on&lev_800_mb=on&lev_850_mb=on&lev_900_mb=on' + 
                '&lev_925_mb=on&lev_950_mb=on&lev_975_mb=on&lev_surface=on')

    lvars = ('&var_APCP=on&var_HGT=on&var_PRES=on&var_PRMSL=on&var_RH=on&var_TMP=on' + 
             '&var_UGRD=on&var_VGRD=on')
        
    subset = '&subregion=&leftlon={}&rightlon={}&toplat={}&bottomlat={}'.format(
                minlon, maxlon, maxlat, minlat)
        
    urllist = []
    fhrs = fhrs_gfs
    for fhr in fhrs:
        file = 'file=gfs.t{:02d}z.pgrb2.0p25.f{:03d}'.format(init_hour, fhr)
        urllist.append(base + file + levs + lvars + subset + ext)
        
    
    p = ThreadPool(len(urllist))
    download_mp = partial(download, init=init_date, mod=_model)
    flist = p.map(download_mp, urllist)
    p.close()
    p.join()

    flist = np.array(flist)[np.argsort(flist)]

    p = ThreadPool(len(flist))
    open_dataset = partial(xr.open_dataset, engine='cfgrib', 
                           backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa'}})
    _isobaric = p.map(open_dataset, flist)
    p.close()
    p.join()

    isobaric = xr.concat(_isobaric, dim='valid_time').rename(
        {'isobaricInhPa':'level', 'latitude':'lat', 'longitude':'lon'})

    _surface = xr.open_mfdataset(flist, engine='cfgrib', combine='nested', concat_dim='valid_time', 
                                backend_kwargs={'filter_by_keys':
                                                {'typeOfLevel': 'surface', 'stepType':'instant'}}, )

    tp = [xr.open_dataset(f, engine='cfgrib', drop_variables=['sp', 'orog', 't', 'step', 'time', 'surface'], 
                          backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface'}}) for f in flist]
    tp = [t.reset_coords('valid_time') for t in tp]
    tp[0]['tp'] = tp[1].tp.copy()
    tp[0]['tp'].values = np.zeros(tp[0].tp.shape)
    tp = xr.concat(tp, dim='valid_time')
    _surface['qpf'] = tp['tp']
        
    _surface2m = [xr.open_dataset(f, engine='cfgrib',
                                backend_kwargs={'filter_by_keys':
                                    {'typeOfLevel':'heightAboveGround', 'level':2}}) for f in flist]
    _surface2m = xr.concat(_surface2m, dim='valid_time')

    _surface10m = [xr.open_dataset(f, engine='cfgrib',
                                backend_kwargs={'filter_by_keys':
                                    {'typeOfLevel':'heightAboveGround', 'level':10}}) for f in flist]
        
    _surface10m = xr.concat(_surface10m, dim='valid_time').rename({'u10':'u10m', 'v10':'v10m'})

    del _surface2m['heightAboveGround'], _surface10m['heightAboveGround']
    surface = xr.merge([_surface, _surface2m, _surface10m], compat='override').rename({'latitude':'lat', 'longitude':'lon'})
    del surface['surface'], _surface, _surface2m, _surface10m

    isobaric['lon'] = isobaric.lon - 360
    surface['lon'] = surface.lon - 360

    del isobaric['time'], surface['time']
    del isobaric['step'], surface['step']

    isobaric = isobaric.rename({'valid_time':'time'}).load()
    surface = surface.rename({'valid_time':'time'}).load()
    
    # Deconstruct the 6-hour totals at 6, 12, etc into 3-hour totals
    qpf = [np.zeros(surface.qpf.isel(time=0).shape)]
    for i in range(len(flist[1:])):
    
        i+=1
        print(i, flist[i])

        grbs = pygrib.open(flist[i])
        grb = grbs.message(104)
        print('Used %s'%grb)
    
        if i>1:
            priorgrbs = grbs = pygrib.open(flist[i-1])
            priorgrb = grbs.message(104)
            qpf.append(grb.values - priorgrb.values)
        else:
            qpf.append(grb.values)
        
    surface['qpf'].values = np.array(qpf)
    
    return surface, isobaric

def get_NAM(_inittime):
    from multiprocessing import cpu_count, Pool, get_context
    from functools import partial
    import pygrib

    _model = 'NAMDS'
    init_date = _inittime[:-2]
    init_hour = int(_inittime[-2:])

    base = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_nam.pl?'
    
    levs = ('&lev_1000_mb=on&lev_950_mb=on&lev_975_mb=on&lev_500_mb=on&lev_525_mb=on' + 
            '&lev_575_mb=on&lev_600_mb=on&lev_625_mb=on&lev_650_mb=on&lev_675_mb=on' + 
            '&lev_700_mb=on&lev_725_mb=on&lev_750_mb=on&lev_775_mb=on&lev_800_mb=on' + 
            '&lev_825_mb=on&lev_850_mb=on&lev_875_mb=on&lev_900_mb=on&lev_925_mb=on' + 
            '&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_surface=on')
    
    lvars = ('&var_APCP=on&var_GUST=on&var_HGT=on&var_PRES=on&var_PRMSL=on&var_RH=on' + 
             '&var_TMP=on&var_UGRD=on&var_VGRD=on')

    subset = '&subregion=&leftlon={}&rightlon={}&toplat={}&bottomlat={}'.format(
                minlon, maxlon, maxlat, minlat)
    
    ext = '&dir=%2Fnam.{}'.format(init_date)
    
    urllist = []
    fhrs = fhrs_nam
    for fhr in fhrs:

        file = 'file=nam.t{:02d}z.awphys{:02d}.tm00.grib2'.format(
                    init_hour, fhr)
        
        urllist.append(base + file + levs + lvars + subset + ext)

    with get_context("spawn").Pool(len(urllist)) as p:
        download_mp = partial(download, init=init_date, mod=_model)
        flist = p.map(download_mp, urllist)
        p.close()
        p.join()

    flist = np.array(flist)[np.argsort(flist)]

    with get_context("spawn").Pool(len(flist)) as p:
        open_dataset = partial(xr.open_dataset, engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa'}})
        _isobaric = p.map(open_dataset, flist)
        p.close()
        p.join()

    isobaric = xr.concat(_isobaric, dim='valid_time').rename(
        {'isobaricInhPa':'level', 'latitude':'lat', 'longitude':'lon'})

    _surface = xr.open_mfdataset(flist, engine='cfgrib', combine='nested',
                                backend_kwargs={'filter_by_keys':
                                                {'typeOfLevel': 'surface', 'stepType':'instant'}}, 
                                concat_dim='valid_time')

    _surface_tp = xr.open_mfdataset(flist, engine='cfgrib', combine='nested',
                                backend_kwargs={'filter_by_keys':
                                                {'typeOfLevel': 'surface', 'stepType':'accum'}}, 
                                concat_dim='valid_time')['tp']

    _surface2m = [xr.open_dataset(f, engine='cfgrib',
                                backend_kwargs={'filter_by_keys':
                                    {'typeOfLevel':'heightAboveGround', 'level':2}}) for f in flist]

    _surface2m = xr.concat(_surface2m, dim='valid_time').rename({'r2':'r2m'})
        
    _surface10m = [xr.open_dataset(f, engine='cfgrib',
                                backend_kwargs={'filter_by_keys':
                                    {'typeOfLevel':'heightAboveGround', 'level':10}}) for f in flist]
        
    _surface10m = xr.concat(_surface10m, dim='valid_time').rename({'u10':'u10m', 'v10':'v10m'})

    del _surface2m['heightAboveGround'], _surface10m['heightAboveGround']
    surface = xr.merge([_surface, _surface_tp, _surface2m, _surface10m]).rename({'latitude':'lat', 'longitude':'lon', 'tp':'qpf'})
    del surface['surface'], _surface, _surface2m, _surface10m, _surface_tp, surface['gust']
    
    isobaric['lon'] = isobaric.lon - 360
    surface['lon'] = surface.lon - 360

    del isobaric['time'], surface['time']
    del isobaric['step'], surface['step']

    isobaric = isobaric.rename({'valid_time':'time'}).load()
    surface = surface.rename({'valid_time':'time'}).load()

    qpf = []
    for f in flist:
        grbs = pygrib.open(f)
        try:
            grb = grbs.message(110)
        except:
            grb = grbs.message(109)
            qpf.append(grb.values)
            print('Used msg 109', grbs.message(109))
        else:
            qpf.append(grb.values)
            print('Used msg 110', grbs.message(110))

    surface['qpf'].values = np.array(qpf)

    return surface, isobaric

def interpolate_prism_daily(doy, year, bounds):
    from netCDF4 import Dataset
    from datetime import datetime
    
    """ Interpolates monthly PRISM totals to a daily total. Assumes the 15th
        (14th for February) is most representative of the month.
        ## Parameters
        doy: The day of year as an int

        year: The year with century as a int

        bounds: A tuple containing boundaries of the domain as indices of
                the PRISM grid. In order xmin, xmax, ymin, ymax.

        prism_dir: The directory the PRISM files are in
        ## Returns
        pclimo: A 2D grid representing a monthly PRISM total were that month
                centered around doy

        Alex Weech
    """

    # Unpack the bounds
    xmin, xmax, ymin, ymax = bounds

    # List of centers of each month
    prism_day = [15] * 12
    prism_day[1] = 14

    # Convert doy and year to a datetime object
    date = datetime.strptime(str(doy) + '-' + str(year), '%j-%Y')

    # Simple case of it being the center day
    center = prism_day[date.month-1]
    if date.day == center:
        prism_path = prism_dir + '/us_' + date.strftime('%m') + '_prcp.nc'
        with Dataset(prism_path, 'r') as prism_cd:
            pclimo = prism_cd.variables['prcp'][0, :, xmin:xmax]
            pclimo = np.flipud(pclimo)[ymin:ymax, :]

    # Else interpolate the two closest months
    else:
        # Check which side of center today is
        if date.day > center:
            month1 = date.month
            year_wrap, month2 = divmod(date.month + 1, 12)
            if month2 == 0:
                year_wrap = 0
                month2 = 12
            centdt1 = datetime(int(year), month1, center)
            centdt2 = datetime(int(year) + year_wrap, month2,
                                  prism_day[month2 - 1])
            weight1 = (date - centdt1).days / (centdt2 - centdt1).days
            weight2 = (centdt2 - date).days / (centdt2 - centdt1).days

        # Else today is before the center
        else:
            month1 = date.month
            year_wrap, month2 = divmod(date.month - 1, 12)
            if month2 == 0:
                year_wrap = -1
                month2 = 12
            centdt1 = datetime(int(year), month1, center)
            centdt2 = datetime(int(year) + year_wrap, month2,
                                  prism_day[month2 - 1])
            weight1 = (centdt1 - date).days / (centdt1 - centdt2).days
            weight2 = (date - centdt2).days / (centdt1 - centdt2).days

        # Open the two files
        file1 = prism_dir + '/us_' + str(month1).zfill(2) + '_prcp.nc'
        file2 = prism_dir + '/us_' + str(month2).zfill(2) + '_prcp.nc'
        with Dataset(file1, 'r') as prism_cd:
            pclimo1 = prism_cd.variables['prcp'][0, :, xmin:xmax]
            pclimo1 = np.flipud(pclimo1)[ymin:ymax, :]

        with Dataset(file2, 'r') as prism_cd:
            pclimo2 = prism_cd.variables['prcp'][0, :, xmin:xmax]
            pclimo2 = np.flipud(pclimo2)[ymin:ymax, :]

        # Interpolate
        pclimo = weight1 * pclimo1 + weight2 * pclimo2

    return pclimo

def downscale_prism(init, _model, minclip=0.3, maxclip=5.0):
    import warnings
    warnings.filterwarnings("ignore")
    
    import cv2
    from scipy import ndimage
    from pandas import to_datetime
    from datetime import datetime, timedelta
    
    # Get the PRISM lats and lons from a sample file
    print('Getting PRISM lats and lons')
    prism = xr.open_dataset(prism_dir + 'us_05_prcp.nc', decode_times=False)
    
    # Get boundary max and mins using full domain
    xmin = np.max(np.argwhere(prism['lon'].values < -125))
    xmax = np.min(np.argwhere(prism['lon'].values > -100))
    ymin = np.max(np.argwhere(prism['lat'][::-1].values < 30))
    ymax = len(prism['lat'].values) - 1 # Go all the way up
    bounds = (xmin, xmax, ymin, ymax)

    # Subset and mesh
    grid_lons, grid_lats = np.meshgrid(
        prism['lon'][xmin:xmax], prism['lat'][::-1][ymin:ymax])

    # Figure out which days are in this run and put them in a set
    print('Getting PRISM climo')
    date_set = set()

    fhrs = fhrs_nam if 'NAM' in _model else fhrs_gfs
    for i in fhrs:
        hour_time = init + timedelta(hours=int(i))
        day_of_year = int(hour_time.strftime('%j'))
        date_set.add((day_of_year, hour_time.year))

    # Smoothing algebra
    efold = res_model[_model] * 2 / res_prism + 1
    sigma = efold / (np.pi*np.sqrt(2))

    # Loop through the days of this run gathering the climo ratios
    ratios = list()

    for day in date_set:

        pclimo = interpolate_prism_daily(day[0], day[1], bounds)

        # Clip out the missing data
        fixed_prism = np.where(np.logical_and(np.greater(pclimo, 0),
                                              np.isfinite(pclimo)), pclimo, 0)

        # Wyndham's algorithim 
        print('Downscaling PRISM for day of year: {}'.format(
            datetime.strptime(str(day[0]),'%j').strftime('%m/%d')))

        # Create an image smoothed to the model resolution
        # Override sigma
        sigma = sigma_override if sigma_override is not None else sigma
        print('Sigma: ', sigma)
        
        if grid_filter == 'gaussian':
            smooth_prism = ndimage.filters.gaussian_filter(
                fixed_prism, sigma, mode='nearest')
            
        elif grid_filter == 'box':
            smooth_prism = cv2.blur(fixed_prism, (int(sigma), int(sigma)))
        
        smooth_prism = np.where(np.logical_and(np.greater(smooth_prism, 0),
                                               np.isfinite(smooth_prism)),
                                smooth_prism, 0)

        # Divide the real data by the smoothed data to get ratios
        ratios.append([np.where(np.logical_and(np.greater(smooth_prism, 0),
                                               np.greater(fixed_prism, 0)),
                                fixed_prism/smooth_prism, 0), day[0]])
    
    # Sort the prism data back into days (was produced as an unordered set)
    ratios = np.array(ratios)
    ratios = ratios[np.argsort(ratios[:,1].astype(int))]
    
    prism_doy = ratios[:,1]
    prism_data = np.array([x for x in ratios[:,0]])
    
    # Shape into an xarray for easy manipulation
    # Can also save with .to_netcdf if desired
    prism_climo = xr.DataArray(prism_data,
             coords={"time":("time", prism_doy),
                     "lat":(("y", "x"), grid_lats),
                     "lon":(("y", "x"), grid_lons)},
             dims=["time", "y", "x"])
    
    # Do some clipping (Based on Trevor's limits)
    # Not present in old SREF code, added by MW 01/2019
    prism_climo = xr.where(prism_climo < minclip, minclip, prism_climo)
    prism_climo = xr.where(prism_climo > maxclip, maxclip, prism_climo)

    return prism_climo

def get_elev(prism_grid):    
    # Load the elevation DEM
    # Terrainfile is set in config.py
    dem = xr.open_dataset(terrainfile)
    dem = dem.rename({'latitude':'lat', 'longitude':'lon'})

    demlats = dem['lat']
    demlons = dem['lon']

    final_lats = prism_grid.lat.values
    final_lons = prism_grid.lon.values

    # As trevor noted, the DEM isn't a perfect match -- 
    # Need to find something better
    xmin = np.where(demlons == demlons.sel(
        lon=final_lons.min(), method='ffill').values)[0][0]
    xmax = np.where(demlons == demlons.sel(
        lon=final_lons.max(), method='bfill').values)[0][0]
    ymin = np.where(demlats == demlats.sel(
        lat=final_lats.min(), method='ffill').values)[0][0]
    ymax = np.where(demlats == demlats.sel(
        lat=final_lats.max(), method='bfill').values)[0][0]
    bounds = (xmin, xmax, ymin, ymax)

    elev = dem['elevation'][ymin:ymax, xmin:xmax]

    try:    
        elevxr = xr.DataArray(elev.values,
            coords={"lat":(("y", "x"), final_lats),
                    "lon":(("y", "x"), final_lons)},
            dims=["y", "x"], name='elev')
    except:
        # Correct for mismatched grids
        if elev.lon.shape != final_lons.shape[1]:
            elev = dem['elevation'][ymin:ymax+1, xmin:xmax]

        if elev.lat.shape != final_lats.shape[0]:
            elev = dem['elevation'][ymin:ymax, xmin:xmax+1]

        elevxr = xr.DataArray(elev.values,
            coords={"lat":(("y", "x"), final_lats),
                    "lon":(("y", "x"), final_lons)},
            dims=["y", "x"], name='elev')

    dem.close()
    
    return elevxr

# def calcwbz(_tw, _gh):
#     import warnings
#     warnings.filterwarnings("ignore")
#     from xarray.ufuncs import logical_and
    
#     wbz = []
#     for i in range(_tw.level.size)[:0:-1]:
        
#         # Hi is 'prior' level
#         levLO = _tw.level[i-1]
#         levHI = _tw.level[i]
#         twLO = _tw.isel(level=i-1)
#         twHI = _tw.isel(level=i)
#         ghLO = _gh.isel(level=i-1)
#         ghHI = _gh.isel(level=i)
        
#         print('Searching for WBZ between %d and %d hPa'%(levHI, levLO))
        
#         twdiff = twLO / (twLO - twHI)
#         wbzh = ghHI * twdiff + ghLO * (1 - twdiff)
        
#         select = logical_and(twHI < wbzparam, twLO > wbzparam)
#         wbzi = xr.where(select, wbzh, np.nan)
#         wbz.append(wbzi)

#     return xr.concat(wbz, dim='level').sum(dim='level')

def calcwbz(_tw, _gh, _orog):
    for i, level in enumerate(_tw.level.values):
        if i > 0:

            level_top = _tw.isel(level=i).level.values
            level_bot = _tw.isel(level=i-1).level.values
            print('Searching for WBZ between %d and %d hPa'%(level_bot, level_top))

            gh_top = _gh.isel(level=i)
            gh_bot = _gh.isel(level=i-1)

            tw_top = _tw.isel(level=i)
            tw_bot = _tw.isel(level=i-1)

            # Find where multiple wbz exist (Not used currently)
            # if i == 1:
            #     num_wbz = xr.where( (tw_bot >= wbzparam) & (tw_top <= wbzparam), 1, 0 )
            # else:
            #     num_wbz = xr.where( (tw_bot >= wbzparam) & (tw_top <= wbzparam), num_wbz+1, num_wbz)

            interp_wbzh = gh_bot + ((wbzparam - tw_bot)*((gh_top - gh_bot)/(tw_top - tw_bot)))

            if i == 1:
                # First level, establish wbzh array
                # If WBZ between these two levels, use interpolated WBZH, else np.nan
                wbzh = xr.where( (tw_bot >= wbzparam) & (tw_top <= wbzparam), interp_wbzh, np.nan)

            else:
                # Other levels, modify wbzh array
                # If WBZ between these two levels, use interpolated WBZH, else np.nan

                # If does not exist:
                wbzh = xr.where( ((tw_bot >= wbzparam) & (tw_top <= wbzparam)) & (np.isnan(wbzh)), interp_wbzh, wbzh)

                # If exists and wbzh subterrainian
                wbzh = xr.where( ((tw_bot >= wbzparam) & (tw_top <= wbzparam)) & (~np.isnan(wbzh) & (wbzh >= _orog.min())), interp_wbzh, wbzh)

    # Where nans remain because entire column Tw < wbzparam, fill with 0 m AMSL
    wbzh = xr.where(np.isnan(wbzh) & (_tw.max(dim='level') < wbzparam), 0, wbzh)
    
    return wbzh

def calct500(_t, _gh, topo):
    
    # Geo Height - Surface Elev + 500 m
    # Gives Geo Heights ABOVE GROUND LEVEL + 500 m buffer
    gh_agl = (_gh - (topo + 500.0)).compute()
    
    # Where this is zero, set to 1.0
    gh_agl = xr.where(gh_agl == 0.0, 1.0, gh_agl)
    
    # If the 1000mb height is > 0, use the 1000 mb temperature to start
    # Otherwise assign t=0
    tvals = xr.where(gh_agl.sel(level=1000) > 0, _t.sel(level=1000), 0) # - 273.15, 0)
    
    for i in range(_t.level.size)[:0:-1]:
        
        # current level
        lc = _t.level.isel(level=i).values
        zc = gh_agl.isel(level=i)
        tc = _t.isel(level=i)# - 273.15
        
        # level above (correct for 'wraparound')
        up = i+1 if i+1 < _t.level.size else 0
        lup = _t.level.isel(level=up).values
        zup = gh_agl.isel(level=up)
        tup = _t.isel(level=up)# - 273.15
        
        # level below
        ldn = _t.level.isel(level=i-1).values
        zdn = gh_agl.isel(level=i-1)
        tdn = _t.isel(level=i-1)# - 273.15
        
        # print(i, lc, lup, ldn)
        
        # Where the geo height AGL > 0 at this level and geo height AGL < 0 at level below...
        tvals = xr.where(((zc > 0.0) & (zdn < 0.0)),
        
        # Do this
        ( ( zc / ( zc - zup ) ) * ( tup - tc ) + tc ),
        
        # Else use tvals already determined
        tvals )
        
    tvals = xr.where(gh_agl.sel(level=500) < 0, _t.sel(level=500), tvals)
        
    return tvals

def calc_slr(t500, wbz, elev):
    ''' Sometimes the old fashioned way of doing things is still the best way.
    Sticking to Trevor's stepwise method which is a little slower but produces
    a reliable result.'''
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # SLR Params
    allsnow = 0
    melt = 200

    snowlevel = wbz - allsnow
    snowlevel = xr.where(snowlevel < 0., 0., snowlevel)

    initslr = xr.where(t500 < 0., 5. - t500, 5.)
    initslr = xr.where(t500 < -15., 20. + (t500 + 15.), initslr)
    initslr = xr.where(t500 < -20., 15., initslr)

    slr = xr.where(elev >= snowlevel, initslr, 0.)

    slr = xr.where(
        ((elev < snowlevel) & (elev > (snowlevel - melt))),
        (initslr * (elev - (snowlevel - melt)) / melt), slr)

    return slr

def downscale_calc_grids(ti, model, init_req, temp):
    import os
    from scipy.interpolate import griddata
    from pandas import to_datetime
    import time as pytime
    from glob import glob

    ti, fhr = ti[0], ti[1]

    print('Processing fhr %03d v%s'%(int(fhr), ti))
    t_start = pytime.time()

    fi = glob(temp + '*%s*.nc'%ti)[0]
    ds = xr.open_dataset(fi)

    hr = xr.open_dataset(temp + 'hires_cache.nc').sel(time=ti)
    
    xy = np.load(temp + 'xy.npy', allow_pickle=True)
    lrxy, hrxy = (xy[0][0], xy[0][1]), (xy[1][0], xy[1][1])

    # print('Downscaling T, Z')
    hrt = xr.DataArray(
        np.array([griddata(
            lrxy, lr[1].values.flatten(), hrxy, 
            method='cubic', fill_value=np.nan) 
                  for lr in ds['t'].groupby('level')][::-1]), 
        dims={'level':ds.level, 'y':ds.y, 'x':ds.x}, 
        coords={'level':ds.level})

    hrgh = xr.DataArray(
        np.array([griddata(
            lrxy, lr[1].values.flatten(), hrxy, 
            method='cubic', fill_value=np.nan) 
                  for lr in ds['gh'].groupby('level')][::-1]), 
        dims={'level':ds.level, 'y':ds.y, 'x':ds.x}, 
        coords={'level':ds.level})

    # print('Calculating T_layer')
    tlayer = calct500(hrt, hrgh, hr.elev)

    del hrt, hrgh
    gc.collect()

    # print('Downscaling WBZ')
    hrwbz = xr.DataArray(
        np.array(
            griddata(
            lrxy, ds.wbz.values.flatten(), hrxy, 
            method='cubic', fill_value=np.nan)), 
        dims={'y':ds.y, 'x':ds.x})
    hr['wbzh'] = hrwbz

    # print('Calculating SLR')
    slr = calc_slr(tlayer, hrwbz, hr.elev)
    slr.values[slr.values < 0.] = 0.
    hr['slr'] = slr

    del hrwbz, hr['elev']
    gc.collect()

    # print('Downscaling QPF')
    hrqpf = xr.DataArray(
        np.array(
            griddata(
            lrxy, ds.qpf.values.flatten(), hrxy, 
            method='cubic', fill_value=np.nan)), 
        dims={'y':ds.y, 'x':ds.x})

    hrqsf = hrqpf * slr
    dqpf = hrqpf * hr.prism
    dqsf = dqpf * slr

    hr['qpf'] = hrqpf
    hr.attrs['qpf_units'] = 'mm'

    hr['qsf'] = hrqsf
    hr.attrs['qsf_units'] = 'mm'

    hr['dqpf'] = dqpf
    hr.attrs['dqpf_units'] = 'mm'

    hr['dqsf'] = dqsf
    hr.attrs['dqsf_units'] = 'mm'
    
    hr.attrs['slr_units'] = 'ratio(qpf/qsf)'
    hr.attrs['wbzh_units'] = 'm'
    hr.attrs['prism_units'] = 'orographic_precipitation_ratio'

    try:
        del hr['level']
    except:
        pass

    lats, lons = np.unique(hr['lat']), np.unique(hr['lon'])
    del hr['lat'], hr['lon']
    hr = hr.rename({'x':'lon', 'y':'lat'})
    hr['lat'], hr['lon'] = lats, lons
    
    # # Write the file with a time dimension re-added to concat later
    fo = temp + '{}_{}_{:02d}00_{:03d}.nc'.format(
        model, init_req[:-2], int(init_req[-2:]), fhr)    
    hr.expand_dims('time').to_netcdf(fo)
    
    del hrqpf, hrqsf, dqpf, dqsf
    os.remove(fi)
    ds.close()
    hr.close()
    del hr, ds

    print('%s completed in %d seconds...'%(ti, pytime.time()-t_start))
    
    return fo

def make_plots(fhrdata, model, imgdir, domain='WE'):
    from os import remove as rm
    from subprocess import call
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib import colors
    from mpl_toolkits.basemap import Basemap, maskoceans
    import matplotlib.image as mimage
    plt.style.use('classic')
    import matplotlib
    matplotlib.use('Agg')

    tinfo, data = fhrdata
    init, init_req, fhr, plotstep = tinfo
    var = data.name

    valid = datetime.strftime(pd.to_datetime(data.time.values[-1]), '%Y-%m-%d %H:%M UTC')
    valid_fname = datetime.strftime(pd.to_datetime(data.time.values[-1]), '%Y%m%d%H')

    lons, lats = data.lon.values, data.lat.values
    
    print(data)

    if var in ['qpf', 'dqpf']:
        vals = data.sum(dim='time').values/25.4 
        levs, norml, cmap, tickloc, ticks = (
            qpflevs, qpfnorml, qpfcmap, qpflevloc, qpfticks)
        tacc = fhr if fhr <= plotstep else plotstep
        titlevar = '%dH ACC DOWNSCALED QPF'%tacc if var[0] == 'd' else '%dH ACC QPF'%tacc
        clab = 'QPF [in]'
        
    elif var in ['qsf', 'dqsf']:
        vals = data.sum(dim='time').values/25.4 
        levs, norml, cmap, tickloc, ticks = (
            snowlevs, snownorml, snowcmap, snowlevloc, snowticks)
        tacc = fhr if fhr <= plotstep else plotstep
        titlevar = '%dH ACC DOWNSCALED SNOW'%tacc if var[0] == 'd' else '%dH ACC SNOW'%tacc
        clab = 'SNOW [in]'
        
    elif var == 'slr':
        vals = data.sel(time=valid).values
        levs, norml, cmap, tickloc, ticks = (
            slrlevs, slrnorml, slrcmap, slrlevloc, slrticks)
        titlevar = 'INSTANTANEOUS SLR'
        clab = 'SLR\nWhite where SLR < 2.5'
        
    elif var in ['wbzh_agl', 'wbzh']:                   
        vals = data.sel(time=valid).values
        vals[vals < 0.] = 0.
        
        levs, norml, cmap, tickloc, ticks = (
            wbzlevs, wbznorml, wbzcmap, wbzlevloc, wbzticks)
        vmin, vmax = wbzlevs[0], wbzlevs[-1]

        clab = '\nHEIGHT AMSL [m]\nWhite where ≤ 100 m'
        clab_agl = '\nHEIGHT AGL [m]\nWhite where ≤ 100 m'
        clab = clab_agl if var == 'wbzh_agl' else clab
        
        titlevar = 'INSTANTANEOUS WB0.5 HEIGHT AMSL'
        titlevar_agl = 'INSTANTANEOUS WB0.5 HEIGHT ABOVE TERRAIN'
        titlevar = titlevar_agl if var == 'wbzh_agl' else titlevar
        
    title = 'filter type: %s sigma override: %s\n%s INIT: %s\n\n%s\nFHR %02d VALID: %s\n'%(
        grid_filter, sigma_override, model[:-2], init, titlevar, fhr, valid)

    # Initialize the figure frame
    fig = plt.figure(num=None, figsize=(16.0/1.5, 12.0/1.5), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    # Give the plot some reasonable title
    ax.set_title(title)

    # Initialize the basemap object
    if domain:
        minlon, maxlon, minlat, maxlat = map_regions[domain]
    else:
        minlon, maxlon, minlat, maxlat = lons.min(), lons.max(), lats.min(), lats.max()
        domain = 'WE'

    bmap = Basemap(
        llcrnrlon=minlon-0.01,
        llcrnrlat=minlat-0.01,
        urcrnrlon=maxlon+0.01,
        urcrnrlat=maxlat+0.01,
        resolution='i',
        # Project as 'aea', 'lcc', or 'mill'
        projection='mill')

    # Plot geography
    bmap.drawcoastlines(linewidth=1.0, color='black')
    bmap.drawcountries(linewidth=0.85, color='black')
    bmap.drawstates(linewidth=1.25, color='black')
    
    # Project the lat/lon grid
    meshlon, meshlat = np.meshgrid(lons, lats)
    X, Y = bmap(meshlon, meshlat)

    # Mask the oceans, inlands=False to prevent masking lakes (e.g. GSL)
    vals = maskoceans(meshlon, meshlat, vals, inlands=False)#, grid=1.25)

    if var in ['wbzh_agl', 'wbzh']:
        # bmap.contour(X, Y, vals, linestyles='-', levels=levs, colors='k', linewidths=0.4, alpha=0.75)
        cb = bmap.contourf(X, Y, vals, levels=levs, vmin=vmin, vmax=vmax, norm=norml, 
                            extend='both', cmap=cmap, alpha=0.8)
        cb.cmap.set_under('white')
        cb.cmap.set_over('#662506')

    else:
        # bmap.contour(X, Y, vals, linestyles='-', levels=levs, colors='k', linewidths=0.4, alpha=0.75)
        cb = bmap.contourf(X, Y, vals, levels=levs, cmap=cmap, norm=norml, alpha=0.8, extend='both')
        cb.cmap.set_under('white')
        if var == 'slr':
            cb.cmap.set_over('#040e2a')
    
    cbar = plt.colorbar(cb, extend='both', label='\n'+clab)
    cbar.set_ticks(tickloc)
    cbar.set_ticklabels(ticks)
    cbar.ax.tick_params(size=1)

    # Add the U watermark
    logo = mimage.imread(chpcdir + 'Ulogo_400p.png')
    axs_water = [0.15, 0.12, 0.10, 0.15]
    ax_water = fig.add_axes(axs_water)
    ax_water.axis('off')
    ax_water.imshow(logo, aspect='auto', zorder=10, alpha=0.30)

    webvar = webvars[var]
    webvar = webvar if plotstep == 24 else webvar+'S'
    filename = '{}{}_{}{}F{:03d}.png'.format(model, webvar, domain, init_req, fhr)
    filepath = imgdir + filename

    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    filepath_gif = filepath[:-4] + '.gif'
    call('convert ' + filepath + ' ' + filepath_gif, shell=True)
    rm(filepath)

    print('Saved %s'%filepath_gif.split('/')[-1])

    data.close()
    del data
    gc.collect()

    return None
    
