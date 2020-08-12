

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

print('Script complete')