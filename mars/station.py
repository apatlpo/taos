# qsub station.pbs
#

import os, sys

import numpy as np
import xarray as xr
import pandas as pd

import dask
from dask.distributed import performance_report
from dask import delayed

import taos.utils as ut
import taos.mars as ms

def combine_files(df, _preprocess):
    with dask.config.set(scheduler="threads"):    
        ds = xr.open_mfdataset(df["files"],
                               concat_dim="time", 
                               preprocess=_preprocess, 
                               combine="nested",
                               coords="minimal",
                               compat="override",
                              )
        return ds

combine_files_delayed = delayed(combine_files)

def process_top(files, label, _preprocess, freq="5D", overwrite=False):
    
    file_out = os.path.join(ms.diag_dir, "station_{}.nc".format(label))
        
    if not os.path.isfile(file_out) or overwrite:

        delayed_outputs = [combine_files_delayed(g, _preprocess) 
                           for label, g in files.groupby(pd.Grouper(freq=freq))
                          ]
        outputs = dask.compute(delayed_outputs)

        ds = (xr.combine_nested(outputs[0], 
                                concat_dim=["time"], 
                                coords="minimal",
                                compat="override",                         
                                #combine_attrs="override",
                               )
              .chunk(dict(time=-1))
              )
        # add vertical coordinate and eos variables
        _z = ms.get_z(ds).transpose("time", "level")
        ds = ds.assign_coords(z = _z)
        ds = ms.add_eos(ds)

        ds.to_netcdf(file_out, mode="w")
        # could append to a single zarr
    
    print("{} done".format(label))

if __name__ == '__main__':

    cluster, client = ut.spin_up_cluster("local", n_workers=7)
    print(client)
    
    # station position
    lon=-.6
    lat=49.7
    
    files = ms.browse_files()
    files = files.loc["2011":]
    print("Number of data files = {} ".format(len(files)))
    
    # select one location
    f = files.iloc[0]["files"]
    ds = xr.open_dataset(f)
    idx = ms.get_horizontal_indices(ds, lon=lon, lat=lat)
    
    def _preprocess(ds):
        ds = (ds.sel(**idx["rho"], **idx["u"], **idx["v"])
              .drop_dims(["ni_f", "nj_f"])
             )
        exclude = ["adapt_imp_ts", "adapt_imp_uz", "adapt_imp_vz"]
        exclude = [v for v in exclude if v in ds]
        ds = ds.drop_vars(exclude)
        return ds
    
    i=0
    for label, _files in files.groupby(pd.Grouper(freq="M")):
        process_top(_files, label.strftime("%Y%m"), _preprocess, freq="5D", overwrite=True)
        i+=1

    # create empty file to indicate processing was completed
    #with open(log_file, "w+") as f:
    #    pass

    print("Congrats, processing is over !")