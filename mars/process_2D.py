# qsub process_2D.pbs
#

import os

import numpy as np
import xarray as xr
import pandas as pd

from tqdm import tqdm

import taos.utils as ut
import taos.mars as ms


def potential_energy_anomaly(ds):
    # a bit inaccurate numerically
    g = 9.81
    h = (ds.sigma0*0+1).integrate("level")
    rho_bar = (ds.sigma0).integrate("level")/h
    Ep = (g*(ds.sigma0-rho_bar)*ds.z).integrate("level")/h
    return Ep.rename("phi")

if __name__ == '__main__':

    cluster, client = ut.spin_up_cluster("local", n_workers=7)
    print(client)
    
    files = ms.browse_files()
    print("Number of data files = {} ".format(len(files)))
    
    zarr = os.path.join(ms.diag_dir, "fields_2d.zarr")

    i=0
    for label, _files in files.groupby(pd.Grouper(freq="4D")):

        _kwargs= dict(z=True, eos=True, chunks=dict(time=1, level=-1))
        ds = xr.concat([ms.read_one_file(f, **_kwargs) for f in _files["files"]], 
                       dim="time",
                       coords="minimal",
                       compat="override",
                      )

        ds2D = xr.merge([ds[["U", "V", "XE"]],
                         potential_energy_anomaly(ds),
                        ]
                        +[ds[v].sel(level=0, method="nearest").rename(v+"_surf")
                          for v in ["UZ", "VZ", "SAL", "TEMP", "sigma0"]]
                       )
        ds2D = ds2D.drop_vars("z")
        ds2D = ds2D.chunk(dict(time=24*2))

        label.strftime("%Y%m%d")

        if i==0:
            ds2D.to_zarr(zarr, mode="w")
        else:
            ds2D.to_zarr(zarr, mode="a", append_dim="time")

        print("{} done".format(i))

        i+=1

    print("Congrats, processing is over !")