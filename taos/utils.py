import os

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.colors as colors
import matplotlib.cm as cmx

import pytide

from .mars import diag_dir

# ---------------------------- tides ------------------------------------

def predict_tides(time, real=True, summed=True):
    
    wt = pytide.WaveTable() # not working on months like time series, need to restrict
    # to restrict the constituents used
    #wt = pytide.WaveTable(["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "S1", "M4"]) 

    time = time.data.astype("datetime64[us]")
    f, vu = wt.compute_nodal_modulations(time)    
    
    dsh = xr.open_dataset(os.path.join(diag_dir,"station_tide.nc"))
    dsh["amplitude"] = dsh.amplitude_real + 1j * dsh.amplitude_imag
    dsp = dsh.assign_coords(time=("time", time))
    _time = [(pd.Timestamp(t)-pd.Timestamp(1970,1,1)).total_seconds() for t in dsp.time.data]
    dsp = dsp.assign_coords(time_seconds=("time", _time))
    dsp["f"] = (("constituent", "time"), f)
    dsp["vu"] = (("constituent", "time"), vu)
    dsp["complex"] =  dsp.f * np.exp(1j*dsp.vu) * np.conj(dsp.amplitude)
    dsp["real"] = np.real(dsp.complex)
    dsp["imag"] = np.imag(dsp.complex)
    
    if summed:
        dsp = dsp.sum("constituent").drop_vars(["f", "vu"])
    if real:
        return dsp["real"]
    else:
        dsp["prediction"] = dsp["real"].sum("constituent")
        dsp["prediction_quad"] = dsp["imag"].sum("constituent")
        return dsp


def compute_tidal_range(da, window=1):
    """ Compute tidal range over a rolling window
    
    Parameters
    ----------
    da: xr.DataArray
        input data
    window: float, optional
        Size of the window in days, default is 1
    """
    dt = float((da.time[1]-da.time[0]).dt.seconds)/86400 # days
    delta = round(window/dt)
    h_max = da.rolling(time=delta, center=True).max()
    h_min = da.rolling(time=delta, center=True).min()
    return (h_max-h_min).rename("range")

# ---------------------------- dask related ------------------------------------

def spin_up_cluster(type=None, **kwargs):
    """ Spin up a dask cluster ... or not
    Waits for workers to be up for distributed ones
    Paramaters
    ----------
    type: None, str
        Type of cluster: None=no cluster, "local", "distributed"
    """

    if type is None:
        return
    elif type=="local":
        from dask.distributed import Client, LocalCluster
        dkwargs = dict(n_workers=14, threads_per_worker=1)
        dkwargs.update(**kwargs)
        cluster = LocalCluster(**dkwargs) # these may not be hardcoded
        client = Client(cluster)
    elif type=="distributed":
        from dask_jobqueue import SLURMCluster
        from dask.distributed import Client
        assert "processes" in kwargs, "you need to specify a number of processes"
        processes = kwargs["processes"]
        assert "jobs" in kwargs, "you need to specify a number of dask-queue jobs"
        jobs = kwargs["jobs"]
        dkwargs = dict(cores=28,
                       name='pangeo',
                       walltime='03:00:00',
                       job_extra=['--constraint=HSW24',
                                  '--exclusive',
                                  '--nodes=1'],
                       memory="118GB",
                       interface='ib0',
                       )
        dkwargs.update(**kwargs)
        dkwargs = _removekey(dkwargs, "jobs")
        cluster = SLURMCluster(**dkwargs)
        cluster.scale(jobs=jobs)
        client = Client(cluster)

        flag = True
        while flag:
            wk = client.scheduler_info()["workers"]
            print("Number of workers up = {}".format(len(wk)))
            sleep(5)
            if len(wk)>=processes*jobs*0.8:
                flag = False
                print("Cluster is up, proceeding with computations")

    return cluster, client

def print_graph(da, name, flag):
    """ print dask graph of a DataArray as png
    see: https://docs.dask.org/en/latest/diagnostics-distributed.html
    Parameters
    ----------
    da: dask.DataArray
        Array of which we want the graph
    name: str
        Name used for figure name
    flag: boolean
        turn actual graph printing on/off
    """
    if not flag:
        return
    da.data.visualize(filename='graph_{}.png'.format(name),
                      optimize_graph=True,
                      color="order",
                      cmap="autumn",
                      node_attr={"penwidth": "4"},
                      )

# ---------------------------- misc ------------------------------------

def _removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def _reset_chunk_encoding(ds):
    ''' Delete chunks from variables encoding.
    This may be required when loading zarr data and rewriting it with different chunks
    Parameters
    ----------
    ds: xr.DataArray, xr.Dataset
        Input data
    '''
    if isinstance(ds, xr.DataArray):
        return _reset_chunk_encoding(ds.to_dataset()).to_array()
    #
    for v in ds.coords:
        if 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']
    for v in ds:
        if 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']
    return ds


# -------------------------------- plot -------------------------------

def get_cmap_colors(Nc, cmap='plasma'):
    """ load colors from a colormap to plot lines
    
    Parameters
    ----------
    Nc: int
        Number of colors to select
    cmap: str, optional
        Colormap to pick color from (default: 'plasma')
    """
    scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=Nc),
                                   cmap=cmap)
    return [scalarMap.to_rgba(i) for i in range(Nc)]

