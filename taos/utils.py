import os

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.colors as colors
import matplotlib.cm as cmx

import pytide

from .mars import diag_dir, rotate, get_grid_angle

# ---------------------------- tides ------------------------------------

def _harmonic_analysis(data, f=None, vu=None, wt=None):
    """ core method to be wrapped by harmonic_analysis
    """
    hanalysis = lambda data: wt.harmonic_analysis(data, f, vu)
    return np.apply_along_axis(hanalysis, -1, data)

def harmonic_analysis(da, constituents=[]):
    """ Distributed harmonic analysis
    
    Parameters
    ----------
    da: xr.Dataarray
    
    """
    #constituents = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "S1", "M4"]
    wt = pytide.WaveTable(constituents) # not working on months like time series, need to restrict
    da = da.dropna("time")
    time = da.time.values.astype("datetime64")
    f, vu = wt.compute_nodal_modulations(time)
    a = xr.apply_ufunc(_harmonic_analysis, da,
                       input_core_dims=[["time"],],
                       kwargs={"f": f, "vu": vu, "wt": wt},
                       output_core_dims=[["constituent"]],
                       vectorize=False,
                       dask="parallelized",
                       dask_gufunc_kwargs=dict(output_sizes=dict(constituent=f.shape[0])),
                       output_dtypes=[np.complex128]
                      )
    a = a.assign_coords(constituent=("constituent", wt.constituents()),
                        frequency=("constituent", wt.freq()*86400/2/np.pi),
                        frequency_rad=("constituent", wt.freq()), 
                       )
    return a.rename("amplitudes")

def predict_tides(time,
                  har=None,
                  realimag=None,
                  real=True, 
                  summed=True,
                  suffix="",
                  name="x",
                  constituents=None,
                  ignore=None,
                 ):
    """ Predict tides based on pytide outputs
    
    v = Re ( conj(amplitude) * dsp.f * np.exp(1j*vu) ) 
    
    see: https://pangeo-pytide.readthedocs.io/en/latest/pytide.html#pytide.WaveTable.harmonic_analysis
    
    Parameters
    ----------
    time: xr.DataArray
        Target time
    har: xr.DataArray, xr.Dataset, optional
        Complex amplitudes. Load constituents from a reference station otherwise
    realimag: tuple
        Contains real and imaginary part variable names when har is a dataset
    real: boolean, optional
        returns a real time series if True (default)
    summed: boolean, optional
        returns the sum over constituent contributions if True (default)
    """

    if har is None:
        dsh = xr.open_dataset(os.path.join(diag_dir,"station_tide.nc"))
        har = (dsh.amplitude_real + 1j * dsh.amplitude_imag).rename("amplitude")
        har = har.assign_coords(frequency=dsh["frequency"])
    else:
        if isinstance(har, xr.Dataset):
            assert realimag, "You need to specify real and imaginary names via realimag kwarg"
            _har = har[realimag[0]] + 1j * har[realimag[1]]
            har = _har.assign_coords(frequency=har["frequency"]).rename(name)
        else:
            assert isinstance(har, xr.DataArray), "har should be an xarray Dataset or DataArray"
            name = har.name
    if suffix=="":
        suffix = name+"_"

    constituents = list(har.constituent.values)    
    wt = pytide.WaveTable(constituents) 

    time = time.data.astype("datetime64")
    f, vu = wt.compute_nodal_modulations(time)    
        
    dsp = har.to_dataset().assign_coords(time=("time", time))
    _time = [(pd.Timestamp(t)-pd.Timestamp(1970,1,1)).total_seconds() for t in dsp.time.data]
    dsp = dsp.assign_coords(time_seconds=("time", _time))
    dsp["f"] = (("constituent", "time"), f)
    dsp["vu"] = (("constituent", "time"), vu)
    cplx =  dsp.f * np.exp(1j*dsp.vu) * np.conj(dsp[name])
    dsp[suffix+"real"] = np.real(cplx)
    dsp[suffix+"imag"] = np.imag(cplx)
    
    if constituents is not None:
        dsp = dsp.sel(constituent=constituents)
    if ignore is not None:
        dsp = dsp.drop_sel(constituent=ignore)
    
    if summed:
        dsp = dsp.sum("constituent").drop_vars(["f", "vu"])
    else:
        dsp["frequency"] = har["frequency"]
        
    if real:
        dsp = dsp[suffix+"real"]
    else:
        dsp[suffix+"prediction"] = dsp[suffix+"real"]
        dsp[suffix+"prediction_quad"] = dsp[suffix+"imag"]
        
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

def get_ellipse_properties(u, v):
    """ Compute tidal ellipse properties
    
    Parameters
    ----------
    u, v: xr.DataArray
        velocity complex amplitudes
        u(t) = Re( conj(u) e^{i\omega t} )
    """
    wp = (np.conj(u) + 1j*np.conj(v))/2
    wn = (u+1j*v)/2
    Wp, Wn = np.abs(wp), np.abs(wn)
    A = (Wp + Wn).rename("A")
    a = (Wp - Wn).rename("a")
    thetap = xr.apply_ufunc(np.angle, wp, dask="parallelized")
    thetan = xr.apply_ufunc(np.angle, wn, dask="parallelized")
    inclinaison = ((thetap+thetan)/2).rename("inclinaison")
    phase = ((-thetap+thetan)/2).rename("phase")
    return xr.merge([A, a, inclinaison, phase])

def compute_plot_ellipses(u,v,lon,lat,xgrid,
                          dij=10, 
                          lon_ref=-.6, lat_ref=49.3, 
                          u_ref=1., 
                          v_ref=0.1*1j,
                          T = .5 * 12/2/np.pi/24*86400,
                         ):

    u = xgrid.interp(u, "x")
    v = xgrid.interp(v, "y")

    phi = get_grid_angle(lon, lat)
    u, v = rotate(u, v, phi)

    slices = dict(ni=slice(0,None,dij), nj=slice(0,None,dij))
    u = u.isel(**slices).rename("u")
    v = v.isel(**slices).rename("v")
    lat = lat.isel(**slices)

    ds = xr.merge([u, v]).chunk(dict(ni=-1, nj=-1))
    ds = ds.assign_coords(time=("time", np.arange(0, 2*np.pi, .1)),
                          lon=lon, lat=lat,
                          lon_scale = 111e3*np.cos(np.pi/180*lat),
                          lat_scale = 111e3*(1+lat*0),
                         )
    ds = ds.stack(point=["ni", "nj"])

    ds["x"] = ds.lon + T*np.real( np.conj(ds.u) *np.exp(1j*ds.time) )/ds.lon_scale
    ds["y"] = ds.lat + T*np.real( np.conj(ds.v) *np.exp(1j*ds.time) )/ds.lat_scale

    lon_scale = 111e3*np.cos(np.pi/180*lat_ref)
    lat_scale = 111e3
    ds["x_ref"] = lon_ref + T*np.real( np.conj(u_ref) *np.exp(1j*ds.time) )/lon_scale
    ds["y_ref"] = lat_ref + T*np.real( np.conj(v_ref) *np.exp(1j*ds.time) )/lat_scale
    ds.attrs.update(u_ref=u_ref, v_ref=v_ref)

    return ds

def plot_ellipses(ax, ds, markersize=1):
    ax.plot(ds["x"].T, ds["y"].T, color="k", transform=ms.ccrs.PlateCarree())
    # high tide
    ax.plot(ds["x"].sel(time=0), 
            ds["y"].sel(time=0), 
            "o", color="r", markersize=markersize,
            transform=ms.ccrs.PlateCarree())
    # 90deg (3 hours) before high tide
    ax.plot(ds["x"].sel(time=3*np.pi/2, method="nearest"), 
            ds["y"].sel(time=3*np.pi/2, method="nearest"),
            "o", color="orange", markersize=markersize,
            transform=ms.ccrs.PlateCarree())
    #
    ax.plot(ds.x_ref, ds.y_ref, color="k", transform=ms.ccrs.PlateCarree(), zorder=20)
    ax.text(ds.x_ref[0], ds.y_ref[0], "{} m/s".format(ds.u_ref), 
            verticalalignment="center", transform=ms.ccrs.PlateCarree(), zorder=20)

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

