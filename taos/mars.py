import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

import threading

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo

import cmocean.cm as cm

import gsw

from xgcm import Grid

diag_dir = "/home/datawork-lops-osi/aponte/taos/mars"

# -----------------------------

def browse_files(year=None, daily=False):
    """ Browse simulation directory and figure out output files
    """
    if daily:
        root_dir = "/home/datawork-lops-oc/MARC/MARC_F1-MARS3D-SEINE_FILTRE/"
    else:
        root_dir = "/home/ref-oc-public/modeles_marc/f1_e2500_agrif/MARC_F1-MARS3D-SEINE/best_estimate/"
    suff = "MARC_F1-"
    if year is not None:
        dpath = os.path.join(root_dir, str(year), suff+"*.nc")
    else:
        dpath = os.path.join(root_dir, "*/"+suff+"*.nc")
    files = sorted(glob(dpath))
    dates = [extract_date(f, daily) for f in files]
    df = pd.DataFrame({"files": files}, index=dates)
    return df

def extract_date(f, daily):
    """ extract date from file name
    """
    if daily:
        # MARC_F1-MARS3D-SEINE_20100702T1200Z_FILTRE.nc
        _date = f.split("/")[-1].split("_")[-2].replace(".nc","")
    else:
        _date = f.split("/")[-1].split("_")[-1].replace(".nc","")
    return pd.Timestamp(_date)

def load_date(date):
    """ Load one file from date
    
    Parameters
    ----------
    date: pd.Timestamp, str
        Selected date
    """
    
    # convert to date
    if isinstance(date, str):
        date = pd.Timestamp(date)

    files = browse_files(date.year)
    #print("Number of data files = {} ".format(len(files)))    

    ds = xr.open_dataset(files.loc[date, "files"])

    # add vertical coordinate and eos variables
    ds = ds.assign_coords(z = get_z(ds).transpose("time", "level", "nj", "ni"))
    ds = add_eos(ds)

    return ds

def read_one_file(f, 
                  i=None, 
                  j=None, 
                  z=True, 
                  eos=True,
                  drop=True,
                  chunks={"level": 1}
                 ):
    """ Read one netcdf mars file
    
    Parameters
    ----------
    f: str
        path to netcdf file
    i, j: int
        index for slicing along i or j directions
    z: boolean, True by default
        add depth coordinate
    eos: boolean, True by default
        add equations of station variables
    drop; boolean, True by default
        drops variables that are not in all netcdf files
    """
    
    ds = xr.open_dataset(f, chunks=chunks)
    ds = _rename_dims(ds)
    
    if i is not None:
        ds = ds.isel(ni=i)
    if j is not None:
        ds = ds.isel(nj=j)
        
    if z:
        ds = ds.assign_coords(z = get_z(ds))

    ds = ds.transpose("time", "level", "nj", "nj_v", "ni", "ni_u")    

    if eos:
        ds = add_eos(ds)
        
    if drop is True or isinstance(drop, list):
        drop_vars = ["adapt_imp_ts", "adapt_imp_uz", "adapt_imp_vz"]
        if isinstance(drop, list):
            drop_vars = drop
        ds = ds.drop_vars(drop_vars, errors="ignore")
    
    return ds

def _rename_dims(ds):
    """ Some dimensions are not necessary and may complicate the use of xgcm
    """
    ds = ds.copy()
    for v in ds.reset_coords():
        if "ni_v" in ds[v].dims:
            ds[v] = ds[v].rename(ni_v="ni")
        if "nj_u" in ds[v].dims:
            ds[v] = ds[v].rename(nj_u="nj")
        if "ni_f" in ds[v].dims:
            ds[v] = ds[v].rename(ni_f="ni_u")
        if "nj_f" in ds[v].dims:
            ds[v] = ds[v].rename(nj_f="nj_v")
    ds = ds.drop_dims(["ni_v", "nj_u", "ni_f", "nj_f"])
    return ds

def get_z(ds, s=None):
    # z(n,k,j,i) = eta(n,j,i)*(1+s(k)) + depth_c*s(k) + (depth(j,i)-depth_c)*C(k)
    if s is None:
        s = ds.SIG
    eta, depth, depth_c, C  = ds.XE, ds.H0, ds.hc, ds.Csu_sig
    # fill land values with 0
    eta = eta.fillna(0.)
    depth = depth.fillna(1.)
    depth_c = depth_c.fillna(0.)
    return eta*(1+s) + depth_c*s + (depth-depth_c)*C

def get_horizontal_indices(ds, lon=-.6, lat=49.7):
    """ returns grid index of a position
    
    Parameters
    ----------
    ds: xr.Dataset
        dataset containing grid information ("longitude", "longitude_u", ...)
    lon, lat: float, optional
        Coordinate of the point of interest, default: lon=-.6, lat=49.7
    """
    coords = {}
    for v, suffix in zip(["rho", "u", "v"], ["","_u","_v"]):
        dl2 = (ds["longitude"+suffix]-lon)**2 *np.sin(np.pi/180.*lat)**2 \
            +(ds["latitude"+suffix]-lat)**2
        _min = dl2.where(dl2==dl2.min(), drop=True)
        coords[v] = {c:_min[c].values[0] for c in _min.dims}
    coords["position"] = (lon, lat)
    return coords

def get_xgrid(ds):
    """ Create xgcm grid object
    """ 
    coords={'x': {'center':'ni', 'left':'ni_u'}, 
            'y': {'center':'nj', 'left':'nj_v'}, 
            's': {'center':'level', 'outer':'level_w'}}
    xgrid = Grid(ds, periodic=False, coords=coords, boundary='extend')
    return xgrid

def get_ij_dims(da):
    i = next((d for d in da.dims if "ni" in d))
    j = next((d for d in da.dims if "nj" in d))
    return i, j

def get_grid_angle(lon, lat):
    """ get grid orientation
    """
    #lon, lat = ds.longitude, ds.latitude
    sc = np.cos(np.pi/180*lat)
    tan = (lat.shift(ni=-1) - lat)/(lon.shift(ni=-1) - lon)/sc
    phi = np.arctan(tan)
    return phi

def rotate(u, v, phi):
    _u = np.cos(phi)*u - np.sin(phi)*v
    _v = np.sin(phi)*u + np.cos(phi)*v
    return _u, _v

# -----------------------------

def add_eos(ds, S="SAL", PT="TEMP"):
    assert "z" in ds, "You must first add the vertical coordinate z with get_z"
    lon, lat = 0, 49.5
    ds["SA"] = (ds[S].dims, gsw.SA_from_SP(ds[S].data, -ds.z.data, lon, lat))
    ds["CT"] = (ds[S].dims, gsw.CT_from_pt(ds[S].data, ds[PT].data))
    ds["sigma0"] = (ds[S].dims, gsw.sigma0(ds[S].data, ds[PT].data))
    return ds

def TS_plot(ds, t_range=None, s_range=None, figsize=(5, 5)):
    
    if t_range is None:
        t_range = (ds.TEMP.min().values, ds.TEMP.max())
    if s_range is None:
        s_range = (ds.SAL.min().values, ds.SAL.max())
    
    nt, ns = 100, 100
    _ds = xr.Dataset(coords=dict(theta=(("theta",), np.linspace(t_range[0], t_range[1], num=nt)),
                                 salinity=(("salinity",), np.linspace(s_range[0], s_range[1], num=ns)),
                                ))
    _ds["salinity2"], _ds["theta2"] = xr.broadcast(_ds["salinity"], _ds["theta"])
    _ds["z"] = 0.*_ds["salinity2"]
    _ds = add_eos(_ds, S="salinity2", PT="theta2")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    CS = _ds.sigma0.plot.contour(y="theta", x="salinity", ax=ax, levels=20, add_labels=True, cmap=cm.dense)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.grid()
    
    return fig, ax

def potential_energy_anomaly(ds):
    # a bit inaccurate numerically
    g = 9.81
    h = (ds.sigma0*0+1).integrate("level")
    rho_bar = (ds.sigma0).integrate("level")/h
    Ep = (g*(ds.sigma0-rho_bar)*ds.z).integrate("level")/h
    return Ep.rename("phi")

# -----------------------------

def get_cmap_colors(Nc, cmap="plasma"):
    """load colors from a colormap to plot lines

    Parameters
    ----------
    Nc: int
        Number of colors to select
    cmap: str, optional
        Colormap to pick color from (default: 'plasma')
    """
    scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=Nc), cmap=cmap)
    return [scalarMap.to_rgba(i) for i in range(Nc)]

def plot_bs(da, 
            uv=None,
            zoom=0, 
            title=None,
            fig=None,
            ax=None,
            colorbar=True,
            colorbar_kwargs={},
            center_colormap=False,
            gridlines=True,
            dticks=(1, 1),
            land=True,
            coast_resolution="10m",
            offline=False,
            figsize=0,
            savefig=None,
            **kwargs,
           ):
    
    #
    MPL_LOCK = threading.Lock()
    with MPL_LOCK:
        
        if offline:
            plt.switch_backend("agg")
        
        if figsize==0:
            _figsize = (10, 5)
        elif figsize==1:
            _figsize = (20, 10)
        else:
            _figsize = figsize
        if fig is None and ax is None:
            fig = plt.figure(figsize=_figsize)
            ax = fig.add_subplot(111, projection=ccrs.Orthographic(0., 49.5))
        
        if center_colormap:
            vmax = float(abs(da).max())
            vmin = -vmax
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax        
        
        im = (da
             .squeeze()
             .plot
             .pcolormesh(x="longitude", y="latitude", 
                         ax=ax,
                         transform=ccrs.PlateCarree(),
                         add_colorbar=False,
                         **kwargs,
                        )
            )
        
        if uv is not None:
                dij = uv[1]
                u = uv[0].u.isel(ni=slice(0,None, dij), nj=slice(0,None, dij))
                v = uv[0].v.isel(ni=slice(0,None, dij), nj=slice(0,None, dij))
                u_src_crs = u / np.cos(u.latitude / 180 * np.pi)
                v_src_crs = v
                magnitude = np.sqrt(u**2 + v**2)
                magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
                _kwargs = dict(transform = ccrs.PlateCarree(), units="width", scale=5)
                if len(uv)>2:
                    _kwargs.update(**uv[2])
                Q = ax.quiver(u.longitude.values, u.latitude.values,
                              (u_src_crs * magnitude / magn_src_crs).values, 
                              (v_src_crs * magnitude / magn_src_crs).values,
                              **_kwargs,
                             )
                # for reference arrow, should use:
                # https://matplotlib.org/stable/gallery/images_contours_and_fields/quiver_demo.html
                if len(uv)>3:
                    uv_ref = uv[3]
                else:
                    uv_ref = 0.5
                qk = ax.quiverkey(Q, 0.3, 0.2, uv_ref, r'{:.2f} m/s'.format(uv_ref),
                                  labelpos='E',
                                  coordinates='figure',
                                  transform = ccrs.PlateCarree(),
                                 )
        
        if isinstance(zoom, int):
            if zoom==0:
                _extent = ax.get_extent()
                set_extent = False
            elif zoom==1:
                _extent = [-1., 0.2, 49.25, 49.7]
                set_extent = True
        elif isinstance(zoom, list):
            _extent = zoom
            set_extent = True

        # coastlines and land:
        if land:
            if isinstance(land, dict):
                #land = dict(args=['physical', 'land', '10m'],
                #            kwargs= dict(edgecolor='face', facecolor=cfeature.COLORS['land']),
                #           )
                land_feature = cfeature.NaturalEarthFeature(*land['args'], 
                                                            **land['kwargs'],
                                                           )
            else:
                land_feature = cfeature.LAND
            ax.add_feature(land_feature, zorder=10)        
        if coast_resolution is not None:
            ax.coastlines(resolution=coast_resolution, color='k')

        if set_extent:
            ax.set_extent(_extent)
        
        if colorbar:
            #cbar = fig.colorbar(im, extend="neither", shrink=0.7, **colorbar_kwargs)
            axins = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )            
            #cbar = fig.colorbar(im, extend="neither", shrink=0.9, 
            cbar = fig.colorbar(im,
                                extend="neither",
                                cax=axins,
                                **colorbar_kwargs)            
        else:
            cbar = None
            
        if gridlines:
            gl = ax.gridlines(draw_labels=True, dms=False, 
                         x_inline=False, y_inline=False, 
                        )
            gl.right_labels=False
            gl.top_labels=False

        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 12, }) #"fontweight": "bold"
        #
        if savefig is not None:
            fig.savefig(savefig, dpi=150, bbox_inches = "tight")
            plt.close(fig)
        #
        return {"fig": fig, "ax": ax, "cbar": cbar}

def prepare_uv(u, v, ds, xgrid, dij=5, **kwargs):
    """ prepare uv for quiver plot in plot_bs
    """
    # interpolate at the cell center
    uv = (xr.merge([xgrid.interp(u, "x").rename("u"), 
                    xgrid.interp(v, "y").rename("v"),
                   ])
          .assign_coords(longitude=ds.longitude, latitude=ds.latitude),
          dij,
          kwargs,
         )
    return uv

def plot_section(da,
                 x,
                 title=None,
                 fig=None,
                 ax=None,
            colorbar=True,
            colorbar_kwargs={},
            center_colormap=False,
            xlabel=True,
            ylabel=True,
            gridlines=True,
            offline=False,
            figsize=(10,4),
            savefig=None,
            **kwargs,
           ):
    """ Plot a vertical section
    """
    
    #
    MPL_LOCK = threading.Lock()
    with MPL_LOCK:
        
        if offline:
            plt.switch_backend("agg")
        
        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            #ax.set_facecolor("slategray")
            ax.set_facecolor("0.9")
            
        if center_colormap:
            vmax = float(abs(da).max())
            vmin = -vmax
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
            
        im=ax.pcolormesh(da.level*0.+da[x], da.z, da, 
                         shading="gouraud", 
                         **kwargs,
                        )

        ax.invert_xaxis()
        ax.set_ylim(top=0.)
        if not ylabel:
            ax.set_yticklabels([])
        if not xlabel:
            ax.set_xticklabels([])
        
        if colorbar:
            axins = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )            
            #cbar = fig.colorbar(im, extend="neither", shrink=0.9, 
            cbar = fig.colorbar(im,
                                extend="neither",
                                cax=axins,
                                **colorbar_kwargs)
        else:
            cbar = None
        if gridlines:
            ax.grid()

        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 12, }) #"fontweight": "bold"
        #
        if savefig is not None:
            fig.savefig(savefig, dpi=150) #bbox_inches = "tight"
            plt.close(fig)
        #
        return {"fig": fig, "ax": ax, "cbar": cbar}
    
