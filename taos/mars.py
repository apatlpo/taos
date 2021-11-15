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


# -----------------------------

def browse_files(year=None):
    """ Browse simulation directory and figure out output files
    """
    root_dir = "/home/ref-oc-public/modeles_marc/f1_e2500_agrif/MARC_F1-MARS3D-SEINE/best_estimate/"
    suff = "MARC_F1-"
    if year is not None:
        dpath = os.path.join(root_dir, str(year), suff+"*.nc")
    else:
        dpath = os.path.join(root_dir, "*/"+suff+"*.nc")
    files = sorted(glob(dpath))
    dates = [extract_date(f) for f in files]
    df = pd.DataFrame({"files": files}, index=dates)
    return df

def extract_date(f):
    """ extract date from file name
    """
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

def read_one_file(f, i=None, j=None, z=None, drop=True):
    """ Read one netcdf file
    """
    
    ds = xr.open_dataset(f, chunks={"level": 1})
    
    if i is not None:
        ds = ds.isel(ni=i)
    if j is not None:
        ds = ds.isel(nj=j)
        
    if z is not None:
        ds = ds.assign_coords(z = get_z(ds))
        
    if drop is True or isinstance(drop, list):
        drop_vars = ["adapt_imp_ts", "adapt_imp_uz", "adapt_imp_vz"]
        if isinstance(drop, list):
            drop_vars = drop
        ds = ds.drop_vars(drop_vars, errors="ignore")
    return ds

def get_z(ds):
    # z(n,k,j,i) = eta(n,j,i)*(1+s(k)) + depth_c*s(k) + (depth(j,i)-depth_c)*C(k)
    eta, s, depth, depth_c, C  = ds.XE, ds.SIG, ds.H0, ds.hc, ds.Csu_sig
    # fill land values with 0
    eta = eta.fillna(0.)
    depth = depth.fillna(1.)
    depth_c = depth_c.fillna(0.)
    return eta*(1+s) + depth_c*s + (depth-depth_c)*C



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
        
        if zoom==0:
            _extent = ax.get_extent()
        elif zoom==1:
            _extent = [-1., 0.2, 49.25, 49.7]
            
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
            ax.add_feature(land_feature, zorder=0)        
        if coast_resolution is not None:
            ax.coastlines(resolution=coast_resolution, color='k')

        if zoom>0:
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
    
