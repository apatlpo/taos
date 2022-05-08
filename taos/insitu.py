import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize, Bounds, shgo

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from datetime import datetime
import time

import pyproj

import urllib.request

from .utils import plot_bs

rad2deg = 180./np.pi
knot = 0.514

one_second = pd.Timedelta("1s")
now = lambda: pd.to_datetime(datetime.utcnow())

# ----------------------------- reference locations ---------------------------------

_stations = {"Epave nord Ouistreham":dict(System="release", lon=-0.246717, lat=49.374433),
             "LSTOuistreham":dict(System="buoy", lon=-0.246754, lat=49.340379),
             "Luc":dict(System="buoy", lon=-0.307225, lat=49.345653),
            }
stations = pd.DataFrame(_stations).T

# ----------------------------- project, metrics ---------------------------------

lonc, latc = -0.4, 49.5
proj = pyproj.Proj(proj="aeqd", lat_0=latc, lon_0=lonc, datum="WGS84", units="m")

def ll2xy(lon, lat):
    return proj.transform(lon, lat)
    
def xy2ll(x, y):
    _inv_dir = pyproj.enums.TransformDirection.INVERSE
    return proj.transform(x, y, direction=_inv_dir)

def metrics_cheatsheet(lon, lat):
    """print useful info for navigation"""
    
    proj = pyproj.Proj(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84", units="m")

    def dl(dlon, dlat):
        x0, y0 = proj.transform(lon, lat)
        x1, y1 = proj.transform(lon+dlon, lat+dlat)
        return np.sqrt((x1-x0)**2+(y1-y0)**2)

    print(f" lon: 1 deg = {dl(1,0)/1e3:.2f}km,  0.1 deg = {dl(.1,0)/1e3:.1f}km"+ \
          f",  0.01 deg = {dl(.01,0):.1f}m,  0.001 deg = {dl(.001,0):.1f}m"
         )
    print(f" lon: 1 deg = {dl(1,0)/1e3:.2f}km,  1 min = {dl(1/60,0):.1f}m"+ \
          f",  .1 min = {dl(.1/60,0):.1f}m,  0.01 min = {dl(.01/60,0):.1f}m"
         )
    print(f" lon: 1 deg = {dl(1,0)/1e3:.2f}km,  1 sec = {dl(1/3600,0):.1f}m"+ \
          f",  .1 sec = {dl(.1/3600,0):.1f}m"
         )
    print("-----------------------------------------------------------------")
    print(f" lat: 1 deg = {dl(0,1)/1e3:.2f}km,  0.1 deg = {dl(0,.1)/1e3:.1f}km"+ \
          f",  0.01 deg = {dl(0,.01):.1f}m,  0.001 deg = {dl(0,.001):.1f}m"
         )
    print(f" lat: 1 deg = {dl(0,1)/1e3:.2f}km,  1 min = {dl(0,1/60):.1f}m"+ \
          f",  .1 min = {dl(0,.1/60):.1f}m,  0.01 min = {dl(0,.01/60):.1f}m"
         )
    print(f" lat: 1 deg = {dl(0,1)/1e3:.2f}km,  1 sec = {dl(0,1/3600):.1f}m"+ \
          f",  .1 sec = {dl(0,.1/3600):.1f}m"
         )
    
def _deg_mindec(v):
    deg = np.trunc(v)
    mind = abs(v - deg)*60
    return deg, mind

def to_deg_mindec(df, lon, lat):
    """add lon/lat in deg/min decimals to a dataframe"""
    df[lon+"_deg"], df[lon+"_min"] = _deg_mindec(df[lon])
    df[lat+"_deg"], df[lat+"_min"] = _deg_mindec(df[lat])
    return df

# ---------------------------- route scheduling ----------------------------------

def _match_displacements(x, dX, Uo, speed, t_wait):
    theta, dt = x
    Us = speed*np.exp(1j*theta)
    return np.abs( dX + Uo*(dt + t_wait) - Us*dt )

def solve_route_heading(dX, speed, Uo, time_waiting, time_max=10, **kwargs):
    """ Solve where we need to head given:
    
    Parameters
    ----------
    dX: complex
        distance relative to initial point
    speed: float
        ship speed (m/s)
    Ui: complex
        ocean current (m/s)
    time_waiting: float
        time we wish to wait at the deployment point (seconds)
    dt_max
    **kwargs: sent to optimizing method
    
    Returns
    -------
    theta: float
        Heading in radians
    dt_route: float
        Time to launch location
    dt_total: float
        Time to launch location + time_waiting
    X_launch: complex
        Launch position
    Us: complex
        Complex ship velocity (i.e. include calculated heading)
    """

    # local optimization
    # this turn out not to work well, don't know why
    #bounds = Bounds([-2*np.pi, 2*np.pi], [0., np.inf])
    # https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization
    #minopts = dict(method='trust-constr', bounds=bounds)
    #minopts = dict(bounds=bounds)
    #minopts.update(**kwargs)
    #res = minimize(_match, [np.angle(dX), time_waiting], args=(dX, Uo, speed, time_waiting), **minopts)

    # global optimization
    bounds = [(-1.1*np.pi, np.pi), (1., 3600*time_max)]
    minopts = dict(sampling_method='sobol') # options=dict(f_tol=1), 
    minopts.update(**kwargs)
    res = shgo(_match_displacements, bounds, (dX, Uo, speed, time_waiting), **minopts)

    theta, dt_route = res["x"]
    Us = speed*np.exp(1j*theta)
    X_launch = Us*dt_route
    
    return theta, dt_route, dt_route+time_waiting, X_launch, Us

def solve_route_heading_xr(label, *args):
    """ wrapper around solve_route_heading that spits a dataset instead"""
    theta, dt_route, dt_total, X_launch, _ = solve_route_heading(*args)
    ds = xr.Dataset(dict(theta=(("route",), [theta*rad2deg]),
                         dt_route=(("route",), [pd.Timedelta(seconds=dt_route)]),
                         dt_total=(("route",), [pd.Timedelta(seconds=dt_total)]),
                         X_launch=(("route",), [X_launch]),
                        )
                   )
    return ds

def plot_route_solution(dX, Us, Uo, dt, time_waiting):
    """ plot route solution"""

    fig, ax = plt.subplots(1,1)

    opt = {'head_width': 50, 'head_length': 50, 'width': 20, 'length_includes_head': False}

    ax.arrow(0, 0, dX.real, dX.imag, fc="k", **opt)
    ax.arrow(0, 0, (dt)*Uo.real, (dt)*Uo.imag, fc="b", **opt)
    ax.arrow(0, 0, (dt+time_waiting)*Uo.real, (dt+time_waiting)*Uo.imag, fc="b", **opt)
    ax.arrow(0, 0, dt*Us.real, dt*Us.imag, fc="orange", **opt)

    ax.set_aspect("equal")
    ax.grid()
    
    

def deployments_route_schedule(lon_ship, lat_ship, ship_speed,
                               lon_vertices, lat_vertices, time_waiting,
                               Uo,
                               lon_a=None, lat_a=None, time_a=None, 
                               skip=0,
                              ):
    """ Compute and schedule a route for drifter deployments
    2 use cases:
    - initial deployment
    - underway deployment (lon_a, lat_a, time_a not None and skip>0)

    Parameters
    ----------
    lon_ship, lat_ship: float
        ship position
    ship_speed: float
        ship cruise speed [m/s]
    lon_vertices, lat_vertices: np.array
        vertex positions
    time_waiting: float
        time we wish to wait at each deployment locations
    Uo: complex
        ocean current in m/s
    lon_a, lat_a, time_a: float, optional
        anchor point (first vertex) position and corresponding time
        If lon_a and lat_a are provided but not time_a, this is the first
        deployment.
        Otherwise, lon_a, lat_a corresponds to a passed position fix and this
        implies skip>0
    skip: int, optional
        Number of vertex (deployments) to skip. 
        Default is 0 i.e. first deployment
    """
    
    # positions (initial, anchor, vertices)    
    x, y = ll2xy(lon_ship, lat_ship)
    #
    if lon_a is None:
        lon_a = lon_ship
    if lat_a is None:
        lat_a = lat_ship
    x_a, y_a = ll2xy(lon_a, lat_a)
    #
    x_v, y_v = ll2xy(lon_vertices, lat_vertices)
    X = x_v + 1j*y_v

    # solve for one segment
    def _route_core(dx, Uo, X_deployment, time_deployment):
        theta_route, dt_route, dt_total, dX_deployment, Us = \
                    solve_route_heading(dx, ship_speed, Uo, time_waiting)
        # update
        X_deployment = X_deployment + dX_deployment
        time_deployment = time_deployment + dt_total*one_second
        # store
        x_deployment, y_deployment = X_deployment.real, X_deployment.imag
        lon_deployment, lat_deployment = xy2ll(x_deployment, y_deployment)
        se = pd.Series(dict(time=time_deployment,
                            #x=x_deployment, y=y_deployment, 
                            lon=lon_deployment, lat=lat_deployment,
                            heading=theta_route,
                            dt_route=dt_route/60,
                            dt_total=dt_total/60,
                           ))
        return se, X_deployment, time_deployment
            
    # "first" deployment is treated separately
    time_initial = now() # time where last position was recorded, to be updated
    # skip points if need be
    if skip==0:
        # deployment at anchor position
        _dX = x_a - x + 1j * (y_a - y)
        Uo_init = 0.*1j
    else:
        X = X[skip:]
        _dX = X[0] - (x + 1j*y) # from current position
        assert time_a is not None, "time_a (and lon_a, lat_a) are required when skip>0"
        _dX += Uo * ((now()-time_a)/one_second) # correct anchor position to account for ocean flow displacement
        Uo_init = Uo
    # store initial position
    se = pd.Series(dict(time=time_initial,
                        #x=x_deployment, y=y_deployment, 
                        lon=lon_ship, lat=lat_ship,
                        heading=np.NaN,
                        dt_route=np.NaN,
                        dt_total=np.NaN,
                       )
                  )
    S = [se]
    # compute route to first deployment
    se, X_deployment, time_deployment = _route_core(_dX, Uo_init, x+1j*y, time_initial)
    S.append(se)
    
    # following deployments
    dX = np.diff(X)
    for dx in dX:
        # could also update velocity depending on time_deployment
        se, X_deployment, time_deployment = _route_core(dx, Uo, X_deployment, time_deployment)
        #print(dx, X_deployment) # debug
        S.append(se)

    # concatenate results
    df = pd.DataFrame(S, index=pd.Index(np.arange(skip-1, skip+X.size), name="deployment"))
    
    # add degrees and minute with decimals
    to_deg_mindec(df, "lon", "lat")
    
    return df

def plot_deployments_route(lon, lat, df, arrows=True, **kwargs):
    """ Plot deployment routes as compute from deployments_route_schedule
    """

    dkwargs = dict(bathy=False, zoom=[-.32, -.2, 49.29, 49.36], vmax=30, figsize=(10,10))
    dkwargs.update(kwargs)
    fac = plot_bs(**dkwargs)
    ax = fac["ax"]

    ax.scatter(lon, lat, 40, color="orange", transform=ccrs.PlateCarree())
    ax.scatter(df.lon, df.lat, 40, color="red", transform=ccrs.PlateCarree())

    # arrows
    if arrows:
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        for i in df.index[:-1]:
            ax.annotate('', xy=(df.loc[i+1, "lon"], df.loc[i+1, "lat"]), 
                        xytext=(df.loc[i, "lon"], df.loc[i, "lat"]),
                        xycoords=transform,
                        size=20,
                        arrowprops=dict(facecolor='0.5', ec = 'none', 
                                        #arrowstyle="fancy",
                                        connectionstyle="arc3,rad=-0.3"),
                       )

    # labels
    for i in range(lon.size):
        ax.text(lon[i]+1e-3, lat[i], f"{i}", transform=ccrs.PlateCarree())   

    ax.set_aspect("equal")
    ax.grid()

# ---------------------------- deployment geometry -------------------------------

def build_polygon(L, theta, N, rotation=1):
    """build a polygon with radius L and rotation angle theta
    
    Parameters
    ----------
    L: float
        Radius
    theta: float
        rotation angle of the first vertex
    N: int
        Number of vertices
    rotation: -1,1
        sense of rotation: 1 = counter-clockwise, -1 = clockwise
    """
    X = L*(np.exp(rotation*1j*2*np.arange(N)*np.pi/N) - 1) * np.exp(1j*theta)
    dX = np.diff(X)
    return X, dX

def build_square_with_center(L, theta, rotation=1, center_loc=1):
    """ build a square with center
    
    Parameters
    ----------
    L: float
        Radius
    theta: float
        rotation angle of the first vertex
    N: int
        Number of vertices
    rotation: -1,1
        sense of rotation: 1 = counter-clockwise, -1 = clockwise
    center_loc: int
        location of the center
    """
    # build square without center
    X, _ = build_polygon(L, theta, 4, rotation=rotation)
    # add center
    X = np.insert(X, center_loc, X.mean())
    # put first vertex at origin
    X = X - X[0]
    return X, np.diff(X)

def plot_polygon(X, dX=None):
    """ plot polygon, not geographic """
    fig, ax = plt.subplots(1,1)
    ax.scatter(X.real, X.imag, s=300, marker=".")

    if dX is not None:
        opt = {'head_width': 50, 'head_length': 50, 'width': 10, 'length_includes_head': False}
        for i in range(len(X)-1):
            ax.arrow(X[i].real, X[i].imag, dX[i].real, dX[i].imag, fc="k", **opt)

    for i in range(X.size):
        ax.text(X[i].real+100, X[i].imag, f"{i}")    

    ax.set_aspect("equal")
    ax.grid()
    
def build_square_geo(lon_a, lat_a, L, theta, **kwargs):
    """build square geographic coordinates, along with other useful informations
    
    Parameters
    ----------
    lon_a, lat_a: floats
        Anchor point (first vertex) longitude and latitude
    L: float
        Radius
    theta: float
        rotation angle of the first vertex
    **kwargs: passed to build_square_with_center
    
    Returns
    -------
    lon, lat: np.array
        vertex coordinates
    X: np.array
        vertex positions as complex numbers
    """
    # convert lon/lat to local coordinates
    x_a, y_a = ll2xy(lon_a, lat_a)
    # build polygon
    X, dX = build_square_with_center(L, theta, **kwargs)
    # offset local coordinates and compute lon/lat
    x, y = x_a+X.real, y_a+X.imag
    lon, lat = xy2ll(x, y)
    return lon, lat, x + 1j*y


# ---------------------------- drifter data -------------------------------



def fetch_drifter_data(timestamp=True, verbose=True):
    """ fetch drifter data from pacific gyre website"""

    with open('pacific_url', 'r') as f:
        # do things with your file
        url = f.read().strip()

    if timestamp:
        t = now()
        tstamp = "_"+t.strftime("%Y%m%d_%H%M%S")
    else:
        tstamp = ""
    file = "drifter_data"+tstamp+".csv"
    
    urllib.request.urlretrieve(url, file)
    
    if verbose:
        print(f" Drifter data downloaded as {file}")
        
    return file
    
def find_latest_drifter_file():
    """find latest drifter file"""
    local_dir = os.getcwd()
    drifter_files = glob(os.path.join(local_dir,"drifter_data*.csv"))
    mtimes = [os.path.getmtime(f) for f in drifter_files]
    latest_file = drifter_files[mtimes.index(max(mtimes))]
    return latest_file

def load_drifter_data(file=None):
    """Load drifter data into a dict of dataframes"""
    if file is None:
        file = find_latest_drifter_file()
    df = pd.read_csv(file, parse_dates=["DeviceDateTime"])
    dr = {k: v.sort_values("DeviceDateTime") for k, v in df.groupby("CommId")}
    for k, v in dr.items():
        # compute local coordinates x, y
        v.loc[:,["x", "y"]] = v.apply(lambda r: pd.Series(ll2xy(r["Longitude"], r["Latitude"]), index=["x","y"]),
                                      axis=1)
        # compute velocity
        #print(dr[anchor]["DeviceDateTime"].diff()/pd.Timedelta("1s"))
        v.loc[:,"dt"] = v.DeviceDateTime.diff()/pd.Timedelta("1s")
        v.loc[:,"u"] = v.x.diff()/v.dt
        v.loc[:,"v"] = v.y.diff()/v.dt
        dr[k] = (v.rename(columns=dict(Longitude="longitude", Latitude="latitude", DeviceDateTime="time"))
                 .set_index("time")
                )
    return dr

def extrapolate_one(df, time):
    """Extrapolate positions for one drifter"""
    dfl = df.reset_index().iloc[-1]
    time_last = dfl.time
    dt = (time - time_last)/pd.Timedelta("1s")
    #
    dx, dy = dfl.u*dt, dfl.v*dt
    x, y = dfl.x+dx, dfl.y+dy
    #
    lon, lat = xy2ll(x, y)
    return pd.Series(dict(time=time, longitude=lon, latitude=lat, x=x, y=y))

def extrapolate(dr, time=None):
    """extrapolate drifter position to new one"""
    if time is None:
        time = now()
    return {k: extrapolate_one(df, time) for k, df in dr.items()}