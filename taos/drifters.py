import os, sys
import shutil
from glob import glob
import subprocess
import time

import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt
import cartopy.crs as ccrs

from xhistogram.xarray import histogram

import xml.etree.ElementTree as ET

from functools import reduce

_default_ichthyop_path = os.path.join(
    os.getenv("HOME"),
    "ichthyop/target/",
    "ichthyop-3.3.10-jar-with-dependencies.jar",
)

_root_nc_path = "/home/ref-oc-public/modeles_marc/f1_e2500_agrif/MARC_F1-MARS3D-SEINE/best_estimate/"


def _get_input_files(params):
    """Return a dataframe with the path to required input files
    given simulation parameters
    """
    # extract time line from Parameters
    itime = params["initial_time"]
    for s in ["year", "month", "day", "at"]:
        itime = itime.replace(s, "")
    itime = pd.Timestamp(itime)
    delt = pd.Timedelta(params["transport_duration"].replace("(s)", ""))
    hour = pd.Timedelta("1H")
    times = pd.date_range(itime - 2 * hour, itime + delt + 2 * hour, freq="1H")
    # build list of input file paths
    df = times.to_frame()
    df["dir"] = df.index.map(lambda t: os.path.join(_root_nc_path, str(t.year)))
    df["file"] = df.index.map(
        lambda t: t.strftime("MARC_F1-MARS3D-SEINE_%Y%m%dT%H%MZ.nc")
    )
    df = df.drop(columns=0)
    return df


def prepare_input_links(params, rpath=None, input_path=None):
    """generates an input dir with symbolic links to original file"""
    if isinstance(params, dict):
        df = _get_input_files(params)
    elif isinstance(params, list):
        df = reduce(
            lambda left, right: pd.merge(left, right, how="outer"),
            [_get_input_files(p) for p in params],
        )
    # create symbolic links
    if rpath is not None:
        bpath = os.path.join(rpath, "inputs")
    elif input_path is not None:
        bpath = input_path
    os.mkdir(bpath)
    for t, row in df.iterrows():
        subprocess.run(
            [
                "ln",
                "-s",
                os.path.join(row["dir"], row["file"]),
                os.path.join(bpath, row["file"]),
            ],
            check=True,
        )
    return bpath


def update_config(
    file_in, file_out=None, overwrite=False, base_nc_path=None, verbose=True, **params
):
    """Modify an Ichthyop xml input file

    Parameters
    ----------
    file_in: str
        Input file
    file_out: str, optional
        Output file
    overwrite: boolean, optional
    **kwargs:
        parameters to be modified, e.g.
        initial_time="year 2011 month 01 day 01 at 01:00"

    """

    if file_out is None:
        file_out = file_in.replace(".xml", "_new.xml")

    _params = dict(**params)

    tree = ET.parse(file_in)
    root = tree.getroot()

    if base_nc_path is None:
        # need to update input paths
        base_nc_path = (
            "/home/datawork-lops-osi/aponte/taos/ichthy/MARC_F1-MARS3D-SEINE/"
        )
    _params["input_path"] = base_nc_path
    _params["file_filter"] = "MARC_F1-MARS3D-SEINE_*.nc"

    modified = {k: False for k in _params}

    for p in root.iter("parameter"):
        for k in p.iter("key"):
            if k.text in _params:
                v = p.find("value")
                v.text = str(_params[k.text])
                modified[k.text] = True

    assert all(
        [b for k, b in modified.items()]
    ), "One or several parameters were not modified"

    if overwrite or not os.path.isfile(file_out):
        tree.write(file_out)
        if verbose:
            print("File {} has been generated".format(file_out))
    else:
        print("Nothing done")


def format_date(date):
    """Format date for Ichthyop"""
    if isinstance(date, str):
        date = pd.Timestamp(date)
    return date.strftime("year %Y month %m day %d at %H:%M")


class ichthy(object):
    """Object to automate Ichthyop run launchings

    Parameters
    ----------
    rundir: str
        Name of the run directory
    jobname: str
        Name of the PBS job
    workdir: str, optional
        Path to the working directory, default to scratch
    launch: boolean, optional
        Launch simulation, default is True
    ichthyop_path: str, optional
        Path to jar executable
    cfg: dict, optional
        Parameters used to update the configuration file
    configurations: dict, optional
        Dict of different configurations (dict analogous to cfg) to be launched at once.
        Keys are used for simulations names
    """

    def __init__(
        self,
        rundir,
        jobname="icht",
        workdir=None,
        launch=True,
        ichthyop_path=None,
        **params,
    ):
        #
        self.startdir = os.getcwd()
        #
        if workdir is None:
            self.workdir = os.getenv("SCRATCH")
        elif not os.path.isdir(workdir):
            self.workdir = os.getenv(workdir)
        else:
            self.workdir = workdir
        # main directory where the simulation will be run:
        self.rpath = os.path.join(self.workdir, rundir)
        print("Run will be stored in {}".format(self.rpath))
        self._create_rpath()
        self.base_nc_path = prepare_input_links(params, rpath=self.rpath)
        # change input parameters
        self.update_cfg_file(**params)
        # guess config if necessary and create job files
        self.jobname = jobname
        if ichthyop_path is None:
            ichthyop_path = _default_ichthyop_path
        self._create_job_files(ichthyop_path)
        # launch runs
        if launch:
            self.launch()
        os.chdir(self.startdir)

    def _create_rpath(self):
        if os.path.exists(self.rpath):
            os.system("rm -Rf " + self.rpath)
        os.mkdir(self.rpath)
        # move to run dir
        os.chdir(self.rpath)

    def update_cfg_file(self, **params):
        """Update config"""
        cfg_in = os.path.join(self.startdir, "taos_mars3d.xml")
        update_config(
            cfg_in,
            file_out="cfg.xml",
            overwrite=True,
            base_nc_path=self.base_nc_path,
            **params,
        )

    def _create_job_files(self, ichthyop_path):

        # RAM usage < 7GB
        # elapse time = 7min for 10d run, extrapolate to 21min for 10d run

        with open("job.pbs", "w") as f:
            f.write("#!/bin/csh\n")
            f.write("#PBS -N " + self.jobname + "\n")
            f.write("#PBS -q sequentiel\n")
            f.write("#PBS -l mem=10g\n")
            f.write("#PBS -l walltime=01:00:00\n")
            f.write("\n")
            f.write("# cd to the directory you submitted your job\n")
            f.write("cd $PBS_O_WORKDIR\n")
            f.write("\n")
            f.write("setenv PATH ${HOME}/.miniconda3/envs/ichthy/bin:${PATH}\n")
            f.write("\n")
            f.write("date\n")
            f.write("java -jar {} cfg.xml\n".format(ichthyop_path))
            f.write("\n")
            f.write("date\n")

    def launch(self):
        """Launch simulations"""
        time.sleep(1)
        # os.chdir(join(self.rpath,"t1"))
        os.system("qsub job.pbs")
        os.chdir(self.startdir)


class ichthys(object):
    """Object to automate the launch of multiple ichthyop runs

    Parameters
    ----------
    dir_suffix: str
        Base name of the run directories
    configurations: dict
        Dict of different configurations (dict analogous to cfg) to be launched at once.
        Keys are used for simulations names
    jobname: str, optional
        Base name of the PBS job
    workdir: str, optional
        Path to the working directory, default to scratch
    launch: boolean, optional
        Launch simulations, default is True
    ichthyop_path: str, optional
        Path to jar executable
    """

    def __init__(
        self,
        dir_suffix,
        configurations,
        jobname="icht",
        workdir=None,
        launch=True,
        ichthyop_path=None,
        overwrite=False,
    ):
        #
        self.startdir = os.getcwd()
        #
        self.dir_suffix = dir_suffix
        if workdir is None:
            self.workdir = os.getenv("SCRATCH")
        elif not os.path.isdir(workdir):
            self.workdir = os.getenv(workdir)
        else:
            self.workdir = workdir
        # build directories where simulations will be run:
        flag = self._create_rpath(dir_suffix, configurations, overwrite)
        if not flag:
            # exit if nothing has to be done
            # print(dir_suffix+": nothing to be done")
            return
        # change input parameters
        self.update_cfg_files()
        # move to work dir
        os.chdir(self.workdir)
        # guess config if necessary and create job files
        self.jobname = jobname
        if ichthyop_path is None:
            ichthyop_path = _default_ichthyop_path
        self._create_job_files(ichthyop_path)
        # launch runs
        if launch:
            self.launch()
        os.chdir(self.startdir)

    def _create_rpath(self, dir_suffix, configurations, overwrite):
        self.rpath = {}
        for c in configurations:
            rpath = os.path.join(self.workdir, dir_suffix + c)
            if not os.path.exists(rpath) or overwrite:
                self.rpath[c] = rpath
        if not self.rpath:
            return False
        self.configurations = {c: configurations[c] for c in self.rpath}
        # main directory where the simulation will be run:
        for c, rpath in self.rpath.items():
            if os.path.exists(rpath):
                os.system("rm -Rf " + rpath)
            os.mkdir(rpath)
        input_path = os.path.join(self.workdir, dir_suffix + "inputs")
        self.base_nc_path = prepare_input_links(
            [cfg for c, cfg in self.configurations.items()],
            input_path=input_path,
        )
        return True

    def update_cfg_files(self):
        """Update config"""
        cfg_in = os.path.join(self.startdir, "taos_mars3d.xml")
        for c, cfg in self.configurations.items():
            fout = os.path.join(self.rpath[c], "cfg.xml")
            update_config(
                cfg_in,
                file_out=fout,
                overwrite=True,
                verbose=False,
                base_nc_path=self.base_nc_path,
                **cfg,
            )

    def _create_job_files(self, ichthyop_path):

        # RAM usage < 7GB
        # elapse time = 7min for 10d run, extrapolate to 21min for 10d run

        elapse_one_simulation = 1  # in hours
        elapse_total = len(self.rpath) * elapse_one_simulation

        self.job = self.dir_suffix + "job.pbs"
        with open(self.job, "w") as f:
            f.write("#!/bin/csh\n")
            f.write("#PBS -N " + self.jobname + "\n")
            f.write("#PBS -q sequentiel\n")
            f.write("#PBS -l mem=10g\n")
            f.write("#PBS -l walltime={}:00:00\n".format(elapse_total))
            f.write("\n")
            # f.write("# cd to the directory you submitted your job\n")
            # f.write("cd $PBS_O_WORKDIR\n")
            # f.write("\n")
            f.write("setenv PATH ${HOME}/.miniconda3/envs/ichthy/bin:${PATH}\n")
            f.write("date\n")
            f.write("\n")
            for c, rpath in self.rpath.items():
                f.write("# cd to the directory you submitted your job\n")
                f.write("cd " + rpath + " \n")
                f.write("java -jar {} cfg.xml\n".format(ichthyop_path))
                f.write("date\n")
                f.write("touch " + os.path.join(rpath, "done") + " \n")
                f.write("\n")

    def launch(self):
        """Launch simulations"""
        time.sleep(1)
        os.system("qsub " + self.job)
        os.chdir(self.startdir)


# -------------------------------- post-processing --------------------------------------------


def load_run_simple(run_dir):
    """Load one Ichthyop run"""
    f = glob(os.path.join(run_dir, "output/taos*.nc"))
    # assert len(f)==1, \
    #    "Erroneous number of files ({}) in {}".format(len(f), run_dir)
    if len(f) == 0 or len(f) > 1:
        return None
    ds = xr.open_dataset(f[0])
    # test if "done" flag file is present (means the job executed normaly on the pbs side)
    ds.attrs["done"] = os.path.isfile(os.path.join(run_dir, "done"))
    return ds


def normalize_time(ds):
    ds["date"] = ds["time"]
    ds["time"] = (ds.time - ds.time[0]) / pd.Timedelta("1D")
    return ds


def plot_trajectories(
    ds,
    ax=None,
    nmax=50,
    dt=None,
    track_color="0.5",
    ms=5,
    **kwargs,
):
    """Plot trajectories

    Parameters
    ----------
    ds: xr.Dataset
        Ichthyop trajectories
    ax: GeoAxesSubplot, optional
        Existing cartopy axes plot
    nmax: int, optional
        Max number of trajectories to plot. Default: 50
    dt: float, optional
        Plot locations every dt days. Default: None
    track_color: str, optional
        Track color
    ms: int
        Marker size
    **kwargs: other arguments passed to plotting methods
    """

    t_start = ds.time.isel(time=0)
    t_end = ds.time.isel(time=-1)

    opts = dict(**kwargs)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = None
        opts["transform"] = ccrs.PlateCarree()

    dr = ds.drifter.values[:nmax]
    for d in dr:
        _ds = ds.sel(drifter=d)
        ax.plot(_ds.lon, _ds.lat, lw=0.5, color=track_color, **opts)

    for d in dr:
        _ds = ds.sel(drifter=d, time=t_start)
        ax.plot(
            _ds.lon, _ds.lat, "o", ms=ms, color="orange", markeredgecolor="k", **opts
        )

    if dt is not None:
        _dt = pd.Timedelta(dt, unit="days")
        t = t_start + _dt
        while t < t_end:
            # print(t.values)
            for d in dr:
                _ds = ds.sel(drifter=d, time=t)
                ax.plot(_ds.lon, _ds.lat, "o", ms=ms, color="k", **opts)
            t += _dt

    return fig, ax


def bin_geographically(
    ds,
    dl=2e3,
    dims=["date_start", "drifter"],
    persist=True,
):
    """Bin locations geographically

    Parameters
    ----------
    ds: xr.Dataset
        contains lon, lat
    dl: float
        bin size in meters
    dims: list of str
        Dimensions to bin over
    persist: boolean
        Flag in order to bin data
    """

    edges = ds.region_edge
    for v in ["date_start", "lat_start"]:
        if v in edges.dims:
            edges = edges.isel(**{v: 0})
    # edges = ds.region_edge.isel(date_start=0, lat_start=0)
    _lat = edges[:, 0]
    _lon = edges[:, 1]

    dla = 111e3
    dlo = float(dla * np.cos(np.pi / 180 * _lat.mean()))

    lon_bins = np.arange(_lon.min() - dl / dlo, _lon.max() + dl / dlo, dl / dlo)
    lat_bins = np.arange(_lat.min() - dl / dla, _lat.max() + dl / dla, dl / dla)

    h = histogram(ds.lon, ds.lat, bins=[lon_bins, lat_bins], dim=dims)

    if persist:
        h = h.persist()

    return h


def bin_displacements(
    dx,
    dy,
    dl_step=500,
    dl_max=20e3,
    dims=["date_start", "drifter"],
    persist=True,
):
    """Bin displacements in two direction

    Parameters
    ----------
    dx, dy: xr.Dataarray
        horizontal displacements
    dl_step: float
        bin size in meters
    dl_max: float
        max displacement considered
    dims: list of str
        dimensions to bin over
    persist: boolean
        flag to persist data
    """
    x_bins = np.arange(-dl_max, dl_max, dl_step)
    y_bins = np.arange(-dl_max, dl_max, dl_step)

    h = histogram(
        dx,
        dy,
        bins=[x_bins, y_bins],
        dim=dims,
        density=True,
    )

    if persist:
        h = h.persist()

    return h


def bin_displacement(
    dl,
    dl_step=500,
    dl_max=20e3,
    dims=["date_start", "drifter"],
    persist=True,
):
    """Bin displacement (distance)

    Parameters
    ----------
    dx, dy: xr.Dataarray
        horizontal displacements
    dl_step: float
        bin size in meters
    dl_max: float
        max displacement considered
    dims: list of str
        dimensions to bin over
    persist: boolean
        flag to persist data
    """
    dl_bins = np.arange(0.0, dl_max, dl_step)

    h = histogram(
        dl,
        bins=[
            dl_bins,
        ],
        dim=dims,
        density=True,
    ).rename("density")
    ds = h.to_dataset()

    dim = dl.name + "_bin"
    ds["mean"] = (h * h[dim]).integrate(dim)
    ds["cum"] = h.cumulative_integrate(dim)
    # bad form: ensures cum finishes at 1:
    ds["cum"] = ds["cum"] / ds["cum"].isel(**{dim: -1})

    if persist:
        ds = ds.persist()

    return ds
