import os, sys
import shutil

#from glob import glob
import time

import xml.etree.ElementTree as ET

def update_config(file_in, file_out=None, overwrite=False, **params):
    """ Modify an Ichthyop xml input file

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
    
    if "initial_time" in _params:
        # need to update input paths
        base_nc_path = "/home/ref-oc-public/modeles_marc/f1_e2500_agrif/MARC_F1-MARS3D-SEINE/best_estimate/"
        year = _params["initial_time"].split()[1]
        month = _params["initial_time"].split()[3]
        _params["input_path"] = base_nc_path + year + "/"
        _params["file_filter"] = "*MARC_F1-MARS3D-SEINE_"+year+month+"*.nc"
                
    modified = {k: False for k in _params}

    for p in root.iter("parameter"):
        for k in p.iter("key"):
            if k.text in _params:
                v = p.find("value")
                v.text = str(_params[k.text])
                modified[k.text] = True

    assert all([b for k, b in modified.items()]), "One or several parameters were not modified"

    if overwrite or not os.path.isfile(file_out):
        tree.write(file_out)
        print("File {} has been generated".format(file_out))
    else:
        print("Nothing done")

def format_date(date):
    """ Format date for Ichthyop
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)
    return date.strftime("year %Y month %m day %d at %H:%M")

_default_ichthyop_path = os.path.join(os.getenv("HOME"),
                                      "ichthyop/target/",
                                      "ichthyop-3.3.10-jar-with-dependencies.jar",
                                      )

class ichthy(object):
    """ Object to automate Ichthyop run launchings

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
    """

    def __init__(self,
                 rundir,
                 jobname='icht',
                 workdir=None,
                 launch=True,
                 ichthyop_path=None,
                 **params,
                 ):
        #
        self.startdir = os.getcwd()
        #
        if workdir is None:
            self.workdir = os.getenv('SCRATCH')
        elif not os.path.isdir(workdir):
            self.workdir = os.getenv(workdir)
        else:
            self.workdir = workdir
        # main directory where the simulation will be run:
        self.rpath = os.path.join(self.workdir, rundir)
        print('Run will be stored in {}'.format(self.rpath))
        self._create_rpath()
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
        if os.path.exists(self.rpath) :
            os.system('rm -Rf '+self.rpath)
        os.mkdir(self.rpath)
        # move to run dir
        os.chdir(self.rpath)

    def update_cfg_file(self, **params):
        """ Update config
        """
        cfg_in = os.path.join(self.startdir, "taos_mars3d.xml")
        update_config(cfg_in, file_out="cfg.xml", overwrite=True, **params)

    def _create_job_files(self, ichthyop_path):

        # RAM usage < 7GB
        # elapse time = 7min for 10d run, extrapolate to 21min for 10d run
        
        with open('job.pbs','w') as f:
            f.write('#!/bin/csh\n')
            f.write('#PBS -N '+self.jobname+'\n')
            f.write('#PBS -q sequentiel\n')
            f.write('#PBS -l mem=10g\n')
            f.write('#PBS -l walltime=00:30:00\n')
            f.write('\n')
            f.write('# cd to the directory you submitted your job\n')
            f.write('cd $PBS_O_WORKDIR\n')
            f.write('\n')
            f.write('setenv PATH ${HOME}/.miniconda3/envs/ichthy/bin:${PATH}\n')
            f.write('\n')
            f.write('date\n')
            f.write('java -jar {} cfg.xml\n'.format(ichthyop_path))
            f.write('\n')
            f.write('date\n')

    def launch(self):
        """ Launch simulations
        """
        time.sleep(1)
        #os.chdir(join(self.rpath,'t1'))
        os.system('qsub job.pbs')
        os.chdir(self.startdir)