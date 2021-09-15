import xml.etree.ElementTree as ET
import os

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

    tree = ET.parse(file_in)
    root = tree.getroot()

    modified = {k: False for k in params}

    for p in root.iter("parameter"):
        for k in p.iter("key"):
            if k.text in params:
                v = p.find("value")
                v.text = str(params[k.text])
                modified[k.text] = True

    assert all([b for k, b in modified.items()]), "One or several parameters were not modified"

    if overwrite or not os.path.isfile(file_out):
        tree.write(file_out)
        print("File {} has been generated".format(file_out))
    else:
        print("Nothing done")
