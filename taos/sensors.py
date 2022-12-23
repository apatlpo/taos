import pandas as pd


# ------------------------------------------- ship tracks ----------------------------------------------------

def read_lern_ship(file, correct_day=None):
    """read lern ship (hydrophone, parceque, delphy) data

    Parameters
    ----------
    file: str
        path to data file
    correct_day: tuple, optional
        fix day of first data point (year, month, day)
    """
    with open(file, encoding="unicode_escape") as f:
        d = list(f.readlines())

    # replace NUL, strip whitespace from the end of the strings, split each string into a list
    d = [v.replace("\n", "").split(",") for v in d]

    columns = d[0]
    print("columns:")
    print(columns)
    # ['ID', 'trksegID', 'lat', 'lon', 'ele', 'time', 'magvar', 'geoidheight', 'name', 'cmt', 'desc', 'src', 'sym', 'type', 'fix', 'sat', 'hdop', 'vdop', 'pdop', 'ageofdgpsdata', 'dgpsid', 'Temperature', 'Depth', 'wtemp', 'hr', 'cad', '']
    # ID,trksegID,lat,lon,ele,time,magvar,geoidheight,name,cmt,desc,src,sym,type,fix,sat,hdop,vdop,pdop,ageofdgpsdata,dgpsid,Temperature,Depth,wtemp,hr,cad,

    df = pd.DataFrame(d[1:], columns=columns)
    df["time"] = pd.to_datetime(df["time"])

    # fix day
    if correct_day:
        if isinstance(correct_day, dict):
            day = pd.Timestamp(**correct_day, tz="UTC")
        else:
            # assumes Timestamp
            day = correct_day
        t0 = df["time"][0]
        dt = day - t0.round("1D")
        print(f"Fix first time from {t0} to {t0 + dt}")
        df["time"] = (df["time"] + dt).dt.tz_localize(
            None
        )  # drop timezone as well (xarray complications down the line)

    # only keep non zero columns
    selected_columns = ["time", "lon", "lat", "Depth", "Temperature"]
    df = df[selected_columns].set_index("time").replace("", "0").astype("float")

    # rename critical columns
    df = df.rename(
        columns=dict(
            lon="longitude",
            lat="latitude",
            Depth="water_depth",
            Temperature="air_temperature",
        )
    )

    return df


# ------------------------------------------- ctd ----------------------------------------------------

def read_lern_sonde(file, tz_offset=0, stype=0):
    """ Load LERN CTD data
    """
    
    with open(file, encoding="unicode_escape") as f:
        d = list(f.readlines())

    # replace NUL, strip whitespace from the end of the strings, split each string into a list
    d = [v.replace('\x00', '').strip().replace(",",".").split('\t') for v in d]

    # remove some empty rows
    d = [v for v in d if len(v) > 1]
    
    #d[0]:
    #['Date', 'Time', 'Site', 'Unit ID', 'User ID',
    # '°C-19K104425', 'mmHg-19C102825', 'DO %-20A103151', 'DO mg/L-20A103151', 
    # 'C-mS/cm-19K104425', 'SAL-PSU-19K104425', 'FNU-19M102353',
    # 'Chl RFU-19M101982', 'Chl ug/L-19M101982',
    # 'DEP m-19K105267']

    # benji deployments
    # ['Date', 'Time', 'Site', 'Unit ID', 'User ID', 
    # '°C-21A101173', 'mmHg-20L100404', 'DO %-21A100555', 'DO mg/L-21A100555', 
    # 'C-mS/cm-21A101173', 'SAL-PSU-21A101173', 
    # 'DEP m-20H100561']

    if stype==0:
        columns = ['Date', 'Time', 'Site', 'Unit ID', 'User ID',
         'temperature', 'air_pressure', 'DO_p', 'DO_mgL', 
         'conductivity', 'salinity', 'FNU',
         'Chl_RFU', 'Chl_ugL',
         'depth', 'latitude', 'longitude']
    elif stype==1:
        columns = ['Date', 'Time', 'Site', 'Unit ID', 'User ID',
         'temperature', 'air_pressure', 'DO_p', 'DO_mgL', 
         'conductivity', 'salinity',
         'depth', 'latitude', 'longitude']

    df = pd.DataFrame(d[1:], columns=columns)
    
    # convert column 0 and 1 to a datetime
    df['time'] = pd.to_datetime(df["Date"] + ' ' + df["Time"]) + pd.Timedelta("1H")*tz_offset

    df.set_index("time", inplace=True)
    df = df.drop(columns=["Date", "Time", "Site", "Unit ID", "User ID"])

    df = df.astype('float')

    # convert air pressure from mmHg to bar
    df["air_pressure"] = df["air_pressure"]/750.06
    
    # bad longitude, latitude values are 0.0
    df.loc[:, "longitude"] = df.loc[:, "longitude"].where(df.loc[:, "longitude"] != 0.)
    df.loc[:, "latitude"] = df.loc[:, "latitude"].where(df.loc[:, "latitude"] != 0.)

    return df


# ------------------------------------------- drifters ----------------------------------------------------


def read_carthe_drifters(file):
    
    df = pd.read_csv(file, parse_dates=[1])
    
    df = (df
          .rename(columns=dict(DeviceName="id", DeviceDateTime="time", 
                               Latitude="latitude", Longitude="longitude"))
          .sort_values("id")
         )
    df = df.set_index("id")
        
    return df