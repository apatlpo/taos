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
        dt = day - t0.floor("1D")
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

    # taos2-drifters1 
    # Date (MM/DD/YYYY)	Time (HH:mm:ss)	Time (Fract. Sec)	Site Name	Chlorophyll RFU	Chlorophyll ug/L	Cond µS/cm	Depth m	nLF Cond µS/cm	ODO % sat	ODO % local	ODO mg/L	Pressure psi a	Sal psu	SpCond µS/cm	BGA PE RFU	BGA PE ug/L	TDS mg/L	Turbidity FNU	TSS mg/L	Temp °C	Vertical Position m	GPS Latitude °	GPS Longitude °	Altitude m	Battery V	Cable Pwr V	Barometer mmHg
    #13/06/2023	14:34:18	0.0	frank	0	02	0	55	43640	0	0	596	51416	8	104	1	107	6	8	13	0	868	33	39	50794	7	1	84	5	16	33017	-0	63	0	17	626	0	633	49	35327	-0	46210	5	0	5	25	0	0	759	0

    
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
    elif stype==2:
        columns = ['Date', 'Time', 'Site', 'Chl_RFU', 'Chl_ugL', 'conductivity','depth',
                   'conductivity_mS', 'DO_p', 'DO_l', 'pressure', 'salinity', 'conductivity_Sp',
                   'BGA_PE_RFU', 'BGA_PE_ug', 'TDS', 'Turbidity_FNU', 'TSS', 'temperature', 'vertical_position', 
                   'latitude', 'longitude', 'altitude', 'battery', 'cable_pwr', 'air_pressure'
                  ]
        # reprocess data

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

    # make sure time is a datetime
    df["time"] = pd.to_datetime(df["time"])
    
    return df