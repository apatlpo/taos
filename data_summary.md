# TAOS - available data summary

- mars - drifter comparison: 
	- basic KE maps
	- direct comparison: tidal extraction vs actual predictions
- dispersion estimates: mars vs drifters

- update over sea level
- inspect different bathymetric data sources (compare to gebco)


## campaigns

### TAOS0 - drifters0: 

Dates: 2022/05/11 to 2022/05/13

Meteo: strong westerly event, mostly cloudy

Data cleaning status:

| data type | raw data availability | cleaned | quantity |
|:-----|:------:|:---------:|:---:|
| drifters | X | X | 6 |
| ctd parceque | X | X | 2 casts - 5 underway|
| met - arome | X | X | - |
| met - insitu | X | O | - |
| sea level | X | X | - |

Drifter movie created.
Drifters move east.

```
-----  taos0_drifters0 
drifter0 None d0 / start 2022-05-11 05:10:00 -0.24 49.34 / end 2022-05-13 07:10:00 -0.01 49.36
drifter1 None d0 / start 2022-05-11 05:40:00 -0.25 49.33 / end 2022-05-13 07:20:00 -0.04 49.35
drifter2 None d0 / start 2022-05-11 05:50:00 -0.24 49.34 / end 2022-05-13 07:30:00 -0.04 49.35
drifter3 None d0 / start 2022-05-11 05:50:00 -0.24 49.34 / end 2022-05-13 07:30:00 -0.04 49.35
drifter4 None d0 / start 2022-05-11 05:50:00 -0.25 49.34 / end 2022-05-11 08:40:00 -0.25 49.34
drifter4 None d1 / start 2022-05-11 09:20:00 -0.27 49.34 / end 2022-05-13 07:30:00 -0.04 49.35
drifter5 None d0 / start 2022-05-11 06:00:00 -0.24 49.35 / end 2022-05-11 06:20:00 -0.23 49.35
drifter5 None d1 / start 2022-05-11 06:40:00 -0.23 49.35 / end 2022-05-13 08:00:00 -0.04 49.35
parceque None d0 / start 2022-05-11 04:26:00 / end 2022-05-11 07:15:00
parceque None d1 / start 2022-05-13 05:26:00 / end 2022-05-13 09:20:00
parceque ctd u0 / start 2022-05-11 05:39:00 -0.22 49.36 / end 2022-05-11 07:03:00 -0.24 49.33
parceque ctd u1 / start 2022-05-11 07:06:00 -0.23 49.33 / end 2022-05-11 07:07:00 -0.23 49.33
parceque ctd u2 / start 2022-05-11 07:09:20 -0.22 49.32 / end 2022-05-11 07:11:10 -0.22 49.32
parceque ctd u3 / start 2022-05-11 07:16:38 -0.26 49.32 / end 2022-05-11 07:18:00 -0.26 49.32
parceque ctd u4 / start 2022-05-13 06:41:00 -0.02 49.36 / end 2022-05-13 07:36:00 -0.01 49.36
parceque ctd c5 / start 2022-05-13 07:44:00 -0.03 49.36 / end 2022-05-13 07:46:00 -0.03 49.36
parceque ctd c6 / start 2022-05-13 07:52:00 -0.04 49.35 / end 2022-05-13 07:55:00 -0.03 49.35
hydrophone None d0 / start 2022-05-11 04:26:00 / end 2022-05-11 07:15:00
hydrophone None d1 / start 2022-05-13 05:26:00 / end 2022-05-13 09:20:00
```



### TAOS0 - drifters1: 

Dates: 2022/05/17 to 2022/05/19

Meteo: weak wind, few sunny moments

Data cleaning status:

| data type | raw data availability | cleaned | quantity |
|:-----|:------:|:---------:|:---:|
| drifters | X | X | 6 |
| ctd hydrophone | X | X | 23 casts - 5 underway|
| ctd parceque | X | X | 2 underway|
| met - arome | X | X | - |
| met - insitu | O | O | - |
| sea level | X | X | - |

Drifter movie created.
Drifters move west and oscillate.

```
-----  taos0_drifters1 
drifter0 None d0 / start 2022-05-17 09:30:00 -0.25 49.34 / end 2022-05-19 11:10:00 -0.24 49.37
drifter1 None d0 / start 2022-05-17 08:30:00 -0.25 49.34 / end 2022-05-19 10:20:00 -0.28 49.40
drifter2 None d0 / start 2022-05-17 09:40:00 -0.22 49.34 / end 2022-05-19 11:00:00 -0.24 49.38
drifter3 None d0 / start 2022-05-17 09:20:00 -0.24 49.34 / end 2022-05-19 10:50:00 -0.26 49.39
drifter4 None d0 / start 2022-05-17 09:10:00 -0.25 49.34 / end 2022-05-19 10:50:00 -0.26 49.39
drifter5 None d0 / start 2022-05-17 09:00:00 -0.26 49.33 / end 2022-05-19 11:40:00 -0.27 49.35
parceque None d0 / start 2022-05-17 07:38:00 / end 2022-05-17 12:00:00
parceque None d1 / start 2022-05-19 10:30:00 / end 2022-05-19 12:30:00
parceque ctd u0 / start 2022-05-17 08:44:20 -0.23 49.35 / end 2022-05-17 10:15:00 -0.25 49.35
parceque ctd u1 / start 2022-05-19 10:25:00 -0.25 49.37 / end 2022-05-19 11:54:00 -0.29 49.37
hydrophone None d0 / start 2022-05-17 07:38:00 / end 2022-05-17 12:00:00
hydrophone None d1 / start 2022-05-19 10:30:00 / end 2022-05-19 12:30:00
hydrophone ctd c0 / start 2022-05-17 07:31:00 -0.24 49.34 / end 2022-05-17 07:37:40 -0.24 49.34
hydrophone ctd c1 / start 2022-05-17 07:44:40 -0.26 49.33 / end 2022-05-17 07:50:50 -0.26 49.33
hydrophone ctd c2 / start 2022-05-17 08:00:26 -0.25 49.34 / end 2022-05-17 08:03:06 -0.25 49.34
hydrophone ctd c3 / start 2022-05-17 08:05:00 -0.25 49.34 / end 2022-05-17 08:10:40 -0.25 49.34
hydrophone ctd c4 / start 2022-05-17 08:14:38 -0.25 49.34 / end 2022-05-17 08:18:16 -0.25 49.34
hydrophone ctd c5 / start 2022-05-17 08:24:33 -0.23 49.34 / end 2022-05-17 08:27:47 -0.23 49.34
hydrophone ctd c6 / start 2022-05-17 08:51:26 -0.24 49.32 / end 2022-05-17 08:53:25 -0.24 49.32
hydrophone ctd c7 / start 2022-05-17 08:56:57 -0.24 49.32 / end 2022-05-17 08:59:46 -0.24 49.32
hydrophone ctd c8 / start 2022-05-17 09:03:17 -0.24 49.31 / end 2022-05-17 09:06:00 -0.24 49.31
hydrophone ctd c9 / start 2022-05-17 09:08:39 -0.25 49.31 / end 2022-05-17 09:11:24 -0.25 49.31
hydrophone ctd c10 / start 2022-05-17 09:13:52 -0.25 49.31 / end 2022-05-17 09:16:20 -0.25 49.31
hydrophone ctd c11 / start 2022-05-17 09:19:02 -0.25 49.30 / end 2022-05-17 09:20:35 -0.25 49.30
hydrophone ctd c12 / start 2022-05-17 09:23:00 -0.25 49.30 / end 2022-05-17 09:24:32 -0.25 49.30
hydrophone ctd c13 / start 2022-05-17 09:27:40 -0.25 49.30 / end 2022-05-17 09:29:40 -0.25 49.30
hydrophone ctd c14 / start 2022-05-17 09:32:17 -0.25 49.29 / end 2022-05-17 09:33:40 -0.25 49.29
hydrophone ctd c15 / start 2022-05-19 10:50:50 -0.26 49.40 / end 2022-05-19 10:53:30 -0.26 49.40
hydrophone ctd c16 / start 2022-05-19 11:03:35 -0.26 49.39 / end 2022-05-19 11:05:44 -0.26 49.39
hydrophone ctd c17 / start 2022-05-19 11:10:39 -0.25 49.39 / end 2022-05-19 11:12:41 -0.25 49.39
hydrophone ctd c18 / start 2022-05-19 11:19:28 -0.24 49.38 / end 2022-05-19 11:22:00 -0.24 49.38
hydrophone ctd c19 / start 2022-05-19 11:28:25 -0.23 49.37 / end 2022-05-19 11:30:43 -0.23 49.37
hydrophone ctd c20 / start 2022-05-19 11:35:55 -0.25 49.36 / end 2022-05-19 11:37:47 -0.25 49.36
hydrophone ctd c21 / start 2022-05-19 11:42:00 -0.27 49.36 / end 2022-05-19 11:43:50 -0.27 49.36
hydrophone ctd c22 / start 2022-05-19 11:49:50 -0.27 49.35 / end 2022-05-19 11:51:23 -0.27 49.35
```

### TAOS1 - drifters0: 

Dates: 2022/11/14 to 2022/11/18

Meteo: strong northerly event, mostly cloudy.

Data cleaning status:

| data type | raw data availability | cleaned | quantity |
|:-----|:------:|:---------:|:---:|
| drifters | X | X | 3 |
| ctd hydrophone | X | X | 3 casts |
| ctd delphy | X | X | 3 casts |
| ctd delphy | X | X | 3 casts |
| yuco | X | X | 3 deployments |
| met - arome | X | X | - |
| met - insitu | O | O | - |
| sea level | O | O | - |

Drifter movie created.
Drifter drift slowly northward then more quickly eastward (with similar wind strength !) and along the coastline.


```
-----  taos1_drifters0 
drifter0 None d0 / start 2022-11-14 11:50:23 -0.26 49.35 / end 2022-12-16 10:34:10 0.42 50.11
drifter1 None d0 / start 2022-11-14 12:06:00 -0.26 49.33 / end 2022-11-17 17:02:35 0.02 49.50
drifter2 None d0 / start 2022-11-14 12:11:30 -0.24 49.34 / end 2022-11-18 11:11:35 -0.17 49.69
hydrophone None d0 / start 2022-11-14 10:34:00 / end 2022-11-14 15:12:00
hydrophone ctd cast0 / start 2022-11-14 12:34:50 -0.24 49.34 / end 2022-11-14 12:36:00 -0.24 49.34
hydrophone ctd cast1 / start 2022-11-14 12:43:45 -0.26 49.35 / end 2022-11-14 12:45:15 -0.26 49.35
hydrophone ctd cast2 / start 2022-11-14 12:53:59 -0.27 49.33 / end 2022-11-14 12:56:11 -0.27 49.33
delphy None d0 / start 2022-11-14 10:34:00 / end 2022-11-14 15:12:00
delphy ctd cast0 / start 2022-11-14 14:20:50 -0.39 49.43 / end 2022-11-14 14:23:43 -0.39 49.43
delphy ctd cast1 / start 2022-11-14 14:46:30 -0.35 49.38 / end 2022-11-14 14:50:58 -0.35 49.38
delphy ctd cast2 / start 2022-11-14 15:19:11 -0.46 49.35 / end 2022-11-14 15:21:30 -0.46 49.35
yuco None d0 / start 2022-11-14 12:34:33 / end 2022-11-14 12:47:44
yuco None d1 / start 2022-11-14 13:03:05 / end 2022-11-14 13:32:26
yuco None d2 / start 2022-11-14 13:59:09 / end 2022-11-14 14:27:37
```


### TAOS1 - drifters1: 

Dates: 2022/11/30 to 2022/12/18

Meteo: moderate north-easterly winds. 
Some sunny moments (on the 22 for instance).

Data cleaning status:

| data type | raw data availability | cleaned | quantity |
|:-----|:------:|:---------:|:---:|
| drifters | X | X | 3 |
| met - arome | X | X | - |
| met - insitu | O | O | - |
| sea level | O | O |  - |

Drifter movie created.
Drifters move westward, one beaches and other two aggregate.

```
-----  taos1_drifters1 
drifter0 None d0 / start 2022-11-30 16:09:50 -0.25 49.34 / end 2022-12-06 09:26:15 -0.53 49.41
drifter1 None d0 / start 2022-11-30 16:25:00 -0.27 49.35 / end 2022-12-06 09:25:30 -0.53 49.41
drifter2 None d0 / start 2022-11-30 16:29:00 -0.27 49.33 / end 2022-12-03 09:26:00 -0.46 49.34
delphy None d0 / start 2022-11-30 12:00:00 / end 2022-11-30 16:50:00
```


### TAOS2 - drifters0:

Dates: 2022/11/14 to 2022/11/18

Meteo: strong northerly event, mostly cloudy.

Data cleaning status:

| data type | raw data availability | cleaned | quantity |
|:-----|:------:|:---------:|:---:|
| drifters | X | X | 10 |
| ctd hydrophone | X | X | 8 casts - 9 surface stations |
| ctd delphy | X | X | 6 casts |
| met - arome | X | X | - |
| met - insitu | O | O | - |
| sea level | O | O | - |

**?? Drifter movie created.**

**?Drifter drift slowly northward then more quickly eastward (with similar wind strength !) and along the coastline.**

```
-----  taos2_drifters0 
None None day0 / start 2023-05-05 07:00:00 / end 2023-05-05 20:00:00
hydrophone None d0 / start 2023-05-05 07:55:00 / end 2023-05-05 11:23:00
hydrophone None d1 / start 2023-05-09 09:16:00 / end 2023-05-09 13:31:00
hydrophone None d2 / start 2023-05-11 10:32:00 / end 2023-05-11 13:35:00
hydrophone ctd s0 / start 2023-05-05 08:17:40 / end 2023-05-05 08:21:40
hydrophone ctd s1 / start 2023-05-05 08:25:10 / end 2023-05-05 08:27:00
hydrophone ctd s2 / start 2023-05-05 08:30:50 / end 2023-05-05 08:32:50
hydrophone ctd s3 / start 2023-05-05 08:37:00 / end 2023-05-05 08:40:40
hydrophone ctd s4 / start 2023-05-05 08:45:10 / end 2023-05-05 08:47:50
hydrophone ctd s5 / start 2023-05-05 08:53:00 / end 2023-05-05 08:54:00
hydrophone ctd s6 / start 2023-05-05 08:59:00 / end 2023-05-05 09:02:00
hydrophone ctd s7 / start 2023-05-05 09:06:00 / end 2023-05-05 09:09:00
hydrophone ctd s8 / start 2023-05-05 09:17:10 / end 2023-05-05 09:18:40
hydrophone ctd c0 / start 2023-05-05 09:22:19 / end 2023-05-05 09:23:35
hydrophone ctd c1 / start 2023-05-05 09:41:58 / end 2023-05-05 09:43:50
hydrophone ctd c2 / start 2023-05-05 09:53:14 / end 2023-05-05 09:54:58
hydrophone ctd c3 / start 2023-05-05 10:02:41 / end 2023-05-05 10:04:43
hydrophone ctd c4 / start 2023-05-05 10:14:21 / end 2023-05-05 10:16:01
hydrophone ctd c5 / start 2023-05-05 10:24:39 / end 2023-05-05 10:26:25
hydrophone ctd c6 / start 2023-05-05 10:32:17 / end 2023-05-05 10:33:10
hydrophone ctd c7 / start 2023-05-05 10:40:42 / end 2023-05-05 10:42:00
delphy None d0 / start 2023-05-05 07:24:00 / end 2023-05-05 11:39:00
delphy None d1 / start 2023-05-09 09:40:00 / end 2023-05-09 14:00:00
delphy ctd c0 / start 2023-05-05 08:11:15 -0.25 49.46 / end 2023-05-05 08:22:49 -0.25 49.46
delphy ctd c1 / start 2023-05-05 08:48:50 -0.27 49.46 / end 2023-05-05 08:54:02 -0.27 49.46
delphy ctd c2 / start 2023-05-05 09:38:05 -0.46 49.35 / end 2023-05-05 09:41:09 -0.46 49.35
delphy ctd c3 / start 2023-05-05 10:02:21 -0.35 49.38 / end 2023-05-05 10:07:10 -0.35 49.46
delphy ctd c4 / start 2023-05-05 10:36:58 -0.31 49.34 / end 2023-05-05 10:39:12 -0.31 49.34
delphy ctd c5 / start 2023-05-05 10:58:33 -0.25 49.34 / end 2023-05-05 11:02:45 -0.25 49.34
drifter0 None d0 / start 2023-05-05 08:17:00 -0.25 49.34 / end 2023-05-09 12:38:00 -0.08 49.69
drifter1 None d0 / start 2023-05-05 08:22:18 -0.26 49.34 / end 2023-05-09 11:56:40 -0.23 49.67
drifter2 None d0 / start 2023-05-05 08:31:58 -0.27 49.33 / end 2023-05-11 12:33:12 0.05 49.65
drifter3 None d0 / start 2023-05-05 08:36:48 -0.27 49.34 / end 2023-05-09 10:30:13 -0.20 49.63
drifter4 None d0 / start 2023-05-05 08:45:38 -0.26 49.34 / end 2023-05-09 12:36:09 -0.26 49.67
drifter5 None d0 / start 2023-05-05 08:47:30 -0.25 49.34 / end 2023-05-09 12:08:44 -0.18 49.68
drifter6 None d0 / start 2023-05-05 08:54:20 -0.24 49.35 / end 2023-05-11 12:00:23 -0.28 49.68
drifter7 None d0 / start 2023-05-05 09:01:08 -0.24 49.35 / end 2023-05-09 11:41:20 -0.23 49.67
drifter8 None d0 / start 2023-05-05 09:06:51 -0.25 49.35 / end 2023-05-11 11:54:13 -0.23 49.65
drifter9 None d0 / start 2023-05-05 09:10:52 -0.26 49.35 / end 2023-05-09 10:34:24 -0.19 49.62
```

### TAOS2 - drifters1:

Dates: 2022/11/14 to 2022/11/18

Meteo: strong northerly event, mostly cloudy.

Data cleaning status:

| data type | raw data availability | cleaned | quantity |
|:-----|:------:|:---------:|:---:|
| drifters | X | X | 8 |
| drifter - proto | X | X | 1 |
| yuco | X | X | 4 transects, 5 vertical casts |
| ctd hydrophone | X | X | 8 |
| ctd delphy | X | X | 7 |
| ctd smile |  |  |  |
| met - arome | X | X | - |
| met - insitu | 0 | 0 | - |
| sea level | 0 | 0 | - |

?Drifter movie created.
?Drifter drift slowly northward then more quickly eastward (with similar wind strength !) and along the coastline.

```
-----  taos2_drifters1 
None None day0 / start 2023-06-12 14:00:00 / end 2023-06-12 19:00:00
None None day1 / start 2023-06-13 07:50:00 / end 2023-06-13 15:30:00
None None day2 / start 2023-06-15 05:50:00 / end 2023-06-15 09:35:00
parceque None d0 / start 2023-06-12 15:48:00 / end 2023-06-12 18:59:00
hydrophone None d0 / start 2023-06-12 14:10:00 / end 2023-06-12 18:48:00
hydrophone ctd c0 / start 2023-06-12 17:24:57 -0.25 49.34 / end 2023-06-12 17:26:23 -0.25 49.34
hydrophone ctd c1 / start 2023-06-12 17:29:30 -0.26 49.33 / end 2023-06-12 17:31:41 -0.26 49.33
hydrophone ctd c2 / start 2023-06-12 17:37:00 -0.26 49.32 / end 2023-06-12 17:39:19 -0.26 49.32
hydrophone ctd c3 / start 2023-06-12 17:44:30 -0.27 49.33 / end 2023-06-12 17:46:43 -0.27 49.33
hydrophone ctd c4 / start 2023-06-12 17:51:35 -0.26 49.33 / end 2023-06-12 17:53:46 -0.26 49.33
hydrophone ctd c5 / start 2023-06-12 18:00:47 -0.25 49.34 / end 2023-06-12 18:02:19 -0.25 49.34
hydrophone ctd c6 / start 2023-06-12 18:06:59 -0.26 49.34 / end 2023-06-12 18:08:45 -0.26 49.34
hydrophone ctd c7 / start 2023-06-12 18:14:17 -0.25 49.34 / end 2023-06-12 18:16:28 -0.25 49.34
hydrophone ctd c8 / start 2023-06-12 18:23:10 -0.28 49.33 / end 2023-06-12 18:25:24 -0.28 49.33
delphy None d0 / start 2023-06-13 07:50:00 / end 2023-06-13 15:28:00
delphy None d1 / start 2023-06-15 05:59:00 / end 2023-06-15 09:32:06
delphy ctd c0 / start 2023-06-13 10:20:20 -0.26 49.46 / end 2023-06-13 10:26:05 -0.26 49.46
delphy ctd c1 / start 2023-06-13 10:59:53 -0.27 49.46 / end 2023-06-13 11:04:58 -0.28 49.33
delphy ctd c2 / start 2023-06-13 13:35:11 -0.46 49.35 / end 2023-06-13 13:36:59 -0.46 49.35
delphy ctd c3 / start 2023-06-13 13:35:11 -0.46 49.35 / end 2023-06-13 13:36:59 -0.46 49.35
delphy ctd c4 / start 2023-06-13 14:09:50 -0.31 49.34 / end 2023-06-13 14:11:50 -0.31 49.34
delphy ctd c5 / start 2023-06-13 14:57:00 -0.25 49.34 / end 2023-06-13 14:58:47 -0.25 49.34
delphy ctd c6 / start 2023-06-15 08:16:28 -0.26 49.45 / end 2023-06-15 08:20:30 -0.26 49.45
delphy ctd c7 / start 2023-06-15 09:00:56 -0.31 49.34 / end 2023-06-15 09:05:58 -0.31 49.34
yuco None d0 / start 2023-06-12 14:58:10 / end 2023-06-12 15:35:02
yuco None d1 / start 2023-06-12 15:50:45 / end 2023-06-12 16:09:04
yuco None d2 / start 2023-06-12 16:49:20 / end 2023-06-12 17:24:44
yuco None d3 / start 2023-06-12 17:45:36 / end 2023-06-12 18:21:38
yuco None c4 / start 2023-06-13 09:20:37 / end 2023-06-13 09:23:46
yuco None c5 / start 2023-06-13 09:59:53 / end 2023-06-13 10:02:48
yuco None c6 / start 2023-06-13 11:07:14 / end 2023-06-13 11:08:33
yuco None c7 / start 2023-06-13 12:38:15 / end 2023-06-13 12:38:45
yuco None c8 / start 2023-06-13 13:40:52 / end 2023-06-13 13:41:58
odo_drifter None station / start 2023-06-12 16:45:00 / end 2023-06-12 18:20:00
odo_drifter None drift / start 2023-06-12 18:50:00 / end 2023-06-14 03:05:00
drifter0 None d0 / start 2023-06-12 16:24:37 -0.25 49.34 / end 2023-06-15 06:14:43 -0.11 49.33
drifter1 None d0 / start 2023-06-12 16:34:57 -0.25 49.33 / end 2023-06-15 06:27:04 -0.12 49.35
drifter2 None d0 / start 2023-06-12 16:43:50 -0.26 49.32 / end 2023-06-15 06:28:53 -0.12 49.35
drifter3 None d0 / start 2023-06-12 16:47:05 -0.27 49.33 / end 2023-06-15 08:55:31 -0.16 49.32
drifter4 None d0 / start 2023-06-12 16:53:13 -0.26 49.33 / end 2023-06-15 08:58:57 -0.15 49.32
drifter5 None d0 / start 2023-06-12 17:12:29 -0.26 49.34 / end 2023-06-15 09:12:31 -0.10 49.32
drifter6 None d0 / start 2023-06-12 17:24:28 -0.27 49.34 / end 2023-06-15 09:05:32 -0.15 49.32
drifter7 None d0 / start 2023-06-12 17:26:17 -0.28 49.33 / end 2023-06-15 08:55:51 -0.16 49.32
```

## other in situ observations

| data type | raw data availability | cleaned | quantity |
|:-----|:------:|:---------:|:---:|
| sea level | X | | |
| river discharge | X |  |  |
| bathymetry | X | X | 2 relevant bathymetries: HOMONIM (SHOM), EDMODNET |
| smile hydrology | X | | |
| smile adcp | X? |  | |
| mars | X | | |


## remote sensing 

### TAOS deployments

- latitude: 49.N - 50.0N
- longitude: 1.5W - 0.5W

**TAOS0-drifters0**

- Dates: 2022/05/11 to 2022/05/13
- Meteo: strong westerly event, mostly cloudy

**TAOS0-drifters1**

- Dates: 2022/05/17 to 2022/05/19
- Meteo: weak wind, few sunny moments

**TAOS1-drifters0**

- Dates: 2022/11/14 to 2022/11/18
- Meteo: strong northerly event, mostly cloudy.

**TAOS1-drifters1**

- Dates: 2022/11/30 to 2022/12/18
- Meteo: moderate north-easterly winds. Some sunny moments (on the 22 for instance).

**TAOS2-drifters0**

- Dates: 2023/05/05 to 2023/05/11
- Meteo: wind?. Mostly cloudy

**TAOS2-drifters1**

- Dates: 2023/06/12 to 2022/06/16
- Meteo: wind?. Very nice sunny days (13/14/15/16/24/25 june for instance).

https://scihub.copernicus.eu/dhus/#/home



## model

- extract tidal currents & have tools to predict currents?



## miscellaneous




## tmp



```
# with additional packages
conda create -n myenv -c conda-forge -c apatlpo python=3.10 \
	pynsitu jupyterlab \
	pyTMD \
	xrft xhistogram \
	seaborn 
conda activate myenv
pip install pynmea2
```

```
# conda create -n pynsitu_direct -c pyviz -c apatlpo python=3.10 pynsitu hvplot geoviews jupyterlab seaborn pyTMD xrft xhistogram
conda create -n pynsitu_direct -c apatlpo python=3.10 pynsitu jupyterlab seaborn pyTMD xrft xhistogram
conda activate pynsitu_direct
conda install -c pyviz hvplot geoviews
# may need to uninstall pynsitu if we want to use a loca copy
```

```
conda create -n pynsitu -f ci/environment.yml
conda activate pynsitu
#conda config --env --add channels pyviz
#conda env update -n pynsitu -c pyviz hvplot geoviews
conda install -c pyviz hvplot geoviews
conda install jupyterlab seaborn pyTMD xrft xhistogram



conda create -n pynsitu python=3.10

conda env update -n pynsitu -f ci/environment.yml
conda env update -n pynsitu jupyterlab \
	pyTMD \
	xrft xhistogram \
	seaborn 


```
