# Documentation

## Merge CSV Export into NetCDF

Merge CSV exports into NetCDF 4 format

```bash
python scripts/merge.py "./data/20241227/Export*.csv" ./data/20241227/reference_gold_20241227.csv ./data/merged-20241227.nc
```

Example result:

```plain
Dataset created!
<xarray.Dataset> Size: 43MB
Dimensions:  (id: 605, time: 8800)
Coordinates:
  * id       (id) int64 5kB 1 2 3 4 5 6 7 8 9 10 ... 53 54 55 56 57 58 59 60 61
    device   (id) int64 5kB 11 11 11 11 11 11 11 11 ... 12 12 12 12 12 12 12 12
    name     (id) <U28 68kB 'MELINDA MAULIDIA RAHMAN' ... 'HERLAMBANG'
Dimensions without coordinates: time
Data variables:
    signal   (id, time) float64 43MB 477.0 489.0 483.0 ... 1.016e+04 1.073e+04
    hb       (id) float64 5kB 15.9 15.5 14.7 9.7 13.3 ... 12.7 14.0 14.1 14.6

Patient count by device
9     121
11    121
12    121
13    121
15    121
dtype: int64
```

## Preprocess Data


