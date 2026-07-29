# Input Data Format

wq_framework expects two files: a station registry and a measurements file.

## 1. `stations.csv`

One row per monitoring station:

| column | type | description |
|---|---|---|
| `station_id` | string | unique identifier, referenced by the measurements file |
| `station_name` | string | human-readable name |
| `latitude` | float | decimal degrees |
| `longitude` | float | decimal degrees |

Example:
```csv
station_id,station_name,latitude,longitude
DAM_UP,Upstream of Dam,32.1234,51.5678
DAM_DOWN,Downstream of Dam,32.1000,51.6000
```

Add a new station by adding a new row — no code or config changes needed.

## 2. Measurements file (`.csv` or `.xlsx`)

Wide format: one row per station/date, one column per parameter.

| column | type | description |
|---|---|---|
| `station_id` | string | must match a row in `stations.csv` |
| `date` | date | **ISO 8601 only** (`YYYY-MM-DD`), Gregorian calendar |
| *(parameter columns)* | float | any subset of the parameters registered in `config/schema.yaml` |

Example:
```csv
station_id,date,TDS,pH,HCO3,Cl,SO4,Ca,Mg,Na,K,EC,discharge
DAM_UP,2010-03-21,450,7.8,210,85,120,60,25,40,3.2,780,12.5
```

**Important**: if your source data uses a non-Gregorian calendar (e.g. the
Persian/Shamsi calendar), converting the dates to `YYYY-MM-DD` is your
responsibility before loading the file — the framework validates the date
format but does not perform calendar conversion.

## 3. Parameter registry (`config/schema.yaml`)

Every parameter column the measurements file may contain must be declared here,
with its unit and (optional) sanity bounds:

```yaml
parameters:
  TDS:
    unit: mg/L
    required: false
    min_value: 0        # values below this are flagged as invalid
    max_value: null      # null = no upper sanity bound
```

- A column present in your measurements file but **not** declared here will
  cause the file to be rejected with a clear error — add it to `schema.yaml`
  first.
- `min_value`/`max_value` are *sanity* bounds (catching impossible values like
  negative concentrations), not statistical outlier detection — that happens
  in a later pipeline stage.
- To add a new parameter (e.g. dissolved oxygen), add a new entry under
  `parameters:` — no code changes needed.

## 4. What happens to invalid values

If a cell falls outside its parameter's `min_value`/`max_value`:
1. **Only that cell** is set to missing — the rest of the row is kept as-is.
2. You'll see a summary of every flagged cell (station, date, parameter, value,
   reason).
3. You'll be asked to confirm before the pipeline continues. If you decline,
   the run stops so you can correct the source file yourself.

No row or file is ever silently dropped because of an out-of-range value.
