import os
import re
import warnings

import geopandas as gpd
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd

# default matplotlib params
params = {
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    'xtick.labelsize': 'xx-large',
    'ytick.labelsize': 'xx-large',
    'lines.markersize': 8,
    'figure.autolayout': True,
}
pylab.rcParams.update(params)

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f791bf', '#a65628',
          '#984ea3', '#e41a1c', '#dede00'] * 10
markers = ['o', 's', 'p', 'd', '*', '^']

MOBILITY_TYPES = ["GoogleCBS", "TomTom", "gravity", "gravity_jobs", "radiation"]
CAPPED_MOBILITY_TYPES = {"gravity", "gravity_jobs", "radiation"}
MOBILITY_COLOR_MAP = {m: colors[i % len(colors)] for i, m in enumerate(MOBILITY_TYPES)}

DATE_STR = "01-01-2021"
MUNI_LIMIT = None
N_ITER = 10
DISPERSION = 200
TOTAL_MOBILITY = 3_000_000.0


def _norm_gm_code(x) -> str:
    """'164' / '164.0' / 'gm164' / 'GM0164' -> 'GM0164'."""
    s = str(x).strip()
    if re.fullmatch(r"\d+(\.0)?", s):
        return f"GM{int(float(s)):04d}"
    m = re.fullmatch(r"GM(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"GM{int(m.group(1)):04d}"
    return s


def make_gemeenteshapes():
    """Load 2021 municipality shapefile, indexed by GM_CODE."""
    shapefile_path = "data/shapefiles/gemeenten_2021_v3.shp"
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.loc[gdf["H2O"] == "NEE"].copy()  # drop water

    if "GM_CODE" not in gdf.columns:
        raise ValueError(f"Expected GM_CODE in shapefile. Columns={list(gdf.columns)}")

    gdf["GM_CODE"] = gdf["GM_CODE"].astype(str).map(_norm_gm_code)

    if "GM_NAAM" in gdf.columns:
        gdf = gdf.rename(columns={"GM_NAAM": "name"})
    elif "name" not in gdf.columns:
        gdf["name"] = gdf["GM_CODE"]

    return gdf.set_index("GM_CODE", drop=True)


def load_mobility_matrix(mobility_type, date_str, municipalities_index,
                        base_dir="mobility_data",
                        normalize=False, drop_self=False,
                        total_mobility=TOTAL_MOBILITY):
    """Load a daily OD mobility CSV: woon, bezoek, totaal_aantal_bezoekers.
    Returns a square DataFrame indexed by GM codes.
    Files are already rescaled and have diagonal 0, so normalize/drop_self
    default to False.
    """
    path = os.path.join(base_dir, mobility_type, f"{date_str}.csv")
    municipalities_index = pd.Index([_norm_gm_code(x) for x in municipalities_index])

    if not os.path.exists(path):
        warnings.warn(f"Missing mobility file {path}. Using zeros.")
        return pd.DataFrame(0.0, index=municipalities_index, columns=municipalities_index)

    df = pd.read_csv(path, dtype={"woon": str, "bezoek": str})

    required = {"woon", "bezoek", "totaal_aantal_bezoekers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}.")

    df["woon"] = df["woon"].map(_norm_gm_code)
    df["bezoek"] = df["bezoek"].map(_norm_gm_code)
    df["totaal_aantal_bezoekers"] = (
        pd.to_numeric(df["totaal_aantal_bezoekers"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    M = df.pivot_table(
        index="woon", columns="bezoek",
        values="totaal_aantal_bezoekers",
        aggfunc="sum", fill_value=0.0,
    )
    M = M.reindex(index=municipalities_index, columns=municipalities_index, fill_value=0.0)

    if drop_self:
        np.fill_diagonal(M.values, 0.0)

    if normalize:
        total = float(M.values.sum())
        if total > 0:
            M = (M / total) * float(total_mobility)
        else:
            M.loc[:, :] = 0.0

    return M


# build the shared shapefile-based objects on import
gemeente_shapes = make_gemeenteshapes()

N_MUNI = MUNI_LIMIT if MUNI_LIMIT is not None else len(gemeente_shapes)
if MUNI_LIMIT is not None:
    gemeente_shapes = gemeente_shapes.iloc[:MUNI_LIMIT]

municipalities_index = gemeente_shapes.index.tolist()

if "AANT_INW" not in gemeente_shapes.columns:
    raise ValueError("Expected AANT_INW (population) in shapefile.")
inhabitants = gemeente_shapes["AANT_INW"].astype(float)

# GM code -> name lookup
_gdf = gpd.read_file("data/shapefiles/gemeenten_2021_v3.shp")
GM_to_name = dict(zip(_gdf["GM_CODE"].astype(str), _gdf["GM_NAAM"].astype(str)))
municipality_names = [GM_to_name.get(code, code) for code in municipalities_index]


def find_label(mob):
    if mob == 'Google':
        return 'Google/CBS'
    if mob == 'gravity_jobs':
        return 'gravity-ext'
    if mob == 'Mix':
        return 'ground-truth'
    return mob


compartments = [
    'susceptible', 'exposed', 'infected_tested', 'infected_nottested',
    'removed_tested', 'removed_nottested',
]