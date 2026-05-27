"""
Plot ABM ground-truth trajectories alongside ABM-MF and Classical MPM
simulations driven by the *estimated* time-varying beta(t) from the
regression.
"""
import os
import pickle
import time
import unicodedata

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon

from utils.util import *

# ------------------------------------------------------------
# settings
# ------------------------------------------------------------
division = 'Medium100'
runs = range(100)
mobility_seeds = range(1)
initialization = 5

start_date = '01012021'
start_date_dt = pd.to_datetime(start_date, format='%d%m%Y')
NDays = 100
end_date = (start_date_dt + pd.Timedelta(days=NDays)).strftime('%d%m%Y')

infection_state = 2

# pull beta_hat from these CI csv files
CI_DIR = f'../ABM_data/Medium/transmission_rates/Initialization{initialization}'
CI_START = '01-01-2021'
CI_END = '11-04-2021'
INIT_FLAG = False  # False = true S and I, True = initialization
N_SEEDS_CSV = 1
N_RUNS_CSV = 100

# fallback beta if a day has no valid beta_hat across any realization
if initialization == 4:
    BETA_FALLBACK = 0.5
elif initialization == 5:
    BETA_FALLBACK = 0.25

# disease + dispersion
nu = 3.0
omega = 9.0
k_disp = 10.0

DATA_DIR = '../Output/PlotData'
MAP_DIR = '../Output/Maps'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

MAP_CMAP = 'rocket_r'
MAP_BIN_EDGES = [0, 1, 25, 50, 75, 100, 125, 150, 175, 200]
MAP_ZERO_COLOR = '#e8e8e8'

_rocket = seaborn.color_palette('rocket', 8)
COLOR_ABM = _rocket[1]
COLOR_MF = _rocket[3]
COLOR_MPM = _rocket[5]

# ------------------------------------------------------------
# output naming: encode beta + mobility level instead of
# division / initialization
# ------------------------------------------------------------
NAME_SUFFIX = 'beta='
if initialization == 5:
    NAME_SUFFIX += '0.25'
else:
    NAME_SUFFIX += '0.5'

if division == 'Medium100' or division == 'High30':
    NAME_SUFFIX += '_mob_30%'
else:
    NAME_SUFFIX += '_mob_80%'


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def nb_sample(rng, mean, k):
    mean = np.asarray(mean, dtype=float)
    out = np.zeros_like(mean)
    pos = mean > 0
    if np.any(pos):
        mu = mean[pos]
        out[pos] = rng.poisson(mu)
    return out


def mean_ci(arr):
    arr = np.array(arr)
    return (arr.mean(axis=0),
            np.quantile(arr, 0.025, axis=0),
            np.quantile(arr, 0.975, axis=0))


def concentration_entropy(I_per_muni, N_per_muni):
    """1 - exp(-D_KL(p_inf || p_pop)) — Gosgens et al. (2021).

    0 = infections distributed proportionally to population
    1 = all infections concentrated in one municipality
    """
    I_per_muni = np.asarray(I_per_muni, dtype=float)
    N_per_muni = np.asarray(N_per_muni, dtype=float)

    total_I = I_per_muni.sum()
    total_N = N_per_muni.sum()
    if total_I <= 0 or total_N <= 0:
        return 0.0

    p_inf = I_per_muni / total_I
    p_pop = N_per_muni / total_N

    mask = (p_inf > 0) & (p_pop > 0)
    d_kl = np.sum(p_inf[mask] * np.log(p_inf[mask] / p_pop[mask]))
    return float(1.0 - np.exp(-d_kl))


def step_classical_mpm_fast(S, E, I, R, N_live, M_bar, beta, rng):
    infect_frac = I / np.maximum(N_live, 1e-12)
    foi = beta * (M_bar @ infect_frac)
    mean_new_E = S * foi
    new_E = np.minimum(S, nb_sample(rng, mean_new_E, k_disp))

    p_EI = 1.0 - np.exp(-1.0 / nu)
    p_IR = 1.0 - np.exp(-1.0 / omega)
    new_I = p_EI * E
    new_R = p_IR * I
    S = S - new_E
    E = E + new_E - new_I
    I = I + new_I - new_R
    R = R + new_R
    return S, E, I, R, new_E


def step_abm_mf_fast(S, E, I, R, N_live, M_bar, w_T, beta, rng):
    infect_frac = I / np.maximum(N_live, 1e-12)
    prevalence_j = w_T @ infect_frac
    foi = beta * (M_bar @ prevalence_j)
    mean_new_E = S * foi
    new_E = np.minimum(S, nb_sample(rng, mean_new_E, k_disp))

    p_EI = 1.0 - np.exp(-1.0 / nu)
    p_IR = 1.0 - np.exp(-1.0 / omega)
    new_I = p_EI * E
    new_R = p_IR * I

    S = S - new_E
    E = E + new_E - new_I
    I = I + new_I - new_R
    R = R + new_R
    return S, E, I, R, new_E


# ------------------------------------------------------------
# load beta_hat(t) from CI csvs
# ------------------------------------------------------------
def _load_beta_series(estimator):
    suffix = 'init' if estimator == 'ABM' else 'MPM_init'
    all_betas = []
    for s in range(N_SEEDS_CSV):
        for r in range(N_RUNS_CSV):
            fn = os.path.join(
                CI_DIR,
                f'CI_{CI_START}-{CI_END}_seed_{s}_perday{r}'
                f'_{suffix}_{INIT_FLAG}.csv'
            )
            if not os.path.exists(fn):
                continue
            df = pd.read_csv(fn)
            if 'beta' not in df.columns:
                continue
            beta = df['beta'].values.astype(float)
            if len(beta) < NDays:
                beta = np.concatenate([beta, np.full(NDays - len(beta), np.nan)])
            else:
                beta = beta[:NDays]
            beta = np.where(beta > 1e-5, beta, np.nan)
            all_betas.append(beta)
    if not all_betas:
        raise RuntimeError(f"No CI csvs found for estimator='{estimator}' in {CI_DIR}")
    arr = np.array(all_betas)
    print(f"  loaded {arr.shape[0]} realizations for {estimator}")
    med = np.nanmedian(arr, axis=0)
    nan_mask = np.isnan(med)
    if nan_mask.any():
        print(f"    {nan_mask.sum()} days have no valid beta; using fallback {BETA_FALLBACK}.")
        med = np.where(nan_mask, BETA_FALLBACK, med)
    return med


print("Loading estimated beta(t) from CI csvs...")
beta_abm_med = _load_beta_series('ABM')
beta_mpm_med = _load_beta_series('MPM')


# ------------------------------------------------------------
# load people, gemeenten, build aggregator + mob matrices
# ------------------------------------------------------------
def _load_people_and_gemeenten(mobility_seed):
    fn = f'../Output/Data/{division}/Seed_{mobility_seed}'
    people_df = pd.read_pickle(f'{fn}/PeopleDF.pkl')
    gemeenten = pd.read_pickle(f'{fn}/Gemeenten.pkl')
    gemeente_list = gemeenten.values.ravel()
    gemeente_to_idx = {g: i for i, g in enumerate(gemeente_list)}
    return fn, people_df, gemeenten, gemeente_list, gemeente_to_idx


_ref_fn, _ref_people, _ref_gemeenten, gemeente_list_ref, gemeente_to_idx_ref = \
    _load_people_and_gemeenten(list(mobility_seeds)[0])
M = len(gemeente_list_ref)

print("Pre-loading per-seed data + mob matrices...")
_t0 = time.time()
_seed_cache = {}
for mobility_seed in mobility_seeds:
    fn, people_df, gemeenten, gemeente_list, gemeente_to_idx = \
        _load_people_and_gemeenten(mobility_seed)

    local_to_canon = np.array(
        [gemeente_to_idx_ref.get(g, -1) for g in gemeente_list], dtype=np.int64
    )
    home = people_df["Home"].values
    muni_idx_local = np.array([gemeente_to_idx[h] for h in home])
    muni_idx = local_to_canon[muni_idx_local]
    valid_persons = (muni_idx >= 0)
    valid_p = np.flatnonzero(valid_persons)
    rows = muni_idx[valid_p]
    cols = valid_p
    data = np.ones_like(valid_p, dtype=np.float64)
    N_persons = len(home)
    aggregator = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(M, N_persons), dtype=np.float64,
    )

    perm = np.full(M, -1, dtype=np.int64)
    for local_i, canon_i in enumerate(local_to_canon):
        if 0 <= canon_i < M:
            perm[canon_i] = local_i
    perm_ok = not (perm < 0).any()

    M_per_day, M_bar_per_day = [], []
    for d in range(NDays + 1):
        date_str = (start_date_dt + pd.Timedelta(days=d)).strftime('%d%m%Y')
        path = f'../Data/Mobility/MobMats/{division}{date_str}.p'
        with open(path, 'rb') as f:
            M_raw_local = pickle.load(f)
        M_raw_local = np.asarray(M_raw_local, dtype=np.float64)
        M_raw = M_raw_local[perm][:, perm] if perm_ok else M_raw_local
        M_per_day.append(M_raw)
        row_sums = M_raw.sum(axis=1, keepdims=True)
        M_bar_per_day.append(M_raw / np.maximum(row_sums, 1e-12))

    N_live_seed = np.asarray(aggregator @ np.ones(N_persons, dtype=np.float64))
    wT_per_day = []
    for M_bar in M_bar_per_day:
        numerator = M_bar * N_live_seed[:, None]
        denominator = numerator.sum(axis=0, keepdims=True)
        w = numerator / np.maximum(denominator, 1e-12)
        wT_per_day.append(w.T)

    _seed_cache[mobility_seed] = {
        'fn': fn, 'aggregator': aggregator, 'muni_idx': muni_idx,
        'valid_persons': valid_persons,
        'M_per_day': M_per_day, 'M_bar_per_day': M_bar_per_day,
        'wT_per_day': wT_per_day, 'N_live': N_live_seed,
    }
print(f"  cached {len(_seed_cache)} mobility seeds in {time.time() - _t0:.2f}s")


# ------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------
print("Running simulations with estimated beta(t)...")
all_abm_new, all_mpm_new, all_mf_new = [], [], []
all_abm_S, all_mpm_S, all_mf_S = [], [], []
abm_I_per_muni_runs, mpm_I_per_muni_runs, mf_I_per_muni_runs = [], [], []
pop_per_muni = None
N_total = None

for run in runs:
    if run % 10 == 0:
        print(f"  run {run}/{len(list(runs))}")
    rng = np.random.default_rng(seed=run)
    for mobility_seed in mobility_seeds:
        cache = _seed_cache[mobility_seed]
        fn = cache['fn']
        aggregator = cache['aggregator']
        M_per_day, M_bar_per_day, wT_per_day = (
            cache['M_per_day'], cache['M_bar_per_day'], cache['wT_per_day']
        )
        N_live = cache['N_live']
        if N_total is None:
            N_total = float(N_live.sum())

        # load ABM ground truth
        status_raw = scipy.sparse.load_npz(
            f'{fn}/Initialization{initialization}/{start_date}-{end_date}/Status_{run}.npz'
        )
        status = status_raw.toarray()
        T, N = status.shape

        # ABM per-muni infectious count via one matmul (for maps)
        is_inf = (status == infection_state).astype(np.float64)
        abm_I_per_muni = (aggregator @ is_inf.T).T.astype(np.int64)
        abm_I_per_muni_runs.append(abm_I_per_muni)

        # ABM new infections per day (national)
        is_R = (status == (infection_state + 1)).astype(np.float64)
        cum_post_E_total = (is_inf + is_R).sum(axis=1)
        new_inf_per_day = np.zeros(T, dtype=np.float64)
        new_inf_per_day[0] = is_inf[0].sum()
        new_inf_per_day[1:] = np.maximum(0, np.diff(cum_post_E_total))
        all_abm_new.append(new_inf_per_day)

        # ABM susceptibles per day (national, as fraction)
        S_abm_per_day = (status == 0).sum(axis=1).astype(np.float64) / float(N)
        all_abm_S.append(S_abm_per_day)

        # initial compartments per muni
        s0 = status[0]
        S0 = aggregator @ (s0 == 0).astype(np.float64)
        E0 = aggregator @ (s0 == 1).astype(np.float64)
        I0 = aggregator @ (s0 == 2).astype(np.float64)
        R0 = aggregator @ (s0 == 3).astype(np.float64)

        S_mpm, E_mpm, I_mpm, R_mpm = S0.copy(), E0.copy(), I0.copy(), R0.copy()
        S_mf, E_mf, I_mf, R_mf = S0.copy(), E0.copy(), I0.copy(), R0.copy()
        if pop_per_muni is None:
            pop_per_muni = N_live.copy()

        mpm_inf_new = np.zeros((T, M))
        mf_inf_new = np.zeros((T, M))
        mpm_inf_new[0] = I0
        mf_inf_new[0] = I0

        # per-muni infectious-count trajectories (for maps)
        mpm_I_track = np.zeros((T, M))
        mf_I_track = np.zeros((T, M))
        mpm_I_track[0] = I0
        mf_I_track[0] = I0

        # S trajectories (national fraction)
        mpm_S = np.zeros(T)
        mf_S = np.zeros(T)
        mpm_S[0] = S0.sum() / N_total
        mf_S[0] = S0.sum() / N_total

        for d in range(T - 1):
            M_bar = M_bar_per_day[d]
            wT = wT_per_day[d]
            beta_mpm_d = beta_mpm_med[d] if d < len(beta_mpm_med) else BETA_FALLBACK
            beta_mf_d = beta_abm_med[d] if d < len(beta_abm_med) else BETA_FALLBACK

            S_mpm, E_mpm, I_mpm, R_mpm, ne = step_classical_mpm_fast(
                S_mpm, E_mpm, I_mpm, R_mpm, N_live, M_bar, beta_mpm_d, rng)
            mpm_inf_new[d + 1] = ne
            mpm_I_track[d + 1] = I_mpm
            mpm_S[d + 1] = S_mpm.sum() / N_total

            S_mf, E_mf, I_mf, R_mf, ne = step_abm_mf_fast(
                S_mf, E_mf, I_mf, R_mf, N_live, M_bar, wT, beta_mf_d, rng)
            mf_inf_new[d + 1] = ne
            mf_I_track[d + 1] = I_mf
            mf_S[d + 1] = S_mf.sum() / N_total

        all_mpm_new.append(mpm_inf_new.sum(axis=1))
        all_mf_new.append(mf_inf_new.sum(axis=1))
        all_mpm_S.append(mpm_S)
        all_mf_S.append(mf_S)
        mpm_I_per_muni_runs.append(mpm_I_track)
        mf_I_per_muni_runs.append(mf_I_track)


# ------------------------------------------------------------
# aggregate
# ------------------------------------------------------------
# per-muni infectious means across realizations (for maps + entropy)
abm_I_mean = np.mean(np.stack(abm_I_per_muni_runs, axis=0), axis=0)
mpm_I_mean = np.mean(np.stack(mpm_I_per_muni_runs, axis=0), axis=0)
mf_I_mean = np.mean(np.stack(mf_I_per_muni_runs, axis=0), axis=0)
T_total = abm_I_mean.shape[0]

# ------------------------------------------------------------
# Concentration entropy over time (driven by estimated beta)
# ------------------------------------------------------------
ent_abm = np.array([concentration_entropy(abm_I_mean[t], pop_per_muni) for t in range(T_total)])
ent_mpm = np.array([concentration_entropy(mpm_I_mean[t], pop_per_muni) for t in range(T_total)])
ent_mf = np.array([concentration_entropy(mf_I_mean[t], pop_per_muni) for t in range(T_total)])

plt.figure(figsize=(9, 5))
plt.plot(np.arange(T_total), ent_abm, label='ABM (ground truth)', linewidth=2, color=COLOR_ABM)
plt.plot(np.arange(T_total), ent_mf,
         label=r'ABM mean-field with $\hat{\beta}_{ABM-MF}(t)$',
         linestyle='-.', color=COLOR_MF)
plt.plot(np.arange(T_total), ent_mpm,
         label=r'Classical MPM with $\hat{\beta}_{MPM}(t)$',
         linestyle='--', color=COLOR_MPM)
plt.xlabel('day')
plt.ylabel('Concentration')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(f'concentration_entropy_estimated_beta_{NAME_SUFFIX}.png')
plt.close()

pd.DataFrame({
    'day': np.arange(T_total),
    'ABM': ent_abm,
    'ABM_mean_field': ent_mf,
    'Classical_MPM': ent_mpm,
}).to_csv(os.path.join(DATA_DIR, f'concentration_entropy_estimated_beta_{NAME_SUFFIX}.csv'), index=False)
print("Wrote concentration-entropy plot + csv.")


# ------------------------------------------------------------
# Daily maps (3 panels: ABM | ABM mean-field | Classical MPM)
# ------------------------------------------------------------
def _norm_name(s):
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = ''.join(ch if ch.isalnum() else ' ' for ch in s)
    return ' '.join(s.split())


# 2018 -> 2021 municipality merges/renames (CBS gemeentelijke herindelingen)
_LEGACY_REDIRECTS = {
    # 2019
    'aalburg': 'altena', 'werkendam': 'altena', 'woudrichem': 'altena',
    'nuth': 'beekdaelen', 'onderbanken': 'beekdaelen', 'schinnen': 'beekdaelen',
    'binnenmaas': 'hoeksche waard', 'cromstrijen': 'hoeksche waard',
    'korendijk': 'hoeksche waard', 'oud beijerland': 'hoeksche waard',
    'strijen': 'hoeksche waard',
    'leerdam': 'vijfheerenlanden', 'vianen': 'vijfheerenlanden',
    'zederik': 'vijfheerenlanden',
    'geldermalsen': 'west betuwe', 'lingewaal': 'west betuwe',
    'neerijnen': 'west betuwe',
    'noordwijkerhout': 'noordwijk',
    'haarlemmerliede en spaarnwoude': 'haarlemmermeer',
    'dongeradeel': 'noardeast fryslan', 'ferwerderadiel': 'noardeast fryslan',
    'kollumerland en nieuwkruisland': 'noardeast fryslan',
    'leeuwarderadeel': 'leeuwarden', 'littenseradiel': 'leeuwarden',
    'bedum': 'het hogeland', 'de marne': 'het hogeland',
    'eemsmond': 'het hogeland', 'winsum': 'het hogeland',
    'grootegast': 'westerkwartier', 'leek': 'westerkwartier',
    'marum': 'westerkwartier', 'zuidhorn': 'westerkwartier',
    'menterwolde': 'midden groningen',
    'hoogezand sappemeer': 'midden groningen', 'slochteren': 'midden groningen',
    'giessenlanden': 'molenlanden', 'molenwaard': 'molenlanden',
    'groningen': 'groningen', 'haren': 'groningen', 'ten boer': 'groningen',
    # 2020
    'appingedam': 'eemsdelta', 'delfzijl': 'eemsdelta', 'loppersum': 'eemsdelta',
    'haaren': 'oisterwijk',
    # 2021
    'boxmeer': 'land van cuijk', 'cuijk': 'land van cuijk',
    'grave': 'land van cuijk', 'mill en sint hubert': 'land van cuijk',
    'sint anthonis': 'land van cuijk',
    'heerhugowaard': 'dijk en waard', 'langedijk': 'dijk en waard',
    'beemster': 'purmerend',
    'landerd': 'maashorst', 'uden': 'maashorst',
    'brielle': 'voorne aan zee', 'hellevoetsluis': 'voorne aan zee',
    'westvoorne': 'voorne aan zee',
    'weesp': 'amsterdam',
}


def _build_geo_df():
    try:
        name_norm_to_code = {_norm_name(nm): code for code, nm in GM_to_name.items()}

        matched_codes = []
        unmatched = []
        for nm in gemeente_list_ref:
            key = _norm_name(nm)
            code = name_norm_to_code.get(key)
            if code is None and key in _LEGACY_REDIRECTS:
                code = name_norm_to_code.get(_norm_name(_LEGACY_REDIRECTS[key]))
            matched_codes.append(code)
            if code is None:
                unmatched.append(nm)

        n_missing = sum(1 for c in matched_codes if c is None)
        if n_missing > 0:
            print(f"[maps] warning: {n_missing}/{len(gemeente_list_ref)} "
                  f"gemeenten could not be matched (first few: {unmatched[:5]}).")

        gdf = gemeente_shapes.reindex(matched_codes)
        gdf = gdf.assign(_gemeente_name=list(gemeente_list_ref))
        return gdf.reset_index(drop=False)
    except Exception as e:
        print(f"[maps] could not build geo dataframe ({e}) skipping maps.")
        return None


geo_df = _build_geo_df()

if geo_df is not None:
    _geom_valid_mask = geo_df.geometry.notna().values
    _geo_df_valid = geo_df[_geom_valid_mask].copy()
else:
    _geom_valid_mask, _geo_df_valid = None, None

# Colormap with a special "zero" bin so empty munis don't blend into the gradient
_BOUNDS = list(MAP_BIN_EDGES)
_N_BINS = len(_BOUNDS) - 1
_grad = plt.get_cmap(MAP_CMAP)(np.linspace(0, 1, _N_BINS - 1))
_CMAP = mcolors.ListedColormap([MAP_ZERO_COLOR] + list(_grad))
_NORM = mcolors.BoundaryNorm(boundaries=_BOUNDS, ncolors=_N_BINS, clip=True)


def _geoms_to_patches(geoms):
    """Convert shapely (Multi)Polygons -> matplotlib Polygon patches.
    Returns (patches, owner_idx) so per-muni values can be broadcast
    to all patches owned by that muni (multipolygon munis).
    """
    patches, owner_idx = [], []
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for part in parts:
            patches.append(MplPolygon(np.asarray(part.exterior.coords)))
            owner_idx.append(i)
    return patches, np.asarray(owner_idx, dtype=np.int64)


if _geo_df_valid is not None:
    _patches_template, _owner_idx = _geoms_to_patches(_geo_df_valid.geometry.values)
else:
    _patches_template, _owner_idx = None, None

# Build the figure once per day, just update colors via set_array.
# Disable autolayout so titles changing length doesn't nudge axes.
_fig, _axes = plt.subplots(1, 3, figsize=(15, 6))
_fig.set_layout_engine('none')
_titles = ['ABM (ground truth)', r'ABM mean-field $\hat{\beta}_{ABM-MF}(t)$',
           r'Classical MPM $\hat{\beta}_{MPM}(t)$']
_collections = []
for ax, title in zip(_axes, _titles):
    if _patches_template is not None:
        coll = PatchCollection(
            [MplPolygon(p.get_xy()) for p in _patches_template],
            cmap=_CMAP, norm=_NORM,
            edgecolor='lightgrey', linewidth=0.2,
        )
        coll.set_array(np.zeros(len(_patches_template)))
        ax.add_collection(coll)
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        ax.set_axis_off()
        _collections.append(coll)
    else:
        _collections.append(None)
    ax.set_title(f'{title}\nday')

# colorbar shows the gradient only (without grey 0-bin)
_BOUNDS_CBAR = _BOUNDS[1:]
_CMAP_CBAR = mcolors.ListedColormap(list(_grad))
_NORM_CBAR = mcolors.BoundaryNorm(
    boundaries=_BOUNDS_CBAR, ncolors=len(_BOUNDS_CBAR) - 1, clip=True
)
_sm = plt.cm.ScalarMappable(cmap=_CMAP_CBAR, norm=_NORM_CBAR)
_sm.set_array([])
_cbar = _fig.colorbar(
    _sm, ax=_axes, orientation='horizontal',
    fraction=0.04, pad=0.04,
    ticks=_BOUNDS_CBAR, spacing='proportional',
)
_cbar.set_label('Infectious agents (mean)')
_tick_labels = [str(b) for b in _BOUNDS_CBAR]
_tick_labels[-1] = f'{_BOUNDS_CBAR[-1]}+'
_cbar.ax.set_xticklabels(_tick_labels, fontsize=9)

_fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.18, wspace=0.01)


def _plot_one_day(t, abm_vals, mf_vals, mpm_vals):
    series = [abm_vals, mf_vals, mpm_vals]
    for ax, title, vals, coll in zip(_axes, _titles, series, _collections):
        if coll is not None:
            vals_valid = np.asarray(vals)[_geom_valid_mask]
            coll.set_array(vals_valid[_owner_idx])
        else:
            ax.clear()
            ax.bar(np.arange(len(vals)), vals, color=_CMAP(_NORM(vals)))
            ax.set_xlabel('municipality index')
            ax.set_ylabel('active infections')
        ax.set_title(f'{title}\nday {t}')

    out_path = os.path.join(MAP_DIR, f'disp_estbeta_{NAME_SUFFIX}_t={t}.png')
    _fig.savefig(out_path, dpi=120)

    pd.DataFrame({
        'gemeente': gemeente_list_ref,
        'population': pop_per_muni,
        'ABM': abm_vals,
        'ABM_mean_field': mf_vals,
        'Classical_MPM': mpm_vals,
    }).to_csv(os.path.join(DATA_DIR, f'disp_estbeta_t={t}_{NAME_SUFFIX}.csv'), index=False)


MAP_EVERY = 10
map_days = list(range(0, T_total, MAP_EVERY))
if map_days[-1] != T_total - 1:
    map_days.append(T_total - 1)

for t in map_days:
    _plot_one_day(t, abm_I_mean[t], mf_I_mean[t], mpm_I_mean[t])
plt.close(_fig)

print(f"Saved {len(map_days)} maps to {MAP_DIR}/")
print(f"Saved per-figure CSVs to {DATA_DIR}/")