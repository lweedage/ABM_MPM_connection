"""
Visualisations of D_j(t) over the Netherlands.

    D_j(t) = I_j^ARE(t) / N_j^ARE(t)  -  I_j^LIVE(t) / N_j^LIVE
           = sum_k w_kj(t) * ( I_k^LIVE(t)/N_k^LIVE - I_j^LIVE(t)/N_j^LIVE )

with visitor shares
    w_kj(t) = M_hat[k,j] * N_k / sum_l M_hat[l,j] * N_l.

Sign interpretation
    D_j > 0 : people PRESENT at j have higher prevalence than j's residents
              (j attracts visitors from higher-prevalence places)
    D_j < 0 : people PRESENT at j have lower prevalence than j's residents
              (j attracts visitors from lower-prevalence places)
    D_j = 0 : the MPM and the ABM-MF agree locally at j.

The sign of the *global* gap between beta-hat-MPM and beta-hat-ABM-MF on day t
is determined by  C_num(t) = sum_i S_i(t) sum_j M_hat[i,j] D_j(t)
    C_num > 0  <=>  beta-hat-MPM > beta-hat-ABM-MF
    C_num < 0  <=>  beta-hat-MPM < beta-hat-ABM-MF

This module provides:
    1. plot_D_snapshots  : choropleth maps of D_j at selected days.
    2. plot_D_time       : heatmap with municipalities on y-axis, days on x-axis.
    3. plot_C_trajectory : the global gap term C_num(t) / C_denom(t) over time.

Wired against:
    estimation.rivm_loader.RivmLoader     (I/N/S arrays)
    estimation.estimate_rates._load_mob   (same row-normalised mobility used by the estimators)
    utils.util.gemeente_shapes            (NL shapefile indexed by GM_CODE)
    utils.util.municipalities_index       (column order of the RIVM arrays)
"""
from __future__ import annotations

import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from estimation import rivm_loader
from estimation.estimate_rates import _load_mob
from utils.util import gemeente_shapes, municipalities_index

# ---------------------------------------------------------------------------
# True beta from init_val (mirrors plot_transmission_rates.py)
# ---------------------------------------------------------------------------

TRUE_BETA = {4: 0.5, 5: 0.25}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_visitor_shares(mob: np.ndarray, N_live: np.ndarray) -> np.ndarray:
    """w_kj = mob[k,j] * N_k / sum_l mob[l,j] * N_l (same as in z_vec_ABM)."""
    num = mob * N_live[:, None]  # (n, n), num[k, j]
    denom = num.sum(axis=0, keepdims=True)  # (1, n), denom[0, j]
    denom = np.maximum(denom, 1e-10)
    return num / denom


def compute_D_t(mob: np.ndarray, I_live: np.ndarray, N_live: np.ndarray) -> np.ndarray:
    """D_j at a single time step. Inputs (n,n), (n,), (n,)."""
    w = compute_visitor_shares(mob, N_live)
    i_frac = I_live / np.maximum(N_live, 1e-10)
    present_prev = w.T @ i_frac  # I^ARE_j / N^ARE_j
    return present_prev - i_frac


def compute_C_num_t(mob: np.ndarray, S_live: np.ndarray,
                    I_live: np.ndarray, N_live: np.ndarray) -> tuple[float, float]:
    """Global gap numerator and denominator at a single time step.

    Returns (C_num, C_denom) with
        C_num   = sum_i S_i sum_j mob[i,j] D_j
        C_denom = sum_i S_i sum_j mob[i,j] I_j/N_j
    so  beta_MPM / beta_ABM = 1 + C_num / C_denom.
    """
    D = compute_D_t(mob, I_live, N_live)
    i_frac = I_live / np.maximum(N_live, 1e-10)
    return (
        float((S_live * (mob @ D)).sum()),
        float((S_live * (mob @ i_frac)).sum()),
    )


def compute_D_full(
        start_date: str,
        scenario: str,
        seed: int,
        run: int,
        init_val: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dt.datetime]:
    """Compute D[t, j] and (C_num[t], C_denom[t]) over the full time range.

    Returns
    -------
    D            : (T, n) array of D_j(t) values.
    C_num        : (T,) array.
    C_denom      : (T,) array.
    start_date_dt: parsed start date.
    """
    start_date_dt = dt.datetime.strptime(start_date, '%d-%m-%Y')
    rivm = rivm_loader.RivmLoader(scenario, start_date_dt, seed, run, init_val)

    I_true = rivm.I_true_arr.astype(np.float64)
    S_true = rivm.S_true_arr.astype(np.float64)
    N_live = rivm.N_arr[0, :].astype(np.float64)

    if I_true.shape[1] != len(municipalities_index):
        raise ValueError(
            f"RIVM array has {I_true.shape[1]} cols but municipalities_index "
            f"has {len(municipalities_index)} entries -- column ordering mismatch."
        )

    T, n = I_true.shape
    D_full = np.full((T, n), np.nan)
    C_num = np.full(T, np.nan)
    C_denom = np.full(T, np.nan)

    for d in range(T):
        current_date = start_date_dt + dt.timedelta(days=d)
        try:
            mob = _load_mob(scenario, current_date.strftime('%d%m%Y'), False)
        except FileNotFoundError:
            continue
        D_full[d] = compute_D_t(mob, I_true[d], N_live)
        cn, cd = compute_C_num_t(mob, S_true[d], I_true[d], N_live)
        C_num[d] = cn
        C_denom[d] = cd

    return D_full, C_num, C_denom, start_date_dt


# ---------------------------------------------------------------------------
# Tag for output filenames
# ---------------------------------------------------------------------------

def _tag(scenario: str, seed: int, run: int, init_val: int) -> str:
    return f'{scenario}_init{init_val}_seed{seed}_run{run}'


# ---------------------------------------------------------------------------
# Plot 1: spatial snapshots
# ---------------------------------------------------------------------------

def plot_D_snapshots(
        start_date: str = '01-01-2021',
        days: tuple[int, ...] = (20, 40, 60, 80),
        scenario: str = 'Medium',
        seed: int = 0,
        run: int = 0,
        init_val: int = 4,
        out_dir: str = '../Output/Plots',
        symmetric_clim: float | None = None,
):
    """Choropleth maps of D_j at selected days."""
    D_full, C_num, C_denom, start_date_dt = compute_D_full(
        start_date, scenario, seed, run, init_val
    )

    days = tuple(d for d in days if d < D_full.shape[0])
    print("day |   C_num         C_denom      beta_MPM/beta_ABMMF - 1")
    print("----+-------------------------------------------------------")
    for d in days:
        ratio = C_num[d] / C_denom[d] if abs(C_denom[d]) > 0 else np.nan
        print(f"{d:3d} | {C_num[d]:+.4e}   {C_denom[d]:.4e}   {ratio:+.4e}")

    if symmetric_clim is None:
        symmetric_clim = float(np.nanmax(np.abs(D_full[list(days)])))
        symmetric_clim = max(symmetric_clim, 1e-12)
    norm = TwoSlopeNorm(vmin=-symmetric_clim, vcenter=0.0, vmax=+symmetric_clim)
    cmap = 'RdBu_r'

    gdf = gemeente_shapes.copy()
    fig, axes = plt.subplots(1, len(days), figsize=(4.0 * len(days), 5.5),
                             constrained_layout=True)
    if len(days) == 1:
        axes = [axes]

    for ax, d in zip(axes, days):
        D_series = dict(zip(municipalities_index, D_full[d]))
        gdf['D'] = gdf.index.map(D_series)
        gdf.plot(
            column='D', ax=ax, cmap=cmap, norm=norm,
            edgecolor='white', linewidth=0.1,
            missing_kwds={'color': 'lightgrey'},
        )
        ax.set_title(f"day {d}", fontsize=12)
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes, shrink=0.7, orientation='vertical',
        label=r'$D_j(t)\;=\;I^{ARE}_j/N^{ARE}_j \;-\; I^{LIVE}_j/N^{LIVE}_j$',
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'D_snapshots_NL_{_tag(scenario, seed, run, init_val)}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: municipality x time heatmap
# ---------------------------------------------------------------------------

def plot_D_time(
        start_date: str = '01-01-2021',
        scenario: str = 'Medium',
        seed: int = 0,
        run: int = 0,
        init_val: int = 4,
        top_k: int | None = 40,
        sort_by: str = 'abs_D',
        out_dir: str = '../Output/Plots',
        symmetric_clim: float | None = None,
):
    """Heatmap with municipalities on y-axis and days on x-axis.

    Parameters
    ----------
    top_k    : show only the top-K municipalities, or None for all ~350.
    sort_by  : 'abs_D'   - by max |D_j| over time (most divergent first)
               'I'       - by peak infectious load over time
               'name'    - alphabetical
    """
    D_full, C_num, C_denom, start_date_dt = compute_D_full(
        start_date, scenario, seed, run, init_val
    )
    T, n = D_full.shape

    # Need names for the y-axis. Build via the same lookup as utils.util.
    import geopandas as gpd
    _gdf = gpd.read_file("../data/shapefiles/gemeenten_2021_v3.shp")
    GM_to_name = dict(zip(_gdf["GM_CODE"].astype(str), _gdf["GM_NAAM"].astype(str)))
    muni_names = np.array([GM_to_name.get(code, code) for code in municipalities_index])

    # Sort municipalities (most interesting first).
    if sort_by == 'abs_D':
        score = np.nanmax(np.abs(D_full), axis=0)
        order = np.argsort(-score)
    elif sort_by == 'I':
        rivm = rivm_loader.RivmLoader(
            scenario, start_date_dt, seed, run, init_val,
        )
        I_true = rivm.I_true_arr.astype(np.float64)
        score = I_true.max(axis=0)
        order = np.argsort(-score)
    elif sort_by == 'name':
        order = np.argsort(muni_names)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by!r}")

    if top_k is not None:
        order = order[:top_k]
    D_plot = D_full[:, order]  # (T, K)
    names_plot = muni_names[order]

    if symmetric_clim is None:
        symmetric_clim = float(np.nanmax(np.abs(D_plot)))
        symmetric_clim = max(symmetric_clim, 1e-12)
    norm = TwoSlopeNorm(vmin=-symmetric_clim, vcenter=0.0, vmax=+symmetric_clim)

    # Figure size scales with the number of rows.
    K = D_plot.shape[1]
    fig_h = max(4.0, 0.16 * K + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h), constrained_layout=True)
    im = ax.imshow(
        D_plot.T, aspect='auto', cmap='RdBu_r', norm=norm,
        interpolation='nearest',
        extent=[0, T, K, 0],  # [xmin, xmax, ymin, ymax] - flipped y so row 0 is at top
    )
    ax.set_yticks(np.arange(K) + 0.5)
    ax.set_yticklabels(names_plot, fontsize=7)
    ax.set_xlabel('day')
    ttl = (
        f'$D_j(t)$ per municipality '
        f'({"top " + str(top_k) if top_k else "all"}, sorted by {sort_by})'
    )
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label=r'$D_j(t) = I^{ARE}_j/N^{ARE}_j - I^{LIVE}_j/N^{LIVE}_j$')

    os.makedirs(out_dir, exist_ok=True)
    suffix = f'top{top_k}' if top_k else 'all'
    out_path = os.path.join(
        out_dir,
        f'D_time_NL_{_tag(scenario, seed, run, init_val)}_{suffix}_by_{sort_by}.png',
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: global gap term over time
# ---------------------------------------------------------------------------

def plot_C_trajectory(
        start_date: str = '01-01-2021',
        scenario: str = 'Medium',
        seed: int = 0,
        run: int = 0,
        init_val: int = 4,
        out_dir: str = '../Output/Plots',
):
    """Plot the ratio beta_MPM / beta_ABM-MF - 1 = C_num / C_denom over time."""
    D_full, C_num, C_denom, _ = compute_D_full(
        start_date, scenario, seed, run, init_val
    )
    ratio = np.where(np.abs(C_denom) > 0, C_num / C_denom, np.nan)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(ratio, color='black', linewidth=1.2)
    ax.axhline(0, color='red', linestyle=':', linewidth=1)
    ax.set_xlabel('day')
    ax.set_ylabel(r'$\hat{\beta}_{MPM}/\hat{\beta}_{ABM\text{-}MF} - 1$')
    true_beta = TRUE_BETA.get(init_val)
    title_extra = f' (true beta = {true_beta})' if true_beta else ''
    ax.set_title('Predicted relative gap between estimators' + title_extra)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f'C_trajectory_{_tag(scenario, seed, run, init_val)}.png',
    )
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")
    return fig


if __name__ == '__main__':
    common = dict(
        start_date='01-01-2021',
        scenario='Medium',
        seed=0,
        run=0,
        init_val=4,  # 4: true beta = 0.5; 5: true beta = 0.25
    )

    plot_D_snapshots(days=(20, 40, 60, 80), **common)
    plot_D_time(top_k=40, sort_by='abs_D', **common)
    plot_D_time(top_k=40, sort_by='I', **common)
    plot_C_trajectory(**common)
