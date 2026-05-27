from __future__ import annotations

import pickle
import time

import matplotlib.pyplot as plt
import scipy.sparse
import seaborn

from utils.util import *

DIVISION = 'Medium100'
RUNS = range(100)
MOBILITY_SEEDS = range(1)
INITIALIZATION = 5
TRUE_BETA = 0.5 if INITIALIZATION == 4 else 0.25

START_DATE = '01012021'
start_dt = pd.to_datetime(START_DATE, format='%d%m%Y')
NDAYS = 100
END_DATE = (start_dt + pd.Timedelta(days=NDAYS)).strftime('%d%m%Y')

INFECTION_STATE = 2

CI_DIR = f'../ABM_data/{DIVISION}/transmission_rates/Initialization{INITIALIZATION}'
CI_START = '01-01-2021'
CI_END = '11-04-2021'
INIT_FLAG = False
N_SEEDS_CSV = 10
N_RUNS_CSV = 1
BETA_FALLBACK = TRUE_BETA

NU = 3.0
OMEGA = 9.0

# Munis to highlight spatially. Will be matched against gemeente_list_ref.
HIGHLIGHT_MUNIS = ['Amsterdam', 'Utrecht', 'Maastricht', 'Groningen']

OUT_DIR = '../Output/Plots'
DATA_DIR = '../Output/PlotData'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

_rocket = seaborn.color_palette('rocket', 8)
COLOR_ABM = _rocket[1]
COLOR_MF = _rocket[3]
COLOR_MPM = _rocket[5]

suffix = 'beta='
if INITIALIZATION == 5:
    suffix += '0.25'
else:
    suffix += '0.5'

if DIVISION == 'Medium100' or DIVISION == 'High30':
    suffix += 'mob_30%'
else:
    suffix += 'mob_80%'

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def nb_sample(rng, mean, k):
    mean = np.asarray(mean, dtype=float)
    out = np.zeros_like(mean)
    pos = mean > 0
    if np.any(pos):
        out[pos] = rng.poisson(mean[pos])
    return out


def step_classical_mpm(S, E, I, R, N_live, M_bar, beta, rng):
    infect_frac = I / np.maximum(N_live, 1e-12)
    foi = beta * (M_bar @ infect_frac)
    mean_new_E = S * foi
    new_E = np.minimum(S, rng.poisson(mean_new_E))
    p_EI = 1.0 - np.exp(-1.0 / NU)
    p_IR = 1.0 - np.exp(-1.0 / OMEGA)
    new_I = p_EI * E
    new_R = p_IR * I
    # if it should also be stochastic
    # new_I = rng.binomial(E.astype(int), p_EI)
    # new_R = rng.binomial(I.astype(int), p_IR)
    S = S - new_E
    E = E + new_E - new_I
    I = I + new_I - new_R
    R = R + new_R
    return S, E, I, R, new_E


def step_abm_mf(S, E, I, R, N_live, M_bar, w_T, beta, rng):
    infect_frac = I / np.maximum(N_live, 1e-12)
    prevalence_j = w_T @ infect_frac
    foi = beta * (M_bar @ prevalence_j)
    mean_new_E = S * foi
    new_E = np.minimum(S, rng.poisson(mean_new_E))
    p_EI = 1.0 - np.exp(-1.0 / NU)
    p_IR = 1.0 - np.exp(-1.0 / OMEGA)
    new_I = p_EI * E
    new_R = p_IR * I
    # if it should also be stochastic
    # new_I = rng.binomial(E.astype(int), p_EI)
    # new_R = rng.binomial(I.astype(int), p_IR)
    S = S - new_E
    E = E + new_E - new_I
    I = I + new_I - new_R
    R = R + new_R
    return S, E, I, R, new_E


def mean_ci(arr):
    arr = np.array(arr)
    return (arr.mean(axis=0),
            np.quantile(arr, 0.025, axis=0),
            np.quantile(arr, 0.975, axis=0))


# ------------------------------------------------------------
# load per-day beta_hat from CI CSVs
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
            if len(beta) < NDAYS:
                beta = np.concatenate([beta, np.full(NDAYS - len(beta), np.nan)])
            else:
                beta = beta[:NDAYS]
            beta = np.where(beta > 1e-5, beta, np.nan)
            all_betas.append(beta)
    if not all_betas:
        raise RuntimeError(f"No CI csvs found for {estimator!r} in {CI_DIR}")
    arr = np.array(all_betas)
    med = np.nanmedian(arr, axis=0)
    lo = np.nanquantile(arr, 0.025, axis=0)
    hi = np.nanquantile(arr, 0.975, axis=0)
    nan_mask = np.isnan(med)
    if nan_mask.any():
        med = np.where(nan_mask, BETA_FALLBACK, med)
        lo = np.where(nan_mask, BETA_FALLBACK, lo)
        hi = np.where(nan_mask, BETA_FALLBACK, hi)
    return med, lo, hi


print("Loading beta_hat(t) from CI csvs...")
beta_abm_med, beta_abm_lo, beta_abm_hi = _load_beta_series('ABM')
beta_mpm_med, beta_mpm_lo, beta_mpm_hi = _load_beta_series('MPM')


# ------------------------------------------------------------
# load per-seed data
# ------------------------------------------------------------
def _load_seed(mobility_seed):
    fn = f'../Output/Data/{DIVISION}/Seed_{mobility_seed}'
    people_df = pd.read_pickle(f'{fn}/PeopleDF.pkl')
    gemeenten = pd.read_pickle(f'{fn}/Gemeenten.pkl')
    gemeente_list = gemeenten.values.ravel()
    gemeente_to_idx = {g: i for i, g in enumerate(gemeente_list)}
    return fn, people_df, gemeenten, gemeente_list, gemeente_to_idx


_ref_fn, _ref_people, _ref_gemeenten, gemeente_list_ref, gemeente_to_idx_ref = \
    _load_seed(list(MOBILITY_SEEDS)[0])
M = len(gemeente_list_ref)

# Resolve highlighted munis to indices in the canonical order
HIGHLIGHT_IDX = []
HIGHLIGHT_NAMES = []
for name in HIGHLIGHT_MUNIS:
    if name in gemeente_to_idx_ref:
        HIGHLIGHT_IDX.append(gemeente_to_idx_ref[name])
        HIGHLIGHT_NAMES.append(name)
    else:
        print(f"  {name!r} not in canonical Gemeenten -- skipping.")

print("Pre-loading per-seed mob matrices...")
_t0 = time.time()
_seed_cache = {}
for mobility_seed in MOBILITY_SEEDS:
    fn, people_df, gemeenten, gemeente_list, gemeente_to_idx = _load_seed(mobility_seed)
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

    M_bar_per_day, wT_per_day = [], []
    N_live_seed = np.asarray(aggregator @ np.ones(N_persons, dtype=np.float64))
    for d in range(NDAYS + 1):
        date_str = (start_dt + pd.Timedelta(days=d)).strftime('%d%m%Y')
        path = f'../Data/Mobility/MobMats/{DIVISION}{date_str}.p'
        with open(path, 'rb') as f:
            M_raw_local = pickle.load(f)
        M_raw_local = np.asarray(M_raw_local, dtype=np.float64)
        M_raw = M_raw_local[perm][:, perm] if perm_ok else M_raw_local
        row_sums = M_raw.sum(axis=1, keepdims=True)
        M_bar = M_raw / np.maximum(row_sums, 1e-12)
        M_bar_per_day.append(M_bar)
        numerator = M_bar * N_live_seed[:, None]
        denominator = numerator.sum(axis=0, keepdims=True)
        w = numerator / np.maximum(denominator, 1e-12)
        wT_per_day.append(w.T)

    _seed_cache[mobility_seed] = {
        'fn': fn, 'aggregator': aggregator,
        'M_bar_per_day': M_bar_per_day,
        'wT_per_day': wT_per_day,
        'N_live': N_live_seed,
    }
print(f"  cached {len(_seed_cache)} seed(s) in {time.time() - _t0:.2f}s")


# ------------------------------------------------------------
# Run simulations for (a) constant TRUE_BETA and (b) per-day beta_hat
# ------------------------------------------------------------
def _run_all(beta_provider_mf, beta_provider_mpm, label):
    """Returns (abm_traj_list, mf_traj_list, mpm_traj_list, abm_spatial,
    mf_spatial, mpm_spatial). 'traj' = national; 'spatial' = (T, len(highlight)).
    `beta_provider_*` is callable d -> beta scalar.
    """
    print(f"Running simulations ({label})...")
    all_abm, all_mpm, all_mf = [], [], []
    abm_sp, mpm_sp, mf_sp = [], [], []

    for run in RUNS:
        rng = np.random.default_rng()
        for mobility_seed in MOBILITY_SEEDS:
            cache = _seed_cache[mobility_seed]
            fn = cache['fn']
            aggregator = cache['aggregator']
            M_bar_per_day = cache['M_bar_per_day']
            wT_per_day = cache['wT_per_day']
            N_live = cache['N_live']

            status_raw = scipy.sparse.load_npz(
                f'{fn}/Initialization{INITIALIZATION}/{START_DATE}-{END_DATE}/'
                f'Status_{run}.npz'
            )
            status = status_raw.toarray()
            T = status.shape[0]

            # ABM new infections
            is_inf = (status == INFECTION_STATE).astype(np.float64)
            is_R = (status == INFECTION_STATE + 1).astype(np.float64)
            cum_post_E = (is_inf + is_R).sum(axis=1)
            new_inf = np.zeros(T)
            new_inf[0] = is_inf[0].sum()
            new_inf[1:] = np.maximum(0, np.diff(cum_post_E))
            all_abm.append(new_inf)

            # ABM spatial (per highlighted muni)
            # ABM agent-level: new infections in muni m on day d = number of
            # agents who entered E on day d and live in m. We approximate
            # via aggregator on the per-day diff of (is_inf + is_R).
            cum_post_E_per_agent = is_inf + is_R  # (T, N_agents)
            new_post_E_per_agent = np.zeros_like(cum_post_E_per_agent)
            new_post_E_per_agent[0] = is_inf[0]
            new_post_E_per_agent[1:] = np.maximum(
                0, np.diff(cum_post_E_per_agent, axis=0)
            )
            new_post_E_per_muni = (aggregator @ new_post_E_per_agent.T).T  # (T, M)
            abm_sp.append(new_post_E_per_muni[:, HIGHLIGHT_IDX])

            # Initial compartments
            s0 = status[0]
            S0 = aggregator @ (s0 == 0).astype(np.float64)
            E0 = aggregator @ (s0 == 1).astype(np.float64)
            I0 = aggregator @ (s0 == 2).astype(np.float64)
            R0 = aggregator @ (s0 == 3).astype(np.float64)

            S_mpm, E_mpm, I_mpm, R_mpm = S0.copy(), E0.copy(), I0.copy(), R0.copy()
            S_mf, E_mf, I_mf, R_mf = S0.copy(), E0.copy(), I0.copy(), R0.copy()
            mpm_new = np.zeros((T, M))
            mf_new = np.zeros((T, M))
            mpm_new[0] = I0
            mf_new[0] = I0

            for d in range(T - 1):
                Mbar = M_bar_per_day[d]
                wT = wT_per_day[d]
                b_mpm = beta_provider_mpm(d)
                b_mf = beta_provider_mf(d)
                S_mpm, E_mpm, I_mpm, R_mpm, ne = step_classical_mpm(
                    S_mpm, E_mpm, I_mpm, R_mpm, N_live, Mbar, b_mpm, rng
                )
                mpm_new[d + 1] = ne
                S_mf, E_mf, I_mf, R_mf, ne = step_abm_mf(
                    S_mf, E_mf, I_mf, R_mf, N_live, Mbar, wT, b_mf, rng
                )
                mf_new[d + 1] = ne

            all_mpm.append(mpm_new.sum(axis=1))
            all_mf.append(mf_new.sum(axis=1))
            mpm_sp.append(mpm_new[:, HIGHLIGHT_IDX])
            mf_sp.append(mf_new[:, HIGHLIGHT_IDX])

    return (np.array(all_abm), np.array(all_mf), np.array(all_mpm),
            np.array(abm_sp), np.array(mf_sp), np.array(mpm_sp))


abm_t, mf_t_true, mpm_t_true, abm_sp, mf_sp_true, mpm_sp_true = _run_all(
    lambda d: TRUE_BETA, lambda d: TRUE_BETA, label=f'beta = {TRUE_BETA}'
)
_, mf_t_hat, mpm_t_hat, _, mf_sp_hat, mpm_sp_hat = _run_all(
    lambda d: beta_abm_med[d] if d < len(beta_abm_med) else BETA_FALLBACK,
    lambda d: beta_mpm_med[d] if d < len(beta_mpm_med) else BETA_FALLBACK,
    label='beta_hat(t)',
)


# ------------------------------------------------------------
# Plotting helper
# ------------------------------------------------------------
def _add_traj(ax, x, mean, lo, hi, color, label, ls='-'):
    ax.plot(x, mean, color=color, linewidth=2, linestyle=ls, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


# Figure 1: national, both beta regimes side by side
m_abm, lo_abm, hi_abm = mean_ci(abm_t)
m_mf_true, lo_mf_true, hi_mf_true = mean_ci(mf_t_true)
m_mpm_true, lo_mpm_true, hi_mpm_true = mean_ci(mpm_t_true)
m_mf_hat, lo_mf_hat, hi_mf_hat = mean_ci(mf_t_hat)
m_mpm_hat, lo_mpm_hat, hi_mpm_hat = mean_ci(mpm_t_hat)

T = abm_t.shape[1]
x = np.arange(T)

fig, axes = plt.subplots(2, 1, figsize=(11, 11), constrained_layout=True)

# Row 1: trajectories at TRUE beta
ax = axes[0]
_add_traj(ax, x, m_abm, lo_abm, hi_abm, COLOR_ABM, 'ABM (ground truth)')
_add_traj(ax, x, m_mf_true, lo_mf_true, hi_mf_true, COLOR_MF,
          f'ABM mean-field at beta = {TRUE_BETA}', ls='-.')
_add_traj(ax, x, m_mpm_true, lo_mpm_true, hi_mpm_true, COLOR_MPM,
          f'Classical MPM at beta = {TRUE_BETA}', ls='--')
ax.set_xlabel('day')
ax.set_ylabel('new infections (national)')
ax.set_title(f'Trajectories at TRUE beta = {TRUE_BETA}')
ax.legend()
ax.grid(alpha=0.3)

# Row 3: trajectories at beta_hat(t)
ax = axes[1]
_add_traj(ax, x, m_abm, lo_abm, hi_abm, COLOR_ABM, 'ABM (ground truth)')
_add_traj(ax, x, m_mf_hat, lo_mf_hat, hi_mf_hat, COLOR_MF,
          r'ABM mean-field at $\hat{\beta}_{ABM-MF}(t)$', ls='-.')
_add_traj(ax, x, m_mpm_hat, lo_mpm_hat, hi_mpm_hat, COLOR_MPM,
          r'Classical MPM at $\hat{\beta}_{MPM}(t)$', ls='--')
ax.set_xlabel('day')
ax.set_ylabel('new infections (national)')
ax.set_title(r'Trajectories at estimated $\hat{\beta}(t)$')
ax.legend()
ax.grid(alpha=0.3)

out = os.path.join(OUT_DIR, f'national_{suffix}.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")

# Figure 2: spatial -- per-muni trajectories at TRUE beta
n_h = len(HIGHLIGHT_IDX)
if n_h > 0:
    fig, axes = plt.subplots(1, n_h, figsize=(4 * n_h, 4),
                             constrained_layout=True, sharex=True)
    if n_h == 1:
        axes = [axes]
    for i, (ax, idx, name) in enumerate(zip(axes, HIGHLIGHT_IDX, HIGHLIGHT_NAMES)):
        m_abm_i, lo_abm_i, hi_abm_i = mean_ci(abm_sp[:, :, i])
        m_mf_i, lo_mf_i, hi_mf_i = mean_ci(mf_sp_true[:, :, i])
        m_mpm_i, lo_mpm_i, hi_mpm_i = mean_ci(mpm_sp_true[:, :, i])
        _add_traj(ax, x, m_abm_i, lo_abm_i, hi_abm_i, COLOR_ABM, 'ABM')
        _add_traj(ax, x, m_mf_i, lo_mf_i, hi_mf_i, COLOR_MF, 'ABM-MF', ls='-.')
        _add_traj(ax, x, m_mpm_i, lo_mpm_i, hi_mpm_i, COLOR_MPM, 'MPM', ls='--')
        n_pop = int(_seed_cache[list(MOBILITY_SEEDS)[0]]['N_live'][idx])
        ax.set_title(f'{name} (N={n_pop})')
        ax.set_xlabel('day')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel('new infections')
            ax.legend(fontsize=9)

    out = os.path.join(OUT_DIR, f'per_muni_{suffix}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")

# CSV with all national-level series
pd.DataFrame({
    'day': x,
    'abm_mean': m_abm, 'abm_lo': lo_abm, 'abm_hi': hi_abm,
    'mf_true_mean': m_mf_true, 'mf_true_lo': lo_mf_true, 'mf_true_hi': hi_mf_true,
    'mpm_true_mean': m_mpm_true, 'mpm_true_lo': lo_mpm_true, 'mpm_true_hi': hi_mpm_true,
    'mf_hat_mean': m_mf_hat, 'mf_hat_lo': lo_mf_hat, 'mf_hat_hi': hi_mf_hat,
    'mpm_hat_mean': m_mpm_hat, 'mpm_hat_lo': lo_mpm_hat, 'mpm_hat_hi': hi_mpm_hat,
    'beta_abm_med': np.r_[beta_abm_med, [np.nan] * (T - len(beta_abm_med))][:T],
    'beta_mpm_med': np.r_[beta_mpm_med, [np.nan] * (T - len(beta_mpm_med))][:T],
}).to_csv(os.path.join(DATA_DIR, f'trajectories.csv'),
          index=False)
print("Wrote CSV.")
