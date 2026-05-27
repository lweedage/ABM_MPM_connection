# ABM vs MPM transmission rate estimation

Code for estimating per-day transmission rates (β) from agent-based
model (ABM) output, comparing three estimator variants against a
known-true β from a stochastic ABM ground truth. Built on Dutch
municipality-level mobility data (TomTom) for the period
01-01-2021 to 11-04-2021 (100 days).

## What this code does

The pipeline runs in four stages:

1. **Mobility model** (`mobility_model/`) — reads daily TomTom
   origin-destination CSVs, builds per-day mobility matrices, and
   positions individual agents in municipalities each day.

2. **Transmission model** (`transmission_model/`) — agent-based
   SEIR simulation on top of the mobility data. This is the **ground
   truth**: it runs with a known β (default 0.5) and produces a
   `Status_{run}.npz` file per stochastic run.

3. **Estimation** (`estimation/`) — given the ABM output, estimate
   β per day via negative-binomial regression on ΔS. Three regressor
   variants are supported:
   - `z_vec_ABM`   — ABM mean-field force of infection
   - `z_vec_MPM`   — classical metapopulation model (MPM)
   - `z_vec_MPM_2` — one-β MPM with home-home, away-home, home-away pathways

4. **Plotting** (`plotting/`) — produces (a) trajectory plots
   comparing ABM ground truth to two compartmental approximations,
   (b) per-day concentration entropy of infections across
   municipalities, (c) daily maps of infectious counts, and (d) plots
   of estimated β / dispersion across all (seed, run) realizations.

Transmission and mobility model comes from https://github.com/MarkMDekker/covid_intervention_evaluation
- Corresponding paper: 
  - Dekker, M. M., Coffeng, L. E., Pijpers, F. P., Panja, D., & de Vlas, S. J. (2023). Reducing societal impacts of SARS-CoV-2 interventions through subnational implementation. Elife, 12, e80819.
Estimation code comes from https://github.com/MartijnGosgens/CovidMobilityTradeOffs
- Corresponding papers: 
  - Martijn Gösgens, Teun Hendriks, Marko Boon, Wim Steenbakkers, Hans Heesterbeek, Remco van der Hofstad and Nelly Litvak (2021). Trade-offs between mobility restrictions and transmission of SARS-CoV-2. Journal of the Royal Society Interface, 18(175), 20200936. DOI: https://doi.org/10.1098/rsif.2020.0936,
  - Schoot Uiterkamp, M. H., Gösgens, M., Heesterbeek, H., van der Hofstad, R., & Litvak, N. (2022). The role of inter-regional mobility in forecasting SARS-CoV-2 transmission. Journal of the Royal Society Interface, 19(193), 20220486.

## Running the pipeline

### Stage 1: build mobility matrices and position people
mobility_model.run_model

### Stage 2: run the ABM transmission model (10 mobility seeds × 100 runs by default)
transmission_model.run_model

### Stage 3: estimate β per day for every (seed, run)
estimation.run_estimation

### Stage 4: plots
plotting.plot_trajectories
plotting.plot_transmission_rates

To switch between estimator variants, edit the `MPM` / `MPM2` flags
near the top of `run_estimation.py` and `plot_transmission_rates.py`.

The full pipeline at the default resolution (`'High'` = 1 agent per
100 inhabitants, 10 mobility seeds × 100 stochastic runs, 100 days)
takes a few hours. For quicker iteration, drop
`N_SEEDS` and `N_RUNS` in `run_estimation.py` and lower the
resolution to `'Low'` or `'Verylow'` in the transmission model.

## Project layout

## Project layout
 
```
├── mobility_model/      stage 1: build per-day mobility matrices, position agents
│   ├── model.py            ModelM — TomTom OD -> daily mobility matrices + agent positions
│   ├── run_model.py        
├── transmission_model/  stage 2: ABM SEIR simulation
│   ├── model.py            ModelT — agent-based SEIR on the municipality network
│   └── run_model.py        
├── estimation/          stage 3: NB regression to estimate beta
│   ├── estimate_rates.py   the Poisson estimators (ABM / MPM / no-mobility)
│   ├── rivm_loader.py      ABM Status -> per-day (S,E,I,R) + reconstructed S_hat/E_hat/I_hat
│   ├── pathways.py         decompose the ABM force of infection into its four pathways
│   ├── run_estimation.py   estimate beta for every (seed, run)
│   └── compare_estimators.py  pool per-(seed,run) estimates into one CSV
├── plotting/            stage 4: trajectories, beta estimates, maps
│   ├── plot_trajectories.py                ABM vs compartmental, entropy, daily maps
│   ├── plot_trajectories_estimated_beta.py same, driven by the *estimated* beta(t)
│   ├── plot_transmission_rates.py          pooled + per-seed beta estimates
│   ├── compare_trajectories.py             extra trajectory comparison
│   └── plot_differences.py                 maps of the D_j(t) visitor-share term
├── seir/                aggregate SEIR helper (MobilitySEIR) used by some plots
├── utils/               constants, shapefiles, mobility-loading helpers
└── make_figures.py      draws the 2x2 estimator-comparison figure from the pooled CSV

data/                    expected layout described in data/README.md (gitignored)
Output/                  pipeline writes outputs here (gitignored)
```

## Notes

- The `'High' / 'Med' / 'Low' / 'Verylow'` labels in the transmission
  model refer to agent resolution (population / divisor). At `'High'`
  one agent = 100 people; at `'Verylow'` one agent = 5000 people (manual entry).

- `Initialization=4` (5 seed infections in Amsterdam) is the default
  for the experiments, with beta = 0.5 `Initialization=5' runs similarly but with beta = 0.25. Other values (1: MPM-initialized compartments;
  2: Tilburg seed; 3: one infection per municipality) are kept for from the original ABM.

- The estimator drops the first 14 days (warm-up) and last 7 days
  (look-ahead window too short for E_hat) by default — see
  `INIT_DAYS` / `END_DAYS` in `plot_transmission_rates.py`.

- TomTom data is not publicly available.

- **Shaded bands.** In the estimator-comparison and trajectory figures
  the shaded band is the 2.5%–97.5% spread of the per-realization
  estimates across simulation runs (run-to-run variability), **not** a
  confidence interval.

