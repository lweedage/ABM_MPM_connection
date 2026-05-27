"""
Driver script: estimate per-day beta for all (seed, run) pairs in a scenario.
Set MPM / MPM2 below to switch estimator variant.
"""
import os
from estimation import estimate_rates

START_DATE = '01-01-2021'
END_DATE = '11-04-2021'  # 100 days later
SCENARIO = 'High'

# Pick at most one of these
MPM = False  # classical MPM
MPM2 = False  # MPM with three pathways (home-home, and both home-away and away-home)
NO_MOBILITY = False

INITIALIZATION = False
CONFIDENCE = False

N_SEEDS = 10
N_RUNS = 100
INIT_VAL = 4  # 4: beta = 0.5, 5: beta = 0.25


def _fn(seed, run):
    if MPM:
        suffix = 'MPM_init'
    elif MPM2:
        suffix = 'MPM2_init'
    elif NO_MOBILITY:
        suffix = 'no_mob_init'
    else:
        suffix = 'init'
    return (f'../ABM_data/{SCENARIO}/transmission_rates/Initialization{INIT_VAL}/'
            f'CI_{START_DATE}-{END_DATE}_seed_{seed}_perday{run}_{suffix}_{INITIALIZATION}.csv')


def main():
    out_dir = f'../ABM_data/{SCENARIO}/transmission_rates/Initialization{INIT_VAL}'
    os.makedirs(out_dir, exist_ok=True)
    for run in range(N_RUNS):
        for seed in range(N_SEEDS):
            print(f'Seed {seed}, run {run}')
            print(f'Estimating beta in scenario {SCENARIO} '
                  f'between {START_DATE} and {END_DATE}.')

            fn = _fn(seed, run)
            if os.path.exists(fn):
                print(f'  -> {fn} exists, skipping.')
                continue

            kw = dict(start_date=START_DATE, confidence=CONFIDENCE,
                      scenario=SCENARIO, seed=seed, run=run,
                      initialization=INITIALIZATION, init_val=INIT_VAL)

            if MPM:
                _, CI = estimate_rates.estimate_rates_poisson_MPM(**kw)
            elif MPM2:
                _, CI = estimate_rates.estimate_rates_per_day_MPM2(**kw)
            elif NO_MOBILITY:
                _, CI = estimate_rates.estimate_rates_per_day_no_mob(**kw)
            else:
                _, CI = estimate_rates.estimate_rates_poisson(**kw)

            CI.to_csv(fn)


if __name__ == '__main__':
    main()
