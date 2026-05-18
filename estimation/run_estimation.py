"""
Driver script: estimate per-day beta for all (seed, run) pairs in a scenario.
Set MPM / MPM2 below to switch estimator variant.
"""
import os
from estimation import estimate_rates

START_DATE = '01-01-2021'
END_DATE = '11-04-2021'  # 100 days later
SCENARIO = 'Medium'

# Pick at most one of these
MPM = False  # classical MPM
MPM2 = False  # MPM with three pathways (home-home, and both home-away and away-home)

INITIALIZATION = True
CONFIDENCE = True

N_SEEDS = 10
N_RUNS = 5


def _fn(seed, run):
    if MPM:
        suffix = 'MPM_init'
    elif MPM2:
        suffix = 'MPM2_init'
    else:
        suffix = 'init'
    return (f'ABM_data/{SCENARIO}/transmission_rates/'
            f'CI_{START_DATE}-{END_DATE}_seed_{seed}_perday{run}_{suffix}_{INITIALIZATION}.csv')


def main():
    out_dir = f'ABM_data/{SCENARIO}/transmission_rates'
    os.makedirs(out_dir, exist_ok=True)

    for seed in range(N_SEEDS):
        for run in range(N_RUNS):
            print(f'Seed {seed}, run {run}')
            print(f'Estimating beta in scenario {SCENARIO} '
                  f'between {START_DATE} and {END_DATE}.')

            fn = _fn(seed, run)
            if os.path.exists(fn):
                print(f'  -> {fn} exists, skipping.')
                continue

            kw = dict(start_date=START_DATE, confidence=CONFIDENCE,
                      scenario=SCENARIO, seed=seed, run=run,
                      initialization=INITIALIZATION)

            if MPM:
                _, CI = estimate_rates.estimate_rates_per_day_MPM(**kw)
            elif MPM2:
                _, CI = estimate_rates.estimate_rates_per_day_MPM2(**kw)
            else:
                _, CI = estimate_rates.estimate_rates_per_day(**kw)

            CI.to_csv(fn)


if __name__ == '__main__':
    main()
