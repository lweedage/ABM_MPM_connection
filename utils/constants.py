import os


class CoronaConstants:
    # disease
    latent_period = 3  # nu
    infectious_period = 9  # omega

    # mobility
    population_nl = 17242149
    average_total_mobility = 3_000_000
    constant_mobility = True

    # how many days to look back/forward when reconstructing E and I
    # from the per-day E->I stream
    lookback = 14
    lookforward = 7
    look_into_past = lookforward

    # NB dispersion
    dispersion = 50

    changed = {}
