"""MobilitySEIR — coarse aggregate SEIR state per municipality. Used by
the comparison plots; not the full ABM simulation."""
import numpy as np
import pandas as pd


class MobilitySEIR:
    compartments = ['susceptible', 'exposed', 'infected', 'removed']
    all_columns = compartments + ['deltaI']

    def __init__(self, init_df, horizon=0, time_dependency=False,
                 start_date=None, constants=None, gemeente_shapes=None):
        if not isinstance(init_df, pd.DataFrame):
            raise TypeError("init_df must be a pandas DataFrame.")
        if 'inhabitants' not in init_df.columns:
            raise ValueError("init_df must contain an 'inhabitants' column.")

        self.init_df = init_df.copy()
        self._original_init_df = init_df.copy()

        self.horizon = horizon
        self.start_date_dt = start_date
        self.time_dependency = time_dependency
        self.gemeente_shapes = gemeente_shapes

        self.preloaded_mobility_dfs = {}
        self.municipalities_order = self.init_df.index.tolist()
        self.num_municipalities = len(self.municipalities_order)

        self.division_no_restrictions = {area: 0 for area in init_df.index}

        self.constants = constants
        self.inv_latent_period = 1 / self.constants.latent_period
        self.inv_infectious_period = 1 / self.constants.infectious_period
        self.transmission_prob = getattr(self.constants, 'transmission_prob', 0.1)

        self.beta = []

        self._current_state = self.init_df[self.compartments].values.astype(np.int64)
        self._inhabitants = self.init_df['inhabitants'].values.astype(np.int64)

        self.coeffs = pd.DataFrame()
        self._simulation_history = pd.DataFrame()

    def reset_state(self, new_init_df=None):
        """Reset state back to initial (or to a new init df)."""
        self.init_df = (new_init_df if new_init_df is not None
                        else self._original_init_df).copy()
        self._current_state = self.init_df[self.compartments].values.astype(np.int64)
        self._inhabitants   = self.init_df['inhabitants'].values.astype(np.int64)

    def simulate_all_contacts(self):
        S = self._current_state[:, self.compartments.index('susceptible')].astype(float)
        I = self._current_state[:, self.compartments.index('infected')].astype(float)
        N = self._inhabitants.astype(float)

        infect_frac = np.nan_to_num(I / np.maximum(N, 1e-10), nan=0.0, posinf=0.0, neginf=0.0)
        sus_frac    = np.nan_to_num(S / np.maximum(N, 1e-10), nan=0.0, posinf=0.0, neginf=0.0)

        self.coeffs = pd.DataFrame({
            'S_live': S,
            'I_live': I,
            'N_live': N,
            'infect_frac': infect_frac,
            'sus_frac':    sus_frac,
        }, index=self.municipalities_order)
