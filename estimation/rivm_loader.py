"""
Loads ABM output (Status sparse matrix + PeopleDF) and turns it into the
per-day (S, E, I, R) compartment counts plus the look-ahead/look-back
reconstructed S_hat, E_hat, I_hat, R_hat used by the NB estimator.

E_hat(t) is a forward-weighted sum of future E->I transitions (a person
in E now will transition with some delay).
I_hat(t) is a backward-weighted sum of past E->I transitions.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import scipy.sparse

from utils.constants import CoronaConstants

INFECTION_STATE = 2


class RivmLoader:
    def __init__(self, scenario, start_date, seed=0, run=0, init_val = 4):
        self.scenario = scenario
        self.folder = f'../Output/Data/{scenario}/Seed_{seed}'

        initialization = init_val

        status_raw = scipy.sparse.load_npz(
            f'{self.folder}/Initialization{initialization}/01012021-11042021/Status_{run}.npz'
        )
        status = status_raw.toarray()

        people_df = pd.read_pickle(f'{self.folder}/PeopleDF.pkl')
        gemeenten = pd.read_pickle(f'{self.folder}/Gemeenten.pkl')
        gemeente_list = gemeenten.values.ravel()
        gemeente_to_idx = {g: i for i, g in enumerate(gemeente_list)}
        M = len(gemeenten)

        self.munis = gemeente_list
        self.M = M
        self.status = status
        self.people_df = people_df.copy()
        self.start_date = start_date

        T, N = self.status.shape
        self.T = T

        # map each person to their municipality index
        home = self.people_df["Home"].values
        self.muni_idx = np.fromiter(
            (gemeente_to_idx[h] for h in home), dtype=np.int64, count=len(home)
        )

        # new E->I (positive test) transitions per person, per day
        new_inf_person = np.zeros((T, N), dtype=np.int8)
        new_inf_person[1:, :] = (
                (status[1:, :] == INFECTION_STATE)
                & (status[:-1, :] == INFECTION_STATE - 1)
        ).astype(np.int8)
        new_inf_person[0, :] = (status[0, :] == INFECTION_STATE).astype(np.int8)

        # scatter-add into (T, M) counts per municipality
        new_inf_arr = np.zeros((T, M), dtype=np.int64)
        np.add.at(new_inf_arr.T, self.muni_idx, new_inf_person.T)

        # date index
        dates = pd.to_datetime(
            [self.start_date + timedelta(days=int(t)) for t in range(T)]
        )
        self.dates = dates
        self.date_to_idx = {d: i for i, d in enumerate(dates)}

        cum_inf_arr = np.cumsum(new_inf_arr, axis=0)
        self.new_exp_arr = new_inf_arr
        self.cum_exp_arr = cum_inf_arr

        # pandas mirror, kept for back-compat with callers that use .loc
        new_inf_df = pd.DataFrame(new_inf_arr, index=dates, columns=self.munis)
        relevant_data = (
            new_inf_df.stack()
            .rename("Total_reported")
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "name"})
        )
        relevant_data["date"] = pd.to_datetime(relevant_data["date"])
        relevant_data = relevant_data.set_index(["date", "name"]).sort_index()
        relevant_data["cumulative_reported"] = (
            relevant_data.groupby(level="name")["Total_reported"].cumsum()
        )
        self.mmdd2area2reported = relevant_data["Total_reported"]
        self.mmdd2cumulatives = relevant_data["cumulative_reported"]

        self.max_lookback = CoronaConstants.lookback
        self.max_lookforward = CoronaConstants.lookforward

        self.true_df = self.build_true_df()
        self._precompute_init_arrays()

    def build_true_df(self):
        """Per-(day, muni) ground-truth compartment counts."""
        status = self.status
        T, _ = status.shape

        S_arr = np.zeros((T, self.M), dtype=np.int64)
        E_arr = np.zeros((T, self.M), dtype=np.int64)
        I_arr = np.zeros((T, self.M), dtype=np.int64)
        R_arr = np.zeros((T, self.M), dtype=np.int64)

        for code, arr in [(0, S_arr), (1, E_arr), (2, I_arr), (3, R_arr)]:
            indicator = (status == code).astype(np.int64)
            np.add.at(arr.T, self.muni_idx, indicator.T)

        self.S_true_arr = S_arr
        self.E_true_arr = E_arr
        self.I_true_arr = I_arr
        self.R_true_arr = R_arr
        self.N_arr = S_arr + E_arr + I_arr + R_arr  # constant in t

        date_col = np.repeat(self.dates.values, self.M)
        name_col = np.tile(self.munis, T)
        true_df = pd.DataFrame({
            "date": date_col,
            "name": name_col,
            "susceptible": S_arr.ravel(),
            "exposed": E_arr.ravel(),
            "infected": I_arr.ravel(),
            "removed": R_arr.ravel(),
            "inhabitants": self.N_arr.ravel(),
        })
        true_df["datum"] = pd.to_datetime(true_df["date"])
        return true_df.set_index(['datum', 'name'])

    def _precompute_init_arrays(self, latent_period=None, infectious_period=None):
        """Compute S_hat, E_hat, I_hat, R_hat for every day.

        E_hat(t) = sum_{k=1..Lf-1} dI(t+k) * (1-1/nu)^(k-1)
                 + dI(t+Lf)       * nu * (1-1/nu)^(Lf-1)         [tail]
        I_hat(t) = sum_{s=0..Lb-1} dI(t-s) * (1-1/omega)^s
                 + dI(t-Lb)       * omega * (1-1/omega)^Lb       [tail]
        """
        if latent_period is None:
            latent_period = CoronaConstants.latent_period
        if infectious_period is None:
            infectious_period = CoronaConstants.infectious_period

        nu = latent_period
        omega = infectious_period
        Lf = self.max_lookforward
        Lb = self.max_lookback
        T, M = self.new_exp_arr.shape

        # E_hat: look-ahead with geometric kernel
        weights_e = np.zeros(Lf + 1, dtype=np.float64)
        weights_e[0] = 0.0  # offset 0 = today's E->I, not in look-ahead
        w = 1.0
        for k in range(1, Lf):
            weights_e[k] = w
            w *= (1 - 1 / nu)
        weights_e[Lf] = nu * ((1 - 1 / nu) ** (Lf - 1))

        E_hat = np.zeros((T, M), dtype=np.float64)
        for k, wk in enumerate(weights_e):
            if wk == 0.0 or k >= T:
                continue
            E_hat[:T - k, :] += wk * self.new_exp_arr[k:, :]

        # I_hat: look-back with geometric kernel
        weights_i = np.zeros(Lb + 1, dtype=np.float64)
        w = 1.0
        for s in range(Lb):
            weights_i[s] = w
            w *= (1 - 1 / omega)
        weights_i[Lb] = omega * ((1 - 1 / omega) ** Lb)

        I_hat = np.zeros((T, M), dtype=np.float64)
        for s, ws in enumerate(weights_i):
            if ws == 0.0 or s >= T:
                continue
            I_hat[s:, :] += ws * self.new_exp_arr[:T - s, :]

        N_per_muni = self.N_arr[0, :]  # constant population over time
        cum = self.cum_exp_arr
        S_hat = N_per_muni[None, :] - cum - E_hat
        R_hat = np.maximum(0.0, cum - I_hat)

        self.S_hat_arr = S_hat
        self.E_hat_arr = E_hat
        self.I_hat_arr = I_hat
        self.R_hat_arr = R_hat
