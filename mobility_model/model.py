"""
ModelM — daily mobility generator.

Reads TomTom OD CSVs, builds per-day mobility matrices (saved as pickles),
then samples positions for each individual on each day.

When use_same_mobility=True, a single date's matrix is reused for every
day.
"""
import configparser
import datetime
import os
import pickle
import re

import numpy as np
import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm

import warnings

import utils.util
from utils.util import gemeente_shapes

warnings.filterwarnings("ignore")


def norm_gm_code(x) -> str:
    """'164' / '164.0' / 'GM164' -> 'GM0164'."""
    s = str(x).strip()
    if re.fullmatch(r"\d+(\.0)?", s):
        return f"GM{int(float(s)):04d}"
    m = re.fullmatch(r"GM(\d+)", s)
    if m:
        return f"GM{int(m.group(1)):04d}"
    return s


class ModelM:
    def __init__(self, params_input):
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        config.read(config_path)

        self.SaveName = params_input['savename']
        self.Path_TomTom = config['PATHS']['MOB']
        self.Path_RawDataGem = config['PATHS']['RAWDATA_GEM']
        self.Path_DemoMat = config['PATHS']['RAWDATA_DEMO']
        self.Path_Data = config['PATHS']['DATA']
        self.Path_Datasave = config['PATHS']['FIG']
        self.Path_RawDataMix = config['PATHS']['RAWDATA_MIX']
        self.Path_RawDataMix2 = config['PATHS']['RAWDATA_MIX2']

        self.Div = np.float64(params_input['division'])
        self.StartDate = dt.strptime(config['PARAMS']['START_DATE'], '%d-%m-%Y')
        self.EndDate = dt.strptime(config['PARAMS']['END_DATE'], '%d-%m-%Y')
        self.Ndays = (self.EndDate - self.StartDate).days

        self.UseSameMobility = params_input.get('use_same_mobility', False)
        self.SameMobilityDate = self.StartDate

        print('# ---------------------------------------------------------- #')
        print('# Starting Mobility model')
        print(f'# ------ Resolution:    {self.SaveName}')
        print(f'# ------ Amount days:   {self.Ndays}')
        print(f'# ------ Same mobility: {self.UseSameMobility}')
        if self.UseSameMobility:
            print(f'# ------ Fixed date:    {self.SameMobilityDate.strftime("%d-%m-%Y")}')
        print('# ---------------------------------------------------------- #')

    def read_data(self):
        self.DF_Gem = pd.read_csv(self.Path_RawDataGem, delimiter=';')
        self.UniLocs = np.unique(self.DF_Gem.Gemeentenaam)
        self.UniIDs = [
            list(self.DF_Gem.Gemeentecode[self.DF_Gem.Gemeentenaam == loc])[0]
            for loc in self.UniLocs
        ]

        self.UniGM = [norm_gm_code(g) for g in self.UniIDs]
        self.GM_to_idx = {gm: i for i, gm in enumerate(self.UniGM)}

        self.gemeente_shapes = utils.util.make_gemeenteshapes()
        shapes = self.gemeente_shapes.copy()
        shapes['GM_norm'] = shapes.index.map(norm_gm_code)
        pop_map = dict(zip(shapes['GM_norm'], shapes['AANT_INW']))

        self.HomePop_full = np.array([
            pop_map.get(gm, 0) for gm in self.UniGM
        ], dtype=float)

        # agent-scale population (one agent per Div people)
        self.DemoMat = (self.HomePop_full / self.Div).astype(int)
        self.HomePop = self.DemoMat
        self.UniGroups = ['total']
        self.N = np.sum(self.DemoMat)

        self.Dates = [self.StartDate + datetime.timedelta(i) for i in range(self.Ndays)]

        if self.UseSameMobility:
            print(f'Loading fixed-date {self.Path_TomTom} mobility for '
                  f'{self.SameMobilityDate.strftime("%d-%m-%Y")} ..')
            self.SameMobMat = self._load_tomtom_day(self.SameMobilityDate)

    # ----- TomTom loading -----

    def _tomtom_path(self, date):
        return os.path.join(self.Path_TomTom, date.strftime('%d-%m-%Y') + '.csv')

    def _load_tomtom_day(self, date):
        path = self._tomtom_path(date)
        if not os.path.exists(path):
            raise FileNotFoundError(f'TomTom file not found: {path}')

        df = pd.read_csv(path)
        df['woon'] = df['woon'].map(norm_gm_code)
        df['bezoek'] = df['bezoek'].map(norm_gm_code)

        TotLocs = len(self.UniGM)
        MobMat = np.zeros((TotLocs, TotLocs), dtype=float)

        mask = df['woon'].isin(self.GM_to_idx) & df['bezoek'].isin(self.GM_to_idx)
        df = df.loc[mask]

        i_idx = df['woon'].map(self.GM_to_idx).to_numpy()
        j_idx = df['bezoek'].map(self.GM_to_idx).to_numpy()
        vals = df['totaal_aantal_bezoekers'].to_numpy(dtype=float)
        np.add.at(MobMat, (i_idx, j_idx), vals)

        # 1. Build the TomTom-implied diagonal at full population scale
        trips_out_per_muni = MobMat.sum(axis=1)
        diag = np.maximum(self.HomePop_full - trips_out_per_muni, 0.0)
        np.fill_diagonal(MobMat, diag)

        # 2. Rescale to agent counts
        MobMat = MobMat / self.Div

        # 3. Correct row sums to HomePop (small flooring drift)
        row_sums = MobMat.sum(axis=1)
        diff = self.HomePop - np.floor(row_sums).astype(int)
        MobMat = np.floor(MobMat).astype(int)  # cast everything to int first
        for i in range(len(diff)):
            MobMat[i, i] += diff[i]

        # 4. NOW optionally rescale the diagonal to a target stay-home fraction
        if self.SaveName == 'Medium100' or self.SaveName == 'High30':
            MobMat = self.rescale_diagonal(MobMat)

        return MobMat

    def mobility_matrix(self, day):
        date = self.StartDate + datetime.timedelta(days=day)
        out_path = ('../Data/Mobility/MobMats/' + self.SaveName
                    + date.strftime('%d%m%Y') + '.p')

        MobMat = self.SameMobMat.copy() if self.UseSameMobility \
            else self._load_tomtom_day(date)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pickle.dump(MobMat, open(out_path, 'wb'))
        self.MobMat = MobMat

    def rescale_diagonal(self, MobMat, target_stay_home = 0.25):
        M = MobMat.copy().astype(float)
        n = M.shape[0]
        out = np.zeros_like(M, dtype=int)
        for i in range(n):
            row_sum = M[i].sum()
            if row_sum <= 0:
                continue
            target_diag = target_stay_home * row_sum
            current_diag = M[i, i]
            off_sum = row_sum - current_diag
            new_off_sum = row_sum - target_diag
            if off_sum > 0:
                scale = new_off_sum / off_sum
                new_row = M[i] * scale
                new_row[i] = target_diag
            else:
                new_row = M[i].copy()
                new_row[i] = target_diag

            # round off-diagonals (stochastic rounding preserves expectation)
            off_mask = np.ones(n, dtype=bool)
            off_mask[i] = False
            off_vals = new_row[off_mask]
            floored = np.floor(off_vals).astype(int)
            frac = off_vals - floored
            bumps = (np.random.random(len(frac)) < frac).astype(int)
            out[i, off_mask] = floored + bumps
            # diagonal absorbs whatever is left so row sum = HomePop[i]
            out[i, i] = int(self.HomePop[i]) - out[i, off_mask].sum()
        return out

    # ----- People & positions -----

    def create_people_DF(self, N=0):
        """Build PeopleDF: one row per individual with Home + Group."""
        homes = np.repeat(np.asarray(self.UniLocs), self.DemoMat)
        groups = np.full(len(homes), self.UniGroups[0])
        return pd.DataFrame({'Home': homes, 'Group': groups})

    def _build_people_arrays(self):
        if hasattr(self, '_people_built'):
            return
        self._PeopleDF = self.create_people_DF(0)
        home_idx = np.repeat(np.arange(len(self.UniLocs)), self.DemoMat)
        self._home_idx = home_idx
        self._people_by_home = [
            np.where(home_idx == h)[0] for h in range(len(self.UniLocs))
        ]
        self._people_built = True

    def position_people(self, seed):
        """Sample where each person is on each day. With UseSameMobility=True,
        the mobility matrix is fixed but per-day positions still vary."""
        self._build_people_arrays()

        if self.UseSameMobility:
            self.mobility_matrix(0)
            row_plans = self._build_row_plans(self.MobMat)

        for N in tqdm(range(self.Ndays)):
            if self.UseSameMobility:
                self._dump_mobmat_for_day(N, self.MobMat)
            else:
                self.mobility_matrix(N)
                row_plans = self._build_row_plans(self.MobMat)

            pos = self._sample_positions(row_plans)
            self.save_positions(pos, N, seed)

    def _build_row_plans(self, MobMat):
        """For each home municipality h, allocate h's residents across destinations.
        """
        plans = []
        for h in range(len(self.UniLocs)):
            people_h = self._people_by_home[h]
            n_people = len(people_h)
            if n_people == 0:
                plans.append(None)
                continue

            row = np.asarray(MobMat[h], dtype=float)
            row_sum = row.sum()
            if row_sum <= 0:
                plans.append(None)
                continue

            # Rescale to agent counts. =
            counts = (row / row_sum) * n_people
            counts = np.floor(counts).astype(int)

            # Don't try to place more people than we have.
            total = counts.sum()
            if total > n_people:
                # Trim the largest-count destinations first.
                excess = total - n_people
                for k in np.argsort(-counts):
                    if excess <= 0:
                        break
                    take = min(int(counts[k]), excess)
                    counts[k] -= take
                    excess -= take

            plans.append(counts)
        return plans

    def _sample_positions(self, row_plans):
        N_total = sum(self.DemoMat)
        pos = np.empty((N_total, 1), dtype=float)

        for h, counts in enumerate(row_plans):
            people_h = self._people_by_home[h]
            if counts is None or len(people_h) == 0:
                pos[people_h] = h
                continue

            shuffled = np.random.permutation(people_h)
            cursor = 0
            for away, n in enumerate(counts):
                if n <= 0:
                    continue
                pos[shuffled[cursor:cursor + n]] = away
                cursor += n
            # any leftover slack stays home
            pos[shuffled[cursor:]] = h
        return pos

    def _dump_mobmat_for_day(self, N, MobMat):
        date = self.StartDate + datetime.timedelta(days=N)
        out_path = ('../Data/Mobility/MobMats/' + self.SaveName
                    + date.strftime('%d%m%Y') + '.p')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pickle.dump(MobMat, open(out_path, 'wb'))

    def save_positions(self, Positions, N, seed):
        path = self.Path_Data + self.SaveName + '/Seed_' + str(seed) + '/'
        os.makedirs(path, exist_ok=True)
        os.makedirs(self.Path_Data + 'General/', exist_ok=True)

        date = self.StartDate + datetime.timedelta(days=N)
        np.save(path + 'Positions' + date.strftime('%d%m%Y'), Positions)

    def count_people(self):
        """Aggregate counts per group/hour/municipality. Diagnostic only."""
        choose = 0
        PeopleDF = self.PeopleDFs[choose]
        Positions = self.Positions_all[choose]
        pops = []
        for g in self.UniGroups:
            whs = np.where(PeopleDF.Group == g)[0]
            totpop = np.zeros((24, len(self.UniLocs)))
            for i in range(24):
                allp = Positions[i][whs]
                for j in allp.astype(int):
                    totpop[i, j] += 1
            pops.append(totpop)
        self.AggPositions = np.array(pops)

    def save(self, seed):
        path = self.Path_Data + self.SaveName + '/Seed_' + str(seed) + '/'
        os.makedirs(path, exist_ok=True)
        os.makedirs(self.Path_Data + 'General/', exist_ok=True)

        pd.DataFrame(self.create_people_DF(0)).to_pickle(path + 'PeopleDF.pkl')
        pd.DataFrame(self.UniLocs).to_pickle(path + 'Gemeenten.pkl')
        pd.DataFrame(self.UniIDs).to_pickle(path + 'GemeentenID.pkl')
