"""
ModelT — agent-based SEIR model on the Dutch municipality network.

Pipeline:
  read_model_data()      load people, mixing matrices, etc.
  read_empirical_data()  set initial S/E/I/R per municipality
  set_parameters()       transmission parameters
  initialise()           assign initial states + Weibull EI/IR timers
  simulate_new()         run T days of transitions
  save(run)              dump Status (sparse) and Lvecs
"""
import configparser
import datetime
import os
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

from transmission_model.functions import (
    new_mixmat, determine_exposed, recalc_positions,
)

warnings.filterwarnings("ignore")


class ModelT:
    def __init__(self, params_input):
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        config.read(config_path)

        self.SaveName = params_input['savename']
        self.Initialization = params_input['initialization']
        self.Seed = int(params_input['seed'])
        self.T = int(params_input['Ndays'])

        self.Path_Data = config['PATHS']['DATA']
        self.Path_Datasave = config['PATHS']['FIG']
        self.Path_RawDataMix = config['PATHS']['RAWDATA_MIX']
        self.Path_RawDataMix2 = config['PATHS']['RAWDATA_MIX2']
        self.Path_InfData = config['PATHS']['INITIALISATION_DATA']
        self.Path_ICData = config['PATHS']['ICDATA']
        self.Path_GoogleData = config['PATHS']['GOOGLEDATA']
        self.Path_PienterData = config['PATHS']['PIENTERDATA']

        self.Prob_hos = np.float64(config['PARAMS']['PROB_HOS'])
        self.Hos_lag_av = np.float64(config['PARAMS']['LAG_HOS_MEAN'])
        self.Hos_lag_sh = np.float64(config['PARAMS']['LAG_HOS_SHAPE'])
        self.Threshold = np.float64(config['PARAMS']['THRESH'])
        self.Thresholdlocal = np.float64(config['PARAMS']['THRESHLOCAL'])

        self.StartDate = datetime.datetime.strptime(
            config['PARAMS']['START_DATE'], '%d-%m-%Y'
        )
        if self.Initialization == 2:
            self.StartDate = datetime.datetime.strptime('27-02-2020', '%d-%m-%Y')
        self.EndDate = self.StartDate + datetime.timedelta(days=int(self.T))

        # resolution -> divisor on population
        self.Div = {
            'High': 100, 'Medium': 500, 'Low': 1000, 'Verylow': 5000, 'Medium100': 500
        }.get(self.SaveName, 100)

        # initialization codes:
        # 1 = MPM compartments on start date
        # 2 = 10 inf + 10 exp in Tilburg (start 27-02-2020)
        # 3 = 1 infected per municipality
        # 4 = 5 infected in Amsterdam
        # 5 = same as 4 but varying beta

        print('# ---------------------------------------------------------- #')
        print('# Starting Transmission model')
        print(f'# ------ Resolution:     {self.SaveName}')
        print(f'# ------ Mobility seed:  {self.Seed}')
        print(f'# ------ Amount days:    {int(self.T)}')
        print(f'# ------ Initialization: {self.Initialization}')
        print(f'# ------ Starting date:  {self.StartDate}')
        print('# ---------------------------------------------------------- #')

    def read_model_data(self):
        path = self.Path_Data + self.SaveName + '/Seed_' + str(self.Seed) + '/'

        self.PeopleDF = pd.read_pickle(path + 'PeopleDF.pkl')
        self.UniLocs = np.array(pd.read_pickle(path + 'Gemeenten.pkl')).T[0]
        self.UniIDs = np.array(pd.read_pickle(path + 'GemeentenID.pkl')).T[0]
        self.Homes = np.array(self.PeopleDF.Home)
        self.Groups = np.array(self.PeopleDF.Group)
        self.UniGroups = np.unique(self.Groups)

        # map group name -> index
        self.GroupsI = np.zeros(len(self.Groups))
        for i, g in enumerate(self.UniGroups):
            self.GroupsI[self.Groups == g] = i
        self.GroupsI = self.GroupsI.astype(int)

        # map home -> index
        self.HomesI = np.zeros(len(self.Homes))
        for i, l in enumerate(self.UniLocs):
            self.HomesI[self.Homes == l] = i
        self.HomesI = self.HomesI.astype(int)

        # contact mixing matrices (Polymod-NL aggregated)
        mix_dir = self.Path_RawDataMix2
        self.Mix_h_r = pd.read_excel(mix_dir + 'MUestimates_home_2.xlsx', sheet_name='Netherlands', header=None)
        self.Mix_s_r = pd.read_excel(mix_dir + 'MUestimates_school_2.xlsx', sheet_name='Netherlands', header=None)
        self.Mix_w_r = pd.read_excel(mix_dir + 'MUestimates_work_2.xlsx', sheet_name='Netherlands', header=None)
        self.Mix_o_r = pd.read_excel(mix_dir + 'MUestimates_other_locations_2.xlsx', sheet_name='Netherlands',
                                     header=None)

        self.Mix_h = new_mixmat(np.array(self.Mix_h_r))
        self.Mix_s = new_mixmat(np.array(self.Mix_s_r))
        self.Mix_w = new_mixmat(np.array(self.Mix_w_r))
        self.Mix_o = new_mixmat(np.array(self.Mix_o_r))
        self.Mix_ws = (self.Mix_s + self.Mix_w) / 2

        # also keep 0-versions (originally used for restrictions diff)
        self.Mix_h0, self.Mix_s0, self.Mix_w0, self.Mix_o0 = (
            self.Mix_h, self.Mix_s, self.Mix_w, self.Mix_o
        )
        self.Mix_ws0 = self.Mix_ws

        self.Lvecs = []

        self.HomePops = np.array([
            np.sum(self.Homes == l) for l in self.UniLocs
        ])

        self.calculatedR = {'date': [], 'effectiveR': []}

    def read_empirical_data(self):
        self.InitialS = np.zeros(len(self.UniLocs))
        self.InitialE = np.zeros(len(self.UniLocs))
        self.InitialI = np.zeros(len(self.UniLocs))
        self.InitialR = np.zeros(len(self.UniLocs))

        if self.Initialization == 1:
            self._init_from_mpm()
        elif self.Initialization == 2:
            tilburg = np.where(self.UniLocs == 'Tilburg')
            self.InitialE[tilburg] = 50
            self.InitialI[tilburg] = 10
        elif self.Initialization == 3:
            for i in range(len(self.UniLocs)):
                self.InitialI[i] = min(self.HomePops[i], 100)
        elif self.Initialization in (4, 5):
            amsterdam = np.where(self.UniLocs == 'Amsterdam')
            self.InitialI[amsterdam] = 5

        self.N = len(self.PeopleDF)

    def set_parameters(self):
        self.N = len(self.PeopleDF)
        self.EI_l = 3  # E->I Weibull scale
        self.EI_k = 1.0  # E->I Weibull shape
        self.IR_l = 9  # I->R Weibull scale
        self.IR_k = 1.0  # I->R Weibull shape

        # beta values for different scenarios / phases
        if self.Initialization == 5:
            self.Beta_f1 = 0.25
        else:
            self.Beta_f1 = 0.5
        self.Beta_f2 = 0.11
        self.Beta_f3 = 0.09
        self.Beta_f4 = 0.11

    def initialise(self):
        self.Init = np.zeros(len(self.Homes))

        # randomly assign initial E/I/R people in each municipality
        for i in range(len(self.UniLocs)):
            for amount, state_code in [
                (int(self.InitialE[i]), 1),
                (int(self.InitialI[i]), 2),
                (int(self.InitialR[i]), 3),
            ]:
                if amount > 0:
                    wh = np.where(self.Homes == self.UniLocs[i])[0]
                    self.Init[np.random.choice(wh, size=amount, replace=False)] = state_code

        self.Gammas = np.zeros((self.N, 2)) + np.nan
        self.Rhos = np.zeros((self.N, 2)) + np.nan

        In = np.where(self.Init == 2)[0]  # initially-infected
        hours = 1
        self.Rhos[In, 0] = hours * np.random.weibull(self.EI_k, size=len(In)) * self.EI_l
        self.Gammas[In, 0] = hours * np.random.weibull(self.IR_k, size=len(In)) * self.IR_l
        self.Gammas[In, 1] = np.random.choice(
            np.arange(-self.IR_k * hours, 0), size=len(In)
        )

        mixmats = np.array([self.Mix_h, self.Mix_s, self.Mix_w, self.Mix_o])
        self.mixav = np.mean(mixmats, axis=0)

        self.HG = 1
        self.Betaf = self.Beta_f1

    def simulate_new(self):
        Status = np.zeros((self.T, self.N)) + np.nan
        Status[0] = self.Init

        self.Inisum = np.sum(self.InitialI)

        # all old data, not used in this simulation.
        self.Predated = 0
        self.Timestep12March = 999999
        self.Homeschoolers = []
        self.Homeworkers = []
        self.Homeschoolers_m = [[]] * len(self.UniLocs)
        self.Homeworkers_m = [[]] * len(self.UniLocs)
        self.Workers = np.where(self.Groups == 'total')[0]

        GroupsMat = np.zeros(self.N)
        GroupsMat[self.GroupsI == 0] = 1
        self.GroupsMat = GroupsMat
        self.GroupsMat_sp = scipy.sparse.csr_matrix(GroupsMat)

        Phases = [1]

        for day in tqdm(range(1, self.T)):
            print(day)
            self.PosMat, self.PosMat0 = recalc_positions(self, day)

            En = determine_exposed(self, Status[day - 1])
            In = np.where(self.Rhos.sum(axis=1) <= day)[0]
            Rn = np.where(self.Gammas.sum(axis=1) <= day)[0]

            Status[day] = Status[day - 1]
            hours = 1

            # newly exposed
            self.Rhos[En, 0] = hours * np.random.weibull(self.EI_k, size=len(En)) * self.EI_l
            self.Rhos[En, 1] = day
            Status[day, En] = 1

            # E -> I
            self.Rhos[In, 1] = np.nan
            self.Gammas[In, 0] = hours * np.random.weibull(self.IR_k, size=len(In)) * self.IR_l
            self.Gammas[In, 1] = day
            Status[day, In] = 2

            # I -> R
            self.Gammas[Rn, 1] = np.nan
            Status[day, Rn] = 3

            Phases.append(1)

        self.Status = Status
        self.Phases = Phases

    def save(self, run):
        print(self.Status.shape)
        Status_sparse = scipy.sparse.csr_matrix(self.Status)

        path = self.Path_Data + self.SaveName + '/Seed_' + str(self.Seed) + '/'
        os.makedirs(path, exist_ok=True)

        out_dir = (path + 'Initialization' + str(self.Initialization) + '/'
                   + self.StartDate.strftime('%d%m%Y') + '-'
                   + self.EndDate.strftime('%d%m%Y'))
        os.makedirs(out_dir, exist_ok=True)

        scipy.sparse.save_npz(out_dir + '/Status_' + str(run) + '.npz', Status_sparse)
        print(f'Saved Status for run {run} in {out_dir}')
