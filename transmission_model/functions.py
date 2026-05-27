"""Helper functions for the ModelT transmission ABM."""
import datetime
import os
import sys

import numpy as np
import scipy.sparse


def windowmean(data, size):
    """Running window mean over a 1D time series."""
    if size == 1:
        return data
    if size == 0:
        print('Size 0 not possible!')
        sys.exit()

    result = np.zeros(len(data)) + np.nan
    half = int(size / 2.)
    for i in range(half, int(len(data) - size / 2.)):
        result[i] = np.nanmean(data[i - half:i + half])
    return np.array(result)


# --------------------------------------------------------------------------- #
# Polymod conversion
# --------------------------------------------------------------------------- #

def rivm_to_model(mat):
    newvec = np.zeros(shape=(11, 11))
    for k in range(11):
        for l in range(11):
            newvec[k, l] = np.mean(mat[iconv(k)][:, iconv(l)])
    return newvec


def iconv(i):
    """Map our 11-class age groups to RIVM age groups."""
    if i == 0:  return [0]
    if i == 1:  return [1]
    if i == 2:  return [2]
    if i in (3, 4): return [3]
    if i in (5, 6): return [4, 5, 6]
    if i in (7, 8): return [6, 7]
    if i == 9:  return [7, 8]
    if i == 10: return [9]


def translate_polymod(g):
    """Map our 11-class groups to Polymod's 16-class groups."""
    if g == 0:  return [0], [1]
    if g == 1:  return [1, 2], [1, 0.5]
    if g == 2:  return [2, 3], [0.5, 1]
    if g in (3, 4): return [4, 5], [1, 0.5]
    if g in (5, 6): return [5, 6, 7, 8, 9, 10], [0.5, 1, 1, 1, 1, 1]
    if g in (7, 8): return [11, 12, 13], [1, 1, 0.5]
    if g == 9:  return [13, 14, 15], [0.5, 1, 1]
    if g == 10: return [15], [1]


def new_mixmat(matraw):
    """Aggregate a 16x16 Polymod-style matrix to 11x11, then pick the
    middle-aged-working entry (assumption: everyone is in group 5).
    """
    i, j = 5, 5
    row, ps = translate_polymod(i)
    ps = np.array(ps)
    matrow = (matraw[row].T * ps) / len(ps)
    col, ps = translate_polymod(j)
    ps = np.array(ps)
    return np.sum(matrow[col].T * ps)


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #

def recalc_positions(self, t):
    date = self.StartDate + datetime.timedelta(days=int(t))
    path = self.Path_Data + self.SaveName + '/Seed_' + str(self.Seed) + '/'

    fname = path + 'Positions' + date.strftime('%d%m%Y') + '.npy'
    if not os.path.exists(fname):
        # fallback to a known-good day
        fname = path + 'Positions01072020.npy'
        print("File did not exist!")

    positions = np.load(fname)
    positions = np.array([p[0] for p in positions.astype(int)])

    PosMat = np.zeros((len(self.UniLocs), self.N))
    for m in range(len(self.UniLocs)):
        PosMat[m, :][positions == m] = 1
    return PosMat, PosMat


# --------------------------------------------------------------------------- #
# Force of infection
# --------------------------------------------------------------------------- #

def determine_exposed(self, Stat):
    """Pick which susceptibles get exposed this step."""
    Svec = np.zeros(self.N)
    Svec[Stat == 0] = 1
    Ivec = np.zeros(self.N)
    Ivec[Stat == 2] = 1
    Lvec = np.zeros(self.N)

    Ipos = scipy.sparse.csr_matrix(self.PosMat)
    Is = scipy.sparse.csr_matrix((self.GroupsMat_sp.multiply(Ivec)).toarray()).T

    infs = (Ipos.dot(Is)).toarray()  # infected per muni
    tots = (Ipos.dot(self.GroupsMat_sp.T)).toarray()  # total people per muni
    sucs = (Ipos.multiply(Svec)).toarray()  # susceptibles per muni

    fracs = infs / (1e-9 + tots)
    beta = self.Betaf

    munis_with_inf = np.unique(np.where(infs >= 1)[0])
    for m in munis_with_inf:
        fracs2 = fracs[m]
        if np.sum(fracs2) > 0:
            people = np.where(sucs[m] == 1)[0]
            for p in people:
                Lvec[p] = force_of_infection2(self, p, fracs2, m) * beta

    if self.Initialization == 6:
        En = np.where(np.random.random(self.N) < 1 - np.exp(-Lvec))[0]
    else:
        En = np.where(np.random.random(self.N) < Lvec)[0]
    del Svec, Ivec, Lvec, Ipos, Is
    return En


def force_of_infection2(self, p, fracs, m):
    group = 5  # assumption: everyone is in middle-aged working group
    mixvec = get_mixmat(self, m, group, p, self.Homes)
    return np.sum(mixvec * fracs * self.HG)


def get_mixmat(self, r, g, p, Homes):
    """Currently always returns 1 — placeholder for time/place-aware mixing."""
    return 1


def daytimemixer(self, r, g, p, Homes):
    """Day-time mixing matrix selector (kept for completeness, not used)."""
    if Homes[p] == self.UniLocs[r]:
        if g + 1 in [1, 7, 9, 10, 11]: return self.Mix_h
        if g + 1 in [2, 3]:            return self.Mix_s
        if g + 1 == 4:                 return self.Mix_ws
        if g + 1 in [5, 6, 8]:         return self.Mix_w
    else:
        if g + 1 in [1, 7, 9, 10, 11]: return self.Mix_o
        if g + 1 in [2, 3]:            return self.Mix_s
        if g + 1 == 4:                 return self.Mix_ws
        if g + 1 in [5, 6, 8]:         return self.Mix_w


def nighttimemixer(self, h, r, g, p, Homes):
    return self.Mix_h if Homes[p] == self.UniLocs[r] else self.Mix_o
