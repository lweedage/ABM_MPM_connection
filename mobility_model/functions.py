"""Helper functions for the ModelM mobility model."""
import numpy as np


def draw_fractions(self, r, g, date):
    """Draw a Dirichlet movement schedule for residents of municipality r."""
    lst = np.copy(self.MobMat[r]) * 1
    lst = lst / self.HomePop[r]
    lst[r] = 1.5  # boost home-municipality probability
    lst[lst <= 0] = 1e-9
    lst = 2.5 * lst / np.sum(lst)
    return np.array(np.random.dirichlet(lst))


def translate_polymod(g):
    """Map 11-class age groups to Polymod's 16-class groups."""
    if g == 0:  return [0], [1]
    if g == 1:  return [1, 2], [1, 0.5]
    if g == 2:  return [2, 3], [0.5, 1]
    if g in (3, 4): return [4, 5], [1, 0.5]
    if g in (5, 6): return [5, 6, 7, 8, 9, 10], [0.5, 1, 1, 1, 1, 1]
    if g in (7, 8): return [11, 12, 13], [1, 1, 0.5]
    if g == 9:  return [13, 14, 15], [0.5, 1, 1]
    if g == 10: return [15], [1]


def new_mixmat(matraw):
    """Aggregate a 16x16 Polymod matrix down to 11x11."""
    mat = np.zeros((11, 11))
    for i in range(11):
        row, ps = translate_polymod(i)
        ps = np.array(ps)
        matrow = (matraw[row].T * ps) / len(ps)
        for j in range(11):
            col, ps = translate_polymod(j)
            ps = np.array(ps)
            mat[i, j] = np.sum(matrow[col].T * ps)
    return mat
