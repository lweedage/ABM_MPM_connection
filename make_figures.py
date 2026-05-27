"""
Make the 2x2 estimator-comparison figures from the pooled CSV.

WHAT THIS DOES
    Reads the CSV that compare_estimators.py writes, and draws the 2x2 figure
    (rows = beta scenario, columns = ABM vs MPM). It can draw two versions:
        - "idealised"  : initialization = False  (true S and I)   -> Figure 1
        - "with init"  : initialization = True   (reconstructed)  -> Figure 3

HOW TO USE
    1. Run run_estimation.py, then compare_estimators.py (these make the CSV).
    2. Edit the SETTINGS block below if you want (you usually don't need to).
    3. Run this file (the green play button, or `python make_figures.py`).
    The figures are saved into the folder set by OUTPUT_FOLDER.

NOTE ON THE SHADED BAND
    The shaded band is the 2.5%-97.5% spread of the per-realization estimates
    across simulation runs (how much the estimate varies from run to run).
    It is NOT a confidence interval.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# SETTINGS  --  edit these, then just run the file
# ============================================================

# where the CSV from compare_estimators.py lives
INPUT_CSV = 'Output/PlotData/estimator_comparison_pooled.csv'

# where to save the figures
OUTPUT_FOLDER = 'Output/Plots'

# which figures to make. Set to True/False.
MAKE_NO_INIT_FIGURE = True  # Figure 1: true S and I
MAKE_WITH_INIT_FIGURE = True  # Figure 3: reconstructed S and I

# colours for the two estimators
COLOR_ABM = '#541e4e'  # dark purple (rocket palette)
COLOR_MPM = '#f06043'  # orange      (rocket palette)

# how far above/below the true beta the y-axis should reach
Y_RANGE = 0.2

# ============================================================
# (you should not need to change anything below here)
# ============================================================

COLOR = {'ABM': COLOR_ABM, 'MPM': COLOR_MPM}
COLUMN_TITLE = {'ABM': 'ABM mean-field', 'MPM': 'Classical MPM'}
LINESTYLE = {4: '-', 5: '--'}  # solid for beta=0.5, dashed for beta=0.25
TRUE_BETA_FOR = {4: 0.5, 5: 0.25}  # init_val 4 -> 0.5, init_val 5 -> 0.25

TOP_ROW_INIT_VAL = 4  # beta = 0.5 on top
BOTTOM_ROW_INIT_VAL = 5  # beta = 0.25 on the bottom
LEFT_COLUMN = 'ABM'
RIGHT_COLUMN = 'MPM'

FIGURE_TITLE = {
    False: 'Estimated transmission rate -- true $S$ and $I$',
    True: 'Estimated transmission rate -- with reconstruction (estimated $S$ and $I$)',
}
FILE_NAME = {
    False: 'estimator_comparison_2x2_no_initialization.png',
    True: 'estimator_comparison_2x2_with_initialization.png',
}


def read_csv(path):
    """Load the pooled CSV and make sure the columns we need exist."""
    table = pd.read_csv(path)

    # fill in true_beta / init_val if an older CSV is missing them
    if 'true_beta' not in table.columns:
        if 'init_val' in table.columns:
            table['true_beta'] = table['init_val'].map(TRUE_BETA_FOR)
        else:
            table['true_beta'] = table['initialization'].map({4: 0.5, 5: 0.25})
    if 'init_val' not in table.columns:
        table['init_val'] = table['true_beta'].map({0.5: 4, 0.25: 5})

    # make sure the initialization column is True/False, not text or 4/5
    if table['initialization'].dtype != bool:
        if set(table['initialization'].unique()) <= {4, 5}:
            table['initialization'] = False  # very old CSV: treat as idealised
        else:
            table['initialization'] = table['initialization'].astype(str).isin(
                ['True', 'true', '1'])
    return table


def get_one_panel(table, estimator, init_val, initialization):
    """Pull out the rows for a single panel, or None if there are none."""
    rows = table[(table['estimator'] == estimator)
                 & (table['init_val'] == init_val)
                 & (table['initialization'] == initialization)]
    rows = rows.sort_values('day')
    return rows if len(rows) else None


def draw_panel(ax, rows, estimator, init_val):
    """Draw one of the four panels."""
    true_beta = TRUE_BETA_FOR[init_val]
    ax.set_ylim(true_beta - Y_RANGE, true_beta + Y_RANGE)

    if rows is None:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes)
        return true_beta

    true_beta = float(rows['true_beta'].iloc[0])
    ax.set_ylim(true_beta - Y_RANGE, true_beta + Y_RANGE)
    days = rows['day'].to_numpy()
    colour = COLOR[estimator]
    n_runs = int(rows['n_realizations'].iloc[0])

    # shaded band (spread across runs) + median line + true-beta line
    ax.fill_between(days, rows['beta_q025'], rows['beta_q975'],
                    color=colour, alpha=0.25)
    ax.plot(days, rows['beta_median'], color=colour,
            linestyle=LINESTYLE[init_val], linewidth=2,
            marker='o', markersize=3, label=f'median (n={n_runs})')
    ax.axhline(true_beta, color='red', linestyle=':', alpha=0.7,
               label=fr'true $\beta = {true_beta}$')
    ax.legend(fontsize=8, loc='upper right')
    return true_beta


def make_figure(table, initialization):
    """Make and save one full 2x2 figure for the chosen initialization."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    layout = [
        [(LEFT_COLUMN, TOP_ROW_INIT_VAL), (RIGHT_COLUMN, TOP_ROW_INIT_VAL)],
        [(LEFT_COLUMN, BOTTOM_ROW_INIT_VAL), (RIGHT_COLUMN, BOTTOM_ROW_INIT_VAL)],
    ]

    for row_index, row in enumerate(layout):
        for col_index, (estimator, init_val) in enumerate(row):
            ax = axes[row_index, col_index]
            rows = get_one_panel(table, estimator, init_val, initialization)
            true_beta = draw_panel(ax, rows, estimator, init_val)

            if row_index == 0:
                ax.set_title(COLUMN_TITLE[estimator])
            if col_index == 0:
                ax.set_ylabel(fr'$\beta = {true_beta}$' + '\n\n' + r'$\beta$')
            if row_index == 1:
                ax.set_xlabel('day')

    fig.suptitle(FIGURE_TITLE[initialization], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if initialization:
        plt.xlim(15, 99 - 8)
    else:
        plt.xlim(0, 99)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    save_path = os.path.join(OUTPUT_FOLDER, FILE_NAME[initialization])
    fig.savefig(save_path, dpi=150)
    print(f'Saved {save_path}')


# ============================================================
# run it
# ============================================================
table = read_csv(INPUT_CSV)

if MAKE_NO_INIT_FIGURE:
    if (table['initialization'] == False).any():
        make_figure(table, initialization=False)
    else:
        print('No initialization rows in the CSV; skipping that figure.')

if MAKE_WITH_INIT_FIGURE:
    if (table['initialization'] == True).any():
        make_figure(table, initialization=True)
    else:
        print('With-initialization rows in the CSV; skipping that figure.')
