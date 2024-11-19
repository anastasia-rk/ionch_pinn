import os
import sys
import shutil
import numpy as np
import scipy as sp
import pandas as pd
import torch as pt
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import gc # garbage collector
from tqdm import tqdm
import torch as pt
import torch.nn as nn
from time import perf_counter
from functools import partial

def check_for_executable(executable_name):
    # make sure executable is a string
    if not isinstance(executable_name, str):
        raise ValueError("executable must be a string")
    # check if executable exists on the machine
    executable_path = shutil.which("latex")
    if executable_path is None:
        print(f"no executable found for command: {executable_name}")
        return False
    else:
        print(f"path to executable found: {executable_path}")
        return True

# figure settings across the project
matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.style.use("ggplot")
# check if we can use latex for figure text
latexInstalled = check_for_executable("latex")
if latexInstalled:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]
            })
else:
    plt.rcParams.update({
        "text.usetex": False
            })

def pretty_axis(ax, legendFlag=True):
    """
    Function to set the axis colour, label, and grid
    :param ax: axis object
    :param legendFlag: flag to show the legend
    :return: axis object
    """
    # set axis labels and grid
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    if legendFlag:
        ax.legend(loc='best', fontsize=12)
    return ax

# set up input and output directories in absolute paths
# get the current working directory
cwd = os.getcwd()
# if cwd ends with methods, we are in the methods folder, so we need to go one level up
if os.path.basename(cwd) == "methods":
    # remove the last folder from the path to make sure we are in the project folder
    cwd = os.path.dirname(cwd)
print(f"Current working directory: {cwd}")
# these prefixes will be used in all other files
direcory_names = {
    "input_data": os.path.join(cwd, "protocols"),
    "models": os.path.join(cwd, "models"),
    "output_data": os.path.join(cwd, "simulated_data"),
    "figures": os.path.join(cwd, "figures"),
    "pickles": os.path.join(cwd, "pickles"),
    "tikzes": os.path.join(cwd, "tikzes")
    }
print("Directories:")
print(direcory_names)
########################################################################################################################
# set the device
device = pt.device('mps' if pt.backends.mps.is_available() else 'cuda' if pt.cuda.is_available() else 'cpu')
# define nn class for a fully connected framework - it works for all the methods at the moment
########################################################################################################################
# ensure reproducibility by setting np and pt seeds, as well as worker seeds
np.random.seed(42)
pt.manual_seed(42)
pt.cuda.manual_seed_all(42)
pt.mps.manual_seed(42)
# pt.use_deterministic_algorithms(True)
# set the default dtype
pt.set_default_dtype(pt.float32)
# set the default generator for workers
# taken directy from pytorch documentation
def seed_worker(worker_id):
    worker_seed = pt.initial_seed() % 2**32
    np.random.seed(worker_seed)
    pt.random.seed(worker_seed)
worker_generator = pt.Generator()
worker_generator.manual_seed(0)


if __name__ == '__main__':
    print('All directories set up.')
