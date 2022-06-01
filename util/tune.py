import os
import gc
import time
import importlib
from datetime import datetime

import optuna
import numpy as np
from tqdm.notebook import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib import pyplot as plt
from mpmath import mp

np.random.seed(1229)
mp.dps = 100
