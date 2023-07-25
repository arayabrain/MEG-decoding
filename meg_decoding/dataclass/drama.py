import os
from torch.utils.data import Sampler, Dataset
import numpy as np
import mne
from tqdm import tqdm
from meg_decoding.matlab_utils.load_meg import get_meg_data, roi, time_window, get_baseline
import tqdm
import torch
from typing import List
from meg_decoding.utils.preproc_utils import (
    check_preprocs,
    scaleAndClamp,
    scaleAndClamp_single,
    baseline_correction_single,
)
import scipy.io
mne.set_log_level(verbose="WARNING")
import bdpy


# get