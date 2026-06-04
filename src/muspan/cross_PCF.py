
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import muspan as ms

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.setup_logger import setup_logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)