import sys
import os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import evt_detect

data_folder = Path(__file__).resolve().parents[1] / 'data'
model_folder = data_folder / 'model'
