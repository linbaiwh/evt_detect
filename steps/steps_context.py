from pathlib import Path
import sys
import os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# topfolder = Path(r'E:\SEC filing')
tag = 'breach'
data_folder = Path(__file__).resolve().parents[1] / 'data'
topfolder = data_folder
input_folder = data_folder / 'input'
label_folder = data_folder / 'label'
model_folder = data_folder / 'model'
compare_folder = data_folder / 'compare'
feature_folder = data_folder / 'feature'
result_folder = data_folder / 'result'

logger_conf = Path(__file__).resolve().parents[1] / 'docs' / 'logging.conf'
