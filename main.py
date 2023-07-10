import os, datetime
import torch
import logging, yaml
from pathlib import Path
import numpy as np
from utils import dataloader, utils
from utils.results_analyzer import *
from models import minirocket_models

######################## Configs and Directory Setup ########################
### Experiment Name
start_time_obj = datetime.datetime.now()
experiment_name = start_time_obj.strftime("%d%b%H_%M_%S")

### Direcotires Setup
home_dir = os.getcwd()
experiment_dir = './logs/'+experiment_name
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path('./data').mkdir(parents=True, exist_ok=True)

### Read Configs
with open('./configs.yml') as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)
configs['experiment_name'] = experiment_name
### Store Configs
with open(experiment_dir+'/'+experiment_name+'.yml', 'w') as outfile:
    yaml.dump(configs, outfile, default_flow_style=False)

######################## Logging ########################
logger = logging.getLogger(__name__)  
logger.setLevel(logging.DEBUG)
extra = {'experiment_name':experiment_name}

formatter = logging.Formatter('Exp: %(experiment_name)s @ %(asctime)s : %(name)s : %(levelname)s : %(message)s')

handler = logging.FileHandler(experiment_dir+'/'+experiment_name +'.log')
handler.setFormatter(formatter)
logger.addHandler(handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger = logging.LoggerAdapter(logger, extra)

######################## Computing Environment Setup ########################
if torch.cuda.is_available():
    logger.info(f'Available GPUs: {torch.cuda.device_count()}')
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = configs['gpu'] 
else:
    logger.info('GPU is not available.')
    device = torch.device("cpu")

### Fix random seeds for reproducibility
SEED = 12345
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)    

######################## Loading Data ########################
"""
Data should be available in ./data directory.
Format after loading into "data_dict" should be:
    data_dict['AP<#>'] should contain CSI samples for AP number <#> (replace <#> with AP index 0,1,2,...)
        -> dimensions should be [number_of_samples, number_of_subcarriers, number_of_time_steps]
    data_dict['activity_labels'] should contain activity labels {0,1,2,3,...} for corresponding CSI samples
        -> dimensions should be [number_of_samples]
    data_dict['orientation_labels'] should contain orientation labels {0,1,...} for corresponding CSI samples
        -> dimensions should be [number_of_samples]
        -> for HM dataset the labels should correpond to orientations as: {0: 0, 45: 1, 90: 2, 180: 3}
        -> for Widar3.0 dataset the labels should correspond to orientations as: {135: 0, 90: 1, 45: 2, 0: 3, -45: 4}
    data_dict['location_labels'] should contain location labels {0,1,...} for corresponding CSI samples
        -> dimensions should be [number_of_samples]
        -> this should only exist for the Widar3.0 dataset, not HM dataset
        -> for Widar3.0 dataset the labels should correspond to locations in the same order as in the IEEE Dataport documentation (pg. 4,5)
"""

if configs['dataset'] == 'hm':
    dataloader_class = dataloader.dataloader(configs,logger)
    data_dict = np.load('./data/data_hm.npy', allow_pickle=True).item()
    configs['n_samples'] = data_dict['AP4'].shape[0]
    configs['n_freq'] = data_dict['AP4'].shape[1]
    configs['n_time'] = data_dict['AP4'].shape[2]
elif configs['dataset'] == 'widar':
    dataloader_class = dataloader.dataloader(configs,logger)
    data_dict = np.load('./data/data_widar.npy', allow_pickle=True).item()
    configs['access_points'] = [0,1,2,3,4,5]
    configs['n_samples'] = data_dict['AP0'].shape[0]
    configs['n_freq'] = data_dict['AP0'].shape[1]
    configs['n_time'] = data_dict['AP0'].shape[2]

######################## Cross-Validation Run ########################
cv_results_dict = {}
predictions = {}
for cv_indx in range(configs['cv']): # cross-validation run
    logger.info(f'Starting run {cv_indx}')
    if configs['model'] == 'Rocket_Ridge_AP': # A classifier per AP
        results_dict = minirocket_models.Rocket_Ridge_AP(configs,data_dict,logger,dataloader_class,device,
                                                         cv_indx,predictions)
        np.save(experiment_dir+'/AP_predictions.npy', predictions, allow_pickle=True)
    elif configs['model'] == 'Rocket_Ridge': # A classifier for mixed features of APs
        results_dict = minirocket_models.Rocket_Ridge(configs,data_dict,logger,dataloader_class,device)
    elif configs['model'] == 'Rocket_Ridge_tf':
        results_dict = minirocket_models.Rocket_Ridge_tf(configs,data_dict,logger,dataloader_class,device)
    elif configs['model'] == 'Rocket_Ridge_AP_tf':
        results_dict = minirocket_models.Rocket_Ridge_AP_tf(configs,data_dict,logger,dataloader_class,device,
                                                            cv_indx,predictions)
        np.save(experiment_dir+'/AP_predictions.npy', predictions, allow_pickle=True)
    cv_results_dict[cv_indx] = results_dict


### Store results
utils.dump_json(experiment_dir,cv_results_dict)

### Print results in readable format
if configs['model'] == 'Rocket_Ridge_AP' or configs['model'] == 'Rocket_Ridge_AP_tf':
    results_rocket_ridge_AP(experiment_dir)
    conf_rocket_ridge_AP(experiment_dir, len(configs['access_points']))
elif configs['model'] == 'Rocket_Ridge' or configs['model'] == 'Rocket_Ridge_tf':
    results_rocket_ridge(experiment_dir)
    conf_rocket_ridge(experiment_dir)