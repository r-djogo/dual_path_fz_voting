from pathlib import Path
import numpy as np
import torch
from utils import utils, dataloader
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score
from models import minirocket


def Rocket_Ridge(configs,data_dict,logger,dataloader_class,device):
    ### Multivar MiniRocket Feature Extractor
    logger.info('MiniRocket Feature extraction')
    features_dict = {}

    if configs['time_features']:
        mrf = minirocket.MiniRocketFeatures(configs['n_freq'], configs['n_time'], num_features=configs['n_kernels']).to(device)
    else:
        mrf = minirocket.MiniRocketFeatures(configs['n_time'], configs['n_freq'], num_features=configs['n_kernels']).to(device)

    for ap in tqdm(configs['access_points']):
        if configs['time_features']:
            X_train = data_dict['AP'+str(ap)]
        else:
            X_train = np.transpose(data_dict['AP'+str(ap)], (0, 2, 1))
        features_dict['AP'+str(ap)] = []
        for ndx in utils.batcher(range(X_train.shape[0]), 64): # 64:batch size
            mrf.fit(X_train[ndx,:,:])
            X_feat = minirocket.get_minirocket_features(X_train[ndx,:,:], mrf, chunksize=1024, to_np=True)
            X_feat = np.squeeze(X_feat)
            features_dict['AP'+str(ap)].extend(X_feat)
        features_dict['AP'+str(ap)] = np.asarray(features_dict['AP'+str(ap)])

    ### Train Test Split
    indx_dict = dataloader_class.train_test_split_generator()
    

    ### Classification & Reporting
    logger.info('Ridge Classification')


    y_train_activ = data_dict['activity_labels'][indx_dict['train']]
    y_test_activ = data_dict['activity_labels'][indx_dict['test']]  


    results_dict = {}
    results_dict['AP_number'] = configs['access_points'] # list of used APs
    results_dict['Activity_class'] = configs['activities'] # list of used Activities

    results_dict['y_test_activ'] = np.asarray(y_test_activ,dtype=int)  # targets activ

    results_dict['all_ap_predictions_activ'] = [] # Collection of all APs Activity predictions 
    
    results_dict['all_ap_acc_activ'] = [] # Accuracy of all AP Activity 
    results_dict['all_ap_mcc_activ'] = [] # MCC of all AP Activity 
    results_dict['all_ap_f1_activ'] = [] # F1 of all AP Activity 
    results_dict['all_ap_bacc_activ'] = [] # Balanced acc of all AP Activity 
    results_dict['all_ap_activ_confusion'] = [] # Confusion of all AP activity


    if configs['dataset'] == 'hm':
        all_features = np.concatenate((features_dict['AP0'],features_dict['AP1'],features_dict['AP2'],features_dict['AP3'],features_dict['AP4']),axis=1)
    elif configs['dataset'] == 'widar':
        all_features = np.concatenate((features_dict['AP0'],features_dict['AP1'],features_dict['AP2'],features_dict['AP3'],features_dict['AP4'],features_dict['AP5']),axis=1)

    X_train = all_features[indx_dict['train'],:]
    X_test = all_features[indx_dict['test'],:]    


    ##### Activity Classifier
    clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train_activ)
    y_pred_activ = clf.predict(X_test)
    results_dict['all_ap_predictions_activ'].append(y_pred_activ)
    results_dict['all_ap_acc_activ'].append(accuracy_score(y_test_activ,y_pred_activ))
    results_dict['all_ap_mcc_activ'].append(matthews_corrcoef(y_test_activ,y_pred_activ))
    results_dict['all_ap_bacc_activ'].append(balanced_accuracy_score(y_test_activ,y_pred_activ))
    results_dict['all_ap_f1_activ'].append(f1_score(y_test_activ,y_pred_activ,average='weighted'))
    results_dict['all_ap_activ_confusion'].append(confusion_matrix(y_test_activ,y_pred_activ))

    results_dict['all_ap_predictions_activ'] = np.asarray(results_dict['all_ap_predictions_activ'])

    print('all_ap_bacc_activ', results_dict['all_ap_bacc_activ'])

    return results_dict


def Rocket_Ridge_AP(configs,data_dict,logger,dataloader_class,device,run,predictions):
    ### Multivar MiniRocket Feature Extractor
    logger.info('MiniRocket Feature extraction')
    features_dict = {}

    if configs['time_features']:
        mrf = minirocket.MiniRocketFeatures(configs['n_freq'], configs['n_time'], num_features=configs['n_kernels']).to(device)
    else:
        mrf = minirocket.MiniRocketFeatures(configs['n_time'], configs['n_freq'], num_features=configs['n_kernels']).to(device)
    
    for ap in tqdm(configs['access_points']):
        if configs['time_features']:
            X_train = data_dict['AP'+str(ap)]
        else:
            X_train = np.transpose(data_dict['AP'+str(ap)], (0, 2, 1))
        features_dict['AP'+str(ap)] = []
        for ndx in utils.batcher(range(X_train.shape[0]), 64): # 64:batch size
            mrf.fit(X_train[ndx,:,:])
            X_feat = minirocket.get_minirocket_features(X_train[ndx,:,:], mrf, chunksize=1024, to_np=True)
            X_feat = np.squeeze(X_feat)
            features_dict['AP'+str(ap)].extend(X_feat)
        features_dict['AP'+str(ap)] = np.asarray(features_dict['AP'+str(ap)])

    ### Train Test Split
    indx_dict = dataloader_class.train_test_split_generator()
    

    ### Classification & Reporting
    logger.info('Ridge Classification')

    
    y_test_orient = data_dict['orientation_labels'][indx_dict['test']]

    y_train_activ = data_dict['activity_labels'][indx_dict['train']]
    y_test_activ = data_dict['activity_labels'][indx_dict['test']]  


    results_dict = {}
    results_dict['AP_number'] = configs['access_points'] # list of used APs
    results_dict['Activity_class'] = configs['activities'] # list of used Activities

    results_dict['y_test_activ'] = np.asarray(y_test_activ,dtype=int)  # targets activ
    results_dict['y_test_orient'] = np.asarray(y_test_orient,dtype=int)  # targets orient

    results_dict['all_ap_predictions_activ'] = [] # Collection of all APs Activity predictions
    results_dict['all_ap_predictions_activ_weighted'] = [] # Collection of all APs Activity predictions for weighted voting
    
    results_dict['per_ap_acc_activ'] = [] # Accuracy of each AP Activity 
    results_dict['per_ap_mcc_activ'] = [] # MCC of each AP Activity 
    results_dict['per_ap_f1_activ'] = [] # F1 of each AP Activity 
    results_dict['per_ap_bacc_activ'] = [] # Balanced acc of each AP Activity 
    results_dict['per_ap_activ_confusion'] = [] # Confusion of each AP

    results_dict['voted_predictions_activ'] = [] # Voted Activity predictions of all APs
    results_dict['voted_acc_activ'] = [] # Voted acc Activity
    results_dict['voted_mcc_activ'] = [] # Voted mcc Activity
    results_dict['voted_f1_activ'] = [] # Voted f1 Activity
    results_dict['voted_bacc_activ'] = [] # Voted bacc Activity
    results_dict['voted_weighted_predictions_activ'] = []
    results_dict['voted_acc_activ_weighted'] = []
    results_dict['voted_mcc_activ_weighted'] = []
    results_dict['voted_f1_activ_weighted'] = []
    results_dict['voted_bacc_activ_weighted'] = []

    results_dict['voted_activ_confusion'] = [] # Voted Confusion activity
    results_dict['voted_activ_confusion_weighted'] = [] # Voted Weighted Confusion activity

    for ap in tqdm(configs['access_points']):  
        logger.info(f'Access Point {ap}')

        X_train = features_dict['AP'+str(ap)][indx_dict['train'],:]
        X_test = features_dict['AP'+str(ap)][indx_dict['test'],:]

        ##### Activity
        clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train_activ)
        y_pred_activ = clf.predict(X_test)
        d = clf.decision_function(X_test)
        results_dict['all_ap_predictions_activ_weighted'].append(np.exp(d) / np.sum(np.exp(d), axis=1)[:,np.newaxis])
        predictions[f'run{run}_ap{ap}'] = np.asarray(np.exp(d) / np.sum(np.exp(d), axis=1)[:,np.newaxis])
        results_dict['all_ap_predictions_activ'].append(y_pred_activ)
        results_dict['per_ap_acc_activ'].append(accuracy_score(y_test_activ,y_pred_activ))
        results_dict['per_ap_mcc_activ'].append(matthews_corrcoef(y_test_activ,y_pred_activ))
        results_dict['per_ap_bacc_activ'].append(balanced_accuracy_score(y_test_activ,y_pred_activ))
        results_dict['per_ap_f1_activ'].append(f1_score(y_test_activ,y_pred_activ,average='weighted'))
        results_dict['per_ap_activ_confusion'].append(confusion_matrix(y_test_activ,y_pred_activ))
    predictions[f'run{run}_label_gest'] = results_dict['y_test_activ']
    predictions[f'run{run}_label_ori'] = results_dict['y_test_orient']
    if configs['dataset'] == 'widar':
        predictions[f'run{run}_label_loc'] = np.asarray(data_dict['location_labels'][indx_dict['test']],dtype=int)

    #### Voting
    results_dict['all_ap_predictions_activ'] = np.asarray(results_dict['all_ap_predictions_activ'])
    results_dict['all_ap_predictions_activ_weighted'] = np.asarray(results_dict['all_ap_predictions_activ_weighted'])

    for i in range(results_dict['all_ap_predictions_activ'].shape[1]):
        count_activ = np.bincount(results_dict['all_ap_predictions_activ'][:,i])
        
        results_dict['voted_predictions_activ'].append(np.argmax(count_activ))

        if configs['dataset'] == 'widar':
            # weighted voting
            # weights are all equal as placeholder - actual voting scheme is implemented in "voting_widar.py"
            weights = np.array([1., 1., 1., 1., 1., 1.])
            pred = np.zeros(10)
            for j in range(6):
                pred += weights[j] * results_dict['all_ap_predictions_activ_weighted'][j,i,:]
            results_dict['voted_weighted_predictions_activ'].append(np.argmax(pred))
        elif configs['dataset'] == 'hm':
            # weighted voting
            # weights are all equal as placeholder - actual voting scheme is implemented in "voting_hm.py"
            weights = np.sqrt([1., 1., 1., 1., 1.])
            pred = np.zeros(5)
            for j in range(5):
                pred += weights[j] * results_dict['all_ap_predictions_activ_weighted'][j,i,:]
            results_dict['voted_weighted_predictions_activ'].append(np.argmax(pred))

    results_dict['voted_acc_activ'] = accuracy_score(y_test_activ,results_dict['voted_predictions_activ']) #np.sum(results_dict['voted_predictions_activ']==results_dict['y_test_activ'])/X_test.shape[0]
    results_dict['voted_mcc_activ'] = matthews_corrcoef(y_test_activ,results_dict['voted_predictions_activ']) # Voted mcc Activity
    results_dict['voted_f1_activ'] = f1_score(y_test_activ,results_dict['voted_predictions_activ'],average='weighted') # Voted f1 Activity
    results_dict['voted_bacc_activ'] = balanced_accuracy_score(y_test_activ,results_dict['voted_predictions_activ']) # Voted bacc Activity

    results_dict['voted_activ_confusion'] = confusion_matrix(y_test_activ,results_dict['voted_predictions_activ']) # Voted Confusion activity

    # weighted
    results_dict['voted_acc_activ_weighted'] = accuracy_score(y_test_activ,results_dict['voted_weighted_predictions_activ']) #np.sum(results_dict['voted_predictions_activ']==results_dict['y_test_activ'])/X_test.shape[0]
    results_dict['voted_mcc_activ_weighted'] = matthews_corrcoef(y_test_activ,results_dict['voted_weighted_predictions_activ']) # Voted mcc Activity
    results_dict['voted_f1_activ_weighted'] = f1_score(y_test_activ,results_dict['voted_weighted_predictions_activ'],average='weighted') # Voted f1 Activity
    results_dict['voted_bacc_activ_weighted'] = balanced_accuracy_score(y_test_activ,results_dict['voted_weighted_predictions_activ']) # Voted bacc Activity
    results_dict['voted_activ_confusion_weighted'] = confusion_matrix(y_test_activ,results_dict['voted_weighted_predictions_activ']) # Voted Confusion activity

    print('Voted BActivity Accuracy', results_dict['voted_bacc_activ'])
    print('Voted Weighted BActivity Accuracy', results_dict['voted_bacc_activ_weighted'])

    return results_dict


def Rocket_Ridge_tf(configs,data_dict,logger,dataloader_class,device):
    ### Time MiniRocket
    logger.info('MiniRocket Feature Extraction-Time')
    features_dict_t = {}
    mrf = minirocket.MiniRocketFeatures(configs['n_freq'], configs['n_time'], num_features=configs['n_kernels']).to(device)
    for ap in tqdm(configs['access_points']):
        X_train = data_dict['AP'+str(ap)]
        features_dict_t['AP'+str(ap)] = []
        for ndx in utils.batcher(range(X_train.shape[0]), 64): # 64:batch size
            mrf.fit(X_train[ndx,:,:])
            X_feat = minirocket.get_minirocket_features(X_train[ndx,:,:], mrf, chunksize=1024, to_np=True)
            X_feat = np.squeeze(X_feat)
            features_dict_t['AP'+str(ap)].extend(X_feat)
        features_dict_t['AP'+str(ap)] = np.asarray(features_dict_t['AP'+str(ap)])

    torch.cuda.empty_cache()
    ### Frequency MiniRocket
    logger.info('MiniRocket Feature Extraction-Frequency')
    features_dict_f = {}
    mrf = minirocket.MiniRocketFeatures(configs['n_time'], configs['n_freq'], num_features=configs['n_kernels']).to(device)
    for ap in tqdm(configs['access_points']):
        X_train = np.transpose(data_dict['AP'+str(ap)], (0, 2, 1))
        features_dict_f['AP'+str(ap)] = []
        for ndx in utils.batcher(range(X_train.shape[0]), 64): # 64:batch size
            mrf.fit(X_train[ndx,:,:])
            X_feat = minirocket.get_minirocket_features(X_train[ndx,:,:], mrf, chunksize=1024, to_np=True)
            X_feat = np.squeeze(X_feat)
            features_dict_f['AP'+str(ap)].extend(X_feat)
        features_dict_f['AP'+str(ap)] = np.asarray(features_dict_f['AP'+str(ap)])

    ### Train Test Split
    indx_dict = dataloader_class.train_test_split_generator()
    

    ### Classification & Reporting
    logger.info('Ridge Classification')


    y_train_activ = data_dict['activity_labels'][indx_dict['train']]
    y_test_activ = data_dict['activity_labels'][indx_dict['test']]  


    results_dict = {}
    results_dict['AP_number'] = configs['access_points'] # list of used APs
    results_dict['Activity_class'] = configs['activities'] # list of used Activities

    results_dict['y_test_activ'] = np.asarray(y_test_activ,dtype=int)  # targets activ

    results_dict['all_ap_predictions_activ'] = [] # Collection of all APs Activity predictions 
    
    results_dict['all_ap_acc_activ'] = [] # Accuracy of all AP Activity 
    results_dict['all_ap_mcc_activ'] = [] # MCC of all AP Activity 
    results_dict['all_ap_f1_activ'] = [] # F1 of all AP Activity 
    results_dict['all_ap_bacc_activ'] = [] # Balanced acc of all AP Activity 
    results_dict['all_ap_activ_confusion'] = [] # Confusion of all AP activity

    
    if configs['dataset'] == 'hm':
        all_features = np.concatenate((features_dict_t['AP0'],
                                        features_dict_t['AP1'],
                                        features_dict_t['AP2'],
                                        features_dict_t['AP3'],
                                        features_dict_t['AP4'],
                                        features_dict_f['AP0'],
                                        features_dict_f['AP1'],
                                        features_dict_f['AP2'],
                                        features_dict_f['AP3'],
                                        features_dict_f['AP4']
                                        ),axis=1)
    elif configs['dataset'] == 'widar':
        all_features = np.concatenate((features_dict_t['AP0'],
                                        features_dict_t['AP1'],
                                        features_dict_t['AP2'],
                                        features_dict_t['AP3'],
                                        features_dict_t['AP4'],
                                        features_dict_t['AP5'],
                                        features_dict_f['AP0'],
                                        features_dict_f['AP1'],
                                        features_dict_f['AP2'],
                                        features_dict_f['AP3'],
                                        features_dict_f['AP4'],
                                        features_dict_f['AP5']
                                        ),axis=1)

    X_train = all_features[indx_dict['train'],:]
    X_test = all_features[indx_dict['test'],:]

    ##### Activity Classifier
    clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train_activ)
    y_pred_activ = clf.predict(X_test)
    results_dict['all_ap_predictions_activ'].append(y_pred_activ)
    results_dict['all_ap_acc_activ'].append(accuracy_score(y_test_activ,y_pred_activ))
    results_dict['all_ap_mcc_activ'].append(matthews_corrcoef(y_test_activ,y_pred_activ))
    results_dict['all_ap_bacc_activ'].append(balanced_accuracy_score(y_test_activ,y_pred_activ))
    results_dict['all_ap_f1_activ'].append(f1_score(y_test_activ,y_pred_activ,average='weighted'))
    results_dict['all_ap_activ_confusion'].append(confusion_matrix(y_test_activ,y_pred_activ))

    results_dict['all_ap_predictions_activ'] = np.asarray(results_dict['all_ap_predictions_activ'])

    results_dict['all_ap_bacc_timefreq'] = []
    results_dict['all_ap_bacc_timefreq'].append(balanced_accuracy_score(y_test_activ,y_pred_activ))
    print('Time + Freq BAcc:', balanced_accuracy_score(y_test_activ,y_pred_activ))

    return results_dict


def Rocket_Ridge_AP_tf(configs,data_dict,logger,dataloader_class,device,run,predictions):
    ### Time MiniRocket (regular)
    logger.info('MiniRocket Feature Extraction-Time')
    features_dict_t = {}
    mrf = minirocket.MiniRocketFeatures(configs['n_freq'], configs['n_time'], num_features=configs['n_kernels']).to(device)
    for ap in tqdm(configs['access_points']):
        X_train = data_dict['AP'+str(ap)]
        features_dict_t['AP'+str(ap)] = []
        for ndx in utils.batcher(range(X_train.shape[0]), 64): # 64:batch size
            mrf.fit(X_train[ndx,:,:])
            X_feat = minirocket.get_minirocket_features(X_train[ndx,:,:], mrf, chunksize=1024, to_np=True)
            X_feat = np.squeeze(X_feat)
            features_dict_t['AP'+str(ap)].extend(X_feat)
        features_dict_t['AP'+str(ap)] = np.asarray(features_dict_t['AP'+str(ap)])

    torch.cuda.empty_cache()
    ### Frequency MiniRocket
    logger.info('MiniRocket Feature Extraction-Frequency')
    features_dict_f = {}
    mrf = minirocket.MiniRocketFeatures(configs['n_time'], configs['n_freq'], num_features=configs['n_kernels']).to(device)
    for ap in tqdm(configs['access_points']):
        X_train = np.transpose(data_dict['AP'+str(ap)], (0, 2, 1))
        features_dict_f['AP'+str(ap)] = []
        for ndx in utils.batcher(range(X_train.shape[0]), 64): # 64:batch size
            mrf.fit(X_train[ndx,:,:])
            X_feat = minirocket.get_minirocket_features(X_train[ndx,:,:], mrf, chunksize=1024, to_np=True)
            X_feat = np.squeeze(X_feat)
            features_dict_f['AP'+str(ap)].extend(X_feat)
        features_dict_f['AP'+str(ap)] = np.asarray(features_dict_f['AP'+str(ap)])
    
    features_dict = {}
    for ap in configs['access_points']:
        features_dict['AP'+str(ap)] = np.concatenate((features_dict_t['AP'+str(ap)],
                                                      features_dict_f['AP'+str(ap)]
                                                      ), axis=1)

    ### Train Test Split
    indx_dict = dataloader_class.train_test_split_generator()
    

    ### Classification & Reporting
    logger.info('Ridge Classification')

    
    y_test_orient = data_dict['orientation_labels'][indx_dict['test']]

    y_train_activ = data_dict['activity_labels'][indx_dict['train']]
    y_test_activ = data_dict['activity_labels'][indx_dict['test']]  


    results_dict = {}
    results_dict['AP_number'] = configs['access_points'] # list of used APs
    results_dict['Activity_class'] = configs['activities'] # list of used Activities

    results_dict['y_test_activ'] = np.asarray(y_test_activ,dtype=int)  # targets activ
    results_dict['y_test_orient'] = np.asarray(y_test_orient,dtype=int)  # targets orient

    results_dict['all_ap_predictions_activ'] = [] # Collection of all APs Activity predictions
    results_dict['all_ap_predictions_activ_weighted'] = [] # Collection of all APs Activity predictions for weighted voting
    
    results_dict['per_ap_acc_activ'] = [] # Accuracy of each AP Activity 
    results_dict['per_ap_mcc_activ'] = [] # MCC of each AP Activity 
    results_dict['per_ap_f1_activ'] = [] # F1 of each AP Activity 
    results_dict['per_ap_bacc_activ'] = [] # Balanced acc of each AP Activity 
    results_dict['per_ap_activ_confusion'] = [] # Confusion of each AP

    results_dict['voted_predictions_activ'] = [] # Voted Activity predictions of all APs
    results_dict['voted_acc_activ'] = [] # Voted acc Activity
    results_dict['voted_mcc_activ'] = [] # Voted mcc Activity
    results_dict['voted_f1_activ'] = [] # Voted f1 Activity
    results_dict['voted_bacc_activ'] = [] # Voted bacc Activity
    results_dict['voted_weighted_predictions_activ'] = []
    results_dict['voted_acc_activ_weighted'] = []
    results_dict['voted_mcc_activ_weighted'] = []
    results_dict['voted_f1_activ_weighted'] = []
    results_dict['voted_bacc_activ_weighted'] = []

    results_dict['voted_activ_confusion'] = [] # Voted Confusion activity
    results_dict['voted_activ_confusion_weighted'] = [] # Voted Weighted Confusion activity

    for ap in tqdm(configs['access_points']):  
        logger.info(f'Access Point {ap}')

        X_train = features_dict['AP'+str(ap)][indx_dict['train'],:]
        X_test = features_dict['AP'+str(ap)][indx_dict['test'],:]

        ##### Activity
        clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train_activ)
        y_pred_activ = clf.predict(X_test)
        d = clf.decision_function(X_test)
        results_dict['all_ap_predictions_activ_weighted'].append(np.exp(d) / np.sum(np.exp(d), axis=1)[:,np.newaxis])
        predictions[f'run{run}_ap{ap}'] = np.asarray(np.exp(d) / np.sum(np.exp(d), axis=1)[:,np.newaxis])
        results_dict['all_ap_predictions_activ'].append(y_pred_activ)
        results_dict['per_ap_acc_activ'].append(accuracy_score(y_test_activ,y_pred_activ))
        results_dict['per_ap_mcc_activ'].append(matthews_corrcoef(y_test_activ,y_pred_activ))
        results_dict['per_ap_bacc_activ'].append(balanced_accuracy_score(y_test_activ,y_pred_activ))
        results_dict['per_ap_f1_activ'].append(f1_score(y_test_activ,y_pred_activ,average='weighted'))
        results_dict['per_ap_activ_confusion'].append(confusion_matrix(y_test_activ,y_pred_activ))
    predictions[f'run{run}_label_gest'] = results_dict['y_test_activ']
    predictions[f'run{run}_label_ori'] = results_dict['y_test_orient']
    if configs['dataset'] == 'widar':
        predictions[f'run{run}_label_loc'] = np.asarray(data_dict['location_labels'][indx_dict['test']],dtype=int)

    #### Voting
    results_dict['all_ap_predictions_activ'] = np.asarray(results_dict['all_ap_predictions_activ'])
    results_dict['all_ap_predictions_activ_weighted'] = np.asarray(results_dict['all_ap_predictions_activ_weighted'])

    for i in range(results_dict['all_ap_predictions_activ'].shape[1]):
        count_activ = np.bincount(results_dict['all_ap_predictions_activ'][:,i])
        
        results_dict['voted_predictions_activ'].append(np.argmax(count_activ))

        if configs['dataset'] == 'widar':
            # weighted voting
            # weights are all equal as placeholder - actual voting scheme is implemented in "voting_widar.py"
            weights = np.array([1., 1., 1., 1., 1., 1.])
            pred = np.zeros(10)
            for j in range(6):
                pred += weights[j] * results_dict['all_ap_predictions_activ_weighted'][j,i,:]
            results_dict['voted_weighted_predictions_activ'].append(np.argmax(pred))
        elif configs['dataset'] == 'hm':
            # weighted voting
            # weights are all equal as placeholder - actual voting scheme is implemented in "voting_widar.py"
            weights = np.sqrt([1., 1., 1., 1., 1.])
            pred = np.zeros(5)
            for j in range(5):
                pred += weights[j] * results_dict['all_ap_predictions_activ_weighted'][j,i,:]
            results_dict['voted_weighted_predictions_activ'].append(np.argmax(pred))

    results_dict['voted_acc_activ'] = accuracy_score(y_test_activ,results_dict['voted_predictions_activ']) #np.sum(results_dict['voted_predictions_activ']==results_dict['y_test_activ'])/X_test.shape[0]
    results_dict['voted_mcc_activ'] = matthews_corrcoef(y_test_activ,results_dict['voted_predictions_activ']) # Voted mcc Activity
    results_dict['voted_f1_activ'] = f1_score(y_test_activ,results_dict['voted_predictions_activ'],average='weighted') # Voted f1 Activity
    results_dict['voted_bacc_activ'] = balanced_accuracy_score(y_test_activ,results_dict['voted_predictions_activ']) # Voted bacc Activity

    results_dict['voted_activ_confusion'] = confusion_matrix(y_test_activ,results_dict['voted_predictions_activ']) # Voted Confusion activity

    # weighted
    results_dict['voted_acc_activ_weighted'] = accuracy_score(y_test_activ,results_dict['voted_weighted_predictions_activ']) #np.sum(results_dict['voted_predictions_activ']==results_dict['y_test_activ'])/X_test.shape[0]
    results_dict['voted_mcc_activ_weighted'] = matthews_corrcoef(y_test_activ,results_dict['voted_weighted_predictions_activ']) # Voted mcc Activity
    results_dict['voted_f1_activ_weighted'] = f1_score(y_test_activ,results_dict['voted_weighted_predictions_activ'],average='weighted') # Voted f1 Activity
    results_dict['voted_bacc_activ_weighted'] = balanced_accuracy_score(y_test_activ,results_dict['voted_weighted_predictions_activ']) # Voted bacc Activity
    results_dict['voted_activ_confusion_weighted'] = confusion_matrix(y_test_activ,results_dict['voted_weighted_predictions_activ']) # Voted Confusion activity

    print('Voted BActivity Accuracy', results_dict['voted_bacc_activ'])
    print('Voted Weighted BActivity Accuracy', results_dict['voted_bacc_activ_weighted'])
    
    return results_dict
