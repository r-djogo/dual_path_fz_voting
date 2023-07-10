import numpy as np
import json, sys


### Reading Json Results as dictionary
def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def results_rocket_ridge_AP(experiment_dir):
    file_path = experiment_dir + '/results.json'
    results = read_json(file_path)
    f_res = open(experiment_dir + '/results.txt', "w")
    f_res.write(str(results['0'].keys()))

    accu_activ = []
    baccu_activ = []
    mcc_activ = []
    f1_activ = []
    confusion_activ = []


    voted_accu_activ = []
    voted_baccu_activ = []
    voted_mcc_activ = []
    voted_f1_activ = []
    voted_confusion_activ = []

    w_voted_accu_activ = []
    w_voted_baccu_activ = []
    w_voted_mcc_activ = []
    w_voted_f1_activ = []
    

    for cv_key in results.keys():
        accu_activ.append(results[cv_key]['per_ap_acc_activ'])
        baccu_activ.append(results[cv_key]['per_ap_bacc_activ'])
        mcc_activ.append(results[cv_key]['per_ap_mcc_activ'])
        f1_activ.append(results[cv_key]['per_ap_f1_activ'])

        voted_accu_activ.append(results[cv_key]['voted_acc_activ'])
        voted_baccu_activ.append(results[cv_key]['voted_bacc_activ'])
        voted_mcc_activ.append(results[cv_key]['voted_mcc_activ'])
        voted_f1_activ.append(results[cv_key]['voted_f1_activ'])
        
        w_voted_accu_activ.append(results[cv_key]['voted_acc_activ_weighted'])
        w_voted_baccu_activ.append(results[cv_key]['voted_bacc_activ_weighted'])
        w_voted_mcc_activ.append(results[cv_key]['voted_mcc_activ_weighted'])
        w_voted_f1_activ.append(results[cv_key]['voted_f1_activ_weighted'])


    f_res.write('\nActiv Accuracy'+str(np.mean(np.asarray(accu_activ),axis=0)))
    f_res.write('\nActive std Accuracy'+str(np.std(np.asarray(accu_activ),axis=0)))

    f_res.write('\nActiv BAccuracy'+str(np.mean(np.asarray(baccu_activ),axis=0)))
    f_res.write('\nActive std BAccuracy'+str(np.std(np.asarray(baccu_activ),axis=0)))

    f_res.write('\nActiv MCC'+str(np.mean(np.asarray(mcc_activ),axis=0)))
    f_res.write('\nActive std MCC'+str(np.std(np.asarray(mcc_activ),axis=0)))

    f_res.write('\nActiv F1'+str(np.mean(np.asarray(f1_activ),axis=0)))
    f_res.write('\nActive std F1'+str(np.std(np.asarray(f1_activ),axis=0)))

    ######## Voting ########
    f_res.write('\nVoting Activ Accuracy'+str(np.mean(np.asarray(voted_accu_activ),axis=0)))
    f_res.write('\nVoting Active std Accuracy'+str(np.std(np.asarray(voted_accu_activ),axis=0)))

    f_res.write('\nVoting Activ BAccuracy'+str(np.mean(np.asarray(voted_baccu_activ),axis=0)))
    f_res.write('\nVoting Active std BAccuracy'+str(np.std(np.asarray(voted_baccu_activ),axis=0)))

    f_res.write('\nVoting Activ MCC'+str(np.mean(np.asarray(voted_mcc_activ),axis=0)))
    f_res.write('\nVoting Active std MCC'+str(np.std(np.asarray(voted_mcc_activ),axis=0)))

    f_res.write('\nVoting Activ F1'+str(np.mean(np.asarray(voted_f1_activ),axis=0)))
    f_res.write('\nVoting Active std F1'+str(np.std(np.asarray(voted_f1_activ),axis=0)))

    # weighted voting
    f_res.write('\nWeighted Voting Activ Accuracy'+str(np.mean(np.asarray(w_voted_accu_activ),axis=0)))
    f_res.write('\nWeighted Voting Active std Accuracy'+str(np.std(np.asarray(w_voted_accu_activ),axis=0)))

    f_res.write('\nWeighted Voting Activ BAccuracy'+str(np.mean(np.asarray(w_voted_baccu_activ),axis=0)))
    f_res.write('\nWeighted Voting Active std BAccuracy'+str(np.std(np.asarray(w_voted_baccu_activ),axis=0)))

    f_res.write('\nWeighted Voting Activ MCC'+str(np.mean(np.asarray(w_voted_mcc_activ),axis=0)))
    f_res.write('\nWeighted Voting Active std MCC'+str(np.std(np.asarray(w_voted_mcc_activ),axis=0)))

    f_res.write('\nWeighted Voting Activ F1'+str(np.mean(np.asarray(w_voted_f1_activ),axis=0)))
    f_res.write('\nWeighted Voting Active std F1'+str(np.std(np.asarray(w_voted_f1_activ),axis=0)))

    f_res.close()

def results_rocket_ridge(experiment_dir):
    file_path = experiment_dir + '/results.json'
    results = read_json(file_path)
    f_res = open(experiment_dir + '/results.txt', "w")
    f_res.write(str(results['0'].keys()))
    tf_flag = False
    if 'all_ap_bacc_freq' in results['0'].keys():
        tf_flag = True

    concat_accu_activ = []
    concat_baccu_activ = []
    concat_mcc_activ = []
    concat_f1_activ = []
    if tf_flag:
        concat_bacc_tf = []
        concat_bacc_t = []
        concat_bacc_f = []

    for cv_key in results.keys():
        concat_accu_activ.append(results[cv_key]['all_ap_acc_activ'])
        concat_baccu_activ.append(results[cv_key]['all_ap_bacc_activ'])
        concat_mcc_activ.append(results[cv_key]['all_ap_mcc_activ'])
        concat_f1_activ.append(results[cv_key]['all_ap_f1_activ'])
        if tf_flag:
            concat_bacc_tf.append(results[cv_key]['all_ap_bacc_timefreq'])
            concat_bacc_t.append(results[cv_key]['all_ap_bacc_time'])
            concat_bacc_f.append(results[cv_key]['all_ap_bacc_freq'])

    ######## Concatenation ########
    f_res.write('\nConcatenation Activ Accuracy'+str(np.mean(np.asarray(concat_accu_activ),axis=0)))
    f_res.write('\nConcatenation Active std Accuracy'+str(np.std(np.asarray(concat_accu_activ),axis=0)))

    f_res.write('\nConcatenation Activ BAccuracy'+str(np.mean(np.asarray(concat_baccu_activ),axis=0)))
    f_res.write('\nConcatenation Active std BAccuracy'+str(np.std(np.asarray(concat_baccu_activ),axis=0)))

    f_res.write('\nConcatenation Activ MCC'+str(np.mean(np.asarray(concat_mcc_activ),axis=0)))
    f_res.write('\nConcatenation Active std MCC'+str(np.std(np.asarray(concat_mcc_activ),axis=0)))

    f_res.write('\nConcatenation Activ F1'+str(np.mean(np.asarray(concat_f1_activ),axis=0)))
    f_res.write('\nConcatenation Active std F1'+str(np.std(np.asarray(concat_f1_activ),axis=0)))

    if tf_flag:
        f_res.write('\nTime+Freq BAcc:'+str(np.asarray(concat_bacc_tf)))
        f_res.write('\nTime+Freq BAcc avg.:'+str(np.mean(np.asarray(concat_bacc_tf),axis=0)))
        f_res.write('\nTime+Freq BAcc std.:'+str(np.std(np.asarray(concat_bacc_tf),axis=0)))

        f_res.write('\nTime BAcc:'+str(np.asarray(concat_bacc_t)))
        f_res.write('\nTime BAcc avg.:'+str(np.mean(np.asarray(concat_bacc_t),axis=0)))
        f_res.write('\nTime BAcc std.:'+str(np.std(np.asarray(concat_bacc_t),axis=0)))

        f_res.write('\nFreq BAcc:'+str(np.asarray(concat_bacc_f)))
        f_res.write('\nFreq BAcc avg.:'+str(np.mean(np.asarray(concat_bacc_f),axis=0)))
        f_res.write('\nFreq BAcc std.:'+str(np.std(np.asarray(concat_bacc_f),axis=0)))

    f_res.close()



#### Confusion matrix
def conf_rocket_ridge_AP(experiment_dir, num_APs):
    file_path = experiment_dir + '/results.json'
    results = read_json(file_path)

    vote_confusion_activ = []

    f_res = open(experiment_dir + '/results.txt', "a")

    for ap in range(num_APs):
        confusion_activ = []

        f_res.write('\nAP'+str(ap))
        for cv_key in results.keys():
            conf = np.asarray(results[cv_key]['per_ap_activ_confusion'])            
            confusion_activ.append(np.diag(conf[ap,:,:])/(np.sum(conf[ap,:,:],axis=1)))


        f_res.write('\nActiv'+str(np.mean(np.asarray(confusion_activ),axis=0)))
        f_res.write('\nActiv std'+str(np.std(np.asarray(confusion_activ),axis=0)))


        conf = np.asarray(results[cv_key]['voted_activ_confusion'])            
        vote_confusion_activ.append(np.diag(conf)/(np.sum(conf,axis=1)))

    f_res.write('\nVoted Activ'+str(np.mean(np.asarray(vote_confusion_activ),axis=0)))
    f_res.write('\nVoted Activ std'+str(np.std(np.asarray(vote_confusion_activ),axis=0)))

    f_res.close()


def conf_rocket_ridge(experiment_dir):
    file_path = experiment_dir + '/results.json'
    results = read_json(file_path)

    concat_confusion_activ = []
        
    for cv_key in results.keys():
        conf = np.asarray(results[cv_key]['all_ap_activ_confusion'])[0]       
        concat_confusion_activ.append(np.diag(conf)/(np.sum(conf,axis=1)))

    f_res = open(experiment_dir + '/results.txt', "a")

    f_res.write('\nConcat Activ'+str(np.mean(np.asarray(concat_confusion_activ),axis=0)))
    f_res.write('\nConcat Activ std'+str(np.std(np.asarray(concat_confusion_activ),axis=0)))

    f_res.close()