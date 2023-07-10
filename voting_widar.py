import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy import stats


experiment_name = "name_of_experiment" # change to name of experiment
predictions = np.load('./logs/{experiment_name}/AP_predictions.npy', allow_pickle=True).item()

APs = [0,1,2,3,4,5] # APs to be included
runs = 10


##### Majority Voting
B_ACC = []

for run in range(runs):
    # voting
    y_true = predictions[f'run{run}_label_gest']

    voted_pred = np.zeros((predictions[f'run{run}_ap0'].shape[0], len(APs)))
    for ap in APs:
        voted_pred[:,ap] += np.argmax(predictions[f'run{run}_ap{ap}'], axis=1)
    m = stats.mode(voted_pred.transpose(), keepdims=False)
    B_ACC.append(balanced_accuracy_score(y_true, m[0]))

print("MAJORITY VOTING avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC), '\n')


##### VOTING - FSPL (location only)
left = -2.5
right = 0
top = 0
bottom = -2.5

tx_x = right-0.5
tx_y = top-0.5

user_x = [tx_x-1.365, tx_x-0.455, tx_x-0.455, tx_x-1.365, tx_x-0.91]
user_y = [tx_y-0.455, tx_y-0.455, tx_y-1.365, tx_y-1.365, tx_y-0.91]

ap_x = [tx_x-0.5, tx_x-1.4, tx_x-2, 0, 0, tx_x]
ap_y = [0, 0, tx_y, tx_y-0.5, tx_y-1.4, tx_y-2]

fr = 5825.0*1e6
c = 2.997925e8
wavelen = c/fr

B_ACC = []

weights = np.zeros([5,6])
for ap in range(6):
    for loc in range(5):
        distance = np.sqrt((tx_x - user_x[loc])**2 + (tx_y - user_y[loc])**2) + \
                np.sqrt((ap_x[ap] - user_x[loc])**2 + (ap_y[ap] - user_y[loc])**2)
        weights[loc][ap] = 1/distance**2 # free space pathloss proportional to: 1/dist^2

for run in range(runs):
    # voting
    y_true = predictions[f'run{run}_label_gest']
    loc = predictions[f'run{run}_label_loc']
    ori = predictions[f'run{run}_label_ori']
    
    voted_pred = np.zeros((predictions[f'run{run}_ap0'].shape[0], predictions[f'run{run}_ap0'].shape[1]))
    for ap in APs:
        voted_pred += [[weights[l][ap]]*10 for l in loc] * predictions[f'run{run}_ap{ap}']
    B_ACC.append(balanced_accuracy_score(y_true, np.argmax(voted_pred, axis=1)))

print("FSPL WEIGHTS avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC), '\n')


##### VOTING - location plus orientation
left = -2.5
right = 0
top = 0
bottom = -2.5

tx_x = right-0.5
tx_y = top-0.5

user_x = [tx_x-1.365, tx_x-0.455, tx_x-0.455, tx_x-1.365, tx_x-0.91]
user_y = [tx_y-0.455, tx_y-0.455, tx_y-1.365, tx_y-1.365, tx_y-0.91]

ap_x = [tx_x-0.5, tx_x-1.4, tx_x-2, 0, 0, tx_x]
ap_y = [0, 0, tx_y, tx_y-0.5, tx_y-1.4, tx_y-2]

fr = 5825.0*1e6
c = 2.997925e8
wavelen = c/fr

normal_dir_loc_ap = np.zeros([5,6])
n = [[79-1, 36-1, 9-1, 101-1, 80-1, 59-1],
     [35-1, 20-1, 10-1, 35-1, 20-1, 10-1],
     [101-1, 80-1, 59-1, 79-1, 36-1, 9-1],
     [128-1, 90-1, 56-1, 128-1, 90-1, 56-1],
     [80-1, 51-1, 28-1, 80-1, 51-1, 28-1]]

for loc in range(5):
    for ap in range(len(APs)):
        x1 = tx_x
        x2 = ap_x[ap]
        y1 = tx_y
        y2 = ap_y[ap]
        
        d = 1/2*np.sqrt((x2-x1)**2+(y2-y1)**2)
        a = d + n[loc][ap]*wavelen/4
        r = np.sqrt((d+n[loc][ap]*wavelen/4)**2 - d**2)

        # find angle to closest point on surface of ellipse to user within margin of error
        accuracy = 1e-8
        w = np.arctan2(y2-y1,x2-x1)
        t = np.linspace(0,2*np.pi,num=300)
        dist2 = (user_x[loc] - ((x1+x2)/2 + a*np.cos(t)*np.cos(w) - r*np.sin(t)*np.sin(w)))**2 \
                + (user_y[loc] - ((y1+y2)/2 + a*np.cos(t)*np.sin(w) + r*np.sin(t)*np.cos(w)))**2
        sort_dist2 = np.sort(dist2)
        ind = np.argmin(dist2)
        best_t = t[ind]
        
        while (sort_dist2[2] - sort_dist2[1]) > accuracy:
            t = np.linspace(t[ind-1], t[ind+1], num=99)
            dist2 = (user_x[loc] - ((x1+x2)/2 + a*np.cos(t)*np.cos(w) - r*np.sin(t)*np.sin(w)))**2 \
                    + (user_y[loc] - ((y1+y2)/2 + a*np.cos(t)*np.sin(w) + r*np.sin(t)*np.cos(w)))**2
            sort_dist2 = np.sort(dist2)
            ind = np.argmin(dist2)
            best_t = t[ind]

        X = a*np.cos(best_t)
        Y = r*np.sin(best_t)
        closest_x = (x1+x2)/2 + X*np.cos(w) - Y*np.sin(w)
        closest_y = (y1+y2)/2 + X*np.sin(w) + Y*np.cos(w)

        normal_dir_loc_ap[loc,ap] = np.arctan2((closest_y-user_y[loc]), (closest_x-user_x[loc]))

orients = np.pi/180 * np.array([135.0, 90.0, 45.0, 0.0, -45.0])

betas = np.linspace(0.0, 1.0, num=21)
best_acc = 0
best_std = 0
best_beta = 0
beta_accs = []
for beta in betas:
    B_ACC = []

    for run in range(runs):
        # voting
        y_true = predictions[f'run{run}_label_gest']
        loc = predictions[f'run{run}_label_loc']
        ori = predictions[f'run{run}_label_ori']
        true_orients = np.array([[orients[int(o)]] for o in ori])
        
        voted_pred = np.zeros((predictions[f'run{run}_ap0'].shape[0], predictions[f'run{run}_ap0'].shape[1]))
        for ap in APs:
            voted_pred += (1.0 + beta*np.cos(true_orients - (np.array([[normal_dir_loc_ap[l,ap]] for l in loc])))) \
                        * np.array([[weights[l][ap]] for l in loc]) * predictions[f'run{run}_ap{ap}']
        B_ACC.append(balanced_accuracy_score(y_true, np.argmax(voted_pred, axis=1)))

    if np.mean(B_ACC) > best_acc:
        best_acc = np.mean(B_ACC)
        best_std = np.std(B_ACC)
        best_beta = beta

    beta_accs.append(np.mean(B_ACC))

print(best_beta)
print("LOC+ORI WEIGHTS avg balanced accuracy:", best_acc, "std balanced accuracy:", best_std, '\n')


##### individual APs
for ap in APs:
    # get performance metrics
    B_ACC = []
    for run in range(runs):
        y_true = predictions[f'run{run}_label_gest']
        B_ACC.append(balanced_accuracy_score(y_true, np.argmax(predictions[f'run{run}_ap{ap}'], axis=1)))
    print(f"AP{ap}", "avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC))