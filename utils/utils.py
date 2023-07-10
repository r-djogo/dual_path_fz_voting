import json

### Batching data for skipping GPU memeory error
def batcher(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

### Array to list for json dump
def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return x.tolist()
    raise TypeError(x)

### Dump dictionary as json
def dump_json(experiment_dir, cv_results_dict):
    with open(experiment_dir+'/results.json', 'w', encoding='utf-8') as f:
        json.dump(cv_results_dict, f, ensure_ascii=False,default=convert)


### Reading Json Results as dictionary
def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # print(data.keys())
    # print(np.mean(data['0']['per_ap_activ_confusion'],axis=0))        