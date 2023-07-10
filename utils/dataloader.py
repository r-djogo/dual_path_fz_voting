import random

class dataloader:
    def __init__(self, configs,logger):
        self.configs = configs
        self.logger = logger
        self.logger.info('Dataset is '+ self.configs['dataset'])

    def train_test_split_generator(self):
        split_indexs={}
        indexs = random.sample(range(self.configs['n_samples']), self.configs['n_samples'])
        # print(indexs,len(indexs))
        self.configs['n_train'] = int(self.configs['n_samples']*self.configs['train_per'])
        self.configs['n_test'] = int(self.configs['n_samples']*self.configs['test_per'])
        self.configs['n_valid'] = int(self.configs['n_samples']-(self.configs['n_train']+self.configs['n_test']))
        
        split_indexs['train'] = indexs[:self.configs['n_train']]
        split_indexs['valid'] = indexs[self.configs['n_train']: self.configs['n_train']+self.configs['n_valid']]
        split_indexs['test'] = indexs[self.configs['n_train'] + self.configs['n_valid']:]
        assert len(split_indexs['train'])==self.configs['n_train'] 
        assert len(split_indexs['valid'])==self.configs['n_valid'] 
        assert len(split_indexs['test'])==self.configs['n_test']
        # print(self.configs)
        return split_indexs

