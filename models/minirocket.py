
__all__ = ['MiniRocketFeatures', 'MRF', 'get_minirocket_features', 'MiniRocketHead', 'MiniRocket']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# Cell
class MiniRocketFeatures(nn.Module):

    kernel_size, num_kernels, fitting = 9, 84, False

    def __init__(self, c_in, seq_len, num_features=10000, max_dilations_per_kernel=32, random_state=None):
        super(MiniRocketFeatures, self).__init__()
        self.c_in, self.seq_len = c_in, seq_len
        self.num_features = num_features // self.num_kernels * self.num_kernels
        print(num_features,self.num_features,self.num_kernels)
        self.max_dilations_per_kernel  = max_dilations_per_kernel
        self.random_state = random_state

        # Convolution
        indices = torch.combinations(torch.arange(self.kernel_size), 3).unsqueeze(1)
        kernels = (-torch.ones(self.num_kernels, 1, self.kernel_size)).scatter_(2, indices, 2)
        self.kernels = nn.Parameter(kernels.repeat(c_in, 1, 1), requires_grad=False)

        # Dilations & padding
        self._set_dilations(seq_len)

        # Channel combinations (multivariate)
        if c_in > 1:
            self._set_channel_combinations(c_in)

        # Bias
        for i in range(self.num_dilations):
            self.register_buffer(f'biases_{i}', torch.empty((self.num_kernels, self.num_features_per_dilation[i])))
        self.register_buffer('prefit', torch.BoolTensor([False]))

    def fit(self, X, chunksize=None):
        num_samples = X.shape[0]
        if chunksize is None:
            chunksize = min(num_samples, self.num_dilations * self.num_kernels)
        else:
            chunksize = min(num_samples, chunksize)
        np.random.seed(self.random_state)
        idxs = np.random.choice(num_samples, chunksize, False)
        self.fitting = True
        if isinstance(X, np.ndarray):
            self(torch.from_numpy(X[idxs]).to(self.kernels.device))
        else:
            self(X[idxs].to(self.kernels.device))
        self.fitting = False

    def forward(self, x):
        _features = []
        # print(self.dilations,self.padding)
        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):
            # print(i)
            _padding1 = i%2
            # print(x.shape)
            # Convolution
            C = F.conv1d(x, self.kernels.double(), padding=padding, dilation=dilation, groups=self.c_in)
            if self.c_in > 1: # multivariate
                # print('C before',C.shape)
                C = C.reshape(x.shape[0], self.c_in, self.num_kernels, -1) # To n_sample x n_cha x n_kern x length
                # print('C after',C.shape)
                channel_combination = getattr(self, f'channel_combinations_{i}') # binary torch.Size([1, n_c, n_ker, 1])
                # print(channel_combination.shape)
                C = torch.mul(C, channel_combination)
                C = C.sum(1) # n_sample x n_kernel x length
                # print('C final',C.shape)


            # Bias
            if not self.prefit or self.fitting:
                num_features_this_dilation = self.num_features_per_dilation[i]
                bias_this_dilation = self._get_bias(C, num_features_this_dilation)
                setattr(self, f'biases_{i}', bias_this_dilation)
                if self.fitting:
                    if i < self.num_dilations - 1:
                        continue
                    else:
                        self.prefit = torch.BoolTensor([True])
                        return
                elif i == self.num_dilations - 1:
                    self.prefit = torch.BoolTensor([True])
            else:
                bias_this_dilation = getattr(self, f'biases_{i}')

            # Features
            _features.append(self._get_PPVs(C[:, _padding1::2], bias_this_dilation[_padding1::2]))
            _features.append(self._get_PPVs(C[:, 1-_padding1::2, padding:-padding], bias_this_dilation[1-_padding1::2]))
        return torch.cat(_features, dim=1)

    def _get_PPVs(self, C, bias):
        # print(C.shape)
        C = C.unsqueeze(-1)
        bias = bias.view(1, bias.shape[0], 1, bias.shape[1])
        return (C > bias).float().mean(2).flatten(1)

    def _set_dilations(self, input_length):
        num_features_per_kernel = self.num_features // self.num_kernels
        true_max_dilations_per_kernel = min(num_features_per_kernel, self.max_dilations_per_kernel)
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel
        max_exponent = np.log2((input_length - 1) / (9 - 1))
        dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)
        remainder = num_features_per_kernel - num_features_per_dilation.sum()
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)
        self.num_features_per_dilation = num_features_per_dilation
        self.num_dilations = len(dilations)
        self.dilations = dilations
        self.padding = []
        for i, dilation in enumerate(dilations):
            self.padding.append((((self.kernel_size - 1) * dilation) // 2))

    def _set_channel_combinations(self, num_channels):
        num_combinations = self.num_kernels * self.num_dilations
        max_num_channels = min(num_channels, 9)
        max_exponent_channels = np.log2(max_num_channels + 1)
        np.random.seed(self.random_state)
        num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent_channels, num_combinations)).astype(np.int32)
        channel_combinations = torch.zeros((1, num_channels, num_combinations, 1))
        for i in range(num_combinations):
            channel_combinations[:, np.random.choice(num_channels, num_channels_per_combination[i], False), i] = 1
        channel_combinations = torch.split(channel_combinations, self.num_kernels, 2) # split by dilation
        for i, channel_combination in enumerate(channel_combinations):
            self.register_buffer(f'channel_combinations_{i}', channel_combination) # per dilation

    def _get_quantiles(self, n):
        return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)]).double()

    def _get_bias(self, C, num_features_this_dilation):
        np.random.seed(self.random_state)
        idxs = np.random.choice(C.shape[0], self.num_kernels)
        # print('Values',C.shape,self.num_kernels)
        samples = C[idxs].diagonal().T
        biases = torch.quantile(samples, self._get_quantiles(num_features_this_dilation).to(C.device), dim=1).T
        return biases

MRF = MiniRocketFeatures

# Cell
def get_minirocket_features(o, model, chunksize=1024, use_cuda=None, to_np=True):
    """Function used to split a large dataset into chunks, avoiding OOM error."""
    use = torch.cuda.is_available() if use_cuda is None else use_cuda
    device = torch.device(torch.cuda.current_device()) if use else torch.device('cpu')
    model = model.to(device)
    if isinstance(o, np.ndarray): o = torch.from_numpy(o).to(device)
    _features = []
    for oi in torch.split(o, chunksize):
        _features.append(model(oi))
    features = torch.cat(_features).unsqueeze(-1)
    if to_np: return features.cpu().numpy()
    else: return features

# Cell
class MiniRocketHead(nn.Sequential):
    def __init__(self, c_in, c_out, seq_len=1, bn=True, fc_dropout=0.):
        layers = [nn.Flatten()]
        if bn:
            layers += [nn.BatchNorm1d(c_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        linear = nn.Linear(c_in, c_out)
        nn.init.constant_(linear.weight.data, 0)
        nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        head = nn.Sequential(*layers)
        super().__init__(OrderedDict(
            [('backbone', nn.Sequential()), ('head', head)]))

# Cell
class MiniRocket(nn.Sequential):
    def __init__(self, c_in, c_out, seq_len, num_features=10_000, max_dilations_per_kernel=32, random_state=None, bn=True, fc_dropout=0):

        # Backbone
        backbone =  MiniRocketFeatures(c_in, seq_len, num_features=num_features, max_dilations_per_kernel=max_dilations_per_kernel,
                                       random_state=random_state)
        num_features = backbone.num_features

        # Head
        self.head_nf = num_features
        layers = [nn.Flatten()]
        if bn: layers += [nn.BatchNorm1d(num_features)]
        if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        linear = nn.Linear(num_features, c_out)
        nn.init.constant_(linear.weight.data, 0)
        nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        head = nn.Sequential(*layers)

        super().__init__(OrderedDict([('backbone', backbone), ('head', head)]))

    def fit(self, X, chunksize=None):
        self.backbone.fit(X, chunksize=chunksize)