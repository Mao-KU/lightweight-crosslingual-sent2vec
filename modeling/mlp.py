import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, idim=300, odim=2, nhid=None,
                 dropout=0.0, use_gpu=False, activation='TANH'):
        super(MLP, self).__init__()
        self.use_gpu = use_gpu
        modules = []

        modules = []
        logger.info(' - mlp {:d}'.format(idim))
        if len(nhid) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = idim
            for nh in nhid:
                if nh > 0:
                    modules.append(nn.Linear(nprev, nh))
                    nprev = nh
                    if activation == 'TANH':
                        modules.append(nn.Tanh())
                        logger.info('-{:d}t'.format(nh))
                    elif activation == 'RELU':
                        modules.append(nn.ReLU())
                        logger.info('-{:d}r'.format(nh))
                    else:
                        raise Exception('Unrecognized activation {activation}')
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nprev, odim))
            logger.info('-{:d}, dropout={:.1f}'.format(odim, dropout))
        else:
            modules.append(nn.Linear(idim, odim))
            logger.info(' - mlp %d-%d'.format(idim, odim))
        self.mlp = nn.Sequential(*modules)
        # Softmax is included in CrossEntropyLoss !
        if self.use_gpu:
            self.mlp = self.mlp.cuda()

    def forward(self, x):
        return self.mlp(x)

    def TestCorpus(self, dset, name=' Dev', nlbl=4, s=''):
        correct = 0
        total = 0
        self.mlp.train(mode=False)
        corr = np.zeros(nlbl, dtype=np.int32)
        for data in dset:
            X, Y = data
            X = X.float()
            Y = Y.long()
            if self.use_gpu:
                X = X.cuda()
                Y = Y.cuda()
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).int().sum()
            for i in range(nlbl):
                corr[i] += (predicted == i).int().sum()
        
        accuracy = 100.0 * correct.float() / total
        s += ' | {:4s}: {:5.2f}%'.format(name, accuracy)
        s += ' | classes:'
        for i in range(nlbl):
            s += ' {:5.2f}'.format(100.0 * corr[i] / total)
        return accuracy, s

