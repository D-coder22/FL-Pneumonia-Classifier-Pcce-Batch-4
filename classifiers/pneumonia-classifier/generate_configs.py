import os

import torch
from torch import nn
import torch.nn.functional as F 

import classifiers.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'IterAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.iter_avg_fusion_handler'
    }
    return fusion


def get_local_training_config():
    local_training_handler = {
        'name': 'LocalTrainingHandler',
        'path': 'ibmfl.party.training.local_training_handler'
    }
    return local_training_handler


def get_hyperparams():
    hyperparams = {
        'gobal': {
            'rounds': 3,
            'termination_accuracy': 0.9,
            'max_timeout': 60
        },
        'local': {
            'trainging': {
                'epochs': 5
            },
            'optimizer': {
                'lr': 0.002
            }
        }
    }


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False):

    SUPPORTED_DATASETS = ['chest-xray']
    if dataset in SUPPORTED_DATASETS:
        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


class Network(nn.Module()):
    def __init__(self):
        super(self, Network).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernal_size = 3, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(32, 32, kernal_size = 3, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(32, 64, kernal_size = 3, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(64, 64, kernal_size = 3, stride = 1, padding = 0)
        self.conv5 = nn.Conv2d(64, 128, kernal_size = 3, stride = 2, padding = 0)
        self.conv6 = nn.Conv2d(128, 128, kernal_size = 3, stride = 2, padding = 0)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4608, 1152)
        self.fc2 = nn.Linear(1152, 288)
        self.fc3 = nn.Linear(288, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.pool(F.relu(self.conv5(x)))

        x = self.pool(F.relu(self.conv6(x)))

        x = x.view(-1, 4608)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc1(x))

        return x

def get_model_config(folder_configs, dataset, is_agg = False, party_id = 0):
    if is_agg:
        return None

    model = Network().cuda()

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    fname = os.path.join(folder_configs, 'complied_torch.h5')
    torch.save(model, fname)

    torch.cuda.empty_cache()


    spec = {
        'model_name': 'pytorch-cnn',
        'model_definition': fname
    }

    model = {
        'name': 'PytorchFLModel',
        'path': 'ibmfl.model.pytorch_fl_model',
        'spec': spec
    }

    return model




