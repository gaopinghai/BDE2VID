import torch
import numpy as np
from .utils.util import normalize_image
from .model.snn_network import EVSNN_LIF_final, PAEVSNN_LIF_AMPLIF_final


class RecSNN:
    def __init__(self, model_name, pretrain_model):
        self.model_name = model_name
        self.pretrain_model = pretrain_model
        self.device = None
        self.init__()

    def init__(self):
        network_kwargs = {'activation_type': 'lif',
                          'mp_activation_type': 'amp_lif',
                          'spike_connection': 'concat',
                          'num_encoders': 3,
                          'num_resblocks': 1,
                          'v_threshold': 1.0,
                          'v_reset': None,
                          'tau': 2.0
                          }
        net = eval(self.model_name)(kwargs=network_kwargs)
        net.load(self.pretrain_model)
        self.net = net
        self.num_bins = 5
        self.states = None
        if self.device is not None:
            self.to(self.device)

    def __call__(self, input):
        event_tensor = input['events']
        mean, stddev = event_tensor[event_tensor != 0].mean(), event_tensor[event_tensor != 0].std()
        event_tensor[event_tensor != 0] = (event_tensor[event_tensor != 0] - mean) / stddev

        for j in range(self.num_bins):
            event_input = event_tensor[:,j,:,:].unsqueeze(dim=1)
            with torch.no_grad():
                if self.model_name == 'EVSNN_LIF_final':
                    membrane_potential = self.net(event_input, self.states)
                    self.states = membrane_potential
                elif self.model_name == 'PAEVSNN_LIF_AMPLIF_final':
                    membrane_potential, self.states = self.net(event_input, self.states)
        result = membrane_potential
        return {'image': result}

    def eval(self):
        self.net.eval()

    def to(self, device):
        self.device = device
        self.net.to(device)

    def reset_states(self):
        self.init__()
