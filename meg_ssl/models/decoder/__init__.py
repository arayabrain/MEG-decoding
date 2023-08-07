
from .eegnet import EEGNet
from .mlp import MLPDecoder
from .mapping import Mapping
from omegaconf import OmegaConf

def get_decoder(name, parameters:OmegaConf):
    if name == 'eegnet':
        return EEGNet(parameters)
    elif name == 'mlp':
        return MLPDecoder(**parameters)
    elif name == 'mapping':
        return Mapping(**parameters)
    else:
        raise ValueError(f'{name} is not supported')