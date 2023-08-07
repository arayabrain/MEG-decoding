import torch.nn as nn

class Mapping(nn.Module):
    """taken from dream diffusion(https://github.com/bbaaii/DreamDiffusion/blob/main/code/sc_mbm/mae_for_eeg.py#L441)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_features:int, middle_features:list, output_features:int):
        super().__init__()
        self.maxpool = nn.Conv1d(input_features, 1, 1, stride=1)#nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(middle_features, output_features)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x
