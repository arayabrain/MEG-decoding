import torch.nn as nn

class Mapping(nn.Module):
    """taken from dream diffusion(https://github.com/bbaaii/DreamDiffusion/blob/main/code/sc_mbm/mae_for_eeg.py#L441)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_features:int, middle_features:list, output_features:int, activation='none'):
        super().__init__()
        reduce_ch = 1
        self.maxpool = nn.Conv1d(input_features, reduce_ch, 12, stride=1, padding='same')#nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(middle_features, output_features)
        if activation == 'relu':
            self.bn = nn.BatchNorm1d(reduce_ch)
            self.act = nn.ReLU()
        else :
            self.act = None

    def forward(self, x):
        h = x.transpose(1,2) # BxCxT -> BxTxC
        h = self.maxpool(h)
        if self.act is not None:
            h = self.bn(h)
            h = self.act(h)
        h = h.squeeze(1)
        h = self.fc(h)
        # import pdb; pdb.set_trace()
        return h
