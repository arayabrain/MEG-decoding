import torch.nn as nn
import torch
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, input_features:int,
                 middle_features:list,
                 output_features:int,
                 activation='relu',
                 norm='layer',
                 dropout=0.5,):
        super(MLPDecoder, self).__init__()

        self.input_features = input_features
        self.middle_features = middle_features
        self.output_features = output_features

        if activation == 'relu':
            activation = nn.ReLU
        else:
            raise ValueError(f'activation {activation} not supported')

        if norm == 'batch':
            norm = nn.BatchNorm1d
        elif norm == 'layer':
            norm = nn.LayerNorm
        else:
            raise ValueError(f'norm {norm} not supported')

        self.first_layer = nn.Sequential(
            nn.Linear(self.input_features, self.middle_features[0]),
            norm(self.middle_features[0]),
            activation(),
        )
        hidden_layer_list = []
        for i in range(len(self.middle_features)-1):
            hidden_layer_list.append(nn.Sequential(
                nn.Linear(self.middle_features[i], self.middle_features[i+1]),
                norm(self.middle_features[i+1]),
                activation(),
                nn.Dropout(dropout),
            ))
        self.hidden_layer_list = self.nn.ModuleList(hidden_layer_list)
        self.final_layer = nn.Linear(self.middle_features[-1], self.output_features)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.first_layer(x)
        for layer in self.hidden_layer_list:
            x = layer(x)
        x = self.final_layer(x)
        return x