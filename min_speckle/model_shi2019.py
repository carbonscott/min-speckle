import torch
import torch.nn as nn

from .utils import TorchModelAttributeParser, NNSize
from functools import reduce

## import logging
## logger = logging.getLogger(__name__)


class SpeckleEmbedding(nn.Module):
    """ Embed speckle patterns into a unified vector space. """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        bias           = config.isbias
        dim_emb        = config.dim_emb

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 32 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 64 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 3...
            nn.Conv2d( in_channels  = 64,
                       out_channels = 128,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 128 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        # Define the embedding layer...
        self.fc = nn.Sequential(
            nn.Linear( in_features  = self.feature_size, 
                       out_features = 512, 
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = bias),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.fc(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class Shi2019(nn.Module):
    ''' Entire prediction model (embedding+prediction) in 
        DOI: 10.1107/S2052252519001854

        The MLP is the prediction head of the whole model.
    '''

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        isbias         = config.isbias

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # Motif network 1
            nn.Conv2d( in_channels  = in_channels,
                       out_channels = 5,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = isbias, ),
            nn.BatchNorm2d( num_features = 5 ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d( kernel_size = 2, 
                          stride = 2 ),

            # Motif network 2
            nn.Conv2d( in_channels  = 5,
                       out_channels = 3,
                       kernel_size  = 3,
                       stride       = 1,
                       padding      = 0,
                       bias         = isbias, ),
            nn.BatchNorm2d( num_features = 3 ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d( kernel_size = 2, 
                          stride = 2 ),

            # Motif network 3
            nn.Conv2d( in_channels  = 3,
                       out_channels = 2,
                       kernel_size  = 3,
                       stride       = 1,
                       padding      = 0,
                       bias         = isbias, ),
            nn.BatchNorm2d( num_features = 2 ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d( kernel_size = 2, 
                          stride = 2 ),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        self.fc = nn.Sequential(
            nn.Linear( in_features = self.feature_size,
                       out_features = 2,
                       bias = isbias ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features = 2,
                       out_features = 1,
                       bias = isbias ),
            ## nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.fc(x)

        return x
