import torch
import torch.nn as nn

from .utils import TorchModelAttributeParser, NNSize
from functools import reduce

## import logging
## logger = logging.getLogger(__name__)


class SpeckleEmbedding(nn.Module):
    """ Embed speckle patterns into a unified vector space. """

    def __init__(self, size_y, size_x, dim_emb = 128):
        super().__init__()

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = False, ),
            nn.BatchNorm2d( num_features = 32 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = False, ),
            nn.BatchNorm2d( num_features = 64 ),
            nn.PReLU(),
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
                       bias         = True),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = True),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.fc(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class SpeckleEmbeddingBackCompatible(nn.Module):
    """ Embed speckle patterns into a unified vector space. """

    def __init__(self, size_y, size_x, dim_emb = 128):
        super().__init__()

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = True, ),
            nn.BatchNorm2d( num_features = 32 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = True, ),
            nn.BatchNorm2d( num_features = 64 ),
            nn.PReLU(),
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
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.feature_size, 
                       out_features = 512, 
                       bias         = True),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = True),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x



