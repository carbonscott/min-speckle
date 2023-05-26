#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class TripletLoss(nn.Module):
    """ 
    A framework that calculates the loss of a triplet embedding.

    This framework doesn't have knowledge of what the underlying embedding
    model is.
    """

    def __init__(self, alpha, ):
        super().__init__()

        self.alpha = alpha


    def forward(self, emb_a, emb_p, emb_n):
        """
        Returns the triplet loss.

        Parameters
        ----------
        emb_a : torch.tensor, shape (B, E)
        emb_p : torch.tensor, shape (B, E)
        emb_n : torch.tensor, shape (B, E)
            - a for anchor, p for positive, n for negative.
            - B is the batch dimension.
            - E is the embedding dimension.


        Returns
        -------
        loss : torch.tensor, shape (B,)

        """
        embdiff_pa = emb_p - emb_a
        dist_pa = torch.sum(embdiff_pa * embdiff_pa, dim = -1)

        embdiff_na = emb_n - emb_a
        dist_na = torch.sum(embdiff_na * embdiff_na, dim = -1)

        loss = torch.relu(dist_pa - dist_na + self.alpha)

        return loss
