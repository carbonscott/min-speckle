import os
import random

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


class SemiHardSelector:

    def __init__(self, model = None, alpha = 0.02):
        self.model = model
        self.alpha = alpha


    def _embed_candidates(self, batch_candidate_list):
        # Get the shape of the batch_candidate_list...
        num_sample_in_mini_batch, num_candidate, size_c, size_y, size_x = batch_candidate_list.shape

        # Convert all input image into embeddings...
        with torch.no_grad():
            ###################################################################
            # These steps are here to facilitate the use of our PyTorch Model.
            ###################################################################
            # Compress sample dim and candidate dim into one example dim...
            batch_example_list = batch_candidate_list.view(-1, size_c, size_y, size_x)

            # Compute embeddings...
            batch_emb_list = self.model(batch_example_list)

            # Reshape the first two dimension back to the original first two dimension...
            # Original dim: [num_sample_in_mini_batch, num_candidate, ...]
            batch_emb_list = batch_emb_list.view(num_sample_in_mini_batch, num_candidate, -1)

        return batch_emb_list


    def __call__(self, batch_label_encode, batch_candidate_list, encode_to_label_dict, batch_metadata_list = None, logs_triplets = False):
        """
        Return a list of entities of three elements -- `(a, p, n)`, which
        stand for anchor, positive and negative respectively.

        Parameters
        ----------
        batch_label_encode : tensor
            List of encoded labels packaged in Torch Tensor.
            - 0 : (1, '6Q5U')
              ^   ^^^^^^^^^^^
              |        :.... label
              |
              |_____________ encode
            Expected shape: [10]
            - num of sample in a mini-batch

        batch_candidate_list : tensor
            List of candidate images packaged in Torch Tensor.
            Expected shape: [ 10, 20, 1, 48, 48 ]
            - num of sample/encode in a mini-batch
            - num of candidates
            - num of torch channel
            - size_y
            - size_x

        Returns
        -------
        triplet_list : tensor
            List of entities of three elements -- `(a, p, n)`, in which each
            elemnt is an image of shape `(1, size_y, size_x)`.
            Expected shape: [20, 3, 2]
            - num of sample in a mini-batch
            - num of examples in a triplet (constant) = 3
            - (4, 12)
               ^  ^^
               |   :........ idx to find the example in the candidate list
               |
               |____________ encode

        """
        # Get the batch size...
        batch_size = len(batch_label_encode)

        # Get the shape of the batch_candidate_list...
        num_sample_in_mini_batch, num_candidate, size_c, size_y, size_x = batch_candidate_list.shape

        # Convert all input image into embeddings...
        with torch.no_grad():
            ###################################################################
            # These steps are here to facilitate the use of our PyTorch Model.
            ###################################################################
            # Compress sample dim and candidate dim into one example dim...
            batch_example_list = batch_candidate_list.view(-1, size_c, size_y, size_x)

            # Compute embeddings...
            batch_emb_list = self.model(batch_example_list)

            # Reshape the first two dimension back to the original first two dimension...
            # Original dim: [num_sample_in_mini_batch, num_candidate, ...]
            batch_emb_list = batch_emb_list.view(num_sample_in_mini_batch, num_candidate, -1)

        # Build a lookup table to locate negative examples...
        encode_to_seqi_dict = {}
        for idx_encode, encode in enumerate(batch_label_encode):
            encode = encode.item()
            if encode not in encode_to_seqi_dict: encode_to_seqi_dict[encode] = []
            encode_to_seqi_dict[encode].append(idx_encode)

        # Go through each item in the mini-batch and find semi-hard triplets...
        triplet_list = []
        dist_list    = []
        for idx_encode, encode in enumerate(batch_label_encode):
            encode = encode.item()

            # Randomly choose an anchor and a positive from a candidate pool...
            idx_a, idx_p = random.sample(range(num_candidate), k = 2)
            emb_a = batch_emb_list[idx_encode][idx_a]
            emb_p = batch_emb_list[idx_encode][idx_p]

            # Calculate emb distance between a and p...
            # emb distance is defined as the squared L2
            diff_emb = emb_a - emb_p
            dist_p = torch.sum(diff_emb * diff_emb).item()

            # Fetch negative sample candidates...
            idx_encode_n_list = []
            for k_encode, v_idx_encode_list in encode_to_seqi_dict.items():
                # Fetch the real label...
                label   = encode_to_label_dict[encode]
                k_label = encode_to_label_dict[k_encode]

                # Skip those have the same hit type...
                hit_type   = label  [1]
                k_hit_type = k_label[1]
                if hit_type == k_hit_type: continue

                idx_encode_n_list += v_idx_encode_list

            idx_encode_n_list = torch.tensor(idx_encode_n_list)

            # Collect all negative embeddings...
            emb_n_list = batch_emb_list[idx_encode_n_list]

            # Calculate emb distance between a and n...
            diff_emb_list = emb_a[None, :] - emb_n_list
            dist_n_list = torch.sum(diff_emb_list * diff_emb_list, dim = -1)

            # Create a logic expression to locate semi hard...
            cond_semihard = (dist_p < dist_n_list) * (dist_n_list < dist_p + self.alpha)

            # If semi hard exists???
            if torch.any(cond_semihard):
                # Sample a semi hard example but represented using seqence id in cond_semihard_numpy...
                pos_semihard = torch.nonzero(cond_semihard)
                seqi_encode, seqi_candidate = random.choice(pos_semihard)

                # Locate the semi hard using idx_encode and idx_candidate...
                idx_encode_n = idx_encode_n_list[seqi_encode].item()
                idx_n        = seqi_candidate.item()

                # Record dist...
                dist_n = dist_n_list[seqi_encode][idx_n].item()

            # Otherwise, randomly select one negative example...
            else:
                seqi_encode  = random.choice(range(len(idx_encode_n_list)))
                idx_encode_n = idx_encode_n_list[seqi_encode].item()
                idx_n        = random.choice(range(dist_n_list.shape[-1]))
                dist_n       = dist_n_list[seqi_encode][idx_n].item()

            # Track variables for output...
            triplet_list.append(((idx_encode, idx_a), (idx_encode, idx_p), (idx_encode_n, idx_n)))
            dist_list.append((dist_p, dist_n))

        if logs_triplets:
            # Logging all cases...
            for idx, triplet in enumerate(triplet_list):
                (idx_encode, idx_a), (idx_encode, idx_p), (idx_encode_n, idx_n) = triplet

                metadata_a = batch_metadata_list[idx_encode  ][idx_a]
                metadata_p = batch_metadata_list[idx_encode  ][idx_p]
                metadata_n = batch_metadata_list[idx_encode_n][idx_n]

                annotate_semihard = ''
                dist_p, dist_n = dist_list[idx]
                diff_pn = dist_n - dist_p
                if 0 < diff_pn and diff_pn < self.alpha: annotate_semihard = f'semi-hard {diff_pn:e}'

                logger.info(f"DATA - {metadata_a:12s}, {metadata_p:12s}, {metadata_n:12s}; {annotate_semihard}")

        # Get the shape of the batch_candidate_list...
        num_sample_in_mini_batch, num_candidate, size_c, size_y, size_x = batch_candidate_list.shape

        batch_candidate_flat_list = batch_candidate_list.view(num_sample_in_mini_batch * num_candidate, size_c, size_y, size_x)
        batch_a = batch_candidate_flat_list[ [ idx_encode * num_candidate + idx_a for (idx_encode, idx_a), _, _ in triplet_list ] ] 
        batch_p = batch_candidate_flat_list[ [ idx_encode * num_candidate + idx_p for _, (idx_encode, idx_p), _ in triplet_list ] ] 
        batch_n = batch_candidate_flat_list[ [ idx_encode * num_candidate + idx_n for _, _, (idx_encode, idx_n) in triplet_list ] ] 

        return batch_a, batch_p, batch_n
