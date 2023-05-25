import os
import random

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


class OfflineSemiHardSelector:

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




class OnlineSemiHardSelector:

    def __init__(self, model = None, alpha = 0.02):
        self.model = model
        self.alpha = alpha


    def __call__(self, batch_imgs, batch_labels, batch_metadata, logs_triplets = True, **kwargs):
        ''' Only supply batch_size of image triplet for training.  This is in
            contrast to the all positive method.  Each image in the batch has
            the chance of playing an anchor.  Negative mining is applied.  
        '''
        # Retrieve the batch size...
        batch_size = batch_imgs.shape[0]

        # Encode all batched images without autograd tracking...
        with torch.no_grad():
            batch_embs = self.model(batch_imgs)

        # Convert batch labels to dictionary for fast lookup...
        batch_label_dict = {}
        batch_label_list = batch_labels.cpu().numpy()
        for i, v in enumerate(batch_label_list):
            if not v in batch_label_dict: batch_label_dict[v] = [i]
            else                        : batch_label_dict[v].append(i)

        # ___/ NEGATIVE MINIG \___
        # Go through each image in the batch and form triplets...
        # Prepare for logging
        triplets = []
        dist_log = []
        for batch_idx_achr, img in enumerate(batch_imgs):
            # Get the label of the image...
            batch_label_achr = batch_label_list[batch_idx_achr]

            # Create a bucket of positive cases...
            batch_idx_pos_list = batch_label_dict[batch_label_achr]

            # Select a positive case from positive bucket...
            batch_idx_pos = random.choice(batch_idx_pos_list)

            # Find positive embedding squared distances..
            emb_achr = batch_embs[batch_idx_achr]
            emb_pos  = batch_embs[batch_idx_pos]
            delta_emb_pos = emb_achr - emb_pos
            dist_pos = torch.sum(delta_emb_pos * delta_emb_pos)

            # Create a bucket of negative cases...
            idx_neg_list = []
            for batch_label, idx_list in batch_label_dict.items():
                if batch_label == batch_label_achr: continue
                idx_neg_list += idx_list
            idx_neg_list = torch.tensor(idx_neg_list)

            # Collect all negative embeddings...
            emb_neg_list = batch_embs[idx_neg_list]

            # Find negative embedding squared distances...
            delta_emb_neg_list = emb_achr[None, :] - emb_neg_list
            dist_neg_list = torch.sum( delta_emb_neg_list * delta_emb_neg_list, dim = -1 )

            # Find negative squared distance satisfying dist_neg > dist_pos (semi hard)...
            # logical_and is only supported when pytorch version >= 1.5
            cond_semihard = (dist_pos < dist_neg_list) * (dist_neg_list < dist_pos + self.alpha)

            # If semi hard exists???
            if torch.any(cond_semihard):
                # Select one random example that is semi hard...
                size_semihard = torch.sum(cond_semihard)
                idx_random_semihard = random.choice(range(size_semihard))

                # Fetch the batch index of the example and its distance w.r.t the anchor...
                batch_idx_neg = idx_neg_list [cond_semihard][idx_random_semihard]
                dist_neg      = dist_neg_list[cond_semihard][idx_random_semihard]

            # Otherwise, randomly select one negative example???
            else:
                idx_reduced   = random.choice(range(len(idx_neg_list)))
                batch_idx_neg = idx_neg_list[idx_reduced]
                dist_neg      = dist_neg_list[idx_reduced]

            # Track triplet...
            triplets.append((batch_idx_achr, batch_idx_pos, batch_idx_neg.tolist()))
            dist_log.append((dist_pos, dist_neg))

        if logs_triplets:
            # Logging all cases...
            for idx, triplet in enumerate(triplets):
                batch_idx_achr, batch_idx_pos, batch_idx_neg = triplet
                metadata_achr = batch_metadata[batch_idx_achr]
                metadata_pos  = batch_metadata[batch_idx_pos]
                metadata_neg  = batch_metadata[batch_idx_neg]
                dist_pos   = dist_log[idx][0]
                dist_neg   = dist_log[idx][1]
                logger.info(f"DATA - {metadata_achr} {metadata_pos} {metadata_neg} {dist_pos:12.6f} {dist_neg:12.6f}")

        batch_a = batch_imgs[ [ triplet[0] for triplet in triplets ] ]
        batch_p = batch_imgs[ [ triplet[1] for triplet in triplets ] ]
        batch_n = batch_imgs[ [ triplet[2] for triplet in triplets ] ]

        return batch_a, batch_p, batch_n
