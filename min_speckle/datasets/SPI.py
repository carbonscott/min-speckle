import random
import numpy as np
import torch

from torch.utils.data import Dataset


class TripletCandidate(Dataset):
    """
    TripletCandidate(dataset_list, num_sample, num_sample_per_label, trans_list)

    This class is only for implementing a specific experiment.

    This class will reutrn a list of `(encode, candidate_list)` so that PyTorch
    DataLoader can pack `label` and `candidate_list` into a batch,
    respectively.  This huge batch can then be divided into a number of
    mini-batch for model training and validation.

    Attributes
    ----------
    label_to_idx_dict : dict
        Dictionary of `label` to `index` mapping, in which `index` is used to
        access data in the `dataset_list`.

    label_to_encode_dict : dict
        Dictionary of `label` to `encode` mapping, where `encode` is an integer
        that encodes a label like `(1, '6Q5U')`.

    encode_to_label_dict : dict
        Dictionary with key-value pair in `label_to_encode_dict` but reversed.

    label_list : list
        List of all values of `label`.

    encode_list : list
        List of all values of `encode`.

    sample_list : list List of all sample that can be returned by the __call__
    function with a supplied index.  Each sample is a tuple of two elements --
    `(encode, candidate_list)`.

    Parameters
    ----------

    dataset_list : list
        List of data points, where a data point is defined as a tuple of three
        elements -- `(image, label, metadata)`.

    num_sample : int
        Number of samples.  See the comment section of `sample_list` for the
        definition of sample.

    num_sample_per_label : int
        Number of samples to be associated with one label.

    trans_list : list
        List of functions that transform an image.

    """

    def __init__(self, dataset_list,
                       num_sample            = 2,
                       num_sample_per_label  = 2,
                       trans_list            = None,
                       mpi_comm              = None):
        self.dataset_list         = dataset_list
        self.num_sample           = num_sample
        self.num_sample_per_label = num_sample_per_label
        self.trans_list           = trans_list
        self.mpi_comm             = mpi_comm

        # Set up mpi...
        if self.mpi_comm is not None:
            self.mpi_size     = self.mpi_comm.Get_size()    # num of processors
            self.mpi_rank     = self.mpi_comm.Get_rank()
            self.mpi_data_tag = 11

        self.label_to_idx_dict = self.build_label_to_idx_dict()

        self.label_to_encode_dict, self.encode_to_label_dict = self.encode_label()

        self.label_list  = []
        self.encode_list = []
        for label, encode in self.label_to_idx_dict.items():
            self.label_list.append(label)
            self.encode_list.append(encode)

        self.sample_list = self.build_sample_list()

        self.dataset_cache_dict = {}


    def encode_label(self):
        label_to_encode_dict = { label  : encode for encode, label in enumerate(self.label_to_idx_dict.keys()) }
        encode_to_label_dict = { encode : label  for encode, label in enumerate(self.label_to_idx_dict.keys()) }

        return label_to_encode_dict, encode_to_label_dict


    def build_label_to_idx_dict(self):
        label_to_idx_dict = {}
        for idx, (img, label, metadata) in enumerate(self.dataset_list):
            if label not in label_to_idx_dict: label_to_idx_dict[label] = []
            label_to_idx_dict[label].append(idx)

        return label_to_idx_dict


    def build_sample_list(self):
        sample_list = []
        for idx in range(self.num_sample):
            label  = random.choice(self.label_list)
            encode = self.label_to_encode_dict[label]
            candidate_list = random.choices(self.label_to_idx_dict[label],
                                               k = self.num_sample_per_label)
            sample_list.append( (encode, candidate_list) )

        return sample_list


    def __len__(self):
        return self.num_sample


    def get_sample(self, idx):
        encode, candidate_list = self.sample_list[idx]

        ## return torch.tensor(encode), torch.tensor(candidate_list)
        return encode, candidate_list


    def get_data(self, idx):
        encode, candidate_list = self.get_sample(idx)

        # Loop through all candidate by index and create a candidate tensor...
        metadata_list = []
        num_candidate = len(candidate_list)
        for i, candidate in enumerate(candidate_list):
            # Fetch original data...
            img, label, metadata = self.dataset_list[candidate]
            img = img[None,]

            # Apply any possible transformation...
            # How to define a custom transform function?
            # Input : img, **kwargs 
            # Output: img_transfromed
            if self.trans_list is not None:
                for trans in self.trans_list:
                    img = trans(img)

            # Normalize the image...
            img  = img - img.mean()
            img /= img.std()

            if i == 0:
                # Preallocate a matrix to hold all data...
                size_y, size_x = img.shape[-2:]
                img_nplist = np.zeros((num_candidate, 1, size_y, size_x), dtype = np.float32)
                #                                     ^
                # Torch Channel ______________________|

            # Keep img in memory...
            img_nplist[i:i+1] = img

            # Save metadata...
            metadata_list.append(metadata)

        return encode, img_nplist, metadata_list


    def __getitem__(self, idx):
        encode, img_nplist, metadata_list =                                \
            self.dataset_cache_dict[idx] if idx in self.dataset_cache_dict \
                                         else                              \
                                         self.get_data(idx)

        return encode, img_nplist, metadata_list


    def mpi_cache_dataset(self, mpi_batch_size = 1):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        # Import chunking method...
        from ..utils import split_list_into_chunk

        # Get the MPI metadata...
        mpi_comm     = self.mpi_comm
        mpi_size     = self.mpi_size
        mpi_rank     = self.mpi_rank
        mpi_data_tag = self.mpi_data_tag

        # If subset is not give, then go through the whole set...
        global_idx_list = range(self.num_sample)

        # Divide all indices into batches and go through them...
        batch_idx_list = split_list_into_chunk(global_idx_list, max_num_chunk = mpi_batch_size)
        for batch_seqi, idx_list in enumerate(batch_idx_list):
            # Split the workload...
            idx_list_in_chunk = split_list_into_chunk(idx_list, max_num_chunk = mpi_size)

            # Process chunk by each worker...
            # No need to sync the dataset_cache_dict across workers
            dataset_cache_dict = {}
            if mpi_rank != 0:
                if mpi_rank < len(idx_list_in_chunk):
                    idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                    dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)

                mpi_comm.send(dataset_cache_dict, dest = 0, tag = mpi_data_tag)

            if mpi_rank == 0:
                print(f'[[[ MPI batch {batch_seqi} ]]]')

                idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)
                self.dataset_cache_dict.update(dataset_cache_dict)

                for i in range(1, mpi_size, 1):
                    dataset_cache_dict = mpi_comm.recv(source = i, tag = mpi_data_tag)
                    self.dataset_cache_dict.update(dataset_cache_dict)

        return None


    def _mpi_cache_data_per_rank(self, idx_list):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        dataset_cache_dict = {}
        for idx in idx_list:
            # Skip those have been recorded...
            if idx in dataset_cache_dict: continue

            print(f"Cacheing data point {idx}...")

            encode, img_nplist, metadata_list = self.get_data(idx)
            dataset_cache_dict[idx] = (encode, img_nplist, metadata_list)

        return dataset_cache_dict
