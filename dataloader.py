import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset
import more_itertools as miter

class TAMKOT_DataLoader:
    def __init__(self, config, data):
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate
        self.metric = config["metric"]

        self.seed = config['seed']

        self.validation_split = config["validation_split"]
        self.mode = config["mode"]

        self.min_seq_len = config["min_seq_len"] if "min_seq_len" in config else None
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else None
        self.stride = config["max_seq_len"] if "max_seq_len" in config else None

        self.init_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
        }

        self.generate_train_test_data(data)

        # define the data format for different perfromance data
        if self.metric == 'rmse':
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).float(),
                                            torch.Tensor(self.train_data_l).long(),
                                            torch.Tensor(self.train_data_d).long(),
                                            torch.Tensor(self.train_target_answers).float(),
                                            torch.Tensor(self.train_target_masks).bool(),
                                            torch.Tensor(self.train_target_masks_l).bool())

            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).float(),
                                           torch.Tensor(self.test_data_l).long(), torch.Tensor(self.test_data_d).long(),
                                           torch.Tensor(self.test_target_answers).float(),
                                           torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_target_masks_l).bool())

        else:
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).long(),
                                            torch.Tensor(self.train_data_l).long(),
                                            torch.Tensor(self.train_data_d).long(),
                                            torch.Tensor(self.train_target_answers).long(),
                                            torch.Tensor(self.train_target_masks).bool(),
                                            torch.Tensor(self.train_target_masks_l).bool())


            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).long(),
                                            torch.Tensor(self.test_data_l).long(), torch.Tensor(self.test_data_d).long(),
                                           torch.Tensor(self.test_target_answers).long(),
                                            torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_target_masks_l).bool())

        # create batched data
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size)

        self.test_loader = DataLoader(self.test_data, batch_size=self.test_data_a.shape[0])


    def generate_train_test_data(self, data):
        """
        read or process data for training and testing
        """

        q_records = data["traindata"]["q_data"]
        a_records = data["traindata"]["a_data"]
        l_records = data["traindata"]["l_data"]
        d_records = data["traindata"]["d_data"]

        self.train_data_q, self.train_data_a, self.train_data_l, self.train_data_d = self.TAMKOT_ExtDataset(q_records,
                                                                                                       a_records,
                                                                                                       l_records,
                                                                                                       d_records,
                                                                                                       self.max_seq_len,
                                                                                                       stride=self.stride)

        self.train_target_answers = np.copy(self.train_data_a)
        self.train_target_masks = (self.train_data_q != 0)
        self.train_target_masks_l = (self.train_data_l != 0)

        if self.mode == "train":
            # n_samples = len(self.train_data_q)
            # split the train data into train and val sets based on the self.n_samples

            self.train_data_q, self.test_data_q, self.train_data_a, self.test_data_a, self.train_data_l, \
            self.test_data_l, self.train_data_d, \
            self.test_data_d, self.train_target_answers, self.test_target_answers, \
            self.train_target_masks, self.test_target_masks, self.train_target_masks_l, self.test_target_masks_l = train_test_split(
                self.train_data_q, self.train_data_a, self.train_data_l, self.train_data_d, self.train_target_answers,
                self.train_target_masks, self.train_target_masks_l)


        elif self.mode == 'test':
            q_records = data["testdata"]["q_data"]
            a_records = data["testdata"]["a_data"]
            l_records = data["testdata"]["l_data"]
            d_records = data["testdata"]["d_data"]


            self.test_data_q, self.test_data_a, self.test_data_l, self.test_data_d = self.TAMKOT_ExtDataset(q_records,
                                                                                                          a_records,
                                                                                                          l_records,
                                                                                                          d_records,
                                                                                                          self.max_seq_len,
                                                                                                          stride=self.stride)

            self.test_target_answers = np.copy(self.test_data_a)
            self.test_target_masks = (self.test_data_q != 0)
            self.test_target_masks_l = (self.test_data_l != 0)



    def TAMKOT_ExtDataset(self, q_records, a_records, l_records, d_records,
                                           max_seq_len,
                                           stride):
        """
        transform the data into feasible input of model,
        truncate the seq. if it is too long and
        pad the seq. with 0s if it is too short
        """

        q_data = []
        a_data = []
        l_data = []
        d_data = []
        for index in range(len(q_records)):
            q_list = q_records[index]
            a_list = a_records[index]
            l_list = l_records[index]
            d_list = d_records[index]

            # if seq length is less than max_seq_len, the windowed will pad it with fillvalue
            # the reason for inserting two padding attempts with 0 and setting stride = stride - 2 is to make sure the
            # first activity of each sequence is included in training and testing, and also for each sequence's first
            # activity there is an activity zero to be t - 1 attempt.

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            l_list.insert(0, 0)
            d_list.insert(0, 0)
            q_patches = list(miter.windowed(q_list, max_seq_len, fillvalue=0, step=stride-2))
            a_patches = list(miter.windowed(a_list, max_seq_len, fillvalue=2, step=stride-2))
            l_patches = list(miter.windowed(l_list, max_seq_len, fillvalue=0, step=stride-2))
            d_patches = list(miter.windowed(d_list, max_seq_len, fillvalue=0, step=stride-2))

            q_data.extend(q_patches)
            a_data.extend(a_patches)
            l_data.extend(l_patches)
            d_data.extend(d_patches)

        return np.array(q_data), np.array(a_data), np.array(l_data), np.array(d_data)



