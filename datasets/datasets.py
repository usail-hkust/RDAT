# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.
from torch_geometric.utils import dense_to_sparse
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
from torchvision.utils import save_image
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import  load_metr_la_data,load_METR_LA_data,load_hk_data,load_PEMS_BAY_data, \
    get_normalized_adj, generate_dataset, generate_semi_dataset, load_PEMS_D4_data

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
random_seed = 0
train_sampler_type = 'default'

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class METRLA(Dataset):
    def __init__(self, mode = 'train',split_train = 0.6, split_val = 0.8, num_timesteps_input = 12, num_timesteps_output = 3):

        A, X, means, stds = load_METR_LA_data()
        A_wave = get_normalized_adj(A)

        max_speed = np.max(X)

        X = X /  max_speed
        self.max_speed = max_speed

        split_line1 = int(X.shape[2] * split_train)
        split_line2 = int(X.shape[2] * split_val)

        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]
        self.A_wave = torch.from_numpy(A_wave)
        self.A = torch.from_numpy(A)

        self.means = means
        self.stds = stds
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices#.numpy()
        values = values#.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        if mode == 'train':
            inputs, targets = generate_dataset(train_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'val':
            inputs, targets = generate_dataset(val_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'test':
            inputs, targets = generate_dataset(test_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        else:
            raise NameError('Invalid mode name: {}'.format(mode))

    def __getitem__(self, index):

        input = self.inputs[index]
        target = self.targets[index]
        return input,target

    def __len__(self):
        return len(self.inputs)



class SemiMETRLA(Dataset):
    def __init__(self, mode = 'train',split_train = 0.6, split_val = 0.8, num_timesteps = 12):

        A, X, means, stds = load_metr_la_data()
        A_wave = get_normalized_adj(A)

        max_speed = np.max(X)

        X = X /  max_speed
        self.max_speed = max_speed

        split_line1 = int(X.shape[2] * split_train)
        split_line2 = int(X.shape[2] * split_val)

        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]
        self.A_wave = torch.from_numpy(A_wave)
        self.A = torch.from_numpy(A)

        self.means = means
        self.stds = stds
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices#.numpy()
        values = values#.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        if mode == 'train':
            current_inputs, current_targets, future_targets = generate_semi_dataset(train_original_data, num_timesteps)
            self.current_inputs = current_inputs
            self.current_targets = current_targets
            self.future_targets = future_targets

        elif mode == 'val':
            current_inputs, current_targets, future_targets = generate_semi_dataset(val_original_data, num_timesteps)
            self.current_inputs = current_inputs
            self.current_targets = current_targets
            self.future_targets = future_targets
        elif mode == 'test':
            current_inputs, current_targets, future_targets = generate_semi_dataset(test_original_data, num_timesteps)
            self.current_inputs = current_inputs
            self.current_targets = current_targets
            self.future_targets = future_targets
        else:
            raise NameError('Invalid mode name: {}'.format(mode))

    def __getitem__(self, index):

        current_inputs = self.current_inputs[index]
        current_targets = self.current_targets[index]
        future_targets = self.future_targets[index]
        return current_inputs, current_targets, future_targets

    def __len__(self):
        return len(self.current_inputs)

class HKSPEED(Dataset):
    def __init__(self, mode = 'train',split_train = 0.6, split_val = 0.8, num_timesteps_input = 12, num_timesteps_output = 3):
        """
        Takes node features for the graph and divides them into multiple samples
        along the time-axis by sliding a window of size (num_timesteps_input+
        num_timesteps_output) across it in steps of 1.
        :param X: Node features of shape (num_vertices, num_features,
        num_timesteps)
        :return:
            - Node features divided into multiple samples. Shape is
              (num_samples, num_vertices, num_features, num_timesteps_input).
            - Node targets for the samples. Shape is
              (num_samples, num_vertices, num_features, num_timesteps_output).
        """
        A, X = load_hk_data()
        X = X.reshape((179, 1, 1996))
       # print('X.shape', X.shape)
        A_wave = get_normalized_adj(A)

        max_speed = np.max(X)
        X = X /  max_speed

        split_line1 = int(X.shape[2] * split_train)
        split_line2 = int(X.shape[2] * split_val)

        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]
        self.A_wave = torch.from_numpy(A_wave).to(torch.float32)
        self.A = torch.from_numpy(A).to(torch.float32)

        self.max_speed = max_speed

        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices#.numpy()
        values = values#.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        if mode == 'train':
            inputs, targets = generate_dataset(train_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'val':
            inputs, targets = generate_dataset(val_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'test':
            inputs, targets = generate_dataset(test_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        else:
            raise NameError('Invalid mode name: {}'.format(mode))

    def __getitem__(self, index):

        input = self.inputs[index]
        target = self.targets[index]
        return input,target

    def __len__(self):
        return len(self.inputs)



class PeMS(Dataset):
    def __init__(self, mode = 'train',split_train = 0.6, split_val = 0.8, num_timesteps_input = 12, num_timesteps_output = 3):

        A, X, means, stds = load_PEMS_BAY_data()
        A_wave = get_normalized_adj(A)

        max_speed = np.max(X)

        X = X /  max_speed
        self.max_speed = max_speed

        split_line1 = int(X.shape[2] * split_train)
        split_line2 = int(X.shape[2] * split_val)

        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]
        self.A_wave = torch.from_numpy(A_wave)
        self.A = torch.from_numpy(A)

        self.means = means
        self.stds = stds
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices#.numpy()
        values = values#.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        if mode == 'train':
            inputs, targets = generate_dataset(train_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'val':
            inputs, targets = generate_dataset(val_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'test':
            inputs, targets = generate_dataset(test_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        else:
            raise NameError('Invalid mode name: {}'.format(mode))

    def __getitem__(self, index):

        input = self.inputs[index]
        target = self.targets[index]
        return input,target

    def __len__(self):
        return len(self.inputs)


class PeMSD4(Dataset):
    def __init__(self, mode = 'train',split_train = 0.6, split_val = 0.8, num_timesteps_input = 12, num_timesteps_output = 3):

        A, X, means, stds = load_PEMS_D4_data()
        A_wave = get_normalized_adj(A)

        max_flow = np.max(X)

        X = X /  max_flow
        self.max_speed = max_flow

        split_line1 = int(X.shape[2] * split_train)
        split_line2 = int(X.shape[2] * split_val)

        train_original_data = X[:, :, :split_line1]

        val_original_data = X[:, :, split_line1:split_line2]

        test_original_data = X[:, :, split_line2:]

        self.A_wave = torch.from_numpy(A_wave)
        self.A = torch.from_numpy(A)

        self.means = means
        self.stds = stds
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices#.numpy()
        values = values#.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        if mode == 'train':
            inputs, targets = generate_dataset(train_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'val':
            inputs, targets = generate_dataset(val_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        elif mode == 'test':
            inputs, targets = generate_dataset(test_original_data, num_timesteps_input, num_timesteps_output)
            self.inputs = inputs
            self.targets = targets
        else:
            raise NameError('Invalid mode name: {}'.format(mode))

    def __getitem__(self, index):

        input = self.inputs[index]
        target = self.targets[index]
        return input,target

    def __len__(self):
        return len(self.inputs)






class SemiPeMS(Dataset):
    def __init__(self, mode = 'train',split_train = 0.6, split_val = 0.8, num_timesteps = 12):

        A, X, means, stds = load_PEMS_BAY_data()
        A_wave = get_normalized_adj(A)

        max_speed = np.max(X)

        X = X /  max_speed
        self.max_speed = max_speed

        split_line1 = int(X.shape[2] * split_train)
        split_line2 = int(X.shape[2] * split_val)

        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]
        self.A_wave = torch.from_numpy(A_wave)
        self.A = torch.from_numpy(A)

        self.means = means
        self.stds = stds
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices#.numpy()
        values = values#.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        if mode == 'train':
            current_inputs, current_targets, future_targets = generate_semi_dataset(train_original_data, num_timesteps)
            self.current_inputs = current_inputs
            self.current_targets = current_targets
            self.future_targets = future_targets

        elif mode == 'val':
            current_inputs, current_targets, future_targets = generate_semi_dataset(val_original_data, num_timesteps)
            self.current_inputs = current_inputs
            self.current_targets = current_targets
            self.future_targets = future_targets
        elif mode == 'test':
            current_inputs, current_targets, future_targets = generate_semi_dataset(test_original_data, num_timesteps)
            self.current_inputs = current_inputs
            self.current_targets = current_targets
            self.future_targets = future_targets
        else:
            raise NameError('Invalid mode name: {}'.format(mode))

    def __getitem__(self, index):

        current_inputs = self.current_inputs[index]
        current_targets = self.current_targets[index]
        future_targets = self.future_targets[index]
        return current_inputs, current_targets, future_targets

    def __len__(self):
        return len(self.current_inputs)




if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
