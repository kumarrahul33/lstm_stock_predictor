import torch
import numpy as np
class StandardScalerLSTM():
    """Standardize data by removing the mean and scaling to
    unit variance.  This object can be used as a transform
    in PyTorch data loaders.

    Args:
        mean (FloatTensor): The mean value for each feature in the data.
        scale (FloatTensor): Per-feature relative scaling.
    """

    def __init__(self, batch_size, sequence_length, device):

        self.mean_ = None # shape = (1, feature_size)
        self.scale_ = None # shape = (1, feature_size)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

    def fit(self, sample):
        """Set the mean and scale values based on the sample data.
            sample: N x feature_size 
                N : batch_size * sequence_length
        """
        self.mean_ = np.mean(sample,axis=0)
        self.scale_ = np.std(sample,axis=0)

        # print("mean",self.mean_.shape)
        # print("scale",self.scale_.shape)

        self.scale_ = torch.from_numpy(self.scale_).float().unsqueeze(0).to(self.device)
        self.mean_ = torch.from_numpy(self.mean_).float().unsqueeze(0).to(self.device)
        # self.min_val =  torch.from_numpy(np.min(sample,axis=0)).float().unsqueeze(0).to(self.device)
        # self.max_val = torch.from_numpy(np.max(sample,axis=0)).float().unsqueeze(0).to(self.device)
        return self

    def __call__(self, sample):
        """
            sample : batch_size x sequence_length x feature_size
        """
        # print("in call")
        # print("sample",sample.shape)
        # print("min_val",self.min_val.shape) 
        # print("max_val",self.max_val.shape)
        # print("scale",self.scale_.shape)
        # print("mean",self.mean_.shape)
        # return (sample - self.min_val) / (self.max_val - self.min_val)
        return (sample - self.mean_) / self.scale_

    def inverse_transform(self, sample):
        """Scale the data back to the original representation
            sample : batch_size x sequence_length x feature_size
        """
        # print("prediction",sample.shape)
        # print("self.scale_",self.scale_.shape) # (feature_size,)
        # print("self.mean_",self.scale_.shape) # (feature_size,)

        # FIXME: A possible bug, when the first element is not the output of the model
        # return sample * self.scale_[0][0] + self.mean_[0][0]
        # do min max scaling
        # print("sample",sample.shape)
        # print("min_val",self.min_val.shape) 
        # print("max_val",self.max_val.shape)
        # return sample * (self.max_val[0][0] - self.min_val[0][0]) + self.min_val[0][0]
        return sample * self.scale_[0][0] + self.mean_[0][0]
