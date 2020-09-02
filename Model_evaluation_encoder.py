#!/usr/bin/env python3
import numpy as np
import h5py
from Model_define_pytorch import AutoEncoder, DatasetFolder
import torch
import os


# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 64
num_workers = 4
# Data parameters
img_height = 16
img_width = 32
img_channels = 2
feedback_bits = 128
model_ID = 'CsiNet'  # Model Number

# load test data
mat = h5py.File('./data/H_test.mat')
data = np.transpose(mat['H_test'] )
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])

# load model
model = AutoEncoder(feedback_bits).cuda()
model_encoder = model.encoder
model_path = './Modelsave/encoder.pth.tar'
model_encoder.load_state_dict(torch.load(model_path)['state_dict'])
print("weight loaded")

#dataLoader for test
test_dataset = DatasetFolder(data)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# test
model_encoder.eval()
encode_feature = []
with torch.no_grad():
    for i, input in enumerate(test_loader):
        # convert numpy to Tensor
        input = input.cuda()
        output = model_encoder(input)
        output = output.cpu().numpy()
        if i == 0:
            encode_feature = output
        else:
            encode_feature = np.concatenate((encode_feature, output), axis=0)
print("feedbackbits length is ", np.shape(encode_feature)[-1])
np.save('./Modelsave/encoder_output.npy', encode_feature)
