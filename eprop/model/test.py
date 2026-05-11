# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:43:48 2023

@author: DELL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import argparse
import train
import setup

def main():
    parser = argparse.ArgumentParser(description='Spiking RNN Pytorch training')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA training and run training on CPU')
    # parser.add_argument('--cpu', action='store_true', default=True, help='Disable CUDA training and run training on CPU')
    # parser.add_argument('--dataset', type=str, choices = ['cue_accumulation'], default='cue_accumulation', help='Choice of the dataset')
    parser.add_argument('--dataset', type=str, choices = ['cue_accumulation, speech'], default='speech', help='Choice of the dataset')
    parser.add_argument('--shuffle', type=bool, default=True, help='Enables shuffling sample order in datasets after each epoch')
    parser.add_argument('--trials', type=int, default=1, help='Nomber of trial experiments to do (i.e. repetitions with different initializations)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, choices = ['SGD', 'NAG', 'Adam', 'RMSProp'], default='Adam', help='Choice of the optimizer')
    parser.add_argument('--loss', type=str, choices = ['MSE', 'BCE', 'CE'], default='BCE', help='Choice of the loss function (only for performance monitoring purposes, does not influence learning)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr-layer-norm', type=float, nargs='+', default=(0.05,0.05,1.0), help='Per-layer modulation factor of the learning rate')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training (limited by the available GPU memory)')
    parser.add_argument('--test-batch-size', type=int, default=5, help='Batch size for testing (limited by the available GPU memory)')
    parser.add_argument('--train-len', type=int, default=200, help='Number of training set samples')
    parser.add_argument('--test-len', type=int, default=500, help='Number of test set samples')
    parser.add_argument('--visualize', type=bool, default=False, help='Enable network visualization')
    parser.add_argument('--visualize-light', type=bool, default=True, help='Enable light mode in network visualization, plots traces only for a single neuron')
    # Network model parameters
    parser.add_argument('--n-rec', type=int, default=100, help='Number of recurrent units')
    parser.add_argument('--model', type=str, choices = ['LIF'], default='LIF', help='Neuron model in the recurrent layer. Support for the ALIF neuron model has been removed.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Firing threshold in the recurrent layer')
    parser.add_argument('--tau-mem', type=float, default=2000e-3, help='Membrane potential leakage time constant in the recurrent layer (in seconds)')
    parser.add_argument('--tau-out', type=float, default=20e-3, help='Membrane potential leakage time constant in the output layer (in seconds)')
    parser.add_argument('--bias-out', type=float, default=0.0, help='Bias of the output layer')
    parser.add_argument('--gamma', type=float, default=0.3, help='Surrogate derivative magnitude parameter')
    parser.add_argument('--w-init-gain', type=float, nargs='+', default=(0.5,0.1,0.5), help='Gain parameter for the He Normal initialization of the input, recurrent and output layer weights')
    
    args = parser.parse_args()

    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)    
    (args_out, model, device_out, train_loader_out, test_loader, optimizer, loss) = train.train(args, device, train_loader, traintest_loader, test_loader)
    
    return (args_out, model, device_out, train_loader_out, optimizer, loss)
    # return (device, train_loader, traintest_loader, test_loader)

if __name__ == '__main__':
    (args, model, device, train_loader, test_loader, optimizer, loss) = main()
   # (device, train_loader, traintest_loader, test_loader) = main()
    model.eval()    # This implementation does not rely on autograd, learning update rules are hardcoded
    batch = args.test_batch_size
    length = args.full_test_len
    with torch.no_grad():   # Same here, we make sure autograd is disabled
       
        neuron_output = []
        # For each batch
        for batch_idx, (data, label) in enumerate(test_loader):  #loader是一个数据加载器，可以对数据集进行迭代。在每次迭代中，它会返回一个批次的数据和标签。

            data, label = data.to(device), label.to(device)     #将获取到的数据data和标签label移动到指定的设备上（CPU/GPU）
            if args.classif:    # Do a one-hot encoding for classification 如果args.classif为True，表示进行分类任务，则将标签进行one-hot编码。
                targets = torch.zeros(label.shape, device=device).unsqueeze(-1).expand(-1,-1,args.n_classes).scatter(2, label.unsqueeze(-1), 1.0).permute(1,0,2)
                # 首先，通过torch.zeros(label.shape, device=device)创建一个与标签label相同形状的全零张量，存储在targets中。这个张量的设备与标签所在的设备一致。
                # 接着，使用unsqueeze(-1)在张量的最后一个维度上添加一个维度，将维度从(batch_size, seq_len)变为(batch_size, seq_len, 1)
                # 然后，使用expand(-1,-1,args.n_classes)将张量在第三个维度上进行扩展，将维度从(batch_size, seq_len, 1)变为(batch_size, seq_len, args.n_classes)，其中args.n_classes表示分类任务的类别数。
                # 接下来，使用scatter(2, label.unsqueeze(-1), 1.0)将标签值在第三个维度上进行散射，将对应的位置的值置为1.0，实现了one-hot编码。
                # 最后，使用permute(1,0,2)对维度进行置换，将维度从(batch_size, seq_len, args.n_classes)变为(seq_len, batch_size, args.n_classes)
            else:
                targets = label.permute(1,0,2) #如果args.classif为False，表示不进行分类任务，则直接对标签进行维度置换，将维度从(batch_size, seq_len)变为(seq_len, batch_size)

            # Evaluate the model for all the time steps of the input data, then either do the weight updates on a per-timestep basis, or on a per-sample basis (sum of all per-timestep updates).
            optimizer.zero_grad() #通过optimizer.zero_grad()将模型的参数梯度置零，以准备接收新的梯度值
            output = model(data.permute(1,0,2), targets, 'test') #通过model(data.permute(1,0,2), targets, do_training)对模型进行评估，其中model调用model类的forward函数
            neuron_output.append(output.numpy())