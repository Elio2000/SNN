# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2020-2022 University of Zurich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "setup.py" - Setup configuration and dataset loading.
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/

------------------------------------------------------------------------------
"""


import torch
import numpy as np
import numpy.random as rd
import sys

class SpeechDataset(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, args, type):
      
        
        # n_cues     = 7  #number of cues in input
        f0         = 400   #input spikes frequency of neuron in input layer     
        # t_cue      = 100    #一个cue的持续时常
        # t_wait     = 1200
        n_symbols  = 40  #input channels Left, right, cue, and Noise
        p_group    = 0.3
        
        self.dt         = 1e-3
        self.t_interval = 100        
        self.seq_len    = 200   #一次实验的时间
        self.n_in       = 200   #网络input层neuron的数量
        self.n_out      = 10    # This is a binary classification task, so using two output units with a softmax activation redundant  #网络output层neuron的数量
        # synapse_Nob = args.NoB
        n_channel       = self.n_in // n_symbols    
        prob0           = f0 * self.dt
        # t_silent        = self.t_interval - t_cue
        
        if (type == 'train'):
            length = args.train_len
        else:
            length = args.test_len
            
    
        # Randomly assign group A and B
        # prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
        # idx = rd.choice([0, 1], length) #这将生成一个大小为(length,1)的一维数组idx，其中的元素值随机为0或1。
        # probs = np.zeros((length, 2), dtype=np.float32)
        # # Assign input spike probabilities
        # probs[:, 0] = prob_choices[idx] #这段代码的目的是根据p_group和length生成一个(length, 2)大小的概率数组probs，其中每个示例的概率由prob_choices随机选择而来。
        # probs[:, 1] = prob_choices[1 - idx]
        
        #load dataset
        
        labels = np.load('./data/label.npy', allow_pickle=True)
        data = np.load('./data/data.npy', allow_pickle=True)
        # Normalisation between [lower_bound, upper_bound]
        for index in range(data.shape[0]):
            min_v = np.min(data[index])
            max_v = np.max(data[index])
            data[index] -= min_v
            data[index] /= max_v - min_v
        
        # Generate input spikes
        input_spike_prob = np.zeros((length, self.seq_len, self.n_in))  #三唯数组用于表示每个input neuron的输入，args.train_len， 一个trainset的时间， input layer的neuron数
        # Generate targets 这段代码的目的是根据提示分配情况生成目标数字矩阵target_nums，其中每一行表示示例中每个时间步的目标数字（1-9）
        target_nums = np.zeros((length, self.seq_len), dtype=int)
        for b in range(length):
            selected_speech = np.random.randint(0,3000)
            input_spike_prob[b, : , :] = prob0 * data[selected_speech//2, 0+200*(selected_speech%2):200+200*(selected_speech%2)].reshape(1,200)
            target_nums[b, :] = labels[selected_speech]
                
        
    
        input_spikes = generate_poisson_noise_np(input_spike_prob)  #根据概率，产生对应的spike input
        self.x = torch.tensor(input_spikes).float()
    
        
        self.y = torch.tensor(target_nums).long()
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    #args.cuda = args.cpu
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    if args.dataset == "cue_accumulation":
        print("=== Loading cue evidence accumulation dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cue_accumulation(args, kwargs)
    elif args.dataset == "speech":
        print("=== Loading speech dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_speech(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
        
    print("Training set length: "+str(args.full_train_len))
    print("Test set length: "+str(args.full_test_len))
    
    return (device, train_loader, traintest_loader, test_loader)


def load_dataset_speech(args, kwargs):
    
    # 创建class CueAccumulationDataset
    trainset = SpeechDataset(args,"train")
    testset  = SpeechDataset(args,"test")

    # 在这段代码中，trainset是训练数据集，args.batch_size是指定的批次大小，args.shuffle是一个布尔值，指示是否在每个epoch之前对数据进行洗牌。**kwargs允许传递其他参数。
    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=args.shuffle, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False       , **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False       , **kwargs)
    
    args.n_classes      = trainset.n_out
    args.n_steps        = trainset.seq_len
    args.n_inputs       = trainset.n_in
    args.dt             = trainset.dt
    args.classif        = True
    args.full_train_len = len(trainset)
    args.full_test_len  = len(testset)
    #args.delay_targets  = trainset.t_interval
    args.delay_targets  = 100
    args.skip_test      = False
    
    return (train_loader, traintest_loader, test_loader)


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):  #检查prob_pattern的类型是否为列表。如果是列表，则递归地对列表中的每个元素调用generate_poisson_noise_np
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape    #获取prob_pattern的维数

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)  #然后，函数根据freezing_seed的值确定使用随机种子还是默认的随机种子来初始化随机数生成器rng
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)    #函数生成一个与prob_pattern具有相同形状的随机数数组，并将其与prob_pattern进行比较。如果随机数大于对应位置的概率值，则该位置对应的元素设为True，表示发生了脉冲；否则设为False，表示没有发生脉冲。最终，返回这个脉冲模式。
    return spikes
