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
import os
import sys

class SpeechDataset(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, args, type, speech_indices):
      

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
        labels = np.load(args.label_file_resolved, allow_pickle=True)
        data = np.load(args.data_file_resolved, allow_pickle=True).astype(np.float32)
        # Normalisation between [lower_bound, upper_bound]
        for index in range(data.shape[0]):
            min_v = np.min(data[index])
            max_v = np.max(data[index])
            data[index] -= min_v
            if max_v != min_v:
                data[index] /= max_v - min_v

        rng = np.random.default_rng(args.seed if type == 'train' else args.seed + 1)
        speech_indices = np.asarray(speech_indices, dtype=np.int64)
        if speech_indices.size == 0:
            raise ValueError("No speech samples available for " + type)
        selected_speeches = rng.choice(speech_indices, size=length, replace=length > speech_indices.size)
        
        # Generate input spikes
        input_spike_prob = np.zeros((length, self.seq_len, self.n_in))  #三唯数组用于表示每个input neuron的输入，args.train_len， 一个trainset的时间， input layer的neuron数
        # Generate targets 这段代码的目的是根据提示分配情况生成目标数字矩阵target_nums，其中每一行表示示例中每个时间步的目标数字（1-9）
        target_nums = np.zeros((length, self.seq_len), dtype=int)
        for b, selected_speech in enumerate(selected_speeches):
            input_spike_prob[b, : , :] = prob0 * get_speech_features(data, labels, int(selected_speech), self.n_in).reshape(1, self.n_in)
            target_nums[b, :] = int(labels[selected_speech])
                
        
    
        input_spikes = generate_poisson_noise_np(input_spike_prob, rng=rng)  #根据概率，产生对应的spike input
        self.x = torch.tensor(input_spikes).float()
    
        
        self.y = torch.tensor(target_nums).long()
        self.selected_speeches = selected_speeches
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def setup(args):
    device = select_device(args)
    
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
    args.data_file_resolved, args.label_file_resolved = resolve_speech_paths(args)
    labels = np.load(args.label_file_resolved, allow_pickle=True)
    train_indices, test_indices = make_split_indices(len(labels), args)
    args.train_pool_len = len(train_indices)
    args.test_pool_len = len(test_indices)
    print("Speech data file: " + args.data_file_resolved)
    print("Speech label file: " + args.label_file_resolved)
    print("Speech train/test pool: " + str(args.train_pool_len) + "/" + str(args.test_pool_len))

    trainset = SpeechDataset(args,"train", train_indices)
    testset  = SpeechDataset(args,"test", test_indices)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # 在这段代码中，trainset是训练数据集，args.batch_size是指定的批次大小，args.shuffle是一个布尔值，指示是否在每个epoch之前对数据进行洗牌。**kwargs允许传递其他参数。
    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=args.shuffle, generator=generator, **kwargs)
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


def select_device(args):
    if args.cpu:
        requested_device = 'cpu'
    else:
        requested_device = args.device

    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if requested_device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif mps_available:
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    elif requested_device == 'cpu':
        device = torch.device('cpu')
    elif requested_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device('cuda')
    elif requested_device == 'mps':
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available")
        device = torch.device('mps')
    else:
        raise ValueError("Unsupported device: " + str(requested_device))

    args.cuda = device.type == 'cuda'
    args.mps = device.type == 'mps'
    print("=== Computation device: " + str(device))
    return device


def resolve_speech_paths(args):
    data_file = args.data_file
    label_file = args.label_file
    if data_file is None:
        data_file = os.path.join(args.data_dir, 'data.npy')
    if label_file is None:
        label_file = os.path.join(args.data_dir, 'label.npy')
    return data_file, label_file


def make_split_indices(n_samples, args):
    if not 0.0 < args.test_split < 1.0:
        raise ValueError("--test-split must be between 0 and 1")
    if n_samples < 2:
        raise ValueError("At least two speech samples are required for train/test split")

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(n_samples)
    n_test = int(round(n_samples * args.test_split))
    n_test = max(1, min(n_samples - 1, n_test))
    return shuffled[n_test:], shuffled[:n_test]


def get_speech_features(data, labels, sample_index, n_in):
    if data.ndim != 2:
        raise ValueError("Speech data must be a 2D array")

    if data.shape[0] == len(labels) and data.shape[1] == n_in:
        return data[sample_index]

    if data.shape[0] * 2 == len(labels) and data.shape[1] == 2 * n_in:
        start = n_in * (sample_index % 2)
        return data[sample_index // 2, start:start + n_in]

    raise ValueError(
        "Unsupported speech data/label shape: data="
        + str(data.shape)
        + ", labels="
        + str(len(labels))
        + ", n_in="
        + str(n_in)
    )


def generate_poisson_noise_np(prob_pattern, freezing_seed=None, rng=None):
    if isinstance(prob_pattern, list):  #检查prob_pattern的类型是否为列表。如果是列表，则递归地对列表中的每个元素调用generate_poisson_noise_np
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed, rng=rng) for pb in prob_pattern]

    shp = prob_pattern.shape    #获取prob_pattern的维数

    if rng is None:
        if not(freezing_seed is None):
            rng = np.random.default_rng(freezing_seed)
        else:
            rng = np.random.default_rng()

    spikes = prob_pattern > rng.random(prob_pattern.size).reshape(shp)    #函数生成一个与prob_pattern具有相同形状的随机数数组，并将其与prob_pattern进行比较。如果随机数大于对应位置的概率值，则该位置对应的元素设为True，表示发生了脉冲；否则设为False，表示没有发生脉冲。最终，返回这个脉冲模式。
    return spikes
