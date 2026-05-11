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

 "train.py" - Initializing the network and proceeding to training and test epochs.
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/
       
------------------------------------------------------------------------------
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models


def train(args, device, train_loader, traintest_loader, test_loader):
    torch.manual_seed(42)
    
    for trial in range(1,args.trials+1):
        
        # Network topology
        model = models.SRNN(n_in=args.n_inputs,
                            n_rec=args.n_rec,
                            n_out=args.n_classes,
                            n_t=args.n_steps,
                            thr=args.threshold,
                            thro=args.threshold_out,
                            v_reset=args.v_reset,
                            vo_reset=args.vo_reset,
                            tau_m=args.tau_mem,
                            tau_o=args.tau_out,
                            b_o=args.bias_out,
                            gamma=args.gamma,
                            dt=args.dt,
                            model=args.model,
                            classif=args.classif,
                            w_init_gain=args.w_init_gain,
                            lr_layer=args.lr_layer_norm,
                            #t_crop=args.delay_targets,
                            t_crop=args.delay_targets,
                            visualize=args.visualize,
                            visualize_light=args.visualize_light,
                            device=device,
                            LIF_tpye=args.LIF_type,
                            Output_type=args.Output_type)

        # Use CUDA for GPU-based computation if enabled
        if args.cuda:
            model.cuda()
        
        # Initial monitoring
        if (args.trials > 1):
            print('\nIn trial {} of {}'.format(trial,args.trials))
        if (trial == 1):
            print("=== Model ===" )
            print(model)
        
        # Optimizer 优化器是深度学习中的一个重要组件，用于优化模型的参数以使其适应训练数据并提高性能
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        
        # Loss function (only for performance monitoring purposes, does not influence learning as e-prop learning is hardcoded)
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")
        
        # Training and performance monitoring
        print("\n=== Starting model training with %d epochs:\n" % (args.epochs,))        
        for epoch in range(1, args.epochs + 1):
            print("\t Epoch "+str(epoch)+"...")
            #Training: do_epoch是自定义的函数
            do_epoch(args, True, model, device, train_loader, optimizer, loss, 'train')           # Will display the average accuracy on the training set during the epoch (changing weights)
            #Check performance on the training set and on the test set:
            if not args.skip_test:
                #do_epoch(args, False, model, device, traintest_loader, optimizer, loss, 'train') # Uncomment to display the final accuracy on the training set after the epoch (fixed weights)
                output = do_epoch(args, False, model, device, test_loader, optimizer, loss, 'test')
        torch.save(model.state_dict(), 'model.pth')        
        return (args, model, device, train_loader, test_loader, optimizer, loss, output)


def do_epoch(args, do_training, model, device, loader, optimizer, loss_fct, benchType):
    # 将模型设置为评估模式。这意味着在评估过程中，模型的参数将不会被更新，也就是不会进行梯度计算和优化器的更新操作。
    model.eval()    # This implementation does not rely on autograd, learning update rules are hardcoded
    score = 0
    #score = torch.tensor(score)
    loss = 0
    # 根据评估类型（训练集评估或测试集评估）选择不同的 batch 大小。通常在训练过程中使用较大的 batch 大小，而在评估过程中使用较小的 batch 大小。
    batch = args.batch_size if (benchType == 'train') else args.test_batch_size
    length = args.full_train_len if (benchType == 'train') else args.full_test_len
    with torch.no_grad():   # Same here, we make sure autograd is disabled
        
        # For each batch
        for batch_idx, (data, label) in enumerate(loader):  #loader是一个数据加载器，可以对数据集进行迭代。在每次迭代中，它会返回一个批次的数据和标签。

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
            output = model(data.permute(1,0,2), targets, do_training, args) #通过model(data.permute(1,0,2), targets, do_training)对模型进行评估，其中model调用model类的forward函数
            if do_training: # 如果do_training为True，表示进行训练模式，通过optimizer.step()对模型的参数进行一次更新
                optimizer.step()
                
            # Compute the loss function, inference and score
            if args.delay_targets:
                loss += loss_fct[0](output[-args.delay_targets:], loss_fct[1](targets[-args.delay_targets:]), reduction='mean')
                print("loss:",loss)
            else:
                loss += loss_fct[0](output, loss_fct[1](targets), reduction='mean')
            if args.classif:
                if args.delay_targets:
                    inference = torch.argmax(torch.sum(output[-args.delay_targets:],axis=0),axis=1)
                    score += torch.sum(torch.eq(inference,label[:,0]))
                else:# 计算正确的个数
                    inference = torch.argmax(torch.sum(output,axis=0),axis=1)
                    score += torch.sum(torch.eq(inference,label[:,0]))
        
    if benchType == "train" and do_training:
        info = "on training set (while training): "
    elif benchType == "train":
        info = "on training set                 : "
    elif benchType == "test":
        info = "on test set                     : "

    if args.classif:
        print("\t\t Score "+info+str(score.item())+'/'+str(length)+' ('+str(score.item()/length*100)+'%), loss: '+str(loss.item()))
    else:
        print("\t\t Loss "+info+str(loss.item()))
        
    return output
            
