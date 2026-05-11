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

 "main.py" - Main file for training a spiking RNN with eligibility propagation (e-prop), as per the original paper:

        [G. Bellec et al., "A solution to the learning dilemma for recurrent networks of spiking neurons,"
         Nature communications, vol. 11, no. 3625, 2020]

    This code demonstrates e-prop on the cue accumulation task described in the e-prop paper by Bellec et al. The 
    e-prop implementation provided here only covers the LIF neuron model, which is presented by Bellec et al. as not
    able to learn the second-long time dependencies of the cue accumulation task.
    We solve this issue by using a second-long leakage time constant. We introduce this technique in the following paper:

        [C. Frenkel and G. Indiveri, "ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network
         processor enabling on-chip learning over second-long timescales," IEEE International Solid-State
         Circuits Conference (ISSCC), 2022]

    Example run: default parameters contained in this file provide suitable convergence for the cue accumulation task
             described in the e-prop paper. The code can just be ran with "python main.py"
 
 Project: PyTorch e-prop

 Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich

 Cite this code: BibTeX/APA citation formats auto-converted from the CITATION.cff file in the repository are available 
       through the "Cite this repository" link in the root GitHub repo https://github.com/ChFrenkel/eprop-PyTorch/
       
------------------------------------------------------------------------------
"""


import argparse
import train
import setup
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='Spiking RNN Pytorch training')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA training and run training on CPU')
    # parser.add_argument('--cpu', action='store_true', default=True, help='Disable CUDA training and run training on CPU')
    # parser.add_argument('--dataset', type=str, choices = ['cue_accumulation'], default='cue_accumulation', help='Choice of the dataset')
    parser.add_argument('--dataset', type=str, choices = ['cue_accumulation, speech'], default='speech', help='Choice of the dataset')
    parser.add_argument('--shuffle', type=bool, default=True, help='Enables shuffling sample order in datasets after each epoch')
    parser.add_argument('--trials', type=int, default=1, help='Nomber of trial experiments to do (i.e. repetitions with different initializations)')
    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, choices = ['SGD', 'NAG', 'Adam', 'RMSProp'], default='Adam', help='Choice of the optimizer')
    parser.add_argument('--loss', type=str, choices = ['MSE', 'BCE', 'CE'], default='BCE', help='Choice of the loss function (only for performance monitoring purposes, does not influence learning)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr-layer-norm', type=float, nargs='+', default=(0.05,0.05,1.0), help='Per-layer modulation factor of the learning rate')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training (limited by the available GPU memory)')
    parser.add_argument('--test-batch-size', type=int, default=5, help='Batch size for testing (limited by the available GPU memory)')
    parser.add_argument('--train-len', type=int, default=200, help='Number of training set samples')
    parser.add_argument('--test-len', type=int, default=200, help='Number of test set samples')
    parser.add_argument('--visualize', type=bool, default=True, help='Enable network visualization')
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
    parser.add_argument('--SynapseNoB', type=int, default=4, help='Number of bits for weight')
    parser.add_argument('--NeuronNoB', type=int, default=7, help='Number of bits for weight')
    parser.add_argument('--LIF_type', type=bool, default=True, help='LIF_tpye = True represent LIF discribe in paper, LIF_type = False represent LIF in the circuit')
    parser.add_argument('--Output_type', type=bool, default=True, help='Output_tpye = True represent output vo as discribe in paper, Output_tpye = False represent output zo as in the circuit')
    parser.add_argument('--threshold_out', type=float, default=0.6, help='Firing threshold in the output layer')
    parser.add_argument('--v_reset', type=float, default=0, help='Reset potential in the recurrent layer')
    parser.add_argument('--vo_reset', type=float, default=0, help='Reset potential in the output layer')
    
    args = parser.parse_args()
    
    args.LIF_type = True
    args.Output_type = True

    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)    
    # #(args_out, model, device_out, train_loader_out, test_loader_out, optimizer, loss, output) = train.train(args, device, train_loader, traintest_loader, test_loader)
    (args_out, model, device_out, train_loader_out, test_loader_out, optimizer, loss, output) = train.train(args, device, train_loader, traintest_loader, train_loader)
    print(f"Running with SynapseNoB = {synapse_nob}, NeuronNoB = {neuron_nob}")
    print(time.localtime(time.time()))
    return args

    # (device, train_loader, traintest_loader, test_loader) = setup.setup(args)    
    # #(args_out, model, device_out, train_loader_out, test_loader_out, optimizer, loss, output) = train.train(args, device, train_loader, traintest_loader, test_loader)
    # (args_out, model, device_out, train_loader_out, test_loader_out, optimizer, loss, output) = train.train(args, device, train_loader, traintest_loader, train_loader)
    
    #return (args_out, model, device_out, train_loader_out, test_loader_out, optimizer, loss, output)

if __name__ == '__main__':
    # original_stdout = sys.stdout
    # # # (args, model, device, train_loader, test_loader, optimizer, loss, output) = main()
    # file = open("E:\\qixiangao\\OneDrive - sjtu.edu.cn\\SNN\\2st流片\\JSSC\\Figure\\SectionIII_algorithm\\speech_class\\accu.txt", "w")
    # sys.stdout = file
    args = main()
    # sys.stdout = original_stdout
    # file.close
    
    