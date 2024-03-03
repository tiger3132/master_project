from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser
import pathlib
import os
import pandas as pd
import seaborn as sns

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--session", type=str, default='test_resnet')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_train_steps", type=int, default=20000)
parser.add_argument("--model_name", type=str, default="ResNet18")
parser.add_argument("--optimizer", type=str, default='adamw')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--wd_schedule_type", type=str, default='cosine')
parser.add_argument("--output_dir", type=str, default="cifar100")
parser.add_argument("--lr_warmup_steps", type=int, default=200)
parser.add_argument("--lr_decay_factor", type=float, default=0.1)
#adamw hyperparams
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.98)

#sgd hyperparams
parser.add_argument("--momentum", type=float, default=0)
#sgd and adamw hyperparams
parser.add_argument("--weight_decay", type=float, default=0.001)

# adamcpr hyperparams
parser.add_argument("--kappa_init_param", type=float, default=1000)
parser.add_argument("--kappa_init_method", type=str, default="warm_start")
parser.add_argument("--reg_function", type=str, default="l2")
parser.add_argument("--kappa_update", type=float, default=1.0)
parser.add_argument("--kappa_adapt", action=argparse.BooleanOptionalAction)
parser.add_argument("--apply_lr", action=argparse.BooleanOptionalAction)


parser.add_argument("--data_transform", type=int, default=1)
parser.add_argument("--batch_norm", type=int, default=1)

args = parser.parse_args()



task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"

if args.wd_schedule_type == "cosine" or args.wd_schedule_type == "linear":
    if args.optimizer == "sgd":
        expt_name = f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}_trans{args.data_transform}_bn{args.batch_norm}"
    elif args.optimizer == "adamw":
        expt_name = f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}_trans{args.data_transform}_bn{args.batch_norm}_b1{args.beta1}_b2{args.beta2}"
    elif args.optimizer == "adamcpr":
        expt_name = f"b{args.batch_size}_{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}_trans{args.data_transform}_bn{args.batch_norm}_t{args.wd_schedule_type}"
else:
    if args.optimizer == "sgd":
        expt_name = f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}_trans{args.data_transform}_bn{args.batch_norm}"
    elif args.optimizer == "adamw":
        expt_name = f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_trans{args.data_transform}_bn{args.batch_norm}_b1{args.beta1}_b2{args.beta2}"
    elif args.optimizer == "adamcpr":
        expt_name = f"b{args.batch_size}_{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}_trans{args.data_transform}_bn{args.batch_norm}_t{args.wd_schedule_type}"


#expt_dir = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}/{expt_name}/version_0"
directory_str = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}"

directory = os.fsencode(directory_str)
y_arr = []
fc_weight_max_20000_arr = []
fc_weight_min_20000_arr = []
fc_weight_std_20000_arr = []
fc_weight_mean_20000_arr = []
layer4_1_bn2_weight_max_20000_arr = []
layer4_1_bn2_weight_min_20000_arr = []
layer4_1_bn2_weight_mean_20000_arr = []
layer4_1_bn2_weight_std_20000_arr = []
layer4_1_conv2_weight_max_20000_arr = []
layer4_1_conv2_weight_min_20000_arr = []
layer4_1_conv2_weight_mean_20000_arr = []
layer4_1_conv2_weight_std_20000_arr = []
bn1_weight_max_20000_arr = []
bn1_weight_min_20000_arr = []
bn1_weight_mean_20000_arr = []
bn1_weight_std_20000_arr = []
bn1_bias_max_20000_arr = []
bn1_bias_min_20000_arr = []
bn1_bias_mean_20000_arr = []
bn1_bias_std_20000_arr = []
layer1_0_conv1_weight_max_20000_arr = []
layer1_0_conv1_weight_min_20000_arr = []
layer1_0_conv1_weight_mean_20000_arr = []
layer1_0_conv1_weight_std_20000_arr = []
task_names = [] 

fc_weight_std_arr = []
layer4_1_bn2_weight_std_arr = []
layer4_1_conv2_weight_std_arr = []
bn1_weight_std_arr = []
bn1_bias_std_arr = []
layer1_0_conv1_weight_std_arr = []

# Mean of std
fc_weight_std_mean_arr = [[], [], [], []] 
layer4_1_bn2_weight_std_mean_arr = [[], [], [], []]
layer4_1_conv2_weight_std_mean_arr = [[], [], [], []] 
bn1_weight_std_mean_arr = [[], [], [], []] 
bn1_bias_std_mean_arr = [[], [], [], []]
layer1_0_conv1_weight_std_mean_arr = [[], [], [], []]

# Magnitude of grad of std
fc_weight_std_grad_norm_arr = [[], [], [], []]
layer4_1_bn2_weight_std_grad_norm_arr = [[], [], [], []]
layer4_1_conv2_weight_std_grad_norm_arr = [[], [], [], []]
bn1_weight_std_grad_norm_arr = [[], [], [], []]
bn1_bias_std_grad_norm_arr = [[], [], [], []]
layer1_0_conv1_weight_std_grad_norm_arr = [[], [], [], []]

# Mean of grad of std
fc_weight_std_grad_mean_arr = [[], [], [], []]
layer4_1_bn2_weight_std_grad_mean_arr = [[], [], [], []]
layer4_1_conv2_weight_std_grad_mean_arr = [[], [], [], []]
bn1_weight_std_grad_mean_arr = [[], [], [], []]
bn1_bias_std_grad_mean_arr = [[], [], [], []]
layer1_0_conv1_weight_std_grad_mean_arr = [[], [], [], []]


# Mean of mean
fc_weight_mean_mean_arr = [[], [], [], []] 
layer4_1_bn2_weight_mean_mean_arr = [[], [], [], []]
layer4_1_conv2_weight_mean_mean_arr = [[], [], [], []] 
bn1_weight_mean_mean_arr = [[], [], [], []] 
bn1_bias_mean_mean_arr = [[], [], [], []]
layer1_0_conv1_weight_mean_mean_arr = [[], [], [], []]

# Magnitude of grad of std
fc_weight_mean_grad_norm_arr = [[], [], [], []]
layer4_1_bn2_weight_mean_grad_norm_arr = [[], [], [], []]
layer4_1_conv2_weight_mean_grad_norm_arr = [[], [], [], []]
bn1_weight_mean_grad_norm_arr = [[], [], [], []]
bn1_bias_mean_grad_norm_arr = [[], [], [], []]
layer1_0_conv1_weight_mean_grad_norm_arr = [[], [], [], []]

# Mean of grad of std
fc_weight_mean_grad_mean_arr = [[], [], [], []]
layer4_1_bn2_weight_mean_grad_mean_arr = [[], [], [], []]
layer4_1_conv2_weight_mean_grad_mean_arr = [[], [], [], []]
bn1_weight_mean_grad_mean_arr = [[], [], [], []]
bn1_bias_mean_grad_mean_arr = [[], [], [], []]
layer1_0_conv1_weight_mean_grad_mean_arr = [[], [], [], []]

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    expt_dir = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}/{filename}/version_0"
    event = EventAccumulator(expt_dir)
    event.Reload()
    y = event.Scalars('test_accuracy')
    y_arr.append(y[0].value)
    """
    fc_weight_max = event.Scalars("param/fc.weight/max")
    fc_weight_min = event.Scalars("param/fc.weight/min")
    fc_weight_mean = event.Scalars("param/fc.weight/mean")
    fc_weight_std = event.Scalars("param/fc.weight/std")
    layer4_1_bn2_weight_max = event.Scalars("param/layer4.1.bn2.weight/max")
    layer4_1_bn2_weight_min = event.Scalars("param/layer4.1.bn2.weight/min")
    layer4_1_bn2_weight_mean = event.Scalars("param/layer4.1.bn2.weight/mean")
    layer4_1_bn2_weight_std = event.Scalars("param/layer4.1.bn2.weight/std")
    layer4_1_conv2_weight_max = event.Scalars("param/layer4.1.conv2.weight/max")
    layer4_1_conv2_weight_min = event.Scalars("param/layer4.1.conv2.weight/min")
    layer4_1_conv2_weight_mean = event.Scalars("param/layer4.1.conv2.weight/mean")
    layer4_1_conv2_weight_std = event.Scalars("param/layer4.1.conv2.weight/std")
    bn1_weight_max = event.Scalars("param/bn1.weight/max")
    bn1_weight_min = event.Scalars("param/bn1.weight/min")
    bn1_weight_mean = event.Scalars("param/bn1.weight/mean")
    bn1_weight_std = event.Scalars("param/bn1.weight/std")
    bn1_bias_max = event.Scalars("param/bn1.bias/max")
    bn1_bias_min = event.Scalars("param/bn1.bias/min")
    bn1_bias_mean = event.Scalars("param/bn1.bias/mean")
    bn1_bias_std = event.Scalars("param/bn1.bias/std")
    layer1_0_conv1_weight_max = event.Scalars("param/layer1.0.conv1.weight/max")
    layer1_0_conv1_weight_min = event.Scalars("param/layer1.0.conv1.weight/min")
    layer1_0_conv1_weight_mean = event.Scalars("param/layer1.0.conv1.weight/mean")
    layer1_0_conv1_weight_std = event.Scalars("param/layer1.0.conv1.weight/std")
    """
    """
    fc_weight_max_20000_arr.append(fc_weight_max[-1].value)
    fc_weight_min_20000_arr.append(fc_weight_min[-1].value)
    fc_weight_mean_20000_arr.append(fc_weight_mean[-1].value)
    fc_weight_std_20000_arr.append(fc_weight_std[-1].value)
    layer4_1_bn2_weight_max_20000_arr.append(layer4_1_bn2_weight_max[-1].value)
    layer4_1_bn2_weight_min_20000_arr.append(layer4_1_bn2_weight_min[-1].value)
    layer4_1_bn2_weight_mean_20000_arr.append(layer4_1_bn2_weight_mean[-1].value)
    layer4_1_bn2_weight_std_20000_arr.append(layer4_1_bn2_weight_std[-1].value)
    layer4_1_conv2_weight_max_20000_arr.append(layer4_1_conv2_weight_max[-1].value) 
    layer4_1_conv2_weight_min_20000_arr.append(layer4_1_conv2_weight_min[-1].value)
    layer4_1_conv2_weight_mean_20000_arr.append(layer4_1_conv2_weight_mean[-1].value) 
    layer4_1_conv2_weight_std_20000_arr.append(layer4_1_conv2_weight_std[-1].value) 
    bn1_weight_max_20000_arr.append(bn1_weight_max[-1].value)
    bn1_weight_min_20000_arr.append(bn1_weight_min[-1].value)
    bn1_weight_mean_20000_arr.append(bn1_weight_mean[-1].value)
    bn1_weight_std_20000_arr.append(bn1_weight_std[-1].value)
    bn1_bias_max_20000_arr.append(bn1_bias_max[-1].value)
    bn1_bias_min_20000_arr.append(bn1_bias_min[-1].value)
    bn1_bias_mean_20000_arr.append(bn1_bias_mean[-1].value)
    bn1_bias_std_20000_arr.append(bn1_bias_std[-1].value)
    layer1_0_conv1_weight_max_20000_arr.append(layer1_0_conv1_weight_max[-1].value) 
    layer1_0_conv1_weight_min_20000_arr.append(layer1_0_conv1_weight_min[-1].value)
    layer1_0_conv1_weight_mean_20000_arr.append(layer1_0_conv1_weight_mean[-1].value) 
    layer1_0_conv1_weight_std_20000_arr.append(layer1_0_conv1_weight_std[-1].value) 
    """
    """
    fc_weight_mean = event.Scalars("param/fc.weight/mean")
    layer4_1_bn2_weight_mean = event.Scalars("param/layer4.1.bn2.weight/mean")
    layer4_1_conv2_weight_mean = event.Scalars("param/layer4.1.conv2.weight/mean")
    bn1_weight_mean = event.Scalars("param/bn1.weight/mean")
    bn1_bias_mean = event.Scalars("param/bn1.bias/mean")
    layer1_0_conv1_weight_mean = event.Scalars("param/layer1.0.conv1.weight/mean")
    """
    """
    fc_weight_std = event.Scalars("param/fc.weight/std")
    layer4_1_bn2_weight_std = event.Scalars("param/layer4.1.bn2.weight/std")
    layer4_1_conv2_weight_std = event.Scalars("param/layer4.1.conv2.weight/std")
    bn1_weight_std = event.Scalars("param/bn1.weight/std")
    bn1_bias_std = event.Scalars("param/bn1.bias/std")
    layer1_0_conv1_weight_std = event.Scalars("param/layer1.0.conv1.weight/std")
    """
    # Mean of last value 
    """
    fc_weight_mean_20000_arr.append(fc_weight_mean[-1].value)
    layer4_1_bn2_weight_mean_20000_arr.append(layer4_1_bn2_weight_mean[-1].value)
    layer4_1_conv2_weight_mean_20000_arr.append(layer4_1_conv2_weight_mean[-1].value) 
    bn1_weight_mean_20000_arr.append(bn1_weight_mean[-1].value)
    bn1_bias_mean_20000_arr.append(bn1_bias_mean[-1].value)
    layer1_0_conv1_weight_mean_20000_arr.append(layer1_0_conv1_weight_mean[-1].value) 
    """
    # Std of last value
    """
    fc_weight_std_20000_arr.append(fc_weight_std[-1].value)
    layer4_1_bn2_weight_std_20000_arr.append(layer4_1_bn2_weight_std[-1].value)
    layer4_1_conv2_weight_std_20000_arr.append(layer4_1_conv2_weight_std[-1].value) 
    bn1_weight_std_20000_arr.append(bn1_weight_std[-1].value)
    bn1_bias_std_20000_arr.append(bn1_bias_std[-1].value)
    layer1_0_conv1_weight_std_20000_arr.append(layer1_0_conv1_weight_std[-1].value) 
    """

    # Contain all 20000 timesteps of std

    fc_weight_std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/fc.weight/std')])
    layer4_1_bn2_weight_std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/layer4.1.bn2.weight/std')])
    layer4_1_conv2_weight_std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/layer4.1.conv2.weight/std')])
    bn1_weight_std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/bn1.weight/std')])
    bn1_bias_std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/bn1.bias/std')])
    layer1_0_conv1_weight_std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/layer1.0.conv1.weight/std')])
    print(f"std fc weight shape: {np.shape(fc_weight_std)}") # shape of (400,)

    # Split into 4 sects of std

    fc_weight_std_split = np.array_split(fc_weight_std, 4)
    layer4_1_bn2_weight_std_split = np.array_split(layer4_1_bn2_weight_std, 4)
    layer4_1_conv2_weight_std_split = np.array_split(layer4_1_conv2_weight_std, 4)
    bn1_weight_std_split = np.array_split(bn1_weight_std, 4)
    bn1_bias_std_split = np.array_split(bn1_bias_std, 4)
    layer1_0_conv1_weight_std_split = np.array_split(layer1_0_conv1_weight_std, 4)
    print(f"split of std fc weight shape: {np.shape(fc_weight_std_split)}") # shape of (100, 4)
    assert fc_weight_std[0] == fc_weight_std_split[0][0]
    assert fc_weight_std[99] == fc_weight_std_split[0][-1]
    assert fc_weight_std[100] == fc_weight_std_split[1][0]

    #Gradient of std

    fc_weight_std_grad = np.gradient(fc_weight_std_split, axis=1)
    layer4_1_bn2_weight_std_grad = np.gradient(layer4_1_bn2_weight_std_split, axis=1)
    layer4_1_conv2_weight_std_grad = np.gradient(layer4_1_conv2_weight_std_split, axis=1)
    bn1_weight_std_grad = np.gradient(bn1_weight_std_split, axis=1)
    bn1_bias_std_grad = np.gradient(bn1_bias_std_split, axis=1)
    layer1_0_conv1_weight_std_grad = np.gradient(layer1_0_conv1_weight_std_split, axis=1)
    print(f"grad of std fc weight shape: {np.shape(fc_weight_std_grad)}") # shape of (100, 4)
    assert np.gradient(fc_weight_std_split[0])[0] == fc_weight_std_grad[0][0]
    assert np.gradient(fc_weight_std_split[1])[0] == fc_weight_std_grad[1][0]

    # Mean of std of grad

    fc_weight_std_mean = np.mean(fc_weight_std_split, axis=1)
    layer4_1_bn2_weight_std_mean = np.mean(layer4_1_bn2_weight_std_split, axis=1)
    layer4_1_conv2_weight_std_mean = np.mean(layer4_1_conv2_weight_std_split, axis=1)
    bn1_weight_std_mean = np.mean(bn1_weight_std_split, axis=1)
    bn1_bias_std_mean = np.mean(bn1_bias_std_split, axis=1)
    layer1_0_conv1_weight_std_mean = np.mean(layer1_0_conv1_weight_std_split, axis=1)
    print(f"mean std fc weight shape: {np.shape(fc_weight_std_mean)}") # shape of (4,)
    assert np.mean(fc_weight_std_split[0]) == fc_weight_std_mean[0]
    assert np.mean(fc_weight_std_split[1]) == fc_weight_std_mean[1]

    # Magnitude of grad of std

    fc_weight_std_grad_norm = np.linalg.norm(fc_weight_std_grad, axis=1)
    layer4_1_bn2_weight_std_grad_norm = np.linalg.norm(layer4_1_bn2_weight_std_grad, axis=1)
    layer4_1_conv2_weight_std_grad_norm = np.linalg.norm(layer4_1_conv2_weight_std_grad, axis=1)
    bn1_weight_std_grad_norm = np.linalg.norm(bn1_weight_std_grad, axis=1)
    bn1_bias_std_grad_norm = np.linalg.norm(bn1_bias_std_grad, axis=1)
    layer1_0_conv1_weight_std_grad_norm = np.linalg.norm(layer1_0_conv1_weight_std_grad, axis=1)
    print(f"mag grad std fc weight shape: {np.shape(fc_weight_std_grad_norm)}") # shape of (4,)
    assert np.linalg.norm(fc_weight_std_grad[0]) == fc_weight_std_grad_norm[0]
    assert np.linalg.norm(fc_weight_std_grad[1]) == fc_weight_std_grad_norm[1]

    # Mean of grad of std

    fc_weight_std_grad_mean = np.mean(fc_weight_std_grad, axis=1)
    layer4_1_bn2_weight_std_grad_mean = np.mean(layer4_1_bn2_weight_std_grad, axis=1)
    layer4_1_conv2_weight_std_grad_mean = np.mean(layer4_1_conv2_weight_std_grad, axis=1)
    bn1_weight_std_grad_mean = np.mean(bn1_weight_std_grad, axis=1)
    bn1_bias_std_grad_mean = np.mean(bn1_bias_std_grad, axis=1)
    layer1_0_conv1_weight_std_grad_mean = np.mean(layer1_0_conv1_weight_std_grad, axis=1)
    print(f"mean grad std fc weight shape: {np.shape(fc_weight_std_grad_mean)}") # shape of (4,)
    assert np.mean(fc_weight_std_grad[0]) == fc_weight_std_grad_mean[0]
    assert np.mean(fc_weight_std_grad[1]) == fc_weight_std_grad_mean[1]

    for i in range(0, 4):
        fc_weight_std_mean_arr[i].append(fc_weight_std_mean[i])
        layer4_1_bn2_weight_std_mean_arr[i].append(layer4_1_bn2_weight_std_mean[i])
        layer4_1_conv2_weight_std_mean_arr[i].append(layer4_1_conv2_weight_std_mean[i])
        bn1_weight_std_mean_arr[i].append(bn1_weight_std_mean[i]) 
        bn1_bias_std_mean_arr[i].append(bn1_bias_std_mean[i])
        layer1_0_conv1_weight_std_mean_arr[i].append(layer1_0_conv1_weight_std_mean[i])
        # Magnitude of grad of std

        fc_weight_std_grad_norm_arr[i].append(fc_weight_std_grad_norm[i]) 
        layer4_1_bn2_weight_std_grad_norm_arr[i].append(layer4_1_bn2_weight_std_grad_norm[i])
        layer4_1_conv2_weight_std_grad_norm_arr[i].append(layer4_1_conv2_weight_std_grad_norm[i])
        bn1_weight_std_grad_norm_arr[i].append(bn1_weight_std_grad_norm[i]) 
        bn1_bias_std_grad_norm_arr[i].append(bn1_bias_std_grad_norm[i])
        layer1_0_conv1_weight_std_grad_norm_arr[i].append(layer1_0_conv1_weight_std_grad_norm[i])
        # Mean of grad of std

        fc_weight_std_grad_mean_arr[i].append(fc_weight_std_grad_mean[i])
        layer4_1_bn2_weight_std_grad_mean_arr[i].append(layer4_1_bn2_weight_std_grad_mean[i])
        layer4_1_conv2_weight_std_grad_mean_arr[i].append(layer4_1_conv2_weight_std_grad_mean[i])
        bn1_weight_std_grad_mean_arr[i].append(bn1_weight_std_grad_mean[i])
        bn1_bias_std_grad_mean_arr[i].append(bn1_bias_std_grad_mean[i])
        layer1_0_conv1_weight_std_grad_mean_arr[i].append(layer1_0_conv1_weight_std_grad_mean[i])

    assert fc_weight_std_mean_arr[3][-1] == fc_weight_std_mean[3]

    """
    # Contain all 20000 timesteps of mean

    fc_weight_mean = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/fc.weight/mean')])
    layer4_1_bn2_weight_mean = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/layer4.1.bn2.weight/mean')])
    layer4_1_conv2_weight_mean = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/layer4.1.conv2.weight/mean')])
    bn1_weight_mean = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/bn1.weight/mean')])
    bn1_bias_mean = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/bn1.bias/mean')])
    layer1_0_conv1_weight_mean = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/layer1.0.conv1.weight/mean')])
    print(f"mean fc weight shape: {np.shape(fc_weight_mean)}") # shape of (400,)

    # Split into 4 sects of mean

    fc_weight_mean_split = np.array_split(fc_weight_mean, 4)
    layer4_1_bn2_weight_mean_split = np.array_split(layer4_1_bn2_weight_mean, 4)
    layer4_1_conv2_weight_mean_split = np.array_split(layer4_1_conv2_weight_mean, 4)
    bn1_weight_mean_split = np.array_split(bn1_weight_mean, 4)
    bn1_bias_mean_split = np.array_split(bn1_bias_mean, 4)
    layer1_0_conv1_weight_mean_split = np.array_split(layer1_0_conv1_weight_mean, 4)
    print(f"split of mean fc weight shape: {np.shape(fc_weight_mean_split)}") # shape of (100, 4)

    #Gradient of mean

    fc_weight_mean_grad = np.gradient(fc_weight_mean_split, axis=1)
    layer4_1_bn2_weight_mean_grad = np.gradient(layer4_1_bn2_weight_mean_split, axis=1)
    layer4_1_conv2_weight_mean_grad = np.gradient(layer4_1_conv2_weight_mean_split, axis=1)
    bn1_weight_mean_grad = np.gradient(bn1_weight_mean_split, axis=1)
    bn1_bias_mean_grad = np.gradient(bn1_bias_mean_split, axis=1)
    layer1_0_conv1_weight_mean_grad = np.gradient(layer1_0_conv1_weight_mean_split, axis=1)
    print(f"grad of mean fc weight shape: {np.shape(fc_weight_mean_grad)}") # shape of (100, 4)

    # Mean of mean of grad

    fc_weight_mean_mean = np.mean(fc_weight_mean_split, axis=1)
    layer4_1_bn2_weight_mean_mean = np.mean(layer4_1_bn2_weight_mean_split, axis=1)
    layer4_1_conv2_weight_mean_mean = np.mean(layer4_1_conv2_weight_mean_split, axis=1)
    bn1_weight_mean_mean = np.mean(bn1_weight_mean_split, axis=1)
    bn1_bias_mean_mean = np.mean(bn1_bias_mean_split, axis=1)
    layer1_0_conv1_weight_mean_mean = np.mean(layer1_0_conv1_weight_mean_split, axis=1)
    print(f"mean mean fc weight shape: {np.shape(fc_weight_mean_mean)}") # shape of (4,)

    # Magnitude of grad of mean

    fc_weight_mean_grad_norm = np.linalg.norm(fc_weight_mean_grad, axis=1)
    layer4_1_bn2_weight_mean_grad_norm = np.linalg.norm(layer4_1_bn2_weight_mean_grad, axis=1)
    layer4_1_conv2_weight_mean_grad_norm = np.linalg.norm(layer4_1_conv2_weight_mean_grad, axis=1)
    bn1_weight_mean_grad_norm = np.linalg.norm(bn1_weight_mean_grad, axis=1)
    bn1_bias_mean_grad_norm = np.linalg.norm(bn1_bias_mean_grad, axis=1)
    layer1_0_conv1_weight_mean_grad_norm = np.linalg.norm(layer1_0_conv1_weight_mean_grad, axis=1)
    print(f"mag grad mean fc weight shape: {np.shape(fc_weight_mean_grad_norm)}") # shape of (4,)
    # Mean of grad of mean

    fc_weight_mean_grad_mean = np.mean(fc_weight_mean_grad, axis=1)
    layer4_1_bn2_weight_mean_grad_mean = np.mean(layer4_1_bn2_weight_mean_grad, axis=1)
    layer4_1_conv2_weight_mean_grad_mean = np.mean(layer4_1_conv2_weight_mean_grad, axis=1)
    bn1_weight_mean_grad_mean = np.mean(bn1_weight_mean_grad, axis=1)
    bn1_bias_mean_grad_mean = np.mean(bn1_bias_mean_grad, axis=1)
    layer1_0_conv1_weight_mean_grad_mean = np.mean(layer1_0_conv1_weight_mean_grad, axis=1)
    print(f"mean grad mean fc weight shape: {np.shape(fc_weight_mean_grad_mean)}") # shape of (4,)


    for i in range(0, 4):
        fc_weight_mean_mean_arr[i].append(fc_weight_mean_mean[i])
        layer4_1_bn2_weight_mean_mean_arr[i].append(layer4_1_bn2_weight_mean_mean[i])
        layer4_1_conv2_weight_mean_mean_arr[i].append(layer4_1_conv2_weight_mean_mean[i])
        bn1_weight_mean_mean_arr[i].append(bn1_weight_mean_mean[i]) 
        bn1_bias_mean_mean_arr[i].append(bn1_bias_mean_mean[i])
        layer1_0_conv1_weight_mean_mean_arr[i].append(layer1_0_conv1_weight_mean_mean[i])
        # Magnitude of grad of std

        fc_weight_mean_grad_norm_arr[i].append(fc_weight_mean_grad_norm[i]) 
        layer4_1_bn2_weight_mean_grad_norm_arr[i].append(layer4_1_bn2_weight_mean_grad_norm[i])
        layer4_1_conv2_weight_mean_grad_norm_arr[i].append(layer4_1_conv2_weight_mean_grad_norm[i])
        bn1_weight_mean_grad_norm_arr[i].append(bn1_weight_mean_grad_norm[i]) 
        bn1_bias_mean_grad_norm_arr[i].append(bn1_bias_mean_grad_norm[i])
        layer1_0_conv1_weight_mean_grad_norm_arr[i].append(layer1_0_conv1_weight_mean_grad_norm[i])
        # Mean of grad of std

        fc_weight_mean_grad_mean_arr[i].append(fc_weight_mean_grad_mean[i])
        layer4_1_bn2_weight_mean_grad_mean_arr[i].append(layer4_1_bn2_weight_mean_grad_mean[i])
        layer4_1_conv2_weight_mean_grad_mean_arr[i].append(layer4_1_conv2_weight_mean_grad_mean[i])
        bn1_weight_mean_grad_mean_arr[i].append(bn1_weight_mean_grad_mean[i])
        bn1_bias_mean_grad_mean_arr[i].append(bn1_bias_mean_grad_mean[i])
        layer1_0_conv1_weight_mean_grad_mean_arr[i].append(layer1_0_conv1_weight_mean_grad_mean[i])

    assert fc_weight_mean_mean_arr[3][-1] == fc_weight_mean_mean[3]
    """

    task_names.append(filename)

#print(f"fc weight std mean arr length: {len(fc_weight_std_mean_arr[0])}") # length of 97 
print(f"fc weight mean mean arr length: {len(fc_weight_mean_mean_arr[0])}") # length of 97 


"""
acc_task_df = pd.DataFrame({
    'name' : task_names,
    'test_accuracy' : y_arr
    })
print(acc_task_df)
"""
"""
df_heat_fc_weight = pd.DataFrame({
    'max': fc_weight_max_20000_arr,
    'min': fc_weight_min_20000_arr,
    'mean': fc_weight_mean_20000_arr,
    'std': fc_weight_std_20000_arr,
    'test_accuracy': y_arr
    })
df_heat_layer4_1_bn2_weight = pd.DataFrame({
    'max' : layer4_1_bn2_weight_max_20000_arr,
    'min' : layer4_1_bn2_weight_min_20000_arr,
    'mean' : layer4_1_bn2_weight_mean_20000_arr,
    'std' : layer4_1_bn2_weight_std_20000_arr,
    'test_accuracy': y_arr
    })
df_heat_layer4_1_conv2_weight = pd.DataFrame({
    'max' : layer4_1_conv2_weight_max_20000_arr,
    'min' : layer4_1_conv2_weight_min_20000_arr,
    'mean' : layer4_1_conv2_weight_mean_20000_arr,
    'std' : layer4_1_conv2_weight_std_20000_arr,
    'test_accuracy': y_arr
    })
df_heat_bn1_weight = pd.DataFrame({
    'max' : bn1_weight_max_20000_arr,
    'min' : bn1_weight_min_20000_arr,
    'mean' : bn1_weight_mean_20000_arr,
    'std' : bn1_weight_std_20000_arr,
    'test_accuracy': y_arr
    })
df_heat_bn1_bias = pd.DataFrame({
    'max' : bn1_bias_max_20000_arr,
    'min' : bn1_bias_min_20000_arr,
    'mean' : bn1_bias_mean_20000_arr,
    'std' : bn1_bias_std_20000_arr,
    'test_accuracy': y_arr
    })
df_heat_layer1_0_conv1_weight = pd.DataFrame({
    'max' : layer1_0_conv1_weight_max_20000_arr,
    'min' : layer1_0_conv1_weight_min_20000_arr,
    'mean' : layer1_0_conv1_weight_mean_20000_arr,
    'std' : layer1_0_conv1_weight_std_20000_arr,
    'test_accuracy': y_arr
    })
"""
# Dataframe for std
"""
df_std = pd.DataFrame({
    'fc_w': fc_weight_std_20000_arr,
    'l41_bn2_w' : layer4_1_bn2_weight_std_20000_arr, 
    'l41_c2_w' : layer4_1_conv2_weight_std_20000_arr, 
    'bn1_w' : bn1_weight_std_20000_arr,
    'bn1_b' : bn1_bias_std_20000_arr,
    'l10_c1_w' : layer1_0_conv1_weight_std_20000_arr,
    'test_acc': y_arr
    }) 
df_std.to_csv("df_std.csv")

df_std_no_outlier = df_std.drop(df_std[df_std.test_acc < 0.05].index)
df_std_no_outlier.to_csv("df_std_no_outlier.csv")
"""
# Dataframe for mean
"""
df_mean = pd.DataFrame({
    'fc_w': fc_weight_mean_20000_arr,
    'l41_bn2_w' : layer4_1_bn2_weight_mean_20000_arr, 
    'l41_c2_w' : layer4_1_conv2_weight_mean_20000_arr, 
    'bn1_w' : bn1_weight_mean_20000_arr,
    'bn1_b' : bn1_bias_mean_20000_arr,
    'l10_c1_w' : layer1_0_conv1_weight_mean_20000_arr,
    'test_acc': y_arr
    }) 
df_mean.to_csv("df_mean.csv")

df_mean_no_outlier = df_mean.drop(df_mean[df_mean.test_acc < 0.05].index)
df_mean_no_outlier.to_csv("df_mean_no_outlier.csv")
"""
"""
# Dataframes for std
df_std_mean = pd.DataFrame({
    'P0_fc_w': fc_weight_std_mean_arr[0],
    'P1_fc_w': fc_weight_std_mean_arr[1],
    'P2_fc_w': fc_weight_std_mean_arr[2],
    'P3_fc_w': fc_weight_std_mean_arr[3],
    'P0_l41_bn2_w': layer4_1_bn2_weight_std_mean_arr[0],
    'P1_l41_bn2_w': layer4_1_bn2_weight_std_mean_arr[1],
    'P2_l41_bn2_w': layer4_1_bn2_weight_std_mean_arr[2],
    'P3_l41_bn2_w': layer4_1_bn2_weight_std_mean_arr[3],
    'P0_l41_cn2_w': layer4_1_conv2_weight_std_mean_arr[0],
    'P1_l41_cn2_w': layer4_1_conv2_weight_std_mean_arr[1],
    'P2_l41_cn2_w': layer4_1_conv2_weight_std_mean_arr[2],
    'P3_l41_cn2_w': layer4_1_conv2_weight_std_mean_arr[3],
    'P0_bn1_w': bn1_weight_std_mean_arr[0], 
    'P1_bn1_w': bn1_weight_std_mean_arr[1], 
    'P2_bn1_w': bn1_weight_std_mean_arr[2], 
    'P3_bn1_w': bn1_weight_std_mean_arr[3], 
    'P0_bn1_b': bn1_bias_std_mean_arr[0], 
    'P1_bn1_b': bn1_bias_std_mean_arr[1], 
    'P2_bn1_b': bn1_bias_std_mean_arr[2], 
    'P3_bn1_b': bn1_bias_std_mean_arr[3], 
    'P0_l10_cn1': layer1_0_conv1_weight_std_mean_arr[0], 
    'P1_l10_cn1': layer1_0_conv1_weight_std_mean_arr[1], 
    'P2_l10_cn1': layer1_0_conv1_weight_std_mean_arr[2], 
    'P3_l10_cn1': layer1_0_conv1_weight_std_mean_arr[3], 
    'test_acc': y_arr
})
df_std_mean_no_outlier = df_std_mean.drop(df_std_mean[df_std_mean.test_acc < 0.05].index)
df_std_grad_norm = pd.DataFrame({ 
    'P0_fc_w': fc_weight_std_grad_norm_arr[0], 
    'P1_fc_w': fc_weight_std_grad_norm_arr[1], 
    'P2_fc_w': fc_weight_std_grad_norm_arr[2], 
    'P3_fc_w': fc_weight_std_grad_norm_arr[3], 
    'P0_l41_bn2_w': layer4_1_bn2_weight_std_grad_norm_arr[0], 
    'P1_l41_bn2_w': layer4_1_bn2_weight_std_grad_norm_arr[1], 
    'P2_l41_bn2_w': layer4_1_bn2_weight_std_grad_norm_arr[2], 
    'P3_l41_bn2_w': layer4_1_bn2_weight_std_grad_norm_arr[3], 
    'P0_l41_cn2_w': layer4_1_conv2_weight_std_grad_norm_arr[0], 
    'P1_l41_cn2_w': layer4_1_conv2_weight_std_grad_norm_arr[1], 
    'P2_l41_cn2_w': layer4_1_conv2_weight_std_grad_norm_arr[2], 
    'P3_l41_cn2_w': layer4_1_conv2_weight_std_grad_norm_arr[3], 
    'P0_bn1_w': bn1_weight_std_grad_norm_arr[0], 
    'P1_bn1_w': bn1_weight_std_grad_norm_arr[1], 
    'P2_bn1_w': bn1_weight_std_grad_norm_arr[2], 
    'P3_bn1_w': bn1_weight_std_grad_norm_arr[3], 
    'P0_bn1_b': bn1_bias_std_grad_norm_arr[0], 
    'P1_bn1_b': bn1_bias_std_grad_norm_arr[1], 
    'P2_bn1_b': bn1_bias_std_grad_norm_arr[2], 
    'P3_bn1_b': bn1_bias_std_grad_norm_arr[3], 
    'P0_l10_cn1': layer1_0_conv1_weight_std_grad_norm_arr[0], 
    'P1_l10_cn1': layer1_0_conv1_weight_std_grad_norm_arr[1], 
    'P2_l10_cn1': layer1_0_conv1_weight_std_grad_norm_arr[2], 
    'P3_l10_cn1': layer1_0_conv1_weight_std_grad_norm_arr[3], 
    'test_acc': y_arr
})
df_std_grad_norm_no_outlier = df_std_grad_norm.drop(df_std_grad_norm[df_std_grad_norm.test_acc < 0.05].index)
df_std_grad_mean = pd.DataFrame({ 
    'P0_fc_w': fc_weight_std_grad_mean_arr[0], 
    'P1_fc_w': fc_weight_std_grad_mean_arr[1], 
    'P2_fc_w': fc_weight_std_grad_mean_arr[2], 
    'P3_fc_w': fc_weight_std_grad_mean_arr[3], 
    '0_l41_bn2_w': layer4_1_bn2_weight_std_grad_mean_arr[0], 
    'P1_l41_bn2_w': layer4_1_bn2_weight_std_grad_mean_arr[1], 
    'P2_l41_bn2_w': layer4_1_bn2_weight_std_grad_mean_arr[2], 
    'P3_l41_bn2_w': layer4_1_bn2_weight_std_grad_mean_arr[3], 
    'P0_l41_cn2_w': layer4_1_conv2_weight_std_grad_mean_arr[0], 
    'P1_l41_cn2_w': layer4_1_conv2_weight_std_grad_mean_arr[1], 
    'P2_l41_cn2_w': layer4_1_conv2_weight_std_grad_mean_arr[2], 
    'P3_l41_cn2_w': layer4_1_conv2_weight_std_grad_mean_arr[3], 
    'P0_bn1_w': bn1_weight_std_grad_mean_arr[0], 
    'P1_bn1_w': bn1_weight_std_grad_mean_arr[1], 
    'P2_bn1_w': bn1_weight_std_grad_mean_arr[2], 
    'P3_bn1_w': bn1_weight_std_grad_mean_arr[3], 
    'P0_bn1_b': bn1_bias_std_grad_mean_arr[0], 
    'P1_bn1_b': bn1_bias_std_grad_mean_arr[1], 
    'P2_bn1_b': bn1_bias_std_grad_mean_arr[2], 
    'P3_bn1_b': bn1_bias_std_grad_mean_arr[3], 
    'P0_l10_cn1': layer1_0_conv1_weight_std_grad_mean_arr[0], 
    'P1_l10_cn1': layer1_0_conv1_weight_std_grad_mean_arr[1], 
    'P2_l10_cn1': layer1_0_conv1_weight_std_grad_mean_arr[2], 
    'P3_l10_cn1': layer1_0_conv1_weight_std_grad_mean_arr[3], 
    'test_acc': y_arr
})
df_std_grad_mean_no_outlier = df_std_grad_mean.drop(df_std_grad_mean[df_std_grad_mean.test_acc < 0.05].index)
"""
# Dataframes for mean
"""
df_mean_mean = pd.DataFrame({
    'P0_fc_w': fc_weight_mean_mean_arr[0],
    'P1_fc_w': fc_weight_mean_mean_arr[1],
    'P2_fc_w': fc_weight_mean_mean_arr[2],
    'P3_fc_w': fc_weight_mean_mean_arr[3],
    'P0_l41_bn2_w': layer4_1_bn2_weight_mean_mean_arr[0],
    'P1_l41_bn2_w': layer4_1_bn2_weight_mean_mean_arr[1],
    'P2_l41_bn2_w': layer4_1_bn2_weight_mean_mean_arr[2],
    'P3_l41_bn2_w': layer4_1_bn2_weight_mean_mean_arr[3],
    'P0_l41_cn2_w': layer4_1_conv2_weight_mean_mean_arr[0],
    'P1_l41_cn2_w': layer4_1_conv2_weight_mean_mean_arr[1],
    'P2_l41_cn2_w': layer4_1_conv2_weight_mean_mean_arr[2],
    'P3_l41_cn2_w': layer4_1_conv2_weight_mean_mean_arr[3],
    'P0_bn1_w': bn1_weight_mean_mean_arr[0], 
    'P1_bn1_w': bn1_weight_mean_mean_arr[1], 
    'P2_bn1_w': bn1_weight_mean_mean_arr[2], 
    'P3_bn1_w': bn1_weight_mean_mean_arr[3], 
    'P0_bn1_b': bn1_bias_mean_mean_arr[0], 
    'P1_bn1_b': bn1_bias_mean_mean_arr[1], 
    'P2_bn1_b': bn1_bias_mean_mean_arr[2], 
    'P3_bn1_b': bn1_bias_mean_mean_arr[3], 
    'P0_l10_cn1': layer1_0_conv1_weight_mean_mean_arr[0], 
    'P1_l10_cn1': layer1_0_conv1_weight_mean_mean_arr[1], 
    'P2_l10_cn1': layer1_0_conv1_weight_mean_mean_arr[2], 
    'P3_l10_cn1': layer1_0_conv1_weight_mean_mean_arr[3], 
    'test_acc': y_arr
})
df_mean_mean_no_outlier = df_mean_mean.drop(df_mean_mean[df_mean_mean.test_acc < 0.05].index)
df_mean_grad_norm = pd.DataFrame({ 
    'P0_fc_w': fc_weight_mean_grad_norm_arr[0], 
    'P1_fc_w': fc_weight_mean_grad_norm_arr[1], 
    'P2_fc_w': fc_weight_mean_grad_norm_arr[2], 
    'P3_fc_w': fc_weight_mean_grad_norm_arr[3], 
    'P0_l41_bn2_w': layer4_1_bn2_weight_mean_grad_norm_arr[0], 
    'P1_l41_bn2_w': layer4_1_bn2_weight_mean_grad_norm_arr[1], 
    'P2_l41_bn2_w': layer4_1_bn2_weight_mean_grad_norm_arr[2], 
    'P3_l41_bn2_w': layer4_1_bn2_weight_mean_grad_norm_arr[3], 
    'P0_l41_cn2_w': layer4_1_conv2_weight_mean_grad_norm_arr[0], 
    'P1_l41_cn2_w': layer4_1_conv2_weight_mean_grad_norm_arr[1], 
    'P2_l41_cn2_w': layer4_1_conv2_weight_mean_grad_norm_arr[2], 
    'P3_l41_cn2_w': layer4_1_conv2_weight_mean_grad_norm_arr[3], 
    'P0_bn1_w': bn1_weight_mean_grad_norm_arr[0], 
    'P1_bn1_w': bn1_weight_mean_grad_norm_arr[1], 
    'P2_bn1_w': bn1_weight_mean_grad_norm_arr[2], 
    'P3_bn1_w': bn1_weight_mean_grad_norm_arr[3], 
    'P0_bn1_b': bn1_bias_mean_grad_norm_arr[0], 
    'P1_bn1_b': bn1_bias_mean_grad_norm_arr[1], 
    'P2_bn1_b': bn1_bias_mean_grad_norm_arr[2], 
    'P3_bn1_b': bn1_bias_mean_grad_norm_arr[3], 
    'P0_l10_cn1': layer1_0_conv1_weight_mean_grad_norm_arr[0], 
    'P1_l10_cn1': layer1_0_conv1_weight_mean_grad_norm_arr[1], 
    'P2_l10_cn1': layer1_0_conv1_weight_mean_grad_norm_arr[2], 
    'P3_l10_cn1': layer1_0_conv1_weight_mean_grad_norm_arr[3], 
    'test_acc': y_arr
})
df_mean_grad_norm_no_outlier = df_mean_grad_norm.drop(df_mean_grad_norm[df_mean_grad_norm.test_acc < 0.05].index)
df_mean_grad_mean = pd.DataFrame({ 
    'P0_fc_w': fc_weight_mean_grad_mean_arr[0], 
    'P1_fc_w': fc_weight_mean_grad_mean_arr[1], 
    'P2_fc_w': fc_weight_mean_grad_mean_arr[2], 
    'P3_fc_w': fc_weight_mean_grad_mean_arr[3], 
    '0_l41_bn2_w': layer4_1_bn2_weight_mean_grad_mean_arr[0], 
    'P1_l41_bn2_w': layer4_1_bn2_weight_mean_grad_mean_arr[1], 
    'P2_l41_bn2_w': layer4_1_bn2_weight_mean_grad_mean_arr[2], 
    'P3_l41_bn2_w': layer4_1_bn2_weight_mean_grad_mean_arr[3], 
    'P0_l41_cn2_w': layer4_1_conv2_weight_mean_grad_mean_arr[0], 
    'P1_l41_cn2_w': layer4_1_conv2_weight_mean_grad_mean_arr[1], 
    'P2_l41_cn2_w': layer4_1_conv2_weight_mean_grad_mean_arr[2], 
    'P3_l41_cn2_w': layer4_1_conv2_weight_mean_grad_mean_arr[3], 
    'P0_bn1_w': bn1_weight_mean_grad_mean_arr[0], 
    'P1_bn1_w': bn1_weight_mean_grad_mean_arr[1], 
    'P2_bn1_w': bn1_weight_mean_grad_mean_arr[2], 
    'P3_bn1_w': bn1_weight_mean_grad_mean_arr[3], 
    'P0_bn1_b': bn1_bias_mean_grad_mean_arr[0], 
    'P1_bn1_b': bn1_bias_mean_grad_mean_arr[1], 
    'P2_bn1_b': bn1_bias_mean_grad_mean_arr[2], 
    'P3_bn1_b': bn1_bias_mean_grad_mean_arr[3], 
    'P0_l10_cn1': layer1_0_conv1_weight_mean_grad_mean_arr[0], 
    'P1_l10_cn1': layer1_0_conv1_weight_mean_grad_mean_arr[1], 
    'P2_l10_cn1': layer1_0_conv1_weight_mean_grad_mean_arr[2], 
    'P3_l10_cn1': layer1_0_conv1_weight_mean_grad_mean_arr[3], 
    'test_acc': y_arr
})

df_mean_grad_mean_no_outlier = df_mean_grad_mean.drop(df_mean_grad_mean[df_mean_grad_mean.test_acc < 0.05].index)
"""
"""
corr_1 = df_heat_fc_weight.corr()
heatmap_1 = sns.heatmap(corr_1, cmap = "viridis", annot = True)
fig1 = heatmap_1.get_figure()
fig1.savefig("heatmap_fc_weight.png")

corr_2 = df_heat_layer4_1_bn2_weight.corr()
heatmap_2 = sns.heatmap(corr_2, cmap = "viridis", annot = True)
fig2 = heatmap_2.get_figure()
fig2.savefig("heat_layer4_1_bn2_weight.png")

corr_3 = df_heat_layer4_1_conv2_weight.corr()
heatmap_3 = sns.heatmap(corr_3, cmap = "viridis", annot = True)
fig3 = heatmap_3.get_figure()
fig3.savefig("heat_layer4_1_conv2_weight.png")

corr_4 = df_heat_bn1_weight.corr()
heatmap_4 = sns.heatmap(corr_4, cmap = "viridis", annot = True)
fig4 = heatmap_4.get_figure()
fig4.savefig("heat_bn1_weight.png")

corr_5 = df_heat_layer1_0_conv1_weight.corr()
heatmap_5 = sns.heatmap(corr_5, cmap = "viridis", annot = True)
fig5 = heatmap_5.get_figure()
fig5.savefig("heat_layer1_0_conv1_weight.png")
"""
# Std heatmap and scatter for last timestep
"""
corr_1 = df_std_no_outlier.corr()
heatmap_1 = sns.heatmap(corr_1, cmap = "viridis", annot = True)
fig1 = heatmap_1.get_figure()
fig1.savefig("heatmap_std_no_outlier.png")

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
axs[0, 0].scatter(df_std_no_outlier.fc_w, df_std_no_outlier.test_acc)
axs[0, 0].set_xlabel("fc weight std")
axs[0, 0].set_ylabel("y")
axs[0, 1].scatter(df_std_no_outlier.l41_c2_w, df_std_no_outlier.test_acc)
axs[0, 1].set_xlabel("4.1 conv2 weight std")
axs[0, 1].set_ylabel("y")
axs[0, 2].scatter(df_std_no_outlier.bn1_w, df_std_no_outlier.test_acc)
axs[0, 2].set_xlabel("bn1 weight std")
axs[0, 2].set_ylabel("y")
axs[1, 0].scatter(df_std_no_outlier.bn1_b, df_std_no_outlier.test_acc)
axs[1, 0].set_xlabel("bn1 bias std")
axs[1, 0].set_ylabel("y")
axs[1, 1].scatter(df_std_no_outlier.l41_bn2_w, df_std_no_outlier.test_acc)
axs[1, 1].set_xlabel("4.1 bn2 weight std")
axs[1, 1].set_ylabel("y")
axs[1, 2].scatter(df_std_no_outlier.l10_c1_w, df_std_no_outlier.test_acc)
axs[1, 2].set_xlabel("1.0 conv1 weight std")
axs[1, 2].set_ylabel("y")
plt.savefig("scatter_std_no_outlier.png")
"""
"""
corr_1 = df_mean_no_outlier.corr()
heatmap_1 = sns.heatmap(corr_1, cmap = "viridis", annot = True)
fig1 = heatmap_1.get_figure()
fig1.savefig("heatmap_mean_no_outlier.png")
"""
# std heatmap
"""
plt.subplots(figsize=(12,10))
corr_1 = df_std_mean_no_outlier.corr()
sns.heatmap(corr_1, cmap = "viridis", annot = True)
plt.tight_layout()
plt.savefig("heatmap_std_mean_no_outlier.png")
"""
"""
# heatmap for std gradient norm with no outlier
plt.subplots(figsize=(12,10))
corr_1 = df_std_grad_norm_no_outlier.corr()
sns.heatmap(corr_1, cmap = "viridis", annot = True)
plt.tight_layout()
plt.savefig("heatmap_std_grad_norm_no_outlier.png")
"""
"""
plt.subplots(figsize=(12,10))
corr_1 = df_std_grad_mean_no_outlier.corr()
sns.heatmap(corr_1, cmap = "viridis", annot = True)
plt.tight_layout()
plt.savefig("heatmap_std_grad_mean_no_outlier.png")
"""
# Scatter of best features from std according to correlation matrix
"""
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
axs[0, 0].scatter(df_std_grad_norm_no_outlier.P0_l41_bn2_w, df_std_grad_norm_no_outlier.test_acc)
axs[0, 0].set_xlabel("0_l4_1_bn2_w")
axs[0, 0].set_ylabel("y")
axs[0, 1].scatter(df_std_grad_norm_no_outlier.P1_bn1_w, df_std_grad_norm_no_outlier.test_acc)
axs[0, 1].set_xlabel("1_bn1_w")
axs[0, 1].set_ylabel("y")
axs[0, 2].scatter(df_std_grad_norm_no_outlier.P2_bn1_w, df_std_grad_norm_no_outlier.test_acc)
axs[0, 2].set_xlabel("2_bn1_w")
axs[0, 2].set_ylabel("y")
axs[1, 0].scatter(df_std_grad_norm_no_outlier.P3_bn1_w, df_std_grad_norm_no_outlier.test_acc)
axs[1, 0].set_xlabel("3_bn1_w")
axs[1, 0].set_ylabel("y")
axs[1, 1].scatter(df_std_grad_norm_no_outlier.P0_bn1_b, df_std_grad_norm_no_outlier.test_acc)
axs[1, 1].set_xlabel("0_bn1_b")
axs[1, 1].set_ylabel("y")
axs[1, 2].scatter(df_std_grad_norm_no_outlier.P1_bn1_b, df_std_grad_norm_no_outlier.test_acc)
axs[1, 2].set_xlabel("1_bn1_b")
axs[1, 2].set_ylabel("y")
plt.savefig("scatter_std_grad_norm_no_outlier.png")
"""

# mean heatmaps
"""
plt.subplots(figsize=(12,10))
corr_1 = df_mean_mean_no_outlier.corr()
sns.heatmap(corr_1, cmap = "viridis", annot = True)
plt.tight_layout()
plt.savefig("heatmap_mean_mean_no_outlier.png")
"""
"""
# heatmap for mean gradient norm with no outlier
plt.subplots(figsize=(12,10))
corr_1 = df_mean_grad_norm_no_outlier.corr()
sns.heatmap(corr_1, cmap = "viridis", annot = True)
plt.tight_layout()
plt.savefig("heatmap_mean_grad_norm_no_outlier.png")
"""
"""
plt.subplots(figsize=(12,10))
corr_1 = df_mean_grad_mean_no_outlier.corr()
sns.heatmap(corr_1, cmap = "viridis", annot = True)
plt.tight_layout()
plt.savefig("heatmap_mean_grad_mean_no_outlier.png")
"""
#print(event.Tags())
# Show all tags in the log file
