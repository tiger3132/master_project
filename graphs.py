from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser
import pathlib

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

args.kappa_adapt = args.kappa_adapt == 1
args.apply_lr = args.apply_lr == 1
args.data_transform = args.data_transform == 1
args.batch_norm = args.batch_norm == 1

"""
if args.wd_schedule_type == "cosine" or args.wd_schedule_type == "linear":
    if args.optimizer == "sgd":
        event = EventAccumulator(f'/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}/version_0')
    else:
        event = EventAccumulator(f'/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}/version_0')
else:
    if args.optimizer == "sgd":
        event = EventAccumulator(f'/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}/version_0')
    else:
        event = EventAccumulator(f'/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}/version_0')
"""
task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"
#expt_dir.mkdir(parents=True, exist_ok=True)
"""
if args.wd_schedule_type == "cosine" or args.wd_schedule_type == "linear":
    if args.optimizer == "sgd":
        expt_name = f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}"
    elif args.optimizer == "adamw":
        expt_name = f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}"
    elif args.optimizer == "adamcpr":
        expt_name = f"{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}"
else:
    if args.optimizer == "sgd":
        expt_name = f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}"
    elif args.optimizer == "adamw":
        expt_name = f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}"
    elif args.optimizer == "adamcpr":
        expt_name = f"{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}"
"""


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


expt_dir = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{task_name}/{expt_name}/version_0"
#event = EventAccumulator(f'/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}/version_2')
event = EventAccumulator(expt_dir)
event.Reload()

#print(event.Tags())
# Show all tags in the log file

param_name = ["conv1", "bn1", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2", "layer1.1.conv1", "layer1.1.bn1", "layer1.1.conv2", "layer1.1.bn2",
                "layer2.0.conv1", "layer2.0.bn1", "layer2.0.conv2", "layer2.0.bn2", "layer2.0.downsample.0", "layer2.0.downsample.1", "layer2.1.conv1", "layer2.1.bn1",
                "layer2.1.conv2", "layer2.1.bn2", "layer3.0.conv1", "layer3.0.bn1", "layer3.0.conv2", "layer3.0.bn2", "layer3.0.downsample.0", "layer3.0.downsample.1",
                "layer3.1.conv1", "layer3.1.bn1", "layer3.1.conv2", "layer3.1.bn2", "layer4.0.conv1", "layer4.0.bn1", "layer4.0.conv2", "layer4.0.bn2", "layer4.0.downsample.0",
                "layer4.0.downsample.1", "layer4.1.conv1", "layer4.1.bn1", "layer4.1.conv2", "layer4.1.bn2", "fc"]

# print(event.Tags()['scalars'])

fig = plt.figure(figsize=(15, 75), layout="constrained")
# fig, ax = plt.subplots(figsize=(15, 75), layout="constrained")
gs = fig.add_gridspec(len(param_name), 4)

for i, param in enumerate(param_name):
    diff = (((len(param_name) - i)/len(param_name)) - ((len(param_name) - i + 1)/len(param_name))) / 2

    text = fig.text(0, ((len(param_name) - i)/len(param_name)) + diff, param, horizontalalignment="right")

    if param.rfind("conv") != -1 or param.rfind("downsample.0") != -1:


        weight = fig.add_subplot(gs[i, 1:3], ylim=(-1, 2))
        x = np.array([event_scalar.step for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
        y = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
        min = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/min')])
        std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/std')])
        max = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/max')])
        weight.axes.set_title("Weight")
        weight.fill_between(x, y-std, y+std, alpha=0.2)
        weight.plot(x, min, '--')
        weight.plot(x, max, '--')
        weight.plot(x, y)



    elif param.rfind("bn") != -1 or param.rfind("downsample.1") != -1 or param.rfind("fc") != -1:

        weight1 = fig.add_subplot(gs[i, 0:2], ylim=(-1, 2))
        x = np.array([event_scalar.step for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
        y = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/mean')])
        min = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/min')])
        std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/std')])
        max = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.weight/max')])
        weight1.axes.set_title("Weight")
        weight1.fill_between(x, y-std, y+std, alpha=0.2)
        weight1.plot(x, min, '--')
        weight1.plot(x, max, '--')
        weight1.plot(x, y)

        bias1 = fig.add_subplot(gs[i, 2:], ylim=(-1, 2))
        x = np.array([event_scalar.step for event_scalar in event.Scalars(f'param/{param}.bias/mean')])
        y = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/mean')])
        min = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/min')])
        std = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/std')])
        max = np.array([event_scalar.value for event_scalar in event.Scalars(f'param/{param}.bias/max')])
        bias1.axes.set_title("Bias")
        bias1.fill_between(x, y-std, y+std, alpha=0.2)
        bias1.plot(x, min, '--')
        bias1.plot(x, max, '--')
        bias1.plot(x, y)


task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"
expt_dir = pathlib.Path("hparam_graphs") / args.output_dir / args.session / task_name
expt_dir.mkdir(parents=True, exist_ok=True)
base_dir = f"hparam_graphs/{args.output_dir}/{args.session}"
if args.wd_schedule_type == "cosine" or args.wd_schedule_type == "linear":
    if args.optimizer == "sgd":
        expt_name = base_dir + "/" + task_name + "/" + f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}_trans{args.data_transform}_bn{args.batch_norm}.png"
    elif args.optimizer == "adamw":
        expt_name = base_dir + "/" + task_name + "/" + f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}_trans{args.data_transform}_bn{args.batch_norm}_b1{args.beta1}_b2{args.beta2}.png"
    elif args.optimizer == "adamcpr":
        expt_name = base_dir + "/" + task_name + "/" + f"b{args.batch_size}_{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}_trans{args.data_transform}_bn{args.batch_norm}_t{args.wd_schedule_type}.png"
else:
    if args.optimizer == "sgd":
        expt_name = base_dir + "/" + task_name + "/" + f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}_trans{args.data_transform}_bn{args.batch_norm}.png"
    elif args.optimizer == "adamw":
        expt_name = base_dir + "/" + task_name + "/" + f"b{args.batch_size}_{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_trans{args.data_transform}_bn{args.batch_norm}_b1{args.beta1}_b2{args.beta2}.png"
    elif args.optimizer == "adamcpr":
        expt_name = base_dir + "/" + task_name + "/" + f"b{args.batch_size}_{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}_trans{args.data_transform}_bn{args.batch_norm}_t{args.wd_schedule_type}.png"
plt.savefig(expt_name, bbox_inches="tight")
"""
# task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"
expt_dir = pathlib.Path("hparam_graphs") / args.output_dir / args.session / task_name
expt_dir.mkdir(parents=True, exist_ok=True)

if args.wd_schedule_type == "cosine" or args.wd_schedule_type == "linear":
    if args.optimizer == "sgd":
        plt.savefig(f'hparam_graphs/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_momentum{args.momentum}_lrwarm{args.lr_warmup_steps}_lrdecay_{args.lr_decay_factor}.png', bbox_inches='tight')
    else:
        plt.savefig(f'hparam_graphs/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_lrwarm{args.lr_warmup_steps}_lrdecay_{args.lr_decay_factor}.png', bbox_inches='tight')
else:
    if args.optimizer == "sgd":
        plt.savefig(f'hparam_graphs/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_momentum{args.momentum}.png', bbox_inches='tight')
    else:
        plt.savefig(f'hparam_graphs/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}.png', bbox_inches='tight')
"""
