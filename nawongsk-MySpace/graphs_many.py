from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pathlib

parser = ArgumentParser()
parser.add_argument("--num_of_exps", type=int, default=2)
parser.add_argument("--sessions", nargs="*", type=str, default=['test_resnet', 'test_resnet'])
parser.add_argument("--seeds", nargs="*", type=int, default=[1, 1])
parser.add_argument("--max_train_steps", nargs="*", type=int, default=[20000,20000])
parser.add_argument("--model_names", nargs="*", type=str, default=["ResNet18", "ResNet18"])
parser.add_argument("--optimizers", nargs="*", type=str, default=["adamw", "adamw"])
parser.add_argument("--momentums", nargs="*", type=float, default=[0, 0])
parser.add_argument("--lrs", nargs="*", type=float, default=[0.001, 0.001])
parser.add_argument("--weight_decays", nargs="*", type=float, default=[0.001, 0.001])
parser.add_argument("--wd_schedule_types", nargs="*", type=str, default=["cosine", "cosine"])
parser.add_argument("--output_dirs", nargs="*", type=str, default=["cifar100", "cifar100"])
parser.add_argument("--lr_warmup_steps", nargs="*", type=int, default=[200, 200])
parser.add_argument("--lr_decay_factors", nargs="*", type=float, default=[0.1, 0.1])
args = parser.parse_args()

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
events = []
for i in range(args.num_of_exps):
    task_name = f"{args.model_names[i]}_seed{args.seeds[i]}_steps{args.max_train_steps[i]}"
    if args.wd_schedule_types[i] == "cosine" or args.wd_schedule_types[i] == "linear":
        if args.optimizers[i] == "sgd":
            expt_name = f"{args.optimizers[i]}_l{args.lrs[i]}_w{args.weight_decays[i]}_t{args.wd_schedule_types[i]}_moment{args.momentums[i]}_lrwarm{args.lr_warmup_steps[i]}_lrdecay{args.lr_decay_factors[i]}"
        elif args.optimizers[i] == "adamw":
            expt_name = f"{args.optimizers[i]}_l{args.lrs[i]}_w{args.weight_decays[i]}_t{args.wd_schedule_types[i]}_lrwarm{args.lr_warmup_steps[i]}_lrdecay{args.lr_decay_factors[i]}"
        elif args.optimizers[i] == "adamcpr":
            expt_name = f"{args.optimizers[i]}_p{args.kappa_init_params[i]}_m{args.kappa_init_methods[i]}_kf{args.reg_functions[i]}_r{args.kappa_updates[i]}_l{args.lr[i]}_adapt{args.kappa_adapts[i]}_g{args.apply_lrs[i]}_lrwarm{args.lr_warmup_steps[i]}_lrdecay{args.lr_decay_factors[i]}"
    else:
        if args.optimizers[i] == "sgd":
            expt_name = f"{args.optimizers[i]}_l{args.lrs[i]}_w{args.weight_decays[i]}_t{args.wd_schedule_types[i]}_moment{args.momentums[i]}"
        elif args.optimizers[i] == "adamw":
            expt_name = f"{args.optimizers[i]}_l{args.lrs[i]}_w{args.weight_decays[i]}_t{args.wd_schedule_types[i]}"
        elif args.optimizers[i] == "adamcpr":
            expt_name = f"{args.optimizers[i]}_p{args.kappa_init_params[i]}_m{args.kappa_init_methods[i]}_kf{args.reg_functions[i]}_r{args.kappa_updates[i]}_l{args.lrs[i]}_adapt{args.kappa_adapts[i]}_g{args.apply_lrs[i]}"

    expt_dir = f"/work/dlclarge1/nawongsk-MySpace/{args.output_dirs[i]}/{args.sessions[i]}/{task_name}/{expt_name}/version_0"
    #event = EventAccumulator(f'/work/dlclarge1/nawongsk-MySpace/{args.output_dir}/{args.session}/{args.model_name}_seed{args.seed}_steps{args.max_train_steps}/{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}/version_2')
    event = EventAccumulator(expt_dir)
    event.Reload()
    events.append(event)

#print(event.Tags())
# Show all tags in the log file

param_name = ["conv1", "bn1", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2", "layer1.1.conv1", "layer1.1.bn1", "layer1.1.conv2", "layer1.1.bn2",
                "layer2.0.conv1", "layer2.0.bn1", "layer2.0.conv2", "layer2.0.bn2", "layer2.0.downsample.0", "layer2.0.downsample.1", "layer2.1.conv1", "layer2.1.bn1",
                "layer2.1.conv2", "layer2.1.bn2", "layer3.0.conv1", "layer3.0.bn1", "layer3.0.conv2", "layer3.0.bn2", "layer3.0.downsample.0", "layer3.0.downsample.1",
                "layer3.1.conv1", "layer3.1.bn1", "layer3.1.conv2", "layer3.1.bn2", "layer4.0.conv1", "layer4.0.bn1", "layer4.0.conv2", "layer4.0.bn2", "layer4.0.downsample.0",
                "layer4.0.downsample.1", "layer4.1.conv1", "layer4.1.bn1", "layer4.1.conv2", "layer4.1.bn2", "fc"]

# print(event.Tags()['scalars'])

fig = plt.figure(figsize=(15, 75), layout="constrained")
colors = plt.colormaps["RdYlGn"](np.linspace(0, 1, args.num_of_exps))
# fig, ax = plt.subplots(figsize=(15, 75), layout="constrained")
gs = fig.add_gridspec(len(param_name), 4)

for i, param in enumerate(param_name):
    diff = (((len(param_name) - i)/len(param_name)) - ((len(param_name) - i + 1)/len(param_name))) / 2

    text = fig.text(0, ((len(param_name) - i)/len(param_name)) + diff, param, horizontalalignment="right")

    if param.rfind("conv") != -1 or param.rfind("downsample.0") != -1:

        weight = fig.add_subplot(gs[i, 1:3], ylim=(-1, 2))
        for j, color in enumerate(colors):
            x = np.array([event_scalar.step for event_scalar in events[j].Scalars(f'param/{param}.weight/mean')])
            y = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/mean')])
            min = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/min')])
            std = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/std')])
            max = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/max')])
            weight.axes.set_title("Weight")
            weight.fill_between(x, y-std, y+std, alpha=0.1, color=color)
            weight.plot(x, min, '--', color=color)
            weight.plot(x, max, '--', color=color)
            weight.plot(x, y, color=color)



    elif param.rfind("bn") != -1 or param.rfind("downsample.1") != -1 or param.rfind("fc") != -1:

        weight1 = fig.add_subplot(gs[i, 0:2], ylim=(-1, 2))
        for j, color in enumerate(colors):
            x = np.array([event_scalar.step for event_scalar in events[j].Scalars(f'param/{param}.weight/mean')])
            y = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/mean')])
            min = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/min')])
            std = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/std')])
            max = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.weight/max')])
            weight1.axes.set_title("Weight")
            weight1.fill_between(x, y-std, y+std, alpha=0.1, color=color)
            weight1.plot(x, min, '--', color=color)
            weight1.plot(x, max, '--', color=color)
            weight1.plot(x, y, color=color)

        bias1 = fig.add_subplot(gs[i, 2:], ylim=(-1, 2))
        for j, color in enumerate(colors):
            x = np.array([event_scalar.step for event_scalar in events[j].Scalars(f'param/{param}.bias/mean')])
            y = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.bias/mean')])
            min = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.bias/min')])
            std = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.bias/std')])
            max = np.array([event_scalar.value for event_scalar in events[j].Scalars(f'param/{param}.bias/max')])
            bias1.axes.set_title("Bias")
            bias1.fill_between(x, y-std, y+std, alpha=0.1, color=color)
            bias1.plot(x, min, '--', color=color)
            bias1.plot(x, max, '--', color=color)
            bias1.plot(x, y, color=color)

"""
task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"
expt_dir = pathlib.Path("hparam_graphs") / args.output_dir / args.session / task_name
expt_dir.mkdir(parents=True, exist_ok=True)
base_dir = f"hparam_graphs/{args.output_dir}/{args.session}"
if args.wd_schedule_type == "cosine" or args.wd_schedule_type == "linear":
    if args.optimizer == "sgd":
        expt_name = base_dir + "/" + task_name + "/" + f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}.png"
    elif args.optimizer == "adamw":
        expt_name = base_dir + "/" + task_name + "/" + f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}.png"
    elif args.optimizer == "adamcpr":
        expt_name = base_dir + "/" + task_name + "/" + f"{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}_lrwarm{args.lr_warmup_steps}_lrdecay{args.lr_decay_factor}.png"
else:
    if args.optimizer == "sgd":
        expt_name = base_dir + "/" + task_name + "/" + f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}_moment{args.momentum}.png"
    elif args.optimizer == "adamw":
        expt_name = base_dir + "/" + task_name + "/" + f"{args.optimizer}_l{args.lr}_w{args.weight_decay}_t{args.wd_schedule_type}.png"
    elif args.optimizer == "adamcpr":
        expt_name = base_dir + "/" + task_name + "/" + f"{args.optimizer}_p{args.kappa_init_param}_m{args.kappa_init_method}_kf{args.reg_function}_r{args.kappa_update}_l{args.lr}_adapt{args.kappa_adapt}_g{args.apply_lr}.png"
"""

plt.savefig("1 + 2", bbox_inches="tight")

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
