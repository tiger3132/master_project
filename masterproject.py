import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from typing import List, Optional, Type, Callable
from torch import Tensor
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as data
import math
from torch.utils.data import DataLoader
import pathlib, argparse
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_cpr import apply_CPR

class Block(nn.Module):
    """Block layer module

    Args:

    """
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample: Optional[nn.Module]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(num_features=out_channels)
        self.downsample=downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x                         # residual
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

def accuracy(predicted, labels):
    _, predicted = torch.max(predicted, 1)
    return torch.tensor((predicted == labels).sum().item() / labels.size(0)).to(torch.float32)

class ResNet18(pl.LightningModule):
    def __init__(self, block: Type[Block], layers: List[int], config):
        super().__init__()

        self.cfg = config

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=3, padding=1, bias=False)
        self._norm_layer = nn.BatchNorm2d
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 100)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """


    def _make_layer(self, block: Type[Block], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:

        downsample = None
        norm_layer = self._norm_layer

        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample, norm_layer)
        )
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        if self.cfg.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def training_step(self, batch):
        input, label = batch
        prediction = self.forward(input)
        train_loss = F.cross_entropy(prediction, label, label_smoothing=0.1)
        #self.log("conv1.weight.mean", self.conv1.weight.mean())
        """
        for param in self.parameters():
            print(param)
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.log(f"param/{name}/max", param.max())
                self.log(f"param/{name}/mean", param.mean())
                self.log(f"param/{name}/std", param.std())
                self.log(f"param/{name}/min", param.min())
            """
            if param.grad is not None:
                self.log(f"param/{name}/grad/max", param.grad.max())
                self.log(f"param/{name}/grad/mean", param.grad.mean())
                self.log(f"param/{name}/grad/min", param.grad.min())
            """
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        input, label = batch
        prediction = self.forward(input)
        valid_loss = F.cross_entropy(prediction, label)
        valid_accuracy = accuracy(prediction, label)
        values = {"validation_loss": valid_loss, "validation_accuracy": valid_accuracy}
        self.log_dict(values)

    def test_step(self, batch, batch_idx):
        input, label = batch
        prediction = self.forward(input)
        test_loss = F.cross_entropy(prediction, label)
        test_accuracy = accuracy(prediction, label)
        values = {"test_loss": test_loss, "test_accuracy": test_accuracy}
        self.log_dict(values)


    def configure_optimizers(self):
        if self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == "adamcpr":
	        optimizer = apply_CPR(self, torch.optim.Adam, self.cfg.kappa_init_param, self.cfg.kappa_init_method,
                    self.cfg.reg_function, 
                    self.cfg.kappa_adapt, self.cfg.kappa_update, self.cfg.apply_lr,
                    embedding_regularization=True,
                    lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        def lr_lambda(current_step: int):
            if current_step < self.cfg.lr_warmup_steps:
                return float(current_step) / float(max(1, self.cfg.lr_warmup_steps)) # Warmup step
            elif self.cfg.wd_schedule_type == "cosine":
                decay_steps = self.cfg.max_train_steps - self.cfg.lr_warmup_steps
                step = current_step - self.cfg.lr_warmup_steps
                cosine_decay = max(0.0, 0.5 * (1 +  math.cos(math.pi * step/float(max(1, decay_steps)))))
                decayed = (1 - self.cfg.lr_decay_factor) * cosine_decay + self.cfg.lr_decay_factor
                return decayed
            elif self.cfg.wd_schedule_type == "linear":
                decay_steps = self.cfg.max_train_steps - self.cfg.lr_warmup_steps
                linear_decay = max(0.0, float(self.cfg.max_train_steps - self.cfg.lr_warmup_steps - current_step)/float(max(1, decay_steps)))
                decayed = (1 - self.cfg.lr_decay_factor) * linear_decay + self.cfg.lr_decay_factor
                return decayed
            elif self.cfg.wd_schedule_type == "constant":
                return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, 'interval': 'step'}}



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--session", type=str, default='test_resnet')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="ResNet18")
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=0.001)

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

    parser.add_argument("--wd_schedule_type", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--lr_decay_factor", type=float, default=0.1)

    parser.add_argument("--data_transform", type=int, default=1)
    parser.add_argument("--batch_norm", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default='cifar100')

    parser.add_argument("--test_run", type=bool, default=False)

    args = parser.parse_args()

    args.kappa_adapt = args.kappa_adapt == 1
    args.apply_lr = args.apply_lr == 1
    args.data_transform = args.data_transform == 1
    args.batch_norm = args.batch_norm == 1

    task_name = f"{args.model_name}_seed{args.seed}_steps{args.max_train_steps}"
    expt_dir = pathlib.Path(args.output_dir) / args.session / task_name
    expt_dir.mkdir(parents=True, exist_ok=True)
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

    print(expt_name)
    print(task_name)


    logger = TensorBoardLogger(save_dir=expt_dir, name=expt_name)


    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], inplace=True)

    if args.data_transform:
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                      transforms.ToTensor(), normalize,])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(),])
        
    train_test = transforms.ToTensor()
    train_set = datasets.CIFAR100(root="CIFAR100", download=True, train=True, transform=transform_train)
    # use 10% of training data for validation
    train_set_size = int(len(train_set) * 0.9)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    train_dataload = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, persistent_workers=True, shuffle=True)
    val_dataload = DataLoader(valid_set, batch_size=args.batch_size, num_workers=2, persistent_workers=True)

    resnet = ResNet18(Block, [2, 2, 2, 2], args)

    trainer = pl.Trainer(enable_progress_bar=True, devices=1, accelerator="gpu", max_steps=args.max_train_steps, logger=logger, fast_dev_run=args.test_run)

    trainer.fit(model=resnet, train_dataloaders=train_dataload, val_dataloaders=val_dataload)
    if args.data_transform:
        transform_test = transforms.Compose([transforms.ToTensor(), normalize,])
    else:
        transform_test = transforms.Compose([transforms.ToTensor(),])

    test_set = datasets.CIFAR100(root="CIFAR100", download=True, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    trainer.test(resnet, test_loader)
