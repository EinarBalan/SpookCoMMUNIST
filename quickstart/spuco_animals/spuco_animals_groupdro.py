import argparse
from copy import deepcopy
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from spuco.datasets import GroupLabeledDatasetWrapper, SpuCoAnimals, SpuCoDogs, SpuCoAnimals
from spuco.evaluate import Evaluator
from spuco.robust_train import GroupDRO
from spuco.models import model_factory
from spuco.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="results/spucoanimals_groupdro.csv")
parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "cliprn50"])
parser.add_argument("--only_train_projection", action="store_true", help="only train projection, applicable only for cliprn50")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

trainset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=transform,
)
trainset.initialize()

valset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="val",
    transform=transform,
)
valset.initialize()

testset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="test",
    transform=transform,
)
testset.initialize()

robust_trainset = GroupLabeledDatasetWrapper(trainset, trainset.group_partition)

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
if args.arch == "cliprn50" and args.only_train_projection:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone._modules['attnpool'].parameters():
        param.requires_grad = True
        
valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
group_dro = GroupDRO(
    model=model,
    val_evaluator=valid_evaluator,
    num_epochs=args.num_epochs,
    trainset=robust_trainset,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)

for epoch in range(group_dro.num_epochs):
    group_dro.train_epoch(epoch)
    if group_dro.val_evaluator is not None:
        group_dro.val_evaluator.evaluate()
        if group_dro.val_evaluator.worst_group_accuracy[1] > group_dro._best_wg_acc:
            group_dro._best_wg_acc = group_dro.val_evaluator.worst_group_accuracy[1]
            group_dro._best_model = deepcopy(group_dro.trainer.model)
            group_dro._best_epoch = epoch
        if group_dro.verbose:
            print('Epoch {}: Val Worst-Group Accuracy: {}'.format(epoch, group_dro.val_evaluator.worst_group_accuracy[1]))
            print('Best Val Worst-Group Accuracy: {}'.format(group_dro._best_wg_acc))
            print(group_dro.group_weighted_loss.group_weights.data)

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results = pd.DataFrame(index=[0])
results["timestamp"] = pd.Timestamp.now()
results["seed"] = args.seed
results["pretrained"] = args.pretrained
results["arch"] = args.arch
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=group_dro.best_model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["early_stopping_average_accuracy"] = evaluator.average_accuracy

print(results)

if os.path.exists(args.results_csv):
    results_df = pd.read_csv(args.results_csv)
else:
    results_df = pd.DataFrame()

results_df = pd.concat([results_df, results], ignore_index=True)
results_df.to_csv(args.results_csv, index=False)

print('Done!')
print('Results saved to', args.results_csv)