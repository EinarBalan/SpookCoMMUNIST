from datetime import datetime
import argparse
import os
import sys
import wandb

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from spuco.datasets import GroupLabeledDatasetWrapper, UrbanCars, UrbanCarsSpuriousLabel
from spuco.evaluate import Evaluator
from spuco.robust_train import PDE
from spuco.models import model_factory
from spuco.utils import set_seed


def main(args):
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=args)
        # remove the stdout_file argument
        del args.stdout_file
        del args.results_csv
    else:
        # check if the stdout file already exists, and if want to overwrite it
        DT_STRING = "".join(str(datetime.now()).split())
        args.stdout_file = f"{args.stdout_file}"
        if os.path.exists(args.stdout_file):
            print(f"stdout file {args.stdout_file} already exists, overwrite? (y/n)")
            response = input()
            if response != "y":
                sys.exit()
        os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
        # redirect stdout to a file
        sys.stdout = open(args.stdout_file, "w")

    print(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    trainset=UrbanCars(root=args.root_dir, split="train", spurious_label_type=UrbanCarsSpuriousLabel.BOTH, verbose=True, transform=train_transform)
    trainset.initialize()

    print(f'Using {args.spurious_label_type} spurious labels for validation')
    valset=UrbanCars(root=args.root_dir, split="val", spurious_label_type=args.spurious_label_type, verbose=True, transform=transform)
    valset.initialize()

    testset=UrbanCars(root=args.root_dir, split="test", spurious_label_type=UrbanCarsSpuriousLabel.BOTH, verbose=True, transform=transform)
    testset.initialize()
    
    robust_trainset = GroupLabeledDatasetWrapper(trainset, trainset.group_partition)

    # initialize the model and the trainer
    model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
    
    # initialize the group weights
    if args.spurious_label_type == UrbanCarsSpuriousLabel.BG:
        group_weights = {}
        for key in valset.group_partition.keys():
            group_weights[key] = trainset.group_weights[(key[0], (0, key[1]))] + trainset.group_weights[(key[0], (1, key[1]))]
    elif args.spurious_label_type == UrbanCarsSpuriousLabel.CO_OCCUR:
        group_weights = {}
        for key in valset.group_partition.keys():
            group_weights[key] = trainset.group_weights[(key[0], (key[1], 0))] + trainset.group_weights[(key[0], (key[1], 1))]
    else:
        group_weights = trainset.group_weights
            
    valid_evaluator = Evaluator(
        testset=valset,
        group_partition=valset.group_partition,
        group_weights=group_weights,
        batch_size=args.batch_size,
        model=model,
        device=device,
        verbose=True
    )

    pde = PDE(
        model=model,
        group_partition=trainset.group_partition,
        val_evaluator=valid_evaluator,
        num_epochs=args.num_epochs,
        trainset=trainset,
        batch_size=args.batch_size,
        optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
        device=device,
        verbose=True,
        use_wandb=args.wandb,
        warmup_epochs=args.warmup_epochs,
        expansion_size=args.expansion_size,
        expansion_interval=args.expansion_interval,
        subsample_cap=args.subsample_cap
    )

    pde.train()

    results = pd.DataFrame(index=[0])

    evaluator = Evaluator(
        testset=valset,
        group_partition=valset.group_partition,
        group_weights=group_weights,
        batch_size=args.batch_size,
        model=model,
        device=device,
        verbose=True
    )
    evaluator.evaluate()
    results[f"val_wg_acc"] = evaluator.worst_group_accuracy[1]
    results[f"val_avg_acc"] = evaluator.average_accuracy

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
    results[f"test_wg_acc"] = evaluator.worst_group_accuracy[1]
    results[f"test_avg_acc"] = evaluator.average_accuracy

    evaluator = Evaluator(
        testset=valset,
        group_partition=valset.group_partition,
        group_weights=group_weights,
        batch_size=args.batch_size,
        model=pde.best_model,
        device=device,
        verbose=True
    )
    evaluator.evaluate()
    results[f"val_early_stopping_wg_acc"] = evaluator.worst_group_accuracy[1]
    results[f"val_early_stopping_avg_acc"] = evaluator.average_accuracy

    evaluator = Evaluator(
        testset=testset,
        group_partition=testset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=args.batch_size,
        model=pde.best_model,
        device=device,
        verbose=True
    )
    evaluator.evaluate()
    results[f"test_early_stopping_wg_acc"] = evaluator.worst_group_accuracy[1]
    results[f"test_early_stopping_avg_acc"] = evaluator.average_accuracy

    print(results)

    if args.wandb:
        # convert the results to a dictionary
        results = results.to_dict(orient="records")[0]
        wandb.log(results)
    else:
        results["alg"] = "pde"
        results["timestamp"] = pd.Timestamp.now()
        args_dict = vars(args)
        for key in args_dict.keys():
            results[key] = args_dict[key]

        results

        if os.path.exists(args.results_csv):
            results_df = pd.read_csv(args.results_csv)
        else:
            results_df = pd.DataFrame()

        results_df = pd.concat([results_df, results], ignore_index=True)
        results_df.to_csv(args.results_csv, index=False)

        print('Results saved to', args.results_csv)

        # close the stdout file
        sys.stdout.close()

        # restore stdout
        sys.stdout = sys.__stdout__

    print('Done!')

if __name__ == '__main__':
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--spurious_label_type", type=UrbanCarsSpuriousLabel, choices=list(UrbanCarsSpuriousLabel), default=UrbanCarsSpuriousLabel.BOTH)
    parser.add_argument("--label_noise", type=float, default=0.0)
    parser.add_argument("--results_csv", type=str, default="results/urbancars_pde.csv")
    parser.add_argument("--stdout_file", type=str, default="urbancars_pde.out")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50", "cliprn50"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="spuco")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="urbancars_pde")

    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--expansion_size", type=int, default=10)
    parser.add_argument("--expansion_interval", type=int, default=10)
    parser.add_argument("--subsample_cap", type=int, default=-1)
    args = parser.parse_args()
    main(args)