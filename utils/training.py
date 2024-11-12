import math
import pickle
import sys
from argparse import Namespace
from typing import Tuple

import copy
import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.distributed import is_distributed, is_master, make_ddp, make_dp, wait_for_master

from utils.loggers import *
from utils.status import ProgressBar

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()

    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if is_distributed():
                    if 'class-il' not in model.COMPATIBILITY:
                        outputs = model.module(inputs, k)
                    else:
                        outputs = model.module(inputs)
                else:
                    if 'class-il' not in model.COMPATIBILITY:
                        outputs = model(inputs, k)
                    else:
                        outputs = model(inputs)
                
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes



def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if is_master():
        if not args.ignore_other_metrics:
            dataset_copy = get_dataset(args)
            for t in range(dataset.N_TASKS):
                model.net.train()
                _, _ = dataset_copy.get_data_loaders()
            if model.NAME != 'icarl' and model.NAME != 'pnn':
                random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        print(f"Task {t+1}/{dataset.N_TASKS} | noise ", 1-(train_loader.dataset.gt_targets==train_loader.dataset.targets).mean(), file=sys.stderr)

        if hasattr(model, 'begin_task'):
            pastvals = model.begin_task(dataset)
        else:
            pastvals = None

        if t==0:
            if args.distributed == 'post_bt':
                model = make_dp(model)
            elif args.distributed == 'post_bt_ddp':
                model = make_ddp(model)

        if pastvals is not None:
            tt, args = pastvals

            if t == 0 and tt is not None and t!=tt:
                if is_master():
                    print(f"Skipping {tt-t} tasks")
                for _ in range(tt-t):
                    train_loader, test_loader = dataset.get_data_loaders()
                    model.begin_task(dataset)

                t = tt

        if is_master():
            if t and not args.ignore_other_metrics:
                accs = evaluate(model, dataset, last=True)
                results[t-1] = results[t-1] + accs[0]
                if dataset.SETTING == 'class-il':
                    results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        epochs = model.args.n_epochs if "gdumb" not in model.args.model else 1
        for epoch in range(epochs):
            wait_for_master()
            if hasattr(model,'model_copy') and model.model_copy is not None:
                if ((hasattr(model.args, 'aer') and model.args.aer) and not epoch % 2 == model.args.start_with_replay) or (model.args.restert_after_buffer_fit and epoch==0): #if replay on
                    model.load_state_dict(model.model_copy)

            if args.model.startswith('joint'):
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs, true_labels, idx = data
                    inputs, labels, true_labels = inputs.to(model.device), labels.to(
                        model.device), true_labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    idx = idx.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, true_labels, idx, epoch)
                    model.iteration += 1

                assert not math.isnan(loss)   
                progress_bar.prog(i, len(train_loader), epoch, t, loss)
            
            if scheduler is not None:
                scheduler.step()
            

            if (hasattr(model.args, 'aer') and model.args.aer) and not epoch % 2 == model.args.start_with_replay: #replay on
                model.model_copy = copy.deepcopy(model.state_dict())

        wait_for_master()
        if hasattr(model, 'end_task'):
            model.iteration +=1
            model.end_task(dataset)
            if hasattr(model.args, 'buffer_fitting_epochs'):
                if model.args.buffer_fitting_epochs > 0:
                    model.model_copy = None

        if is_master():
            accs = evaluate(model, dataset)
            print(accs)
            results.append(accs[0])
            results_mask_classes.append(accs[1])

            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

            if not args.disable_log:
                logger.log(mean_acc)
                logger.log_fullacc(accs)


            if args.savecheck:
                if not os.path.isdir('checkpoints'):
                    create_if_not_exists("checkpoints")
                print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)
                torch.save(model.state_dict(), f'checkpoints/{args.ckpt_name}_{t}.pt')
                if 'buffer_size' in model.args:
                    with open(f'checkpoints/{args.ckpt_name_replace.format("bufferoni")}_{t}.pkl', 'wb') as f:
                        pickle.dump(obj=copy.deepcopy(
                            model.buffer).to('cpu'), file=f)
                with open(f'checkpoints/{args.ckpt_name_replace.format("interpr")}_{t}.pkl', 'wb') as f:
                    pickle.dump(obj=args, file=f)
                with open(f'checkpoints/{args.ckpt_name_replace.format("results")}_{t}.pkl', 'wb') as f:
                    pickle.dump(
                        obj=[results, results_mask_classes, logger.dump()], file=f)

    wait_for_master()
    if is_master():
        if not args.disable_log and not args.ignore_other_metrics:
            logger.add_bwt(results, results_mask_classes)
            logger.add_forgetting(results, results_mask_classes)
            if model.NAME != 'icarl' and model.NAME != 'pnn':
                logger.add_fwt(results, random_results_class,
                        results_mask_classes, random_results_task)
