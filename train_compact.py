import sys, json, os, argparse, time
from shutil import copyfile, rmtree
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
from models.get_model import get_arch

from utils.get_loaders import get_train_val_loaders
from utils.evaluation import evaluate, ewma
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.dice_loss import DiceLoss, TvLoss, SimilarityLoss
# argument parsing
parser = argparse.ArgumentParser()
# as seen here: https://stackoverflow.com/a/15460288/3208255
# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:

parser.add_argument('--csv_train', type=str, default='data/DRIVE/train_av.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='big_wnet', help='architecture')
parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
parser.add_argument('--grad_acc_steps', type=int, default=0, help='gradient accumulation steps (0)')
parser.add_argument('--min_lr', type=float, default=1e-8, help='learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cycle_lens', type=str, default='30/50', help='cycling config (nr cycles/cycle len')
parser.add_argument('--metric', type=str, default='dice', help='which metric to use for monitoring progress (tr_auc/auc/loss/dice)')
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
# these three are for training with pseudo-segmentations
# e.g. --csv_test data/DRIVE/test.csv --path_test_preds results/DRIVE/experiments/wnet_drive
# e.g. --csv_test data/LES_AV/test_all.csv --path_test_preds results/LES_AV/experiments/wnet_drive
parser.add_argument('--csv_test', type=str, default=None, help='path to test data csv (for using pseudo labels)')
parser.add_argument('--path_test_preds', type=str, default=None, help='path to test predictions (for using pseudo labels)')
parser.add_argument('--checkpoint_folder', type=str, default=None, help='path to model to start training (with pseudo labels now)')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel (multiprocessing) workers to launch for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'tr_auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print('Epoch {:5d}: reducing learning rate'
                  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None,
        grad_acc_steps=0, assess=False):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()
    if assess: logits_all, labels_all = [], []

    n_elems, running_loss = 0, 0
    for i_batch, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)

        if isinstance(logits, tuple): # wnet
            logits_aux, logits = logits
            # CrossEntropyLoss() -> A/V segmentation
            if model.compose=='mul':
                labels_aux = labels != 0
                loss_aux = torch.nn.BCEWithLogitsLoss()(logits_aux.squeeze(), labels_aux.float())
            elif model.compose=='cat':
                print(torch.cat([-10 * torch.ones(labels.shape).to(device), logits_aux], dim=1).shape)
                print(labels.squeeze(dim=1).shape)
                loss_aux = criterion(torch.cat([-10 * torch.ones(labels.shape).to(device), logits_aux], dim=1), labels.squeeze(dim=1))
                # loss_aux = criterion(logits_aux, labels)

            loss = loss_aux + criterion(torch.cat([-10 * torch.ones(labels.shape).to(device), logits], dim=1), labels.squeeze())
            # loss = loss_aux + criterion(logits, labels)
        else: # not wnet
            if model.n_classes == 1:
                loss = criterion(logits, labels.unsqueeze(dim=1).float())  # BCEWithLogitsLoss()/DiceLoss()
            else:
                loss = criterion(logits, labels)  # CrossEntropyLoss()

        # if train:  # only in training mode
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()

        if train:  # only in training mode
            (loss / (grad_acc_steps + 1)).backward() # for grad_acc_steps=0, this is just loss
            if i_batch % (grad_acc_steps+1) == 0:  # for grad_acc_steps=0, this is always True
                optimizer.step()
                for _ in range(grad_acc_steps+1): scheduler.step() # for grad_acc_steps=0, this means once
                optimizer.zero_grad()
        if assess:
            logits_all.extend(logits.detach().cpu())
            labels_all.extend(labels.cpu())

        # Compute running loss
        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss = running_loss / n_elems

    if assess: return logits_all, labels_all, run_loss
    return None, None, run_loss

def train_one_cycle(train_loader, model, criterion, optimizer=None, scheduler=None, grad_acc_steps=0, cycle=0):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]
    with trange(cycle_len) as t:
        for epoch in range(cycle_len):
            # print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle+1, epoch+1, cycle_len))
            if epoch == cycle_len-1: assess=True # only get logits/labels on last cycle
            else: assess = False
            tr_logits, tr_labels, tr_loss = run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                                                          scheduler=scheduler, grad_acc_steps=grad_acc_steps, assess=assess)
            t.set_postfix_str("Cycle: {}/{} Ep. {}/{} -- tr. loss={:.4f} / lr={:.6f}".format(cycle + 1,
                                                                                             len(scheduler.cycle_lens),
                                                                                             epoch + 1, cycle_len,
                                                                                             float(tr_loss),
                                                                                             get_lr(optimizer)))
            t.update()
    return tr_logits, tr_labels, tr_loss

def train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, grad_acc_steps, metric, exp_path):

    n_cycles = len(scheduler.cycle_lens)
    best_mcc, best_dice, best_cycle, all_dices = 0, 0, 0, []
    is_better, best_monitoring_metric = compare_op(metric)

    for cycle in range(n_cycles):
        print('Cycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # prepare next cycle:
        # reset iteration counter
        scheduler.last_epoch = -1
        # update number of iterations
        scheduler.T_max = scheduler.cycle_lens[cycle] * len(train_loader)

        # train one cycle, retrieve segmentation data and compute metrics at the end of cycle
        tr_logits, tr_labels, tr_loss = train_one_cycle(train_loader, model, criterion, optimizer, scheduler, grad_acc_steps, cycle)
        # classification metrics at the end of cycle
        tr_dice, tr_mcc = evaluate(tr_logits, tr_labels)  # for n_classes>1, will need to redo evaluate
        del tr_logits, tr_labels
        with torch.no_grad():
            assess=True
            vl_logits, vl_labels, vl_loss = run_one_epoch(val_loader, model, criterion, assess=assess)
            vl_dice, vl_mcc = evaluate(vl_logits, vl_labels)  # for n_classes>1, will need to redo evaluate
            del vl_logits, vl_labels
        print('Train/Val Loss: {:.4f}/{:.4f} -- Train/Val DICE: {:.4f}/{:.4f} -- Train/Val MCC: {:.4f}/{:.4f} -- LR={:.6f}'.format(
                tr_loss, vl_loss, tr_dice, vl_dice, tr_mcc, vl_mcc,  get_lr(optimizer)).rstrip('0'))
        all_dices.append(100 * vl_dice)

        # check if performance was better than anyone before and checkpoint if so
        if metric == 'mcc':
            monitoring_metric = vl_mcc
        elif metric == 'loss':
            monitoring_metric = vl_loss
        elif metric == 'dice':
            monitoring_metric = vl_dice
        if is_better(monitoring_metric, best_monitoring_metric):
            print('Best {} attained. {:.2f} --> {:.2f}'.format(metric, 100*best_monitoring_metric, 100*monitoring_metric))
            best_dice, best_mcc, best_cycle = vl_dice, vl_mcc, cycle+1
            best_monitoring_metric = monitoring_metric
            if exp_path is not None:
                print(15 * '-', ' Checkpointing ', 15 * '-')
                save_model(exp_path, model, optimizer)
        else:
            print('Best {} so far --> {:.2f} (cycle {})'.format(metric, 100*best_monitoring_metric, best_cycle))

    del model
    torch.cuda.empty_cache()
    return best_dice, best_mcc, best_cycle, all_dices

if __name__ == '__main__':
    '''
    Example:
    python train_cyclical.py --csv_train data/DRIVE/train.csv --save_path unet_DRIVE
    '''

    args = parser.parse_args()

    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print('* Training on device {}'.format(args.device))
        device = torch.device("cuda")

    else:  #cpu
        device = torch.device(args.device)

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, args.device.startswith("cuda"))

    # gather parser parameters
    model_name = args.model_name
    max_lr, min_lr, bs, grad_acc_steps = args.max_lr, args.min_lr, args.batch_size, args.grad_acc_steps
    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
    cycle_lens = list(map(int, cycle_lens))

    if len(cycle_lens)==2: # handles option of specifying cycles as pair (n_cycles, cycle_len)
        cycle_lens = cycle_lens[0]*[cycle_lens[1]]

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path=osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)

        config_file_path = osp.join(experiment_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path=None

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    # training for artery-vein segmentation
    if 'av' in csv_train:
        n_classes=3 # we do not predict background here
        label_values=[0, 85, 170, 255]
    else:
        n_classes=1
        label_values = [0, 255]


    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs,args.num_workers))
    train_loader, val_loader = get_train_val_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs, tg_size=tg_size, label_values=label_values, num_workers=args.num_workers)

    # grad_acc_steps: if I want to train with a fake_bs=K but the actual bs I want is bs=N, then you use
    # grad_acc_steps = N/K - 1.
    # Example: bs=4, fake_bs=4 -> grad_acc_steps = 0 (default)
    # Example: bs=4, fake_bs=2 -> grad_acc_steps = 1
    # Example: bs=4, fake_bs=1 -> grad_acc_steps = 3


    print('* Instantiating a {} model'.format(model_name))
    model = get_arch(model_name, n_classes=n_classes, compose='cat')
    model = model.to(device)

    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    ### TRAINING WITH PSEUDO-LABELS
    csv_test = args.csv_test
    path_test_preds = args.path_test_preds
    checkpoint_folder = args.checkpoint_folder
    if csv_test is not None:
        print('Training with pseudo-labels, completing training set with predictions on test set')
        from utils.get_loaders import build_pseudo_dataset
        tr_im_list, tr_gt_list, tr_mask_list = build_pseudo_dataset(csv_train, csv_test, path_test_preds)
        train_loader.dataset.im_list = tr_im_list
        train_loader.dataset.gt_list = tr_gt_list
        train_loader.dataset.mask_list = tr_mask_list
        print('* Loading weights from previous checkpoint={}'.format(checkpoint_folder))
        model, stats, optimizer_state_dict = load_model(model, checkpoint_folder, device=device, with_opt=True)
        optimizer.load_state_dict(optimizer_state_dict)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = max_lr
            param_group['initial_lr'] = max_lr


    scheduler = CosineAnnealingLR(optimizer, T_max=cycle_lens[0] * len(train_loader) // (grad_acc_steps + 1), eta_min=min_lr)
    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)


    criterion = torch.nn.BCEWithLogitsLoss() if model.n_classes == 1 else torch.nn.CrossEntropyLoss(ignore_index=0)


    print('* Instantiating loss function', str(criterion))
    print('* Starting to train\n','-' * 10)


    start = time.time()
    m1, m2, m3, dices=train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, grad_acc_steps, metric, experiment_path)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("val_dice: %f" % m1)
    print("val_mcc: %f" % m2)
    print("best_cycle: %d" % m3)
    if do_not_save is False:
        # file = open(osp.join(experiment_path, 'val_metrics.txt'), 'w')
        # file.write(str(m1)+ '\n')
        # file.write(str(m2)+ '\n')
        # file.write(str(m3)+ '\n')
        # file.close()

        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('Best DICE = {:.2f}\nBest MCC = {:.2f}\nBest cycle = {}'.format(100*m1, 100*m2, m3), file=f)
            for j in range(len(dices)):
                print('\nEpoch = {} -> Dice = {:.2f}'.format(j, dices[j]), file=f)
            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)
