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
from utils.evaluation import evaluate4 as evaluate
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
parser.add_argument('--alpha_tv', type=float, default=0.01, help='weight of the TV loss component')
parser.add_argument('--eps_tv', type=float, default=0.01, help='weight of the TV loss component')
parser.add_argument('--min_lr', type=float, default=1e-8, help='learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cycle_lens', type=str, default='40/50', help='cycling config (nr cycles/cycle len')
parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (tr_auc/auc/loss/dice)')
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel (multiprocessing) workers to launch for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'mcc':
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

def run_one_epoch(loader, model, criterion, tv_criterion, optimizer=None, scheduler=None,
        grad_acc_steps=0, assess=False):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()
    if assess: auc_scs, f1_scs, mcc_scs = [], [], []


    n_elems, running_loss, running_loss_ce, running_loss_tv = 0, 0, 0, 0
    for i_batch, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.unsqueeze(dim=1).to(device)
        logits = model(inputs)

        if isinstance(logits, tuple): # wnet
            logits_aux, logits = logits
            # CrossEntropyLoss() -> A/V segmentation

            loss_aux = criterion(logits_aux, labels.squeeze(dim=1))
            loss_ce = loss_aux + criterion(logits, labels.squeeze(dim=1))

            # tv_loss_aux = tv_criterion(logits_aux, labels)
            # tv_loss = tv_criterion.alpha *(tv_loss_aux + tv_criterion(logits, labels))


            # logits = torch.nn.MaxPool2d(kernel_size=2, stride=2)(logits)
            # logits_aux = torch.nn.MaxPool2d(kernel_size=2, stride=2)(logits_aux)
            logits = torch.nn.UpsamplingNearest2d(scale_factor=1/2)(logits)
            logits_aux = torch.nn.UpsamplingNearest2d(scale_factor=1 / 2)(logits_aux)
            labels = torch.nn.UpsamplingNearest2d(scale_factor=1/2)(labels.float()).long()
            loss_ce += 0.5*criterion(torch.cat([-10 * torch.ones(labels.shape).to(device), logits_aux], dim=1), labels.squeeze(dim=1))
            loss_ce += 0.5*criterion(torch.cat([-10 * torch.ones(labels.shape).to(device), logits], dim=1), labels.squeeze(dim=1))

            # logits = torch.nn.MaxPool2d(kernel_size=2, stride=2)(logits)
            # logits_aux = torch.nn.MaxPool2d(kernel_size=2, stride=2)(logits_aux)
            logits = torch.nn.UpsamplingNearest2d(scale_factor=1/2)(logits)
            logits_aux = torch.nn.UpsamplingNearest2d(scale_factor=1 / 2)(logits_aux)
            labels = torch.nn.UpsamplingNearest2d(scale_factor=1/2)(labels.float()).long()
            loss_ce += 0.25*criterion(torch.cat([-10 * torch.ones(labels.shape).to(device), logits_aux], dim=1), labels.squeeze(dim=1))
            loss_ce += 0.25*criterion(torch.cat([-10 * torch.ones(labels.shape).to(device), logits], dim=1), labels.squeeze(dim=1))


            loss, tv_loss = loss_ce, torch.tensor(0)
            # loss = loss_ce + tv_loss

        else: # not wnet
            sys.exit('code needs to be adapted to train models other than Wnet here')

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
            auc_s, f1_s, mcc_s = evaluate(logits.detach().cpu(), labels.cpu())
            auc_scs.append(auc_s)
            f1_scs.append(f1_s)
            mcc_scs.append(mcc_s)

        # Compute running loss
        running_loss += loss.item() * inputs.size(0)
        running_loss_ce += loss_ce.item() * inputs.size(0)
        running_loss_tv += tv_loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss_ce = running_loss_ce / n_elems
        run_loss_tv = running_loss_tv / n_elems
        run_loss = running_loss / n_elems

    if assess: return np.array(auc_scs).mean(), np.array(f1_scs).mean(), np.array(mcc_scs).mean(), run_loss_ce, run_loss_tv
    return None, None, None, run_loss_ce, run_loss_tv

def train_one_cycle(train_loader, model, criterion, tv_criterion, optimizer=None, scheduler=None, grad_acc_steps=0, cycle=0):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]
    with trange(cycle_len) as t:
        for epoch in range(cycle_len):
            # print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle+1, epoch+1, cycle_len))
            if epoch == cycle_len-1: assess=True # only get logits/labels on last cycle
            else: assess = False
            auc_sc, f1_sc, mcc_sc, tr_loss_ce, tr_loss_tv = run_one_epoch(train_loader, model, criterion, tv_criterion, optimizer=optimizer,
                                                          scheduler=scheduler, grad_acc_steps=grad_acc_steps, assess=assess)
            t.set_postfix_str("Cycle: {}/{} Ep. {}/{} -- tr. loss CE/TV={:.4f}/{:.4f} || lr={:.6f}".format(cycle + 1,
                                                                                             len(scheduler.cycle_lens),
                                                                                             epoch + 1, cycle_len,
                                                                                             float(tr_loss_ce),
                                                                                            float(tr_loss_tv),
                                                                                             get_lr(optimizer)))
            t.update()
    return auc_sc, f1_sc, mcc_sc, tr_loss_ce, tr_loss_tv

def train_model(model, optimizer, criterion, tv_criterion, train_loader, val_loader, scheduler, grad_acc_steps, metric, exp_path):

    n_cycles = len(scheduler.cycle_lens)
    best_auc, best_dice, best_mcc, best_cycle, all_aucs, all_dices, all_mccs = 0, 0, 0, 0, [], [], []
    is_better, best_monitoring_metric = compare_op(metric)

    for cycle in range(n_cycles):
        print('Cycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # prepare next cycle:
        # reset iteration counter
        scheduler.last_epoch = -1
        # update number of iterations
        scheduler.T_max = scheduler.cycle_lens[cycle] * len(train_loader)

        # train one cycle
        _, _, _, _, _= train_one_cycle(train_loader, model, criterion, tv_criterion, optimizer, scheduler, grad_acc_steps, cycle)

        # classification metrics at the end of cycle, both for train and val (it's relatively cheap)
        with torch.no_grad():
            assess=True
            tr_auc, tr_dice, tr_mcc, tr_loss_ce, tr_loss_tv = run_one_epoch(train_loader, model, criterion, tv_criterion, assess=assess)
            vl_auc, vl_dice, vl_mcc, vl_loss_ce, vl_loss_tv = run_one_epoch(val_loader, model, criterion, tv_criterion, assess=assess)

        print('Train/Val Loss CE||TV: {:.4f}/{:.4f}||{:.4f}/{:.4f} -- AUC: {:.2f}/{:.2f} -- '
                                                                               'DICE: {:.2f}/{:.2f} -- '
                                                                               'MCC: {:.2f}/{:.2f}'.format(
                                                                                tr_loss_ce, vl_loss_ce, tr_loss_tv, vl_loss_tv,
                                                                                100*tr_auc, 100*vl_auc, 100*tr_dice,
                                                                                100*vl_dice, 100*tr_mcc, 100*vl_mcc))
        all_aucs.append(100 * vl_auc)
        all_dices.append(100 * vl_dice)
        all_mccs.append(100 * vl_mcc)

        # check if performance was better than anyone before and checkpoint if so
        if metric == 'mcc':
            monitoring_metric = vl_mcc
        elif metric == 'loss':
            monitoring_metric = tr_loss_ce
        elif metric == 'auc':
            monitoring_metric = vl_auc
        elif metric == 'dice':
            monitoring_metric = vl_dice

        if is_better(monitoring_metric, best_monitoring_metric):
            print('Best {} attained. {:.2f} --> {:.2f}'.format(metric, 100*best_monitoring_metric, 100*monitoring_metric))
            best_auc, best_dice, best_mcc, best_cycle = vl_auc, vl_dice, vl_mcc, cycle+1
            best_monitoring_metric = monitoring_metric
            if exp_path is not None:
                print(15 * '-', ' Checkpointing ', 15 * '-')
                save_model(exp_path, model, optimizer)
        else:
            print('Best {} so far --> {:.2f} (cycle {})'.format(metric, 100*best_monitoring_metric, best_cycle))

    del model
    torch.cuda.empty_cache()
    return best_auc, best_dice, best_mcc, best_cycle, all_aucs, all_dices, all_mccs

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
        n_classes=4
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
    # optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=cycle_lens[0] * len(train_loader) // (grad_acc_steps + 1), eta_min=min_lr)
    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)

    criterion = torch.nn.CrossEntropyLoss()
    tv_criterion = TvLoss(reduction='mean', alpha=args.alpha_tv, eps = args.eps_tv)

    print('* Instantiating loss function', str(criterion), str(tv_criterion))
    print('* Starting to train\n','-' * 10)


    start = time.time()
    m1, m2, m3, best_cyc, aucs, dices, mccs = train_model(model, optimizer, criterion, tv_criterion, train_loader, val_loader,
                                                          scheduler, grad_acc_steps, metric, experiment_path)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("val_auc: %f" % m1)
    print("val_dice: %f" % m2)
    print("val_mcc: %f" % m3)
    print("best_cycle: %d" % best_cyc)
    if do_not_save is False:
        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('Best AUC = {:.2f}\nBest DICE = {:.2f}\nBest MCC = {:.2f}\nBest cycle = {}'.format(100*m1, 100*m2, 100*m3, best_cyc), file=f)
            for j in range(len(dices)):
                print('\nEpoch = {} -> AUC = {:.2f}, Dice = {:.2f}, MCC = {:.2f}'.format(j+1, aucs[j],dices[j], mccs[j]), file=f)
            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)
