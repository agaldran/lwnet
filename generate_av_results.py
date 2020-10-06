import os, json, sys
import os.path as osp
import argparse
import warnings
from tqdm import tqdm

import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.color import label2rgb

import torch
from utils.model_saving_loading import str2bool
from models.get_model import get_arch
from utils.get_loaders import get_test_dataset
from utils.model_saving_loading import load_model

# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
required_named.add_argument('--dataset', type=str, help='generate results for which dataset', required=True)
parser.add_argument('--experiment_path', help='experiments/subfolder where checkpoint is', default=None)
parser.add_argument('--tta', type=str, default='from_probs', help='test-time augmentation (no/from_logits/from_preds)')
parser.add_argument('--config_file', type=str, default=None,
                    help='experiments/name_of_config_file, overrides everything')
# in case no config file is passed
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--results_path', type=str, default='results', help='path to save predictions (defaults to results')


def flip_ud(tens):
    return torch.flip(tens, dims=[1])


def flip_lr(tens):
    return torch.flip(tens, dims=[2])


def flip_lrud(tens):
    return torch.flip(tens, dims=[1, 2])


def create_pred(model, tens, mask, coords_crop, original_sz, tta='no'):
    act = torch.nn.Softmax(dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    prob = act(logits)
    # print(logits.shape)
    # print((torch.nn.Softmax(dim=0)(logits)).shape)
    # sys.exit()

    if tta != 'no':
        with torch.no_grad():
            logits_lr = model(tens.flip(-1).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1)
            logits_ud = model(tens.flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-2)
            logits_lrud = model(tens.flip(-1).flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1).flip(-2)
        if tta == 'from_logits':
            mean_logits = torch.mean(torch.stack([logits, logits_lr, logits_ud, logits_lrud]), dim=0)
            prob = act(mean_logits)
        elif tta == 'from_probs':
            prob_lr = act(logits_lr)
            prob_ud = act(logits_ud)
            prob_lrud = act(logits_lrud)
            prob = torch.mean(torch.stack([prob, prob_lr, prob_ud, prob_lrud]), dim=0)
        else:
            raise NotImplementedError
    # prob is now n_classes x h_train x w_train
    prob = prob.detach().cpu().numpy()
    # Orders: 0: NN, 1: Bilinear(default), 2: Biquadratic, 3: Bicubic, 4: Biquartic, 5: Biquintic

    prob_0 = resize(prob[0], output_shape=original_sz, order=3)
    prob_1 = resize(prob[1], output_shape=original_sz, order=3)
    prob_2 = resize(prob[2], output_shape=original_sz, order=3)
    prob_3 = resize(prob[3], output_shape=original_sz, order=3)

    full_prob_0 = np.zeros_like(mask, dtype=float)
    full_prob_1 = np.zeros_like(mask, dtype=float)
    full_prob_2 = np.zeros_like(mask, dtype=float)
    full_prob_3 = np.zeros_like(mask, dtype=float)

    full_prob_0[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = prob_0
    full_prob_0[~mask.astype(bool)] = 0
    full_prob_1[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = prob_1
    full_prob_1[~mask.astype(bool)] = 0
    full_prob_2[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = prob_2
    full_prob_2[~mask.astype(bool)] = 0
    full_prob_3[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = prob_3
    full_prob_3[~mask.astype(bool)] = 0

    # full_prob_1 corresponds to uncertain pixels, we redistribute probability between prob_1 and prob_2
    full_prob_2 += 0.5 * full_prob_1
    full_prob_3 += 0.5 * full_prob_1
    full_prob = np.stack([full_prob_0, full_prob_2, full_prob_3], axis=2)  # background, artery, vein

    full_pred = np.argmax(full_prob, axis=2)
    full_rgb_pred = label2rgb(full_pred, colors=['black', 'red', 'blue'])

    return np.clip(full_prob, 0, 1), full_rgb_pred


def save_pred(prob_pred, save_results_path, im_name):
    prob, pred = prob_pred
    os.makedirs(save_results_path, exist_ok=True)
    im_name = im_name.rsplit('/', 1)[-1]
    save_name = osp.join(save_results_path, im_name[:-4] + '.png')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # this casts preds to int, loses precision
        # we only do this for visualization purposes
        imsave(save_name, img_as_ubyte(prob))

    # we save float predictions in a numpy array for
    # accurate performance evaluation
    # save_name_np = osp.join(save_results_path, im_name[:-4])
    # np.save(save_name_np, full_pred)
    # save also binarized image
    save_name = osp.join(save_results_path, im_name[:-4] + '_binary.png')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(save_name, img_as_ubyte(pred))


if __name__ == '__main__':

    args = parser.parse_args()
    results_path = args.results_path
    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":", 1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print('* Running prediction on device {}'.format(args.device))
        device = torch.device("cuda")
    else:  # cpu
        device = torch.device(args.device)

    dataset = args.dataset
    tta = args.tta

    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file): raise Exception('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__ = json.load(f)
    experiment_path = args.experiment_path  # this should exist in a config file
    model_name = args.model_name

    if experiment_path is None: raise Exception('must specify path to experiment')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    data_path = osp.join('data', dataset)

    csv_path = 'test_all.csv'
    print('* Reading test data from ' + osp.join(data_path, csv_path))
    test_dataset = get_test_dataset(data_path, csv_path=csv_path, tg_size=tg_size)
    print('* Instantiating model  = ' + str(model_name))
    model = get_arch(model_name, n_classes=4).to(device)
    if 'wnet' in model_name: model.mode = 'eval'

    print('* Loading trained weights from ' + experiment_path)
    try:
        model, stats = load_model(model, experiment_path, device)
    except RuntimeError:
        sys.exit('---- bad config specification (check layers, n_classes, etc.) ---- ')
    model.eval()

    save_results_path = osp.join(results_path, dataset, experiment_path)
    print('* Saving predictions to ' + save_results_path)
    for i in tqdm(range(len(test_dataset))):
        im_tens, mask, coords_crop, original_sz, im_name = test_dataset[i]
        prob_pred = create_pred(model, im_tens, mask, coords_crop, original_sz, tta=tta)
        save_pred(prob_pred, save_results_path, im_name)
    print('* Done')