import os, sys
import os.path as osp
import argparse
import warnings
import time

import numpy as np
from utils import paired_transforms_tv04 as p_tr
from PIL import Image
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
import torch
from models.get_model import get_arch
from utils.model_saving_loading import load_model
from skimage.measure import regionprops

# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
parser.add_argument('--model_path', help='experiments/subfolder where checkpoint is', default='experiments/big_wnet_drive_av')
parser.add_argument('--im_path', help='path to image to be segmented', default=None)
parser.add_argument('--mask_path', help='path to FOv mask, will be computed if not provided', default=None)
parser.add_argument('--tta', type=str, default='from_probs', help='test-time augmentation (no/from_logits/from_preds)')
# im_size overrides config file
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--device', type=str, default='cpu', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--result_path', type=str, default=None, help='path to save prediction)')

from skimage import measure, draw
import numpy as np
from torchvision.transforms import Resize
from scipy import optimize
from skimage.filters import threshold_minimum
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hsv
from skimage.exposure import equalize_adapthist
from skimage.color import label2rgb


def get_circ(binary):
    # https://stackoverflow.com/a/28287741
    image = binary.astype(int)
    regions = measure.regionprops(image)
    bubble = regions[0]

    y0, x0 = bubble.centroid
    r = bubble.major_axis_length / 2.

    def cost(params):
        x0, y0, r = params
        coords = draw.circle(y0, x0, r, shape=image.shape)
        template = np.zeros_like(image)
        template[coords] = 1
        return -np.sum(template == image)

    x0, y0, r = optimize.fmin(cost, (x0, y0, r))
    return x0, y0, r

def create_circular_mask(sh, center=None, radius=None):
    # https://stackoverflow.com/a/44874588
    h, w = sh
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_fov(img):
    im_s = img.size
    if max(im_s) > 500:
        img = Resize(500)(img)

    with np.errstate(divide='ignore'):
        im_v = equalize_adapthist(np.array(img))[:, :, 1]
        # im_v = equalize_adapthist(rgb2hsv(np.array(img))[:, :, 2])
    thresh = threshold_minimum(im_v)
    binary = binary_fill_holes(im_v > thresh)

    x0, y0, r = get_circ(binary)
    fov = create_circular_mask(binary.shape, center=(x0, y0), radius=r)

    return Resize(im_s[ : :-1])(Image.fromarray(fov))

def crop_to_fov(img, mask):
    mask = np.array(mask).astype(int)
    minr, minc, maxr, maxc = regionprops(mask)[0].bbox
    im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
    return im_crop, [minr, minc, maxr, maxc]

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

    if tta!='no':
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
        else: raise NotImplementedError
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
    full_prob_2 += 0.5*full_prob_1
    full_prob_3 += 0.5 * full_prob_1
    full_prob = np.stack([full_prob_0, full_prob_2, full_prob_3], axis=2) # background, artery, vein

    full_pred = np.argmax(full_prob, axis=2)
    full_rgb_pred = label2rgb(full_pred, colors=['black', 'red', 'blue'])

    return np.clip(full_prob, 0,1), full_rgb_pred

if __name__ == '__main__':

    args = parser.parse_args()

    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print(f"* Running prediction on device '{args.device}'...")
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(args.device)

    tta = args.tta

    model_name = 'big_wnet'
    model_path = args.model_path
    im_path = args.im_path
    im_loc = osp.dirname(im_path)
    im_name = im_path.rsplit('/', 1)[-1]

    mask_path = args.mask_path
    result_path = args.result_path
    if result_path is None:
        result_path = im_loc
        im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_seg.png')
        im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_bin_seg.png')
    else:
        os.makedirs(result_path, exist_ok=True)
        im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_seg.png')
        im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_seg.png')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    print('* Segmenting image ' + im_path)
    img = Image.open(im_path)
    if mask_path is None:
        print('* FOV mask not provided, generating it')
        mask = get_fov(img)
        print('* FOV mask generated')
    else: mask = Image.open(mask_path).convert('L')
    mask = np.array(mask).astype(bool)

    img, coords_crop = crop_to_fov(img, mask)
    original_sz = img.size[1], img.size[0]  # in numpy convention

    rsz = p_tr.Resize(tg_size)
    tnsr = p_tr.ToTensor()
    tr = p_tr.Compose([rsz, tnsr])
    im_tens = tr(img)  # only transform image

    print('* Instantiating model  = ' + str(model_name))
    model = get_arch(model_name, n_classes=4).to(device)
    if model_name == 'big_wnet': model.mode='eval'

    print('* Loading trained weights from ' + model_path)
    model, stats = load_model(model, model_path, device)
    model.eval()

    print('* Saving prediction to ' + im_path_out)
    start_time = time.perf_counter()
    full_pred, full_pred_bin = create_pred(model, im_tens, mask, coords_crop, original_sz, tta=tta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(im_path_out, img_as_ubyte(full_pred))
        imsave(im_path_out_bin, img_as_ubyte(full_pred_bin))
    print('Done, time spent = {:.3f} secs'.format(time.perf_counter() - start_time))



