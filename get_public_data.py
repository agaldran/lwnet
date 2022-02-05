import os
import os.path as osp
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import io
import numpy as np
from torchvision.transforms.functional import resize

########################################################################################################################
os.makedirs('experiments', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('downloading data')

call = 'cd data && curl https://codeload.github.com/sraashis/deepdyn/tar.gz/master | tar -xz --strip=2 deepdyn-master/data'
os.system(call)
shutil.rmtree('data/VEVIO')
shutil.rmtree('data/DRIVE/splits')
shutil.rmtree('data/STARE/splits')
shutil.rmtree('data/STARE/labels-vk')
shutil.rmtree('data/CHASEDB/splits')
shutil.rmtree('data/AV-WIDE/splits')

call = 'wget http://iflexis.com/downloads/DRIVE_AV_evalmasks.zip && unzip DRIVE_AV_evalmasks.zip -d data/DRIVE ' \
       '&& rm DRIVE_AV_evalmasks.zip && mv data/DRIVE/DRIVE_AV_evalmasks/ZoneB_manual data/DRIVE '\
       '&& mv data/DRIVE/DRIVE_AV_evalmasks/Predicted_AV results/results_av_drive_hemelings && rm -r data/DRIVE/DRIVE_AV_evalmasks'
os.system(call)


call = 'mv data/STARE/labels-ah data/STARE/manual && mv data/STARE/stare-images data/STARE/images'
os.system(call)

call ='wget http://webeye.ophth.uiowa.edu/abramoff/AV_groundTruth.zip ' \
      '&& unzip AV_groundTruth.zip -d data/DRIVE && rm AV_groundTruth.zip ' \
      '&& mkdir data/DRIVE/manual_av && mv data/DRIVE/AV_groundTruth/training/av/* data/DRIVE/manual_av ' \
      '&& mv data/DRIVE/AV_groundTruth/test/av/* data/DRIVE/manual_av && rm -r data/DRIVE/AV_groundTruth'
os.system(call)

call ='wget http://iflexis.com/downloads/HRF_AV_GT.zip ' \
      '&& unzip HRF_AV_GT.zip -d data/HRF && rm HRF_AV_GT.zip && mv data/HRF/HRF_AV_GT data/HRF/manual_av'
os.system(call)

call = '(wget https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip ' \
       '&& unzip all.zip -d data/HRF && mv data/HRF/manual1 data/HRF/manual' \
       '&& rm all.zip)'
os.system(call)


call ='wget http://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/DRHAGIS.zip ' \
      '&& unzip DRHAGIS.zip -d data/DRHAGIS && rm DRHAGIS.zip && mv data/DRHAGIS/DRHAGIS data/DR_HAGIS ' \
      '&& rm -r data/DRHAGIS && mv data/DR_HAGIS/Fundus_Images data/DR_HAGIS/images ' \
      '&& mv data/DR_HAGIS/Mask_images data/DR_HAGIS/mask' \
      '&& mv data/DR_HAGIS/Manual_Segmentations data/DR_HAGIS/manual' \
      '&& rm data/DR_HAGIS/.DS_Store && rm data/DR_HAGIS/images/.DS_Store ' \
      '&& rm data/DR_HAGIS/manual/.DS_Store && rm data/DR_HAGIS/mask/.DS_Store && mv data/DR_HAGIS data/DR-HAGIS'
os.system(call)


print('preparing data')

# #######################################################################################################################
# # process drive data, generate CSVs
# path_ims = 'data/DRIVE/images'
# path_masks = 'data/DRIVE/mask'
# path_gts = 'data/DRIVE/manual'
#
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
#
# # append paths
# num_ims = len(all_im_names)
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
#
# test_im_names = all_im_names[:num_ims//2]
# train_im_names = all_im_names[num_ims//2:]
#
# test_mask_names = all_mask_names[:num_ims//2]
# train_mask_names = all_mask_names[num_ims//2:]
#
# test_gt_names = all_gt_names[:num_ims//2]
# train_gt_names = all_gt_names[num_ims//2:]
#
# df_drive_all = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
#
# df_drive_train = pd.DataFrame({'im_paths': train_im_names,
#                                'gt_paths': train_gt_names,
#                                'mask_paths': train_mask_names})
#
# df_drive_test = pd.DataFrame({'im_paths': test_im_names,
#                               'gt_paths': test_gt_names,
#                                'mask_paths': test_mask_names})
#
# df_drive_train, df_drive_val = df_drive_train[:16], df_drive_train[16:]
#
#
# df_drive_train.to_csv('data/DRIVE/train.csv', index=False)
# df_drive_val.to_csv('data/DRIVE/val.csv', index=False)
# df_drive_test.to_csv('data/DRIVE/test.csv', index=False)
# df_drive_all.to_csv('data/DRIVE/test_all.csv', index=False)
#
# # derive A/V split from vessel split
# df_drive_train.gt_paths = [n.replace('manual/', 'manual_av/')
#                         .replace('manual1.gif', 'training.png')
#                         for n in df_drive_train.gt_paths]
#
# df_drive_val.gt_paths = [n.replace('manual/', 'manual_av/')
#                         .replace('manual1.gif', 'training.png')
#                         for n in df_drive_val.gt_paths]
#
# df_drive_test.gt_paths = [n.replace('manual/', 'manual_av/')
#                         .replace('manual1.gif', 'test.png')
#                         for n in df_drive_test.gt_paths]
# df_drive_train.to_csv('data/DRIVE/train_av.csv', index=False)
# df_drive_val.to_csv('data/DRIVE/val_av.csv', index=False)
# df_drive_test.to_csv('data/DRIVE/test_av.csv', index=False)
# print('DRIVE prepared')
#
# df_drive_train_val = pd.concat([df_drive_train, df_drive_val], axis=0)
#
# for n in df_drive_train_val.gt_paths:
#     x = io.imread(n)
#     arteries = np.zeros_like(x[:, :, 0])
#     veins = np.zeros_like(x[:, :, 0])
#     unk = np.zeros_like(x[:, :, 0])
#
#     av = np.zeros_like(x[:, :, 0])
#
#     arteries[x[:, :, 0] == 255] = 255
#     unk[x[:, :, 1] == 255] = 255
#     veins[x[:, :, 2] == 255] = 255
#
#     av[unk == 255] = 85
#     av[arteries == 255] = 170
#     av[veins == 255] = 255
#
#     io.imsave(n, av)
# print('DRIVE A/V prepared')
#
# # ########################################################################################################################
# path_ims = 'data/CHASEDB/images'
# path_masks = 'data/CHASEDB/masks'
# path_gts = 'data/CHASEDB/manual'
#
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
#
# # append paths
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names if '1st' in n]
#
# num_ims = len(all_im_names)
# train_im_names = all_im_names[ :8]
# test_im_names  = all_im_names[8: ]
#
# train_mask_names = all_mask_names[ :8]
# test_mask_names  = all_mask_names[8: ]
#
# train_gt_names = all_gt_names[ :8]
# test_gt_names  = all_gt_names[8: ]
#
# df_chasedb_all = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
#
# df_chasedb_train = pd.DataFrame({'im_paths': train_im_names,
#                               'gt_paths': train_gt_names,
#                               'mask_paths': train_mask_names})
#
# df_chasedb_test = pd.DataFrame({'im_paths': test_im_names,
#                               'gt_paths': test_gt_names,
#                               'mask_paths': test_mask_names})
#
# num_ims = len(df_chasedb_train)
# tr_ims = int(0.8*num_ims)
# df_chasedb_train, df_chasedb_val = df_chasedb_train[:tr_ims], df_chasedb_train[tr_ims:]
#
# df_chasedb_train.to_csv('data/CHASEDB/train.csv', index=False)
# df_chasedb_val.to_csv('data/CHASEDB/val.csv', index=False)
# df_chasedb_test.to_csv('data/CHASEDB/test.csv', index=False)
# df_chasedb_all.to_csv('data/CHASEDB/test_all.csv', index=False)
# print('CHASE-DB prepared')
# # ########################################################################################################################
# # process HRF data, generate CSVs
path_ims = 'data/HRF/images'
path_masks = 'data/HRF/mask'
path_gts = 'data/HRF/manual'

path_ims_resized = 'data/HRF/images_resized'
os.makedirs(path_ims_resized, exist_ok=True)
path_masks_resized = 'data/HRF/mask_resized'
os.makedirs(path_masks_resized, exist_ok=True)
path_gts_resized = 'data/HRF/manual_resized'
os.makedirs(path_gts_resized, exist_ok=True)

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))

# append paths
num_ims = len(all_im_names)
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

df_hrf_all = pd.DataFrame({'im_paths': all_im_names,
                            'gt_paths': all_gt_names,
                            'mask_paths': all_mask_names})

train_im_names = all_im_names[   :3*5]
test_im_names =  all_im_names[3*5:   ]

train_mask_names = all_mask_names[   :3*5]
test_mask_names =  all_mask_names[3*5:   ]

train_gt_names = all_gt_names[   :3*5]
test_gt_names =  all_gt_names[3*5:   ]

# use smaller images for training **only** on HRF
train_im_names_resized = [n.replace(path_ims, path_ims_resized) for n in train_im_names]
train_mask_names_resized = [n.replace(path_masks, path_masks_resized) for n in train_mask_names]
train_gt_names_resized = [n.replace(path_gts, path_gts_resized) for n in train_gt_names]

df_hrf_train = pd.DataFrame({'im_paths': train_im_names_resized,
                             'gt_paths': train_gt_names_resized,
                             'mask_paths': train_mask_names_resized})
df_hrf_test = pd.DataFrame({'im_paths': test_im_names,
                              'gt_paths': test_gt_names,
                               'mask_paths': test_mask_names})

num_ims = len(df_hrf_train)
tr_ims = int(0.8*num_ims)
df_hrf_train, df_hrf_val = df_hrf_train[:tr_ims], df_hrf_train[tr_ims:]

df_hrf_train.to_csv('data/HRF/train.csv', index=False)
df_hrf_val.to_csv('data/HRF/val.csv', index=False)
df_hrf_test.to_csv('data/HRF/test.csv', index=False)
df_hrf_all.to_csv('data/HRF/test_all.csv', index=False)

# need this for AUC analysis on the training set
df_hrf_train_full_res = pd.DataFrame({'im_paths': train_im_names,
                             'gt_paths': train_gt_names,
                             'mask_paths': train_mask_names})
df_hrf_train_full_res, df_hrf_val_full_res = df_hrf_train_full_res[:tr_ims], df_hrf_train_full_res[tr_ims:]
df_hrf_train_full_res.to_csv('data/HRF/train_full_res.csv', index=False)
df_hrf_val_full_res.to_csv('data/HRF/val_full_res.csv', index=False)

print('Resizing HRF images (**only** for training, but we resize all because A/V training set is test set on Vessels)\n')
for i in tqdm(range(len(all_im_names))):
    im_name = all_im_names[i]
    im_name_out = im_name.replace('/images/', '/images_resized/')
    im = Image.open(im_name)
    im_res = resize(im, size=(im.size[1] // 2, im.size[0] // 2), interpolation=Image.BICUBIC)
    im_res.save(im_name_out)

    mask_name = im_name.replace('/images/', '/mask/').replace('.JPG', '_mask.tif').replace('.jpg', '_mask.tif')
    mask_name_out = mask_name.replace('/mask/', '/mask_resized/')
    mask = Image.open(mask_name)
    mask_res = resize(mask, size=(mask.size[1] // 2, mask.size[0] // 2), interpolation=Image.NEAREST)
    # get rid of three channels in mask
    mask = Image.fromarray(np.array(mask)[:,:,0])
    mask_res = Image.fromarray(np.array(mask_res)[:, :, 0])
    mask.save(mask_name)
    mask_res.save(mask_name_out)

    gt_name = im_name.replace('/images/', '/manual/').replace('.JPG', '.tif').replace('.jpg', '.tif')
    gt_name_out = gt_name.replace('/manual/', '/manual_resized/')
    gt = Image.open(gt_name)
    gt_res = resize(gt, size=(gt.size[1] // 2, gt.size[0] // 2), interpolation=Image.NEAREST)
    gt_res.save(gt_name_out)
print('HRF prepared')


# prepare A/V ground-truth for HRF, note that training set becomes test set here
path_gts_resized = 'data/HRF/manual_av_resized'
os.makedirs(path_gts_resized, exist_ok=True)
print('preparing HRF training set for A/V segmentation:')
for i in tqdm(range(len(test_im_names))):
    n= all_gt_names[i]
    n_av = n.replace('manual', 'manual_av').replace('.tif', '_AVmanual.png')

    x = io.imread(n_av)

    arteries = np.zeros_like(x[:, :, 0])
    veins = np.zeros_like(x[:, :, 0])
    unk = np.zeros_like(x[:, :, 0])

    av = np.zeros_like(x[:, :, 0])

    arteries[x[:, :, 0] == 255] = 255
    unk[x[:, :, 1] == 255] = 255
    veins[x[:, :, 2] == 255] = 255

    av[unk == 255] = 85
    av[arteries == 255] = 170
    av[veins == 255] = 255

    # Save also resized versions for faster training
    av = Image.fromarray(av)
    av = resize(av, size=(av.size[1] // 2, av.size[0] // 2), interpolation=Image.NEAREST)

    av.save(n_av.replace('manual_av/', 'manual_av_resized/'))

av_test = pd.concat([df_hrf_train, df_hrf_val], axis=0)

av_im_paths = [n.replace('images_resized/', 'images/') for n in av_test.im_paths]
av_gt_paths = [n.replace('manual_resized/', 'manual_av/') for n in av_test.gt_paths]
av_mask_paths = [n.replace('mask_resized/', 'mask/') for n in av_test.mask_paths]
av_test_df = pd.DataFrame(list(zip(av_im_paths,av_gt_paths,av_mask_paths)), columns=['im_paths','gt_paths', 'mask_paths'])

av_train, av_val = df_hrf_test[:24], df_hrf_test[24:]

av_train_im_paths = [n.replace('images/', 'images_resized/') for n in av_train.im_paths]
av_train_gt_paths = [n.replace('manual_av/', 'manual_av_resized/') for n in av_train.gt_paths]
av_train_mask_paths = [n.replace('mask/', 'mask_resized/') for n in av_train.mask_paths]
av_train_df = pd.DataFrame(list(zip(av_train_im_paths,av_train_gt_paths,av_train_mask_paths)), columns=['im_paths','gt_paths', 'mask_paths'])

av_val_im_paths = [n.replace('images/', 'images_resized/') for n in av_val.im_paths]
av_val_gt_paths = [n.replace('manual_av/', 'manual_av_resized/') for n in av_val.gt_paths]
av_val_mask_paths = [n.replace('mask/', 'mask_resized/') for n in av_val.mask_paths]
av_val_df = pd.DataFrame(list(zip(av_val_im_paths,av_val_gt_paths,av_val_mask_paths)), columns=['im_paths','gt_paths', 'mask_paths'])

av_train_df.to_csv('data/HRF/train_av.csv', index=False)
av_val_df.to_csv('data/HRF/val_av.csv', index=False)
av_test_df.to_csv('data/HRF/test_av.csv')

print('HRF A/V prepared')
# ########################################################################################################################
path_ims = 'data/STARE/images'
path_masks = 'data/STARE/mask'
path_gts = 'data/STARE/manual'

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))

# append paths
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

df_stare_all = pd.DataFrame({'im_paths': all_im_names,
                             'gt_paths': all_gt_names,
                             'mask_paths': all_mask_names})
df_stare_all.to_csv('data/STARE/test_all.csv', index=False)
print('STARE prepared')
########################################################################################################################
path_ims = 'data/AV-WIDE/images'
path_masks = 'data/AV-WIDE/masks'
os.makedirs(path_masks, exist_ok=True)
path_gts = 'data/AV-WIDE/manual'

test_im_names = sorted(os.listdir(path_ims))
test_gt_names = sorted(os.listdir(path_gts))

for n in test_im_names:
    im = Image.open(osp.join(path_ims, n))
    mask = 255*np.ones((im.size[1], im.size[0]), dtype=np.uint8)
    Image.fromarray(mask).save(osp.join(path_masks, n))

num_ims = len(test_im_names)
test_mask_names = [osp.join(path_masks, n) for n in test_im_names]
test_im_names = [osp.join(path_ims, n) for n in test_im_names]
test_gt_names = [osp.join(path_gts, n) for n in test_gt_names]

df_wide_test = pd.DataFrame({'im_paths': test_im_names,
                              'gt_paths': test_gt_names,
                              'mask_paths': test_mask_names})

df_wide_test.to_csv('data/AV-WIDE/test_all.csv', index=False)
print('AV-WIDE prepared')
#######################################################################################################################
path_ims = 'data/DR-HAGIS/images'
path_masks = 'data/DR-HAGIS/mask'
path_gts = 'data/DR-HAGIS/manual'

all_im_names = sorted(os.listdir(path_ims), key=lambda s: s.split("_")[0] )
all_mask_names = sorted(os.listdir(path_masks), key=lambda s: s.split("_")[0] )
all_gt_names = sorted(os.listdir(path_gts), key=lambda s: s.split("_")[0] )

all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

df_drhagis_all = pd.DataFrame({'im_paths': all_im_names,
                             'gt_paths': all_gt_names,
                             'mask_paths': all_mask_names})
df_drhagis_all.to_csv('data/DR-HAGIS/test_all.csv', index=False)
print('DR-HAGIS prepared')

print('All public data prepared, ready to go.')


print(104*'-')
print('NOTE: The Les-AV dataset is hosted at figshare now, see get_public_data.py Line 405 forward for details.')
print(104*'-')
########################################################################################################################
# What you can see below are the old instructions to download the LES-AV dataset. Since the release of this codebase,
# LES-AV has been removed from the public url we used to employ for downloading it and hosted at figshare, which allows
# free downloading, but as far as I know it needs to be done manually. Please head to the following url:
#
# https://figshare.com/articles/dataset/LES-AV_dataset/11857698
#
# download the LES-AV.zip file and then reproduce the steps below accordingly, sorry for the inconvenience. Adrian.
########################################################################################################################
# call = '(wget https://ignaciorlando.github.io/static/data/LES-AV.zip && unzip LES-AV.zip -d data/LES-AV ' \
#        '&& rm LES-AV.zip && mv data/LES-AV data/LES_AV' \
#        '&& rm -r data/LES_AV/__MACOSX)'
# os.system(call)
#
# call = '(mkdir data/LES-AV ' \
#        '&& mv data/LES_AV/LES-AV/images data/LES-AV/images ' \
#        '&& mv data/LES_AV/LES-AV/masks data/LES-AV/mask ' \
#        '&& mv data/LES_AV/LES-AV/vessel-segmentations data/LES-AV/manual' \
#        '&& mv data/LES_AV/LES-AV/arteries-and-veins data/LES-AV/manual_av ' \
#        '&& rm -r data/LES_AV)'
# os.system(call)

# path_ims = 'data/LES-AV/images'
# path_masks = 'data/LES-AV/mask'
# path_gts = 'data/LES-AV/manual'
#
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
#
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
#
# df_lesav_all = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
# df_lesav_all.to_csv('data/LES-AV/test_all.csv', index=False)
#
# # create data/LES_AV/test_av.csv:
# df_lesav_all.gt_paths = [n.replace('manual', 'manual_av') for n in df_lesav_all.gt_paths]
# df_lesav_all.to_csv('data/LES-AV/test_all_av.csv', index=None)
# print('LES-AV prepared')

