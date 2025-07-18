import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import time

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import torch.nn.utils.prune as prune

'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    print("iterations : ", init_iter_optimizerG, init_path_optimizerG)
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) create_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            # Randomly select 150 images
            train_set = torch.utils.data.Subset(train_set, random.sample(range(len(train_set)), min(150, len(train_set))))
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                print('Number of train images for fine-tuning: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

    pruning_iteration = 0
    iteraton_psnr = 1000

    while iteraton_psnr > 34.45:
        pruning_iteration += 1

        print("Pruning iteration: ", pruning_iteration)
        params_to_prune = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                params_to_prune.append((module, 'weight'))

        print("Params to prune: ", params_to_prune)

        # Dynamic pruning amount based on iteration
        pruning_amount = 0.05 

        print("Pruning amount: ", pruning_amount)
        for module, param in params_to_prune:
            prune.ln_structured(module, name=param, amount=pruning_amount, n=2, dim=0)

        '''
        # ----------------------------------------
        # Step--4 (main training)
        # ----------------------------------------
        '''
        e_pochs = opt['fine_tune']['L2_ft_epochs']

        print("Fine-tuning epochs: ", e_pochs)

        for epoch in range(e_pochs):
            print("epoch : ", epoch)
            if opt['dist']:
                train_sampler.set_epoch(epoch + seed)

            for i, train_data in enumerate(train_loader):

                print("Current step: ", current_step)
                current_step += 1

                # -------------------------------
                # 1) Update learning rate
                # -------------------------------
                model.update_learning_rate(current_step)

                # -------------------------------
                # 2) Feed patch pairs
                # -------------------------------
                model.feed_data(train_data)

                # -------------------------------
                # 3) Optimize parameters
                # -------------------------------
                model.optimize_parameters(current_step)

            # -------------------------------
            # Training information
            # -------------------------------
            if opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = ''
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                print(message)

        # -------------------------------
        # Testing
        # -------------------------------
        if opt['rank'] == 0:
            avg_psnr = 0.0
            avg_inference_time = 0.0
            idx = 0

            for test_data in test_loader:
                idx += 1
                image_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                img_dir = os.path.join(opt['path']['images'], img_name)
                util.mkdir(img_dir)

                model.feed_data(test_data)
                start_time = time.time()
                model.test()
                end_time = time.time()
                avg_inference_time += end_time - start_time

                visuals = model.current_visuals()
                E_img = util.tensor2uint(visuals['E'])
                H_img = util.tensor2uint(visuals['H'])

                # Save estimated image E
                save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                util.imsave(E_img, save_img_path)

                # Calculate PSNR
                current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                print('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                avg_psnr += current_psnr

            avg_psnr /= idx
            avg_inference_time /= idx
            iteraton_psnr = avg_psnr

            # Testing log
            print('Average PSNR: {:.2f}dB'.format(avg_psnr))
            print('Average inference time: {:.4f}s'.format(avg_inference_time))
            print("Model sparsity: ", util.compute_sparsity(model))

    # -------------------------------
    # Save model
    # -------------------------------
    if opt['rank'] == 0:
        print('Saving the model.')
        for name, module in model.named_modules():
            if hasattr(module, 'weight_orig'):
                print("Removing pruning mask for: ", name)
                prune.remove(module, 'weight')

        model.save(0)

if __name__ == '__main__':
    main()
