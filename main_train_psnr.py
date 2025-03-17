import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import time
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

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
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                             net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

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
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

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

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
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

    # ----------------------------------------
    # Enhanced attention profiling
    # ----------------------------------------
    attention_metrics = {}

    def forward_pre_hook(module, input):
        module_key = id(module)
        if module_key not in attention_metrics:
            attention_metrics[module_key] = {
                'total_time': 0.0,
                'count': 0,
                'name': str(module),
                'start_event': torch.cuda.Event(enable_timing=True),
                'end_event': torch.cuda.Event(enable_timing=True),
                'max_mem_alloc': 0,
                'mem_alloc_before': 0,
                'mem_cached_before': 0
            }
        metrics = attention_metrics[module_key]
        metrics['mem_alloc_before'] = torch.cuda.memory_allocated()
        metrics['mem_cached_before'] = torch.cuda.memory_reserved()
        metrics['start_event'].record()

    def forward_post_hook(module, input, output):
        module_key = id(module)
        metrics = attention_metrics[module_key]
        metrics['end_event'].record()
        torch.cuda.synchronize()
        
        # Calculate time
        elapsed_time = metrics['start_event'].elapsed_time(metrics['end_event']) / 1000  # Convert to seconds
        metrics['total_time'] += elapsed_time
        metrics['count'] += 1
        
        # Calculate memory
        mem_alloc_after = torch.cuda.memory_allocated()
        mem_cached_after = torch.cuda.memory_reserved()
        metrics['max_mem_alloc'] = max(metrics['max_mem_alloc'], 
                                     mem_alloc_after - metrics['mem_alloc_before'])
        metrics['max_mem_cached'] = max(metrics['max_mem_cached'], 
                                      mem_cached_after - metrics['mem_cached_before'])

    # Register hooks for attention layers
    for name, module in model.netG.named_modules():
        print("layer", name)
        if any(part.lower() == 'attn' for part in name.split('.')):
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)
            if opt['rank'] == 0:
                logger.info(f'Registered profiling hooks for: {name}')

    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    
    # Initialize profiler
    prof = None
    if opt['rank'] == 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=5,
                repeat=2
            ),
            on_trace_ready=tensorboard_trace_handler('./logs/profile'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )
        prof.start()

    for epoch in range(1000000):
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):
            current_step += 1

            # Update learning rate
            model.update_learning_rate(current_step)

            print("current step", current_step)
            # Feed data
            model.feed_data(train_data)

            # Optimize parameters
            with record_function("model_forward"):
                model.optimize_parameters(current_step)

            # Profiler step
            if prof and current_step % 10 == 0 and opt['rank'] == 0:
                prof.step()

            # Logging
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = f'<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}> '
                for k, v in logs.items():
                    message += f'{k}: {v:.3e} '
                
                # Attention metrics
                total_attn_time = sum([v['total_time'] for v in attention_metrics.values()])
                avg_attn_time = total_attn_time / len(attention_metrics) if attention_metrics else 0
                message += f" | AvgAttnTime: {avg_attn_time:.4f}s"
                
                logger.info(message)
                
                # Detailed attention layer analysis
                logger.info("\n=== Attention Layer Performance ===")
                for key, metrics in attention_metrics.items():
                    if metrics['count'] > 0:
                        avg_time = metrics['total_time'] / metrics['count']
                        logger.info(
                            f"{metrics['name']}:\n"
                            f"  - Calls: {metrics['count']}\n"
                            f"  - Avg Time: {avg_time:.4f}s\n"
                            f"  - Peak Memory: {metrics['max_mem_alloc']/1e6:.2f}MB (allocated)\n"
                            f"  - Peak Cache: {metrics['max_mem_cached']/1e6:.2f}MB (reserved)"
                        )
                
                # Reset metrics
                for key in attention_metrics:
                    attention_metrics[key]['total_time'] = 0.0
                    attention_metrics[key]['count'] = 0
                    attention_metrics[key]['max_mem_alloc'] = 0
                    attention_metrics[key]['max_mem_cached'] = 0

            # Save model
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # Testing
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    save_img_path = os.path.join(img_dir, f'{img_name}_{current_step}.png')
                    util.imsave(E_img, save_img_path)

                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                logger.info(f'<epoch:{epoch:3d}, iter:{current_step:8,d}, Avg PSNR: {avg_psnr:.2f}dB\n')

    # Cleanup profiler
    if prof and opt['rank'] == 0:
        prof.stop()
        logger.info("Profiling completed. View results with:")
        logger.info("tensorboard --logdir=logs/profile")

if __name__ == '__main__':
    try:
        main()
    finally:
        dist.destroy_process_group()
