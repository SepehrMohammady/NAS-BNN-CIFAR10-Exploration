# search.py

import argparse
import logging
import os # Make sure os is imported if not already
import os.path as osp
import random
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models as models
from utils import cand2tuple, get_logger, tuple2cand

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

# --- Argument Parser (Keep your existing parser here) ---
parser = argparse.ArgumentParser()
# ... (all your argparse.add_argument calls from your original search.py) ...
# Example line from your log:
parser.add_argument('supernet', type=str) 
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('logdir', metavar='DIR')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='superbnn_cifar10',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: superbnn_cifar10)')
parser.add_argument('--dataset',
                    type=str,
                    default='imagenet',
                    help='imagenet | cifar10')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--max-epochs', type=int, default=20)
parser.add_argument('--population-num', type=int, default=512)
parser.add_argument('--m-prob', type=float, default=0.2)
parser.add_argument('--crossover-num', type=int, default=128)
parser.add_argument('--mutation-num', type=int, default=128)
parser.add_argument('--ops-min', type=float, default=40)
parser.add_argument('--ops-max', type=float, default=250)
parser.add_argument('--step', type=float, default=10)
parser.add_argument('--max-train-iters', type=int, default=10)
parser.add_argument('--train-batch-size', type=int, default=256)
parser.add_argument('--test-batch-size', type=int, default=256)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--world-size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed for initializing training.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
# --- End Argument Parser ---


def is_first_gpu(args, ngpus_per_node):
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)


def seed(seed_val=0): # Renamed parameter to avoid conflict with module
    # import os # Already imported globally
    # import random # Already imported globally
    import sys
    import numpy as np
    # import torch # Already imported globally
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed_val)
    random.seed(seed_val)


class EvolutionSearcher:

    def __init__(self, model, logger, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node

        self.max_epochs = args.max_epochs
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.ops_min = args.ops_min
        self.ops_max = args.ops_max
        self.step = args.step

        if not osp.exists(args.logdir):
            os.makedirs(args.logdir)

        self.logger = logger

        self.model = model
        if hasattr(model, 'module'):
            self.m = model.module
        else:
            self.m = model

        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        if args.dataset == 'imagenet':
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(), normalize
            ])
            train_dataset = datasets.ImageFolder(traindir, train_transform)
            val_dataset = datasets.ImageFolder(valdir, val_transform)
        elif args.dataset == 'cifar10':
            normalize = transforms.Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[0.24703233, 0.24348505, 0.26158768])
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize
            ])
            val_transform = transforms.Compose(
                [transforms.ToTensor(), normalize])
            train_dataset = datasets.ImageFolder(traindir, train_transform)
            val_dataset = datasets.ImageFolder(valdir, val_transform)
        else:
            raise NotImplementedError

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_name = os.path.join(args.logdir, 'info.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.epoch = 0
        self.candidates = []
        self.pareto_global = {}

    def save_checkpoint(self):
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates # Save current candidates list
        info['vis_dict'] = self.vis_dict
        info['epoch'] = self.epoch
        info['pareto_global'] = self.pareto_global
        torch.save(info, self.checkpoint_name)
        if is_first_gpu(self.args, self.ngpus_per_node):
            self.logger.info('Save checkpoint to {}'.format(
                self.checkpoint_name))

    def is_legal(self, cand):
        cand_tuple = cand2tuple(cand)
        
        # Example of how you might add a skip for a KNOWN bad candidate IF identified
        # KNOWN_BAD_CANDIDATE_TUPLE = ( ... tuple values ... ) 
        # if cand_tuple == KNOWN_BAD_CANDIDATE_TUPLE:
        #     if is_first_gpu(self.args, self.ngpus_per_node):
        #         self.logger.warning(f"SKIPPING (hardcoded) problematic candidate: {cand_tuple}")
        #     if cand_tuple not in self.vis_dict: self.vis_dict[cand_tuple] = {}
        #     self.vis_dict[cand_tuple]['acc'] = 0.0 
        #     self.vis_dict[cand_tuple]['ops'] = self.m.get_ops(cand)[2]
        #     self.vis_dict[cand_tuple]['visited'] = True
        #     return False 

        if cand_tuple not in self.vis_dict:
            self.vis_dict[cand_tuple] = {}
        info = self.vis_dict[cand_tuple]
        if 'visited' in info: # If already evaluated and in vis_dict
            return False      # Skip full evaluation

        # If not visited, calculate OPs if not already there (should be done before is_legal normally)
        if 'ops' not in info: 
            _, _, info['ops'] = self.m.get_ops(cand)

        # The actual slow evaluation part
        if is_first_gpu(self.args, self.ngpus_per_node): # Log before slow call
             self.logger.info(f"PERFORMING EVALUATION for candidate: {cand_tuple} with OPs: {info['ops']:.4f}")
        info['acc'], _ = get_cand_acc(self.model, cand, self.train_loader,
                                      self.val_loader, self.args, self.logger if is_first_gpu(self.args, self.ngpus_per_node) else None) # Pass logger
        info['visited'] = True
        return True

    def get_random(self, num):
        cnt = 0
        # self.candidates will be populated here
        temp_candidates_to_add = [] # Use a temporary list
        while len(temp_candidates_to_add) < num and cnt < (num + 50): # Add a safety break for cnt
            if cnt == 0:
                # cnt += 1 # Moved increment after successful add
                cand = self.m.smallest_cand
            elif cnt == 1:
                # cnt += 1 # Moved increment after successful add
                cand = self.m.biggest_cand
            else:
                cand = self.m.get_random_range_cand(self.ops_min, self.ops_max)
            
            if self.args.distributed: # Should not be true for your case
                dist.barrier()
                dist.broadcast(cand, 0)

            # Log before OPs and is_legal for random candidates too
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info(f"DEBUG_RANDOM: About to process random candidate: {cand2tuple(cand)}")

            _, _, ops = self.m.get_ops(cand)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info(f"DEBUG_RANDOM: Generated candidate OPs: {ops:.4f}. Range: [{self.ops_min}, {self.ops_max}]")

            if not (self.ops_min <= ops <= self.ops_max):
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.info(f"DEBUG_RANDOM: Candidate OPs {ops:.4f} out of range. Skipping.")
                cnt +=1 # Increment counter even on skip to avoid infinite loop if range is too tight
                continue
            
            # is_legal now logs before its slow part and handles vis_dict checks
            if not self.is_legal(cand): # This will evaluate if new
                # is_legal returns False if already visited, or if the (optional) hardcoded skip is hit
                if is_first_gpu(self.args, self.ngpus_per_node) and cand2tuple(cand) in self.vis_dict and 'visited' in self.vis_dict[cand2tuple(cand)]:
                     self.logger.info(f"DEBUG_RANDOM: Candidate {cand2tuple(cand)} already visited or marked bad. Skipping.")
                cnt +=1 # Increment counter to avoid potential loops on retrying same bad random candidates
                continue

            cand_tuple = cand2tuple(cand)
            # Ensure we don't add duplicates to the initial list if get_random_range_cand is not perfectly unique
            if cand_tuple not in [c for c in temp_candidates_to_add]:
                 temp_candidates_to_add.append(cand_tuple)
                 if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.info('random {}/{}: {}'.format(
                        len(temp_candidates_to_add), num, cand_tuple))
            cnt +=1 # Increment after processing a candidate
        self.candidates.extend(temp_candidates_to_add)


    def get_mutation(self):
        res = []
        attempts = 0 # Safety break
        max_attempts = self.mutation_num * 50 # Allow more attempts

        while len(res) < self.mutation_num and attempts < max_attempts:
            attempts += 1
            if not self.pareto_global: # Handle empty pareto_global if search is too short
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.warning("Pareto front is empty for mutation. Using a random candidate as parent.")
                if not self.candidates: # If even candidates list is empty
                     parent_cand_tuple = cand2tuple(self.m.get_random_range_cand(self.ops_min, self.ops_max))
                else:
                     parent_cand_tuple = random.choice(self.candidates) # Fallback
                ori_cand = tuple2cand(parent_cand_tuple)
            else:
                ori_cand = tuple2cand(
                    random.choice(list(self.pareto_global.values())))
            
            cand = ori_cand.clone()
            # ... (original mutation logic for cand)
            search_space = self.m.search_space
            stage_first = [0] * len(search_space)
            for stage_num in range(len(search_space)):
                num_blocks = search_space[stage_num][1]
                if stage_num > 0:
                    stage_first[stage_num] += stage_first[stage_num - 1] + max(
                        search_space[stage_num - 1][1])
                if random.random() < self.m_prob:
                    d = random.choice(num_blocks)
                    for block_num in range(max(num_blocks)):
                        if block_num < d:
                            cand[stage_first[stage_num] + block_num, 0] = stage_num
                            cand[stage_first[stage_num] + block_num, 1] = block_num
                        else:
                            cand[stage_first[stage_num] + block_num, 0] = -1
                            cand[stage_first[stage_num] + block_num, 1] = -1
            for i in range(cand.shape[0]):
                stage_num = cand[i][0]
                block_num = cand[i][1]
                if stage_num == -1 or block_num == -1: continue
                if random.random() < self.m_prob:
                    if i == 0: last_channel = -1
                    else: last_channel = cand[i - 1][2]
                    channel_cand_list = search_space[stage_num.item()][0][block_num.item()][0]
                    channel_cand = torch.tensor(channel_cand_list)
                    filtered_channel_cand = channel_cand[channel_cand >= last_channel].tolist()
                    if not filtered_channel_cand: filtered_channel_cand = channel_cand_list # Fallback if no larger channel
                    cand[i][2] = random.choice(filtered_channel_cand)
                    cand[i][3] = random.choice(search_space[stage_num.item()][0][block_num.item()][1])
                    cand[i][4] = random.choice(search_space[stage_num.item()][0][block_num.item()][2])
            # ... (end original mutation logic)

            device = next(self.m.parameters()).device
            cand = cand.to(device)
            if self.args.distributed: dist.barrier(); dist.broadcast(cand, 0)
            
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info(f"DEBUG_MUTATION: About to process mutated candidate: {cand2tuple(cand)}")
            
            _, _, ops = self.m.get_ops(cand)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info(f"DEBUG_MUTATION: Generated candidate OPs: {ops:.4f}. Range: [{self.ops_min}, {self.ops_max}]")

            if not (self.ops_min <= ops <= self.ops_max):
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.info(f"DEBUG_MUTATION: Candidate OPs {ops:.4f} out of range. Skipping.")
                continue
            
            if not self.is_legal(cand):
                if is_first_gpu(self.args, self.ngpus_per_node) and cand2tuple(cand) in self.vis_dict and 'visited' in self.vis_dict[cand2tuple(cand)]:
                     self.logger.info(f"DEBUG_MUTATION: Candidate {cand2tuple(cand)} already visited or marked bad. Skipping in is_legal.")
                # If is_legal hung, we wouldn't reach here for that candidate.
                continue
            
            cand_tuple = cand2tuple(cand)
            res.append(cand_tuple)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('mutation {}/{}, {}'.format(
                    len(res), self.mutation_num, cand_tuple))
        
        if len(res) < self.mutation_num and is_first_gpu(self.args, self.ngpus_per_node):
            self.logger.warning(f"MUTATION: Could only generate {len(res)}/{self.mutation_num} valid new candidates after {max_attempts} attempts.")
        return res

    def get_crossover(self):
        res = []
        attempts = 0 # Safety break
        max_attempts = self.crossover_num * 50 # Allow more attempts

        while len(res) < self.crossover_num and attempts < max_attempts:
            attempts += 1
            if not self.pareto_global: # Handle empty pareto_global
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.warning("Pareto front is empty for crossover. Using random candidates as parents.")
                if not self.candidates:
                     parent1_tuple = cand2tuple(self.m.get_random_range_cand(self.ops_min, self.ops_max))
                     parent2_tuple = cand2tuple(self.m.get_random_range_cand(self.ops_min, self.ops_max))
                else:
                     parent1_tuple = random.choice(self.candidates)
                     parent2_tuple = random.choice(self.candidates)
                cand1 = tuple2cand(parent1_tuple)
                cand2 = tuple2cand(parent2_tuple)
            else:
                cand1 = tuple2cand(random.choice(list(self.pareto_global.values())))
                cand2 = tuple2cand(random.choice(list(self.pareto_global.values())))

            # ... (original crossover logic: d_list, mask, cand generation) ...
            search_space = self.m.search_space
            d_list = []
            for i in range(len(search_space)):
                # Count active blocks for cand1 in stage i
                d1_count = 0
                for row_idx in range(cand1.shape[0]):
                    if cand1[row_idx, 0] == i:
                        d1_count +=1
                # Count active blocks for cand2 in stage i
                d2_count = 0
                for row_idx in range(cand2.shape[0]):
                    if cand2[row_idx, 0] == i:
                        d2_count +=1
                d_list.append(random.choice([d1_count, d2_count]))

            mask = torch.rand_like(cand1.float()).round().int()
            cand = mask * cand1 + (1 - mask) * cand2
            
            # Fix stage and block numbers after crossover based on d_list
            current_row_idx = 0
            for stage_idx in range(len(search_space)):
                num_active_blocks_in_stage = d_list[stage_idx]
                max_blocks_in_stage_template = max(search_space[stage_idx][1])
                for block_template_idx in range(max_blocks_in_stage_template):
                    if current_row_idx >= cand.shape[0]: break # Safety break
                    if block_template_idx < num_active_blocks_in_stage:
                        cand[current_row_idx, 0] = stage_idx
                        cand[current_row_idx, 1] = block_template_idx
                    else:
                        cand[current_row_idx, 0] = -1
                        cand[current_row_idx, 1] = -1
                    current_row_idx +=1
                if current_row_idx >= cand.shape[0]: break
            # ... (end original crossover logic) ...

            # --- BEGIN DETAILED DEBUG FOR CHANNEL SORTING (NON-DECREASING CHECK) ---
            is_non_decreasing = True
            last_active_channel = -1 # Start with a very small value
            active_channels_original_log = []

            for i_row in range(cand.shape[0]):
                if cand[i_row, 0] != -1: # If it's an active layer
                    current_channel_val = cand[i_row, 2].item()
                    active_channels_original_log.append(current_channel_val)
                    # Ensure current channel is not smaller than the previous *active* channel
                    # The ND constraint means channel_i+1 >= channel_i
                    if current_channel_val < last_active_channel : # last_active_channel holds the channel of the previous *active* layer
                        is_non_decreasing = False
                        break 
                    last_active_channel = current_channel_val # Update last_active_channel
            
            if not is_non_decreasing:
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.info(f"DEBUG_CROSSOVER_SKIP (Channel Constraint Violation): Active channels: {active_channels_original_log}. Candidate: {cand2tuple(cand)}")
                continue
            # --- END DETAILED DEBUG FOR CHANNEL SORTING ---

            device = next(self.m.parameters()).device
            cand = cand.to(device)
            if self.args.distributed: dist.barrier(); dist.broadcast(cand, 0)

            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info(f"DEBUG_CROSSOVER: About to process crossover candidate: {cand2tuple(cand)}")

            _, _, ops = self.m.get_ops(cand)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info(f"DEBUG_CROSSOVER: Generated candidate OPs: {ops:.4f}. Range: [{self.ops_min}, {self.ops_max}]")
            
            if not (self.ops_min <= ops <= self.ops_max):
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.info(f"DEBUG_CROSSOVER: Candidate OPs {ops:.4f} out of range. Skipping.")
                continue
            
            if not self.is_legal(cand):
                if is_first_gpu(self.args, self.ngpus_per_node) and cand2tuple(cand) in self.vis_dict and 'visited' in self.vis_dict[cand2tuple(cand)]:
                     self.logger.info(f"DEBUG_CROSSOVER: Candidate {cand2tuple(cand)} already visited or marked bad. Skipping in is_legal.")
                continue
            
            cand_tuple = cand2tuple(cand)
            res.append(cand_tuple)
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('crossover {}/{}, {}'.format(
                    len(res), self.crossover_num, cand_tuple))

        if len(res) < self.crossover_num and is_first_gpu(self.args, self.ngpus_per_node):
            self.logger.warning(f"CROSSOVER: Could only generate {len(res)}/{self.crossover_num} valid new candidates after {max_attempts} attempts.")
        return res

    def update_frontier(self):
        # Ensure self.candidates contains tuples, not tensors, for vis_dict lookup
        current_candidates_tuples = []
        if self.candidates and isinstance(self.candidates[0], torch.Tensor): # If it's list of Tensors
            current_candidates_tuples = [cand2tuple(c) for c in self.candidates]
        elif self.candidates and isinstance(self.candidates[0], tuple): # If it's already list of Tuples
            current_candidates_tuples = self.candidates
            
        for cand_tuple in current_candidates_tuples: # Iterate over tuples
            if cand_tuple not in self.vis_dict or 'acc' not in self.vis_dict[cand_tuple]:
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.warning(f"Candidate {cand_tuple} in candidates list but not in vis_dict or no acc. Skipping for frontier update.")
                continue # Should not happen if is_legal was called

            acc = self.vis_dict[cand_tuple]['acc']
            ops = self.vis_dict[cand_tuple]['ops']
            
            f = int(round(ops / self.args.step) * self.args.step) # Bucket by OPs
            # Ensure f is a valid key, e.g. non-negative
            f = max(0, f) # Or handle other way if ops can be very small leading to negative f with large step

            if f not in self.pareto_global or self.vis_dict[self.pareto_global[f]]['acc'] < acc:
                self.pareto_global[f] = cand_tuple
            elif f in self.pareto_global and self.vis_dict[self.pareto_global[f]]['acc'] == acc:
                 # If accuracies are equal, prefer the one with slightly lower OPs if it falls in the same bucket
                 if ops < self.vis_dict[self.pareto_global[f]]['ops']:
                       self.pareto_global[f] = cand_tuple


    def search(self):
        # --- BEGIN RESUME LOGIC ---
        start_fresh = True
        if os.path.exists(self.checkpoint_name):
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info(f"Found existing checkpoint: {self.checkpoint_name}. Attempting to resume.")
            try:
                checkpoint_data = torch.load(self.checkpoint_name, map_location='cpu')
                if checkpoint_data and 'epoch' in checkpoint_data and checkpoint_data['epoch'] > 0:
                    self.memory = checkpoint_data.get('memory', []) # Restore memory
                    # self.candidates list is for the current generation; don't restore, let it be built
                    self.vis_dict = checkpoint_data.get('vis_dict', {})
                    self.epoch = checkpoint_data.get('epoch', 0) 
                    self.pareto_global = checkpoint_data.get('pareto_global', {})
                    start_fresh = False
                    if is_first_gpu(self.args, self.ngpus_per_node):
                        self.logger.info(f"Successfully resumed from epoch {self.epoch}. Found {len(self.vis_dict)} visited architectures.")
                else:
                    if is_first_gpu(self.args, self.ngpus_per_node):
                        self.logger.info("Checkpoint found but seems to be from an incomplete initial run (epoch 0 or empty) or invalid. Starting fresh.")
            except Exception as e:
                if is_first_gpu(self.args, self.ngpus_per_node):
                    self.logger.error(f"Could not load or parse checkpoint {self.checkpoint_name}: {e}. Starting search from scratch.")
        else:
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('No checkpoint found. Starting search from scratch.')

        if start_fresh:
            self.epoch = 0 
            self.memory = []
            self.vis_dict = {}
            self.candidates = [] 
            self.pareto_global = {}
            if is_first_gpu(self.args, self.ngpus_per_node):
                 self.logger.info('Initializing new search: population_num = {} mutation_num = {} crossover_num = {} max_epochs = {}'.format(
                                 self.population_num, self.mutation_num, self.crossover_num, self.max_epochs))
                 self.logger.info('Init: Generating random population...')
            self.get_random(self.population_num) # Populates self.candidates with initial random set
            self.update_frontier() # Updates pareto_global based on initial self.candidates
            # Save checkpoint after initial population is evaluated
            self.save_checkpoint() # Save state after random init and first frontier update
        # --- END RESUME LOGIC ---
        
        if is_first_gpu(self.args, self.ngpus_per_node) and not start_fresh:
            self.logger.info('Resuming search from epoch {}: population_num = {} mutation_num = {} crossover_num = {} max_epochs = {}'.format(
                             self.epoch, self.population_num, self.mutation_num, self.crossover_num, self.max_epochs))


        while self.epoch < self.max_epochs:
            if is_first_gpu(self.args, self.ngpus_per_node):
                self.logger.info('Epoch: {}/{}'.format(self.epoch, self.max_epochs))

            # self.candidates at this point (if not start_fresh) would be from the restored checkpoint,
            # which might be empty or from a previous partial generation.
            # For a resumed epoch, we want to generate NEW mutations and crossovers.
            # The original self.candidates (from random_get or previous epoch) were used to build pareto_global.
            
            current_generation_parents = list(self.pareto_global.values())
            if not current_generation_parents: # If pareto is empty, fallback
                if self.vis_dict: # Try to get some candidates from vis_dict if pareto empty
                    current_generation_parents = list(self.vis_dict.keys())
                if not current_generation_parents and self.candidates: # Fallback to initial candidates
                     current_generation_parents = self.candidates
                if not current_generation_parents: # Absolute fallback
                    if is_first_gpu(self.args, self.ngpus_per_node):
                        self.logger.warning("No candidates available for mutation/crossover parents. Generating one random.")
                    current_generation_parents = [cand2tuple(self.m.get_random_range_cand(self.ops_min, self.ops_max))]


            # Store current candidates (which are parents for this generation) in memory for this epoch
            self.memory.append([])
            # The actual candidates list that update_frontier uses should be the newly generated ones.
            # The original code had:
            # for cand_tuple in self.candidates: self.memory[-1].append(cand_tuple)
            # This should probably refer to the parents used for this generation or the children generated.
            # Let's assume memory is for logging the state of candidates used or produced.
            # For now, let's log the parents used for this generation.
            for parent_tuple in current_generation_parents:
                 self.memory[-1].append(parent_tuple)


            mutation = self.get_mutation() # Returns list of new cand_tuples
            crossover = self.get_crossover() # Returns list of new cand_tuples

            self.candidates = mutation + crossover # These are the NEWLY evaluated candidates for this generation
                                                 # update_frontier will use these
            self.update_frontier()

            if is_first_gpu(self.args, self.ngpus_per_node):
                ops_stages = sorted(list(self.pareto_global.keys()))
                self.logger.info(f"--- Pareto Front after Epoch {self.epoch} ---")
                for s_ops_key in ops_stages: # Renamed 's' to avoid conflict
                    cand_tuple_pf = self.pareto_global[s_ops_key] # Renamed 'cand_tuple'
                    if cand_tuple_pf in self.vis_dict and 'acc' in self.vis_dict[cand_tuple_pf] and 'ops' in self.vis_dict[cand_tuple_pf]:
                        self.logger.info(
                            'OPs Stage {:.2f}, {}, Top-1 acc: {:.2f}%, OPs: {:.4f}M'.
                            format(s_ops_key * self.args.step if isinstance(s_ops_key, (int, float)) else s_ops_key, # Adjust if key is not direct ops
                                   cand_tuple_pf, 
                                   self.vis_dict[cand_tuple_pf]['acc'],
                                   self.vis_dict[cand_tuple_pf]['ops']))
                    else:
                        self.logger.warning(f"Candidate {cand_tuple_pf} from pareto_global not fully processed in vis_dict for OPs stage {s_ops_key}")
                self.logger.info(f"--- End Pareto Front ---")

            self.epoch += 1
            self.save_checkpoint()

        self.save_checkpoint() # Final save


# --- main() and main_worker() (Keep your existing functions here) ---
# Make sure to pass 'logger' to get_cand_acc if it's available from main_worker
def main():
    args = parser.parse_args()
    # seed(args.seed) # Original placement
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    seed(args.seed) # Moved seed call here to ensure it's called in each worker if multiprocessing_distributed is used
    args.gpu = gpu
    logger = None # Initialize logger to None
    if args.gpu is not None:
        # This print is fine for console, but logger might not be initialized yet for rank > 0
        if not args.distributed or (args.distributed and (args.rank * ngpus_per_node + gpu == 0)): # More robust check for first effective GPU
             print(f'Use GPU: {args.gpu} for training')


    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if is_first_gpu(args, ngpus_per_node):
        if not osp.exists(args.logdir):
            os.makedirs(args.logdir)
        logger = get_logger(name='Search', log_file=osp.join(args.logdir, 'search.log'),
                            log_level=logging.INFO)
        logger.info(args)
    # else: # logger remains None for other processes

    t_start_main_worker = time.time() # For total time logging
    if is_first_gpu(args, ngpus_per_node) and logger:
        logger.info(f"=> creating model '{args.arch}'")
    model = models.__dict__[args.arch]()

    if os.path.isfile(args.supernet):
        if is_first_gpu(args, ngpus_per_node) and logger:
            logger.info(f"=> loading checkpoint '{args.supernet}'")
        checkpoint = torch.load(args.supernet, map_location='cpu')
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        model.load_state_dict(state_dict, strict=False) # Use strict=False if supernet has more params than a specific arch
        if is_first_gpu(args, ngpus_per_node) and logger:
            logger.info(f"=> loaded checkpoint '{args.supernet}'")
    else:
        if is_first_gpu(args, ngpus_per_node) and logger:
            logger.info(f"=> no checkpoint found at '{args.supernet}'")
        exit(0) # Exit if supernet not found

    # --- GPU Setup and Model Placement ---
    if not torch.cuda.is_available():
        if is_first_gpu(args, ngpus_per_node) and logger: # Check logger existence
            logger.info('using CPU, this will be slow')
        elif is_first_gpu(args, ngpus_per_node): # Fallback print if logger somehow not init for first GPU
            print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if is_first_gpu(args, ngpus_per_node) and logger:
                logger.info(f"DIAGNOSTIC (main_worker DDP): Model explicitly moved to {next(model.parameters()).device if list(model.parameters()) else 'N/A - No Params'}")
            args.train_batch_size = int(args.train_batch_size / ngpus_per_node)
            args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            if is_first_gpu(args, ngpus_per_node) and logger:
                logger.info(f"DIAGNOSTIC (main_worker DDP no specific GPU): Model moved to {next(model.parameters()).device if list(model.parameters()) else 'N/A - No Params'}")
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if is_first_gpu(args, ngpus_per_node) and logger:
            logger.info(f"DIAGNOSTIC (main_worker single GPU): Model explicitly moved to {next(model.parameters()).device if list(model.parameters()) else 'N/A - No Params'}")
    else: # Fallback to DataParallel if no specific GPU and not distributed, but CUDA is available
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
        if is_first_gpu(args, ngpus_per_node) and logger:
             logger.info(f"DIAGNOSTIC (main_worker DataParallel): Model moved to {next(model.parameters()).device if list(model.parameters()) else 'N/A - No Params'}")
    # --- End GPU Setup ---

    searcher = EvolutionSearcher(model, logger, args, ngpus_per_node) # Pass logger
    searcher.search()
    
    if is_first_gpu(args, ngpus_per_node) and logger:
        logger.info('total searching time = {:.2f} hours'.format(
            (time.time() - t_start_main_worker) / 3600)) # Use t_start_main_worker
# --- End main_worker ---


# --- accuracy() and no_grad_wrapper() (Keep your existing functions) ---
def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k_val in topk: # Renamed k to k_val
            correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func
# --- End accuracy() and no_grad_wrapper() ---


# --- get_cand_acc() with diagnostic prints ---
@no_grad_wrapper
def get_cand_acc(model, cand, train_loader, val_loader, args, logger=None): # Added logger parameter

    class DataIterator:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.iterator = enumerate(self.dataloader)
        def next(self):
            try: _, data = next(self.iterator)
            except Exception:
                self.iterator = enumerate(self.dataloader)
                _, data = next(self.iterator)
            return data[0], data[1]

    train_provider = DataIterator(train_loader)
    max_train_iters = args.max_train_iters

    model.eval() # Set to eval mode
    # BN calibration part
    for m_module in model.modules(): # Renamed m to m_module
        if isinstance(m_module, nn.BatchNorm2d):
            m_module.training = True # Set BN to training mode for stat update
            m_module.momentum = None  # Use cumulative moving average
            m_module.reset_running_stats()

    # BN Calibration Loop
    with torch.no_grad(): # Ensure no gradients are computed here
        for step in range(max_train_iters):
            images, _ = train_provider.next()
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            
            if step == 0 and logger: # Print only for the first BN calib batch if logger exists
                # Check model parameters device (should be cuda:X from main_worker)
                model_device_str = "N/A"
                try: model_device_str = str(next(model.parameters()).device)
                except StopIteration: pass # No parameters
                logger.info(f"DIAGNOSTIC (get_cand_acc BN calib): model device: {model_device_str}, images device: {images.device}")
            
            model(images, cand) # Forward pass to update BN stats

    # Actual Evaluation Part
    # device_eval = next(model.parameters()).device # Get device after BN calib (should be the same)
    # It's safer to get device from args.gpu or default to cpu if not available
    eval_device_target = torch.device(f"cuda:{args.gpu}") if args.gpu is not None and torch.cuda.is_available() else torch.device("cpu")

    top1 = torch.tensor([0.], device=eval_device_target) # Ensure these are on the correct device
    top5 = torch.tensor([0.], device=eval_device_target)
    total = torch.tensor([0.], device=eval_device_target)

    model.eval() # Ensure model is in eval mode for actual validation
    batch_idx = 0 # For logging eval batch
    for images, target in val_loader:
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        # Target also needs to be on the same device as output for comparison
        if torch.cuda.is_available() and args.gpu is not None:
            target = target.cuda(args.gpu, non_blocking=True)
        else: # If GPU not used for target, ensure it's on CPU if output is also CPU
             target = target.to(eval_device_target, non_blocking=True)


        batchsize = images.shape[0] 
        
        if batch_idx == 0 and logger: # Print only for the first eval batch
            model_device_str = "N/A"
            try: model_device_str = str(next(model.parameters()).device)
            except StopIteration: pass
            logger.info(f"DIAGNOSTIC (get_cand_acc eval): model device: {model_device_str}, images device: {images.device}, target device: {target.device}")
        
        output, _ = model(images, cand)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1 += acc1.item() * batchsize 
        top5 += acc5.item() * batchsize
        total += batchsize
        batch_idx +=1

        del images, target, output, acc1, acc5 # Clean up
    
    if args.distributed: # Should not be true for your case but good to keep
        dist.barrier()
        dist.all_reduce(top1)
        dist.all_reduce(top5)
        dist.all_reduce(total)
    
    # Handle division by zero if total is zero (e.g., empty val_loader)
    res_top1 = (top1.item() / total.item()) if total.item() > 0 else 0.0
    res_top5 = (top5.item() / total.item()) if total.item() > 0 else 0.0
    
    return res_top1, res_top5
# --- End get_cand_acc ---


if __name__ == '__main__':
    main()