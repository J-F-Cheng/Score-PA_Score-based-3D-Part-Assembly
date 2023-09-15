"""
    Training models
"""

import functools
import os
import random
import shutil
import sys
from argparse import ArgumentParser
from subprocess import call

import model_score_based
import numpy as np
import setproctitle
import torch
import torch.utils.data
import tqdm
from quaternion import qrot
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_sum

import utils
from data_dynamic import PartNetPartDataset
from point_cloud_render import point_cloud_render
from samplers import samples_gen
from utils import (diffusion_coeff, marginal_prob_std,
                   quaternion_to_euler_torch_data)

INPUT_DIM = 6

def train(conf):
    # create training and validation datasets and data loaders
    data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'pairs']
    
    train_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.train_data_fn, data_features, \
            max_num_part=conf.max_num_part, level=conf.level)
    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
            num_workers=conf.num_workers, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)
    
    val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, data_features, \
            max_num_part=conf.max_num_part,level=conf.level)
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)

    # create models
    sigma = conf.sigma
    print("sigma: ", sigma)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, conf=conf)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, conf=conf)
    network = model_score_based.Network(conf, marginal_prob_std=marginal_prob_std_fn, input_dim=6)
    if conf.continue_train_epoch > 0:
        print("Continue training")
        network.load_state_dict(torch.load(conf.cont_model_dir))

    utils.printout(conf.flog, '\n' + str(network) + '\n')

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr)

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # train for every epoch
    tqdm_epoch = tqdm.trange(conf.continue_train_epoch, conf.epochs)

    if conf.part_shuffle:
        print("Part shuffle is enabled")

    for epoch in tqdm_epoch:
        avg_loss = 0
        total_items = 0
        
        train_batches = enumerate(train_dataloader, 0)

        val_batches = enumerate(val_dataloader, 0)
        val_batch_ind = -1


        # train for every batch
        for train_batch_ind, batch in train_batches:
            # set models to training mode
            network.train()

            # forward pass (including logging)
            if len(batch)==0:continue
            loss, cur_num_items = forward(batch=batch, data_features=data_features, network=network, conf=conf, margin_fn=marginal_prob_std_fn)
            # optimize one step
            network_opt.zero_grad()
            loss.backward()
            network_opt.step()

            avg_loss += loss.item() * cur_num_items
            total_items += cur_num_items
            tqdm_epoch.set_description('Epoch {}, Average Loss: {:5f}'.format(epoch + 1, avg_loss / total_items))

        utils.printout(conf.flog, 'Epoch {}, Average Loss: {:5f}'.format(epoch + 1, avg_loss / total_items))

        # validate one batch
        if (epoch + 1) % conf.num_epoch_every_val == 0:
            utils.printout(conf.flog, 'Saving checkpoint ...... ')
            # save checkpoint
            torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', 'epoch_{}.pth'.format(epoch + 1)))
            utils.printout(conf.flog, 'DONE')
            print("validating")
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, "epoch_{}".format(epoch + 1))
            input_part_pcs_dir = os.path.join(out_dir, 'input_part_pcs')
            gt_assembly_dir = os.path.join(out_dir, 'gt_assembly')
            pred_assembly_dir = os.path.join(out_dir, 'pred_assembly')
            info_dir = os.path.join(out_dir, 'info')

            # create folders
            os.mkdir(out_dir)
            os.mkdir(input_part_pcs_dir)
            os.mkdir(gt_assembly_dir)
            os.mkdir(pred_assembly_dir)
            os.mkdir(info_dir)

            val_batch_ind, val_batch = next(val_batches)

            # set models to evaluation mode
            network.eval()

            with torch.no_grad():
                # forward pass (including logging)
                if len(val_batch)==0:continue
                val_forward(batch=val_batch, data_features=data_features, network=network, conf=conf, margin_fn=marginal_prob_std_fn,
                            diffusion_coeff_fn=diffusion_coeff_fn, is_val=True, epoch=epoch, batch_ind=val_batch_ind, )
    # save the final models
    utils.printout(conf.flog, 'Saving final checkpoint ...... ')
    # save checkpoint
    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', 'final.pth'))
    utils.printout(conf.flog, 'DONE')


def forward(batch, data_features, network, conf, margin_fn, eps=1e-5):
    # prepare input
    input_part_pcs = torch.cat(batch[data_features.index('part_pcs')], dim=0).to(conf.device)           # B x P x N x 3
    input_part_valids = torch.cat(batch[data_features.index('part_valids')], dim=0).to(conf.device)     # B x P
    # input_part_pairs = torch.cat(batch[data_features.index('pairs')], dim=0).to(conf.device)
    batch_size = input_part_pcs.shape[0] 
    num_part = input_part_pcs.shape[1]
    # num_point = input_part_pcs.shape[2]
    part_ids = torch.cat(batch[data_features.index('part_ids')], dim=0).to(conf.device)      # B x P
    match_ids=batch[data_features.index('match_ids')]
    gt_part_poses = torch.cat(batch[data_features.index('part_poses')], dim=0).to(conf.device)      # B x P x (3 + 4)

    if conf.part_shuffle:
        match_ids = torch.tensor(match_ids)
        for bat_idx in range(batch_size):
            real_num_part = int(input_part_valids[bat_idx].sum().item())
            rand_idx = torch.randperm(real_num_part)
            input_part_pcs[bat_idx, :real_num_part] = input_part_pcs[bat_idx, rand_idx]
            part_ids[bat_idx, :real_num_part] = part_ids[bat_idx, rand_idx]
            match_ids[bat_idx, :real_num_part] = match_ids[bat_idx, rand_idx]
            gt_part_poses[bat_idx, :real_num_part] = gt_part_poses[bat_idx, rand_idx]
        match_ids = match_ids.tolist()


    # get instance label
    instance_label = torch.zeros(batch_size, num_part, num_part).to(conf.device)
    same_class_list = []
    for i in range(batch_size):
        num_class = [ 0 for i in range(160) ]
        cur_same_class_list = [[] for i in range(160)]
        for j in range(num_part):
            cur_class = int(part_ids[i][j])
            if j < input_part_valids[i].sum():
                cur_same_class_list[cur_class].append(j)
            if cur_class == 0: continue
            cur_instance = int(num_class[cur_class])
            instance_label[i][j][cur_instance] = 1
            num_class[int(part_ids[i][j])] += 1
        for i in range(cur_same_class_list.count([])):
            cur_same_class_list.remove([])
        same_class_list.append(cur_same_class_list)
    
    proc_gt_part_poses = []
    proc_input_part_pcs = []
    proc_batch_code = []
    proc_edge = []
    proc_instance_label = []

    for i in range(batch_size):
        real_num_part = int(input_part_valids[i].sum().item())
        batch_code = torch.tensor([i for _ in range(real_num_part)])
        proc_batch_code.append(batch_code)
        pre_poses = gt_part_poses[i, :real_num_part]

        if real_num_part >= 2:
            edge = knn_graph(pre_poses, k=real_num_part - 1)
        else:
            edge = torch.zeros(2, 1, dtype=torch.int64, device=conf.device)

        edge_shift = 0
        for j in range(i):
            edge_shift += int(input_part_valids[j].sum().item())
        edge += edge_shift
        proc_edge.append(edge)
        proc_gt_part_poses.append(pre_poses)
        pre_pcs = input_part_pcs[i, :real_num_part]
        proc_input_part_pcs.append(pre_pcs)
        proc_instance_label.append(instance_label[i, :real_num_part])

    proc_batch_code = torch.cat(proc_batch_code, dim=0)
    proc_edge = torch.cat(proc_edge, dim=1)
    proc_gt_part_poses = torch.cat(proc_gt_part_poses, dim=0)
    proc_gt_part_poses = quaternion_to_euler_torch_data(proc_gt_part_poses, "xyz", conf.device)
    proc_input_part_pcs = torch.cat(proc_input_part_pcs, dim=0)
    proc_instance_label = torch.cat(proc_instance_label, dim=0)

    pose_input = Data(x=proc_gt_part_poses, edge_index=proc_edge, batch=proc_batch_code)

    random_t = torch.rand(batch_size, device=conf.device) * (1. - eps) + eps
    random_t = random_t.unsqueeze(-1)
    random_t = random_t[proc_batch_code]

    z = torch.randn_like(pose_input.x)
    std = margin_fn(random_t)
    pose_input.x = pose_input.x + z * std
    emb_pcs = network.get_part_feature(proc_input_part_pcs.float())
    score = network(x_pose=pose_input, t=random_t, proc_part_pcs=emb_pcs, 
                                        instance_label=proc_instance_label)
    node_l2 = torch.sum((score * std + z) ** 2, dim=-1)
    batch_l2 = scatter_sum(node_l2, pose_input.batch.to(conf.device), dim=0)
    loss = torch.mean(batch_l2)
    return loss, batch_size

def val_forward(batch, data_features, network, conf, margin_fn, diffusion_coeff_fn,\
            is_val=False, epoch=None, batch_ind=0, eps=1e-3, atol=1e-5, rtol=1e-5,):
    # prepare input
    input_part_pcs = torch.cat(batch[data_features.index('part_pcs')], dim=0).to(conf.device)  # B x P x N x 3
    input_part_valids = torch.cat(batch[data_features.index('part_valids')], dim=0).to(conf.device)  # B x P
    input_part_pairs = torch.cat(batch[data_features.index('pairs')], dim=0).to(conf.device)
    batch_size = input_part_pcs.shape[0]
    num_part = input_part_pcs.shape[1]
    num_point = input_part_pcs.shape[2]
    part_ids = torch.cat(batch[data_features.index('part_ids')], dim=0).to(conf.device)  # B x P
    # match_ids = batch[data_features.index('match_ids')]
    gt_part_poses = torch.cat(batch[data_features.index('part_poses')], dim=0).to(conf.device)  # B x P x (3 + 4)
    # get instance label
    instance_label = torch.zeros(batch_size, num_part, num_part).to(conf.device)
    same_class_list = []
    for i in range(batch_size):
        num_class = [0 for i in range(160)]
        cur_same_class_list = [[] for i in range(160)]
        for j in range(num_part):
            cur_class = int(part_ids[i][j])
            if j < input_part_valids[i].sum():
                cur_same_class_list[cur_class].append(j)
            if cur_class == 0: continue
            cur_instance = int(num_class[cur_class])
            instance_label[i][j][cur_instance] = 1
            num_class[int(part_ids[i][j])] += 1
        for i in range(cur_same_class_list.count([])):
            cur_same_class_list.remove([])
        same_class_list.append(cur_same_class_list)

    new_samples = samples_gen(conf, margin_fn, diffusion_coeff_fn, INPUT_DIM,
                input_part_valids, input_part_pcs, instance_label, input_part_pairs, same_class_list,
                network, eps, atol, rtol)

    pred_part_poses = new_samples["pred_part_poses"]

    return_to_orig_x = []
    for i in range(batch_size):

        real_num_part = int(input_part_valids[i].sum().item())
        comple_x = torch.zeros(num_part - real_num_part, 7, device=conf.device)
        batch_shift = 0
        for j in range(i):
            batch_shift += int(input_part_valids[j].sum().item())
        single_orig_x = pred_part_poses[batch_shift:batch_shift + real_num_part]
        single_orig_x = torch.cat([single_orig_x, comple_x], dim=0)
        return_to_orig_x.append(single_orig_x)
    pred_part_poses = torch.stack(return_to_orig_x).float()

    if is_val and (not conf.no_visu):
        visu_dir = os.path.join(conf.exp_dir, 'val_visu')
        out_dir = os.path.join(visu_dir, "epoch_{}".format(epoch + 1))
        input_part_pcs_dir = os.path.join(out_dir, 'input_part_pcs')
        gt_assembly_dir = os.path.join(out_dir, 'gt_assembly')
        pred_assembly_dir = os.path.join(out_dir, 'pred_assembly')
        info_dir = os.path.join(out_dir, 'info')
        utils.printout(conf.flog, 'Visualizing ...')

        for i in range(batch_size):
            fn = 'data-%03d.png' % (batch_ind * batch_size + i)

            cur_input_part_cnt = input_part_valids[i].sum().item()
            cur_input_part_cnt = int(cur_input_part_cnt)
            cur_input_part_pcs = input_part_pcs[i, :cur_input_part_cnt]
            cur_gt_part_poses = gt_part_poses[i, :cur_input_part_cnt]
            cur_pred_part_poses = pred_part_poses[i, :cur_input_part_cnt]

            pred_part_pcs = qrot(cur_pred_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1),
                                 cur_input_part_pcs) + \
                            cur_pred_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
            gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + \
                          cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

            point_cloud_render(os.path.join(input_part_pcs_dir, fn), cur_input_part_pcs, conf)
            point_cloud_render(os.path.join(pred_assembly_dir, fn), pred_part_pcs, conf)
            point_cloud_render(os.path.join(gt_assembly_dir, fn), gt_part_pcs, conf)

            with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                fout.write('shape_id: %s\n' % batch[data_features.index('shape_id')][i])
                fout.write('num_part: %d\n' % cur_input_part_cnt)



if __name__ == '__main__':

    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--train_data_fn', type=str, help='training data file that indexs all data tuples')
    parser.add_argument('--val_data_fn', type=str, help='validation data file that indexs all data tuples')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--data_dir', type=str, default='./prep_data', help='data directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--level',type=str,default='3',help='level of dataset')
    parser.add_argument('--max_num_part', type=int, default=20)

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)
    parser.add_argument('--sigma', type=float, default=25.0)

    # training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)

    # Sampler options
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--snr', type=float, default=0.16)
    parser.add_argument('--t0', type=float, default=1.0)
    parser.add_argument('--cor_steps', type=int, default=1)
    parser.add_argument('--cor_final_steps', type=int, default=1)
    parser.add_argument('--noise_decay_pow', type=int, default=1)

    # visu
    parser.add_argument('--num_epoch_every_val', type=int, default=1, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')
    parser.add_argument('--part_shuffle', action='store_false', default=True, help='shuffle parts?')
    parser.add_argument('--continue_train_epoch', type=int, default=0, help='0 is for train from beginning. Otherwise, continued training')
    parser.add_argument('--cont_model_dir', type=str, help='the path of the model for continued training')

    # parse args
    conf = parser.parse_args()
    print("conf", conf)


    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.category}-level{conf.level}-{conf.exp_suffix}'
    
    # mkdir exp_dir; ask for overwrite if necessary
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    if not os.path.exists(conf.log_dir):
       os.mkdir(conf.log_dir) 
    if os.path.exists(conf.exp_dir):
        if not conf.overwrite:
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
            if response != 'y':
                exit(1)
        shutil.rmtree(conf.exp_dir)
    os.mkdir(conf.exp_dir)
    os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
    if not conf.no_visu:
        os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # control randomness
    if conf.seed < 0 or conf.continue_train_epoch > 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')
     
    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device


    setproctitle.setproctitle("train in " + conf.category)

    ### start training
    train(conf)

    ### before quit
    # close file log
    flog.close()

