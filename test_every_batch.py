import os

import torch
import torch.utils.data
from quaternion import qrot

from point_cloud_render import point_cloud_render
from samplers import samples_gen
from utils import shape_diversity_score

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIM = 6

def forward(batch, data_features, network, conf, margin_fn, diffusion_coeff_fn,\
        is_val=False, batch_ind=0, eps=1e-3, atol=1e-5, rtol=1e-5):
    # prepare input
    input_part_pcs = torch.cat(batch[data_features.index('part_pcs')], dim=0).to(conf.device)           # B x P x N x 3
    input_part_valids = torch.cat(batch[data_features.index('part_valids')], dim=0).to(conf.device)     # B x P
    input_part_pairs = torch.cat(batch[data_features.index('pairs')], dim=0).to(conf.device)
    batch_size = input_part_pcs.shape[0]
    num_part = input_part_pcs.shape[1]
    num_point = input_part_pcs.shape[2]
    part_ids = torch.cat(batch[data_features.index('part_ids')], dim=0).to(conf.device)      # B x P 
    match_ids=batch[data_features.index('match_ids')]  
    gt_part_poses = torch.cat(batch[data_features.index('part_poses')], dim=0).to(conf.device)      # B x P x (3 + 4)
    
    
    contact_points = torch.cat(batch[data_features.index("contact_points")], dim=0).to(conf.device)
    
    # cope with the sym_info
    sym_info = torch.cat(batch[data_features.index("sym")], dim=0)  # B x P x 3

    # clone the input part pcs for shuffle
    input_part_pcs_shuffle = input_part_pcs.clone()

    shuffle_for_each_batch = []

    if conf.part_shuffle:
        # match_ids = torch.tensor(match_ids)
        print("Part shuffle is enabled")
        for bat_idx in range(batch_size):
            real_num_part = int(input_part_valids[bat_idx].sum().item())
            rand_idx = torch.randperm(real_num_part)
            shuffle_for_each_batch.append(rand_idx)
            input_part_pcs_shuffle[bat_idx, :real_num_part] = input_part_pcs_shuffle[bat_idx, rand_idx]
            part_ids[bat_idx, :real_num_part] = part_ids[bat_idx, rand_idx]
    
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

    repeat_times = conf.repeat_times_per_shape
    array_trans_l2_loss_per_data = []
    array_rot_l2_loss_per_data = []
    array_rot_cd_loss_per_data = []
    array_total_cd_loss_per_data = []
    array_shape_cd_loss_per_data = []
    array_contact_point_loss_per_data = []
    array_acc = []
    array_pred_part_poses = []
    array_sds_cd_per_data = []
    
    for repeat_ind in range(repeat_times):
        new_samples = samples_gen(conf, margin_fn, diffusion_coeff_fn, INPUT_DIM,
                    input_part_valids, input_part_pcs_shuffle, instance_label, input_part_pairs, same_class_list,
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

        for i in range(batch_size):
            inv_rand_idx = torch.zeros_like(shuffle_for_each_batch[i], dtype=torch.long)
            real_num_part = int(shuffle_for_each_batch[i].size(0))
            inv_rand_idx[shuffle_for_each_batch[i]] = torch.arange(real_num_part)
            pred_part_poses[i, :real_num_part] = pred_part_poses[i, inv_rand_idx]


        # pred_part_poses = gt_part_poses
        array_pred_part_poses.append(pred_part_poses)

        # matching loss
        for ind in range(len(batch[0])):
            cur_match_ids = match_ids[ind]
            for i in range(1, 10):
                need_to_match_part = []
                for j in range(conf.max_num_part):
                    if cur_match_ids[j] == i:
                        need_to_match_part.append(j)
                if len(need_to_match_part) == 0: break
                cur_input_pts = input_part_pcs[ind, need_to_match_part]
                cur_pred_poses = pred_part_poses[ind, need_to_match_part]
                cur_pred_centers = cur_pred_poses[:, :3]
                cur_pred_quats = cur_pred_poses[:, 3:]
                cur_gt_part_poses = gt_part_poses[ind, need_to_match_part]
                cur_gt_centers = cur_gt_part_poses[:, :3]
                cur_gt_quats = cur_gt_part_poses[:, 3:]
                matched_pred_ids, matched_gt_ids = network.linear_assignment(cur_input_pts, cur_pred_centers,
                                                                             cur_pred_quats, cur_gt_centers,
                                                                             cur_gt_quats)
                match_pred_part_poses = pred_part_poses[ind, need_to_match_part][matched_pred_ids]
                pred_part_poses[ind, need_to_match_part] = match_pred_part_poses
                match_gt_part_poses = gt_part_poses[ind, need_to_match_part][matched_gt_ids]
                gt_part_poses[ind, need_to_match_part] = match_gt_part_poses

        # prepare gt
        input_part_pcs = input_part_pcs[:, :, :1000, :]
        # for each type of loss, compute losses per data
        trans_l2_loss_per_data = network.get_trans_l2_loss(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3],
                                                           input_part_valids)  # B
        rot_l2_loss_per_data = network.get_rot_l2_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                       gt_part_poses[:, :, 3:], input_part_valids)  # B
        rot_cd_loss_per_data = network.get_rot_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                       gt_part_poses[:, :, 3:], input_part_valids, conf.device)  # B

        # prepare gt
        input_part_pcs = input_part_pcs[:, :, :1000, :]
        # if iter_ind == 2:
        total_cd_loss_per_data, acc = network.get_total_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                                gt_part_poses[:, :, 3:],
                                                                input_part_valids, pred_part_poses[:, :, :3],
                                                                gt_part_poses[:, :, :3], conf.device)  # B)
        # total_cd_loss = total_cd_loss_per_data.mean()
        shape_cd_loss_per_data = network.get_shape_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                           gt_part_poses[:, :, 3:],
                                                           input_part_valids, pred_part_poses[:, :, :3],
                                                           gt_part_poses[:, :, :3], conf.device)
        
        contact_point_loss_per_data, count, total_num, batch_count, batch_total_num = network.batch_get_contact_point_loss(pred_part_poses[:, :, :3],
                                                                     pred_part_poses[:, :, 3:], contact_points, sym_info)

        batch_single_ca = batch_count.float() / batch_total_num.float()
        mask_nan = torch.isnan(batch_single_ca)
        batch_single_ca[mask_nan] = 0.0
        array_sds_cd_per_data.append([
            input_part_pcs.clone(),
            pred_part_poses[:, :, :].clone(),
            input_part_valids.clone(),
            shape_cd_loss_per_data.clone(),
            batch_single_ca.to(conf.device)
        ])

        array_trans_l2_loss_per_data.append(trans_l2_loss_per_data)
        array_rot_l2_loss_per_data.append(rot_l2_loss_per_data)
        array_rot_cd_loss_per_data.append(rot_cd_loss_per_data)
        array_total_cd_loss_per_data.append(total_cd_loss_per_data)
        array_shape_cd_loss_per_data.append(shape_cd_loss_per_data)
        array_contact_point_loss_per_data.append(contact_point_loss_per_data)
        # B x P -> B
        acc = torch.tensor(acc)
        acc = acc.sum(-1).float()  # B
        valid_number = input_part_valids.sum(-1).float().cpu()  # B
        acc_rate = acc / valid_number
        array_acc.append(acc_rate)
        count = torch.tensor(count)

        if repeat_ind == 0:
            res_total_cd = total_cd_loss_per_data
            res_shape_cd = shape_cd_loss_per_data
            res_contact_point = contact_point_loss_per_data
            res_acc = acc
            res_count = count
        else:
            res_total_cd = res_total_cd.min(total_cd_loss_per_data)
            res_shape_cd = res_shape_cd.min(shape_cd_loss_per_data)
            res_contact_point = res_contact_point.min(contact_point_loss_per_data)
            res_acc = res_acc.max(acc)  # B
            res_count = res_count.max(count)

    acc_num = res_acc.sum()  # how many parts are right in total in a certain batch
    valid_num = input_part_valids.sum()  # how many parts in total in a certain batch

    cdsV1, cdsV2 = shape_diversity_score(array_sds_cd_per_data, network, conf, batch_size)

    # computer real matric
    real_shape_cd_loss = res_shape_cd.sum()
    real_total_cd_loss = res_total_cd.sum()
    real_contact_point_loss = res_contact_point.sum()
    cdsV1_sum = cdsV1.sum()
    cdsV2_sum = cdsV2.sum()
    real_batch_size = batch_size

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'
    with torch.no_grad():
        # gen visu
        # is_val = False
        if is_val and (not conf.no_visu):
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'test_' + str(conf.out_dir))
            input_part_pcs_dir = os.path.join(out_dir, 'input_part_pcs')
            gt_assembly_dir = os.path.join(out_dir, 'gt_assembly')
            pred_assembly_dir = os.path.join(out_dir, 'pred_assembly')
            info_dir = os.path.join(out_dir, 'info')

            if batch_ind == 0:
                os.mkdir(out_dir)
                os.mkdir(input_part_pcs_dir)
                os.mkdir(gt_assembly_dir)
                os.mkdir(pred_assembly_dir)
                os.mkdir(info_dir)

            for repeat_ind in range(repeat_times):
                for i in range(batch_size):
                    fn = 'data-%03d-%03d.png' % (batch_ind * conf.batch_size + i, repeat_ind)

                    cur_input_part_cnt = input_part_valids[i].sum().item()
                    cur_input_part_cnt = int(cur_input_part_cnt)
                    cur_input_part_pcs = input_part_pcs[i, :cur_input_part_cnt]
                    cur_gt_part_poses = gt_part_poses[i, :cur_input_part_cnt]
                    cur_pred_part_poses = array_pred_part_poses[repeat_ind][i, :cur_input_part_cnt]

                    pred_part_pcs = qrot(cur_pred_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1),
                                            cur_input_part_pcs) + \
                                    cur_pred_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
                    gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1),
                                        cur_input_part_pcs) + \
                                    cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

                    point_cloud_render(os.path.join(input_part_pcs_dir, fn), cur_input_part_pcs, conf)
                    point_cloud_render(os.path.join(pred_assembly_dir, fn), pred_part_pcs, conf)
                    point_cloud_render(os.path.join(gt_assembly_dir, fn), gt_part_pcs, conf)

                    with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                        fout.write('shape_id: %s\n' % batch[data_features.index('shape_id')][i])
                        fout.write('num_part: %d\n' % cur_input_part_cnt)
                        fout.write('trans_l2_loss: %f\n' % array_trans_l2_loss_per_data[repeat_ind][i].item())
                        fout.write('rot_l2_loss: %f\n' % array_rot_l2_loss_per_data[repeat_ind][i].item())
                        fout.write('rot_cd_loss: %f\n' % array_rot_cd_loss_per_data[repeat_ind][i].item())
                        fout.write('total_cd_loss: %f\n' % array_total_cd_loss_per_data[repeat_ind][i].item())
                        fout.write('shape_cd_loss: %f\n' % array_shape_cd_loss_per_data[repeat_ind][i].item())
                        fout.write('contact_point_loss: %f\n' % array_contact_point_loss_per_data[repeat_ind][i].item())
                        fout.write('part_accuracy: %f\n' % array_acc[repeat_ind][i].item())


    return acc_num, valid_num, res_count, total_num, real_shape_cd_loss, real_total_cd_loss, \
            real_contact_point_loss, real_batch_size, cdsV1_sum, cdsV2_sum
