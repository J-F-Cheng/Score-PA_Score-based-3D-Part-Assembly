import functools
import os
import random
import shutil
from argparse import ArgumentParser

import model_score_based
import numpy as np
import setproctitle
import torch
import torch.utils.data
from test_every_batch import forward

import utils
from data_dynamic import PartNetPartDataset
from utils import diffusion_coeff, marginal_prob_std


def parse():
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--train_data_fn', type=str, help='training data file that indexs all data tuples')
    parser.add_argument('--val_data_fn', type=str, help='validation data file that indexs all data tuples')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514,
                        help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    # parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--data_dir', type=str, default='./prep_data', help='data directory')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)
    parser.add_argument('--max_num_part', type=int, default=20)
    parser.add_argument('--sigma', type=float, default=25.0)

    # Sampler options
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--snr', type=float, default=0.16)
    parser.add_argument('--t0', type=float, default=1.0)
    parser.add_argument('--cor_steps', type=int, default=1)
    parser.add_argument('--cor_final_steps', type=int, default=1)
    parser.add_argument('--noise_decay_pow', type=int, default=1)

    # visu
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # data
    parser.add_argument('--level', default='3', help='level of dataset')
    parser.add_argument('--part_shuffle', action='store_false', default=True, help='shuffle parts?')

    #model path
    parser.add_argument('--model_dir', type=str, help='the path of the model')

    # out dir
    parser.add_argument('--out_dir', default='results', type=str, help='the path of the test')

    # set parameters for QDS
    parser.add_argument('--cdsThresh', type=float, default=0.5)

    # repeat times
    parser.add_argument("--repeat_times_per_shape", type=int, help='the repeat times for every shape', default=10)

    # parse args
    conf = parser.parse_args()

    conf.exp_name = f'exp-{conf.category}-level{conf.level}-{conf.exp_suffix}'
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)

    if not os.path.exists(conf.log_dir):
       os.mkdir(conf.log_dir) 
    if os.path.exists(conf.exp_dir):
        if not conf.overwrite:
            response = input('A testing run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
            if response != 'y':
                exit(1)
        shutil.rmtree(conf.exp_dir)
    os.mkdir(conf.exp_dir)
    # os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
    if not conf.no_visu:
        os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    return conf

def test(conf):
    sigma = conf.sigma
    print("sigma: ", sigma)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, conf=conf)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, conf=conf)
    data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'contact_points', 'sym', 'pairs', 'match_ids']

    val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, data_features, \
                                     max_num_part=20, level=conf.level)
                                     
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
                                                 pin_memory=True, \
                                                 num_workers=0, drop_last=False,
                                                 collate_fn=utils.collate_feats_with_none,
                                                 worker_init_fn=utils.worker_init_fn)
    
    network = model_score_based.Network(conf, marginal_prob_std=marginal_prob_std_fn, input_dim=6)
    network.load_state_dict(torch.load(conf.model_dir, map_location=conf.device))

    # send parameters to device
    network.to(conf.device)

    val_num_batch = len(val_dataloader)

    val_batches = enumerate(val_dataloader, 0)
    val_batch_ind = -1
    
    total_acc_num = 0
    total_valid_num = 0
    total_max_count = 0
    total_total_num = 0

    sum_real_part_cd_loss = 0.
    sum_real_shape_cd_loss = 0.
    sum_real_contact_point_loss = 0.
    sum_cdsV1_sum = 0.
    sum_cdsV2_sum = 0.
    
    real_val_data_set = 0

    network.eval()
    # validate one batch
    while val_batch_ind + 1 < val_num_batch:
        val_batch_ind, val_batch = next(val_batches)

        if len(val_batch)==0:
            continue
            
        #ipdb.set_trace()
        with torch.no_grad():
            # forward pass (including logging)
            acc_num, valid_num, res_count, total_num, real_shape_cd_loss, real_total_cd_loss, \
            real_contact_point_loss, real_batch_size, cdsV1_sum, cdsV2_sum \
                    = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, margin_fn=marginal_prob_std_fn,
                        diffusion_coeff_fn=diffusion_coeff_fn, is_val=True, batch_ind=val_batch_ind)
            
            total_acc_num += acc_num
            total_valid_num += valid_num 
            total_max_count += res_count
            total_total_num += total_num 

            sum_real_part_cd_loss += real_total_cd_loss
            sum_real_shape_cd_loss += real_shape_cd_loss
            sum_real_contact_point_loss += real_contact_point_loss
            sum_cdsV1_sum += cdsV1_sum
            sum_cdsV2_sum += cdsV2_sum

            real_val_data_set += real_batch_size
    
    total_max_count = total_max_count.float()
    total_total_num = float(total_total_num)
    total_acc = total_acc_num / total_valid_num
    total_contact = total_max_count / total_total_num

    real_total_shape_loss = sum_real_shape_cd_loss / real_val_data_set
    real_total_cdsV1 = sum_cdsV1_sum / real_val_data_set
    real_total_cdsV2 = sum_cdsV2_sum / real_val_data_set

    print('-'*70)
    print("Test results:")
    print('SCD: ', real_total_shape_loss.item())
    print('PA: ',100 * total_acc.item())
    print('CA: ', 100 * total_contact.item())
    print('QDS: ', real_total_cdsV1.item())
    print('WQDS: ', real_total_cdsV2.item())

if __name__ == '__main__':
    # parse args
    conf = parse()
    print("conf", conf)

    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    setproctitle.setproctitle("test in " + conf.category)

    ### start training
    test(conf)
