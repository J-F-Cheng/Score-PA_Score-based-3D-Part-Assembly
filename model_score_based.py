import random

import numpy as np
import torch
import torch.nn.functional as F
from cd.chamfer import chamfer_distance
from quaternion import qrot
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch_geometric.nn import EdgeConv


class MLP2(nn.Module):
    def __init__(self, feat_len):
        super(MLP2, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Linear(1024, feat_len)
        self.bn6 = nn.BatchNorm1d(feat_len)

    """
        Input: B x N x 3 (B x P x N x 3)
        Output: B x F (B x P x F)
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return x

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None]

class Network(nn.Module):

    def __init__(self, conf, marginal_prob_std, input_dim):
        super(Network, self).__init__()
        self.conf = conf
        self.input_dim = input_dim
        self.cloud_point_encoder = MLP2(conf.feat_len)
        self.x_encoder = nn.Linear(self.input_dim, conf.feat_len)
        self.instance_label_dim = 20

        self.mlp1 = nn.Sequential(
            nn.Linear(conf.feat_len * 2 + conf.feat_len * 2 + conf.feat_len * 2 + self.instance_label_dim * 2, conf.feat_len),
            nn.ReLU(True),
            nn.Linear(conf.feat_len, conf.feat_len),
        )
        self.conv1 = EdgeConv(self.mlp1)
        self.mlp2 = nn.Sequential(
            nn.Linear(conf.feat_len * 2 + conf.feat_len * 2 + conf.feat_len * 2 + self.instance_label_dim * 2, conf.feat_len),
            nn.ReLU(True),
            nn.Linear(conf.feat_len, conf.feat_len),
        )
        self.conv2 = EdgeConv(self.mlp2)
        self.mlp3 = nn.Sequential(
            nn.Linear(conf.feat_len * 2 + conf.feat_len * 2 + conf.feat_len * 2 + self.instance_label_dim * 2, conf.feat_len),
            nn.ReLU(True),
            nn.Linear(conf.feat_len, self.input_dim),
        )
        self.conv3 = EdgeConv(self.mlp3)

        self.marginal_prob_std = marginal_prob_std

        self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=conf.feat_len),
                                   nn.Linear(conf.feat_len, conf.feat_len))
        # self.d1 = Dense(embed_dim, embed_dim)
        self.act = lambda x: x * torch.sigmoid(x)

    """
        Input: B x P x P, B x P, B x P x N x 3, B x P x P
        Output: B x P x (3 + 4)
    """

    def get_part_feature(self, proc_part_pcs):
        return self.cloud_point_encoder(proc_part_pcs)

    def forward(self, x_pose, t, proc_part_pcs, instance_label, lens_part_num=None):
        """x_pose includes x, edge_index and batch """
        t_embed = self.act(self.t_embed(t.squeeze(-1)))

        x = self.x_encoder(x_pose.x)
        emb_pcs = proc_part_pcs
        x = torch.cat([x, t_embed, emb_pcs, instance_label], dim=-1)
        x = torch.relu(self.conv1(x, x_pose.edge_index))
        x = torch.cat([x, t_embed, emb_pcs, instance_label], dim=-1)
        x = torch.relu(self.conv2(x, x_pose.edge_index))
        x = torch.cat([x, t_embed, emb_pcs, instance_label], dim=-1)
        x = self.conv3(x, x_pose.edge_index)
        x = x / (self.marginal_prob_std(t) + 1e-7)
        return x

    """
            Input: * x N x 3, * x 3, * x 4, * x 3, * x 4,
            Output: *, *  (two lists)
    """

    def linear_assignment(self, pts, centers1, quats1, centers2, quats2):
        pts_to_select = torch.tensor(random.sample([i for i  in range(1000)],100))
        pts = pts[:,pts_to_select] 
        cur_part_cnt = pts.shape[0]
        num_point = pts.shape[1]

        with torch.no_grad():

            cur_quats1 = quats1.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers1 = centers1.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts1 = qrot(cur_quats1, pts) + cur_centers1

            cur_quats2 = quats2.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers2 = centers2.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts2 = qrot(cur_quats2, pts) + cur_centers2

            cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cur_part_cnt, 1, 1).view(-1, num_point, 3)
            cur_pts2 = cur_pts2.unsqueeze(0).repeat(cur_part_cnt, 1, 1, 1).view(-1, num_point, 3)
            dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
            dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cur_part_cnt, cur_part_cnt)
            rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

        return rind, cind


    """
        Input: B x P x 3, B x P x 3, B x P
        Output: B
    """

    def get_trans_l2_loss(self, trans1, trans2, valids):
        loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """

    def get_rot_l2_loss(self, pts, quat1, quat2, valids):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """

    def get_rot_cd_loss(self, pts, quat1, quat2, valids, device):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)

        loss_per_data = loss_per_data.to(device)
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data  
        
    def get_total_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part =  pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)
        
        thre = 0.01
        loss_per_data = loss_per_data.to(device)
        acc = [[0 for i in range(num_part)]for j in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_part):
                if loss_per_data[i,j] < thre and valids[i,j]:
                    acc[i][j] = 1
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data , acc

    def get_shape_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        pts1 = pts1.view(batch_size,num_part*num_point,3)
        pts2 = pts2.view(batch_size,num_part*num_point,3)
        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        valids = valids.unsqueeze(2).repeat(1,1,1000).view(batch_size,-1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        
        loss_per_data = loss_per_data.to(device)
        return loss_per_data

        """
            output : B
        """
    def get_sym_point(self, point, x, y, z):

        if x:
            point[0] = - point[0]
        if y:
            point[1] = - point[1]
        if z:
            point[2] = - point[2]

        return point.tolist()

    def get_possible_point_list(self, point, sym):
        sym = torch.tensor([1.0,1.0,1.0]) 
        point_list = []
        #sym = torch.tensor(sym)
        if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
        elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
        elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
        elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
        elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
        else:
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 1))

        return point_list
    def get_min_l2_dist(self, list1, list2, center1, center2, quat1, quat2):

        list1 = torch.tensor(list1) # m x 3
        list2 = torch.tensor(list2) # n x 3
        #print(list1[0])
        #print(list2[0])
        len1 = list1.shape[0]
        len2 = list2.shape[0]
        center1 = center1.unsqueeze(0).repeat(len1, 1)
        center2 = center2.unsqueeze(0).repeat(len2, 1)
        quat1 = quat1.unsqueeze(0).repeat(len1, 1)
        quat2 = quat2.unsqueeze(0).repeat(len2, 1)
        list1 = list1.to(self.conf.device)
        list2 = list2.to(self.conf.device)
        list1 = center1 + qrot(quat1, list1)
        list2 = center2 + qrot(quat2, list2)
        mat1 = list1.unsqueeze(1).repeat(1, len2, 1)
        mat2 = list2.unsqueeze(0).repeat(len1, 1, 1)
        mat = (mat1 - mat2) * (mat1 - mat2)
        #ipdb.set_trace()
        mat = mat.sum(dim=-1)
        return mat.min()

    """    
        Contact point loss metric
        Date: 2020/5/22
        Input B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput B
    """
    def get_contact_point_loss(self, center, quat, contact_points, sym_info):

        batch_size = center.shape[0]
        num_part = center.shape[1]
        contact_point_loss = torch.zeros(batch_size)
        total_num = 0
        count = 0
        for b in range(batch_size):
            #print("Shape id is", b)
            sum_loss = 0
            for i in range(num_part):
                for j in range(num_part):
                    if contact_points[b, i, j, 0]:
                        contact_point_1 = contact_points[b, i, j, 1:]
                        contact_point_2 = contact_points[b, j, i, 1:]
                        sym1 = sym_info[b, i]
                        sym2 = sym_info[b, j]
                        point_list_1 = self.get_possible_point_list(contact_point_1, sym1)
                        point_list_2 = self.get_possible_point_list(contact_point_2, sym2)
                        dist = self.get_min_l2_dist(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :], quat[b, j, :])  # 1
                        #print(dist)
                        if dist < 0.01:
                            count += 1
                        total_num += 1
                        sum_loss += dist
            contact_point_loss[b] = sum_loss


        #print(count, total_num)
        return contact_point_loss, count, total_num

    def batch_get_contact_point_loss(self, center, quat, contact_points, sym_info):

        batch_size = center.shape[0]
        num_part = center.shape[1]
        contact_point_loss = torch.zeros(batch_size)
        total_num = 0
        batch_total_num = torch.zeros(batch_size, dtype=torch.long)
        count = 0
        batch_count = torch.zeros(batch_size, dtype=torch.long)
        for b in range(batch_size):
            #print("Shape id is", b)
            sum_loss = 0
            for i in range(num_part):
                for j in range(num_part):
                    if contact_points[b, i, j, 0]:
                        contact_point_1 = contact_points[b, i, j, 1:]
                        contact_point_2 = contact_points[b, j, i, 1:]
                        sym1 = sym_info[b, i]
                        sym2 = sym_info[b, j]
                        point_list_1 = self.get_possible_point_list(contact_point_1, sym1)
                        point_list_2 = self.get_possible_point_list(contact_point_2, sym2)
                        dist = self.get_min_l2_dist(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :], quat[b, j, :])  # 1
                        #print(dist)
                        if dist < 0.01:
                            count += 1
                            batch_count[b] += 1
                        total_num += 1
                        batch_total_num[b] += 1
                        sum_loss += dist
            contact_point_loss[b] = sum_loss


        #print(count, total_num)
        return contact_point_loss, count, total_num, batch_count, batch_total_num
