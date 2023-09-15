import numpy as np
import torch
from quaternion import euler_to_quaternion, qeuler, qrot
from torch_scatter import scatter_add


def printout(flog, strout):
    print(strout)
    flog.write(strout + '\n')


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def collate_feats(b):
    return list(zip(*b))

def collate_feats_with_none(b):
    b = filter (lambda x:x is not None, b)
    return list(zip(*b))

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

# pc is N x 3, feat is B x 10-dim
def transform_pc_batch(pc, feat, anchor=False):
    batch_size = feat.size(0)
    num_point = pc.size(0)
    pc = pc.repeat(batch_size, 1, 1)
    center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
    shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
    quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
    if not anchor:
        pc = pc * shape
    pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
    if not anchor:
        pc = pc + center
    return pc

def get_surface_reweighting_batch(xyz, cube_num_point):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
                     (y*z).unsqueeze(dim=1).repeat(1, np*2), \
                     (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
    out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
    return out


import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def get_pc_center(pc):
    return np.mean(pc, axis=0)

def get_pc_scale(pc):
    return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0))**2, axis=1)))

def get_pca_axes(pc):
    axes = PCA(n_components=3).fit(pc).components_
    return axes

def get_chamfer_distance(pc1, pc2):
    dist = cdist(pc1, pc2)
    error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    scale = get_pc_scale(pc1) + get_pc_scale(pc2)
    return error / scale

def euler_to_quaternion_torch_data(e, order, device):
    """input: n * 6
        output: n * 7"""
    e_clone = e.clone()
    qua_data = torch.zeros(e_clone.size(0), 7)
    qua_data[:, :3] = e_clone[:, :3]
    # e_clone[:, 3:] = avoid_gimbal_lock(e_clone[:, 3:], order)
    qua_data[:, 3:] = torch.tensor(euler_to_quaternion(e_clone[:, 3:].cpu().numpy(), order), device=device)
    # qua_data[:, 3:] = euler_to_quat(e_clone[:, 3:].to(device), order)
    return qua_data.to(device)

def quaternion_to_euler_torch_data(qua, order, device):
    qua_clone = qua.clone()
    e_data = torch.zeros(qua_clone.size(0), 6)
    e_data[:, :3] = qua_clone[:, :3]
    e_data[:, 3:] = qeuler(qua_clone[:, 3:], order)
    return e_data.to(device)

def to_sparse_batch(x, batch, num_nodes=None):
    batch_size = x.size(0)
    if num_nodes is None:
        num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0, dim_size=batch_size)
    ret_x = []
    for i in range(batch_size):
        ret_x.append(x[i, :num_nodes[i]])
    return torch.cat(ret_x, dim=0)

def marginal_prob_std(t, sigma, conf):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=conf.device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma, conf):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device=conf.device)

def dist_check_one_fun(dist1, dist2, thresh):
    ret = torch.logical_and(dist1 < thresh, dist2 < thresh)
    return ret.float()

def ca_check_one_fun(dist1, dist2, thresh):
    ret = torch.logical_and(dist1 > thresh, dist2 > thresh)
    return ret.float()

def shape_diversity_score(shapes, network, conf, batch_size):
    cdsV1 = torch.zeros([batch_size], device=conf.device)
    cdsV2 = torch.zeros([batch_size], device=conf.device)
    for i in range(len(shapes)):
        for j in range(len(shapes)):
            shape_cd_loss_per_data = network.get_shape_cd_loss(
                shapes[i][0], shapes[i][1][:,:,3:], shapes[j][1][:,:,3:],
                shapes[i][2], shapes[i][1][:,:,:3], shapes[j][1][:,:,:3], 
                conf.device)
            cdsV1 += shape_cd_loss_per_data * ca_check_one_fun(shapes[i][4], shapes[j][4], conf.cdsThresh)
            cdsV2 += shape_cd_loss_per_data * shapes[i][4] * shapes[j][4]

    return cdsV1.cpu()/len(shapes)/len(shapes), cdsV2.cpu()/len(shapes)/len(shapes)

