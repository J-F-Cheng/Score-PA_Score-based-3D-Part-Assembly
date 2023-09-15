import torch
import torch.utils.data
import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_sum

from utils import euler_to_quaternion_torch_data


def samples_gen(conf, margin_fn, diffusion_coeff_fn, input_dim,
                  input_part_valids, input_part_pcs, instance_label, input_part_pairs, same_class_list,
                  network, eps, atol, rtol):
    batch_size = input_part_pcs.shape[0]
    # Data pre-processing
    proc_init_part_poses = []
    proc_input_part_pcs = []
    proc_batch_code = []
    proc_edge = []
    proc_instance_label = []
    lens_part_num = []

    for i in range(batch_size):
        real_num_part = int(input_part_valids[i].sum().item())
        lens_part_num.append(real_num_part)
        batch_code = torch.tensor([i for _ in range(real_num_part)])
        proc_batch_code.append(batch_code)
        pre_poses = torch.randn(real_num_part, input_dim, device=conf.device)
        # edge = knn_graph(pre_poses, k=real_num_part, loop=True)

        if real_num_part >= 2:
            edge = knn_graph(pre_poses, k=real_num_part - 1)
        else:
            edge = torch.zeros(2, 1, dtype=torch.int64, device=conf.device)
            
        edge_shift = 0
        for j in range(i):
            edge_shift += int(input_part_valids[j].sum().item())
        edge += edge_shift
        proc_edge.append(edge)
        proc_init_part_poses.append(pre_poses)
        pre_pcs = input_part_pcs[i, :real_num_part]
        # pre_pcs = pre_pcs.reshape(pre_pcs.size(0), pre_pcs.size(1) * pre_pcs.size(2))
        proc_input_part_pcs.append(pre_pcs)
        proc_instance_label.append(instance_label[i, :real_num_part])

    proc_batch_code = torch.cat(proc_batch_code, dim=0)
    init_x_edge = torch.cat(proc_edge, dim=1)
    init_x = torch.cat(proc_init_part_poses, dim=0)
    proc_input_part_pcs = torch.cat(proc_input_part_pcs, dim=0)
    proc_instance_label = torch.cat(proc_instance_label, dim=0)
    lens_part_num = torch.tensor(lens_part_num, dtype=torch.int)

    x_states = []
    pc_final_cor_steps = []

    t = torch.ones(proc_input_part_pcs.size(0), device=conf.device) * conf.t0
    init_x *= margin_fn(t)[:, None]
    time_steps = torch.linspace(conf.t0, eps, conf.num_steps, device=conf.device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    x = Data(x=x, edge_index=init_x_edge, batch=proc_batch_code).to(conf.device)
    with torch.no_grad():
        iter = 0
        pcs_feature = network.get_part_feature(proc_input_part_pcs.float())
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(proc_input_part_pcs.size(0), device=conf.device) * time_step
            batch_time_step = batch_time_step.unsqueeze(-1)

            if iter < conf.num_steps - 1:
                for _ in range(conf.cor_steps):
                    # Corrector step (Langevin MCMC)
                    grad = network(x_pose=x, t=batch_time_step, proc_part_pcs=pcs_feature, 
                                    instance_label=proc_instance_label, lens_part_num=lens_part_num)
                    grad_norm = torch.square(grad).sum(dim=-1)
                    grad_norm = torch.sqrt(scatter_sum(grad_norm, x.batch, dim=0))[x.batch].unsqueeze(-1)
                    # grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.sqrt(
                        scatter_sum(torch.ones(x.x.size(0), device=conf.device) * input_dim, x.batch, dim=0))
                    noise_norm = noise_norm.unsqueeze(-1)
                    noise_norm = noise_norm[x.batch]
                    # noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 2 * (conf.snr * noise_norm / grad_norm) ** 2
                    x.x = x.x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x.x)
            else:
                total_t = float(conf.cor_final_steps)
                for decay_t in range(conf.cor_final_steps):
                    # Corrector step (Langevin MCMC)
                    grad = network(x_pose=x, t=batch_time_step, proc_part_pcs=pcs_feature, 
                                    instance_label=proc_instance_label, lens_part_num=lens_part_num)
                    grad_norm = torch.square(grad).sum(dim=-1)
                    grad_norm = torch.sqrt(scatter_sum(grad_norm, x.batch, dim=0))[x.batch].unsqueeze(-1)
                    # grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.sqrt(
                        scatter_sum(torch.ones(x.x.size(0), device=conf.device) * input_dim, x.batch, dim=0))
                    noise_norm = noise_norm.unsqueeze(-1)
                    noise_norm = noise_norm[x.batch]
                    # noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 2 * (conf.snr * noise_norm / grad_norm) ** 2
                    noise_decay = ((total_t - float(decay_t)) / total_t) ** conf.noise_decay_pow
                    x.x = x.x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size * noise_decay) * torch.randn_like(x.x)
                    # if iter == conf.num_steps - 1:
                    pc_final_cor_steps.append(euler_to_quaternion_torch_data(x.x, "xyz", conf.device))

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff_fn(batch_time_step)
            x_mean = x.x + (g ** 2) * network(x_pose=x, t=batch_time_step, proc_part_pcs=pcs_feature, 
                                    instance_label=proc_instance_label, lens_part_num=lens_part_num) * step_size
            x.x = x_mean + torch.sqrt(g ** 2 * step_size) * torch.randn_like(x.x)
            x_states.append(euler_to_quaternion_torch_data(x_mean, "xyz", conf.device))
            iter += 1
        pred_part_poses = euler_to_quaternion_torch_data(x_mean, "xyz", conf.device)
        pc_final_cor_steps.append(pred_part_poses)

    return {"x_states": x_states, "pred_part_poses": pred_part_poses, "pc_final_cor_steps": pc_final_cor_steps}
