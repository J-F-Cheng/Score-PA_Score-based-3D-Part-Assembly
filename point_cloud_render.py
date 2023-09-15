import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np
import imageio

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

colors = [[0, 0.8, 0], [0.8, 0, 0], [0, 0.8, 0], \
        [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], \
        [0.3, 0.6, 0], [0.6, 0, 0.3], [0.3, 0, 0.6], \
        [0.6, 0.3, 0], [0.3, 0, 0.6], [0.6, 0, 0.3], \
        [0.8, 0.2, 0.5], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5], \
        [0.2, 0.5, 0.8], [0.5, 0.2, 0.8], [0.5, 0.8, 0.2], \
        [0.3, 0.3, 0.7], [0.3, 0.7, 0.3], [0.7, 0.3, 0.3]]

def pre_render_process(pts, conf):
    """
        pre-rendering processing
        input: pts P x N x 3
        output: processed pts and the corresponding colours
    """
    pts_colors = []
    pre_pts = pts.reshape(pts.size(0) * pts.size(1), pts.size(2))
    for i in range(pts.size(0)):
        pts_colors.append(torch.tensor(colors[i], dtype=torch.float, device=conf.device).repeat(pts.size(1), 1))
    return pre_pts.to(conf.device), torch.cat(pts_colors, dim=0)

def point_cloud_render(out_fn, pts, conf):
    pre_pts, pts_colors = pre_render_process(pts, conf)
    # print("pre_pts: ", pre_pts.size())
    # print("pts_colors: ", pts_colors.size())
    point_cloud = Pointclouds(points=[pre_pts], features=[pts_colors])
    # Initialize a camera.
    R, T = look_at_view_transform(45, 45, 45)
    cameras = FoVOrthographicCameras(device=conf.device, R=R*0.6, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.003,
        points_per_pixel=10
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )
    images = renderer(point_cloud)
    output_img = images[0, ..., :3].cpu().numpy()
    output_img = output_img * 255.0
    output_img = output_img.astype(np.uint8)
    """
    print("output max: ", np.max(output_img))
    print("dtype: ", output_img.dtype)
    """
    imageio.imwrite(out_fn, output_img)

