# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/initialize_config.py

import open3d as o3d

import os
import sys
import json
from os.path import isfile, join, splitext, dirname, basename
from warnings import warn

def extract_rgbd_frames(rgbd_video_file):
    """
    Extract color and aligned depth frames and intrinsic calibration from an
    RGBD video file (currently only RealSense bag files supported). Folder
    structure is:
        <directory of rgbd_video_file/<rgbd_video_file name without extension>/
            {depth/00000.jpg,color/00000.png,intrinsic.json}
    """
    frames_folder = join(dirname(rgbd_video_file),
                         basename(splitext(rgbd_video_file)[0]))
    path_intrinsic = join(frames_folder, "intrinsic.json")
    if isfile(path_intrinsic):
        warn(f"Skipping frame extraction for {rgbd_video_file} since files are"
             " present.")
    else:
        rgbd_video = o3d.t.io.RGBDVideoReader.create(rgbd_video_file)
        rgbd_video.save_frames(frames_folder)
    with open(path_intrinsic) as intr_file:
        intr = json.load(intr_file)
    depth_scale = intr["depth_scale"]
    return frames_folder, path_intrinsic, depth_scale


def set_default_value(config, key, value):
    if key not in config:
        config[key] = value


def initialize_config(config):

    # set default parameters if not specified
    set_default_value(config, "depth_map_type", "redwood")
    set_default_value(config, "n_frames_per_fragment", 100)
    set_default_value(config, "n_keyframes_per_n_frame", 5)
    set_default_value(config, "depth_min", 0.3)
    set_default_value(config, "depth_max", 3.0)
    set_default_value(config, "voxel_size", 0.05)
    set_default_value(config, "depth_diff_max", 0.07)
    set_default_value(config, "depth_scale", 1000)
    set_default_value(config, "preference_loop_closure_odometry", 0.1)
    set_default_value(config, "preference_loop_closure_registration", 5.0)
    set_default_value(config, "tsdf_cubic_size", 3.0)
    set_default_value(config, "icp_method", "color")
    set_default_value(config, "global_registration", "ransac")
    set_default_value(config, "python_multi_threading", True)

    # `slac` and `slac_integrate` related parameters.
    # `voxel_size` and `depth_min` parameters from previous section,
    # are also used in `slac` and `slac_integrate`.
    set_default_value(config, "max_iterations", 5)
    set_default_value(config, "sdf_trunc", 0.04)
    set_default_value(config, "block_count", 40000)
    set_default_value(config, "distance_threshold", 0.07)
    set_default_value(config, "fitness_threshold", 0.3)
    set_default_value(config, "regularizer_weight", 1)
    set_default_value(config, "method", "slac")
    set_default_value(config, "device", "CPU:0")
    set_default_value(config, "save_output_as", "pointcloud")
    set_default_value(config, "folder_slac", "slac/")
    set_default_value(config, "template_optimized_posegraph_slac",
                      "optimized_posegraph_slac.json")

    # path related parameters.
    set_default_value(config, "folder_fragment", "fragments/")
    set_default_value(config, "subfolder_slac",
                      "slac/%0.3f/" % config["voxel_size"])
    set_default_value(config, "template_fragment_posegraph",
                      "fragments/fragment_%03d.json")
    set_default_value(config, "template_fragment_posegraph_optimized",
                      "fragments/fragment_optimized_%03d.json")
    set_default_value(config, "template_fragment_pointcloud",
                      "fragments/fragment_%03d.ply")
    set_default_value(config, "folder_scene", "scene/")
    set_default_value(config, "template_global_posegraph",
                      "scene/global_registration.json")
    set_default_value(config, "template_global_posegraph_optimized",
                      "scene/global_registration_optimized.json")
    set_default_value(config, "template_refined_posegraph",
                      "scene/refined_registration.json")
    set_default_value(config, "template_refined_posegraph_optimized",
                      "scene/refined_registration_optimized.json")
    set_default_value(config, "template_global_mesh", "scene/integrated.ply")
    set_default_value(config, "template_global_traj", "scene/trajectory.log")
