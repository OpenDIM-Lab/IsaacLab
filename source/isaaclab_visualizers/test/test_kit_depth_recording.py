# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless viewport recording alongside a depth-only RTX sensor."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True, video=True)

import numpy as np
import pytest
import torch
from isaaclab_visualizers.kit import KitVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass


@configclass
class DepthSceneCfg(InteractiveSceneCfg):
    """A lit cube seen by a depth sensor and the independent viewport camera."""

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 0.0, 1.0)),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DepthCamera",
        height=64,
        width=64,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), convention="world"),
    )


@pytest.mark.isaacsim_ci
def test_headless_viewport_captures_color_with_depth_only_sensor():
    """A real depth-only camera must not leave RGB video frames black."""
    cfg = SimulationCfg(
        visualizer_cfgs=[
            KitVisualizerCfg(
                headless=True,
                eye=(3.0, 3.0, 3.0),
                lookat=(2.0, 0.0, 1.0),
                window_width=128,
                window_height=128,
            )
        ]
    )
    with build_simulation_context(sim_cfg=cfg) as sim:
        scene = InteractiveScene(DepthSceneCfg(num_envs=1, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        sim.reset()
        for _ in range(8):
            sim.step()
            scene.update(sim.cfg.dt)
            depth = scene["camera"].data.output["distance_to_image_plane"].torch
            frame = sim.visualizers[0].render_rgb_array()
        assert torch.isfinite(depth).any(), "Depth sensor failed to see the cube"
        assert frame.shape == (128, 128, 3)
        assert np.std(frame.astype(np.float32)) > 2.0, "Viewport RGB is blank after depth-only initialization"
