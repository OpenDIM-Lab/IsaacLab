# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering follows articulation pose writes without advancing physics."""

import math

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(headless=True, enable_cameras=True)

import numpy as np
import pytest
import torch
from isaaclab_visualizers.kit import KitVisualizerCfg

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]


def _write_hinge_asset(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, position in [("base", (0.0, 0.0, 0.0)), ("tip", (0.1, 0.0, 0.3))]:
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.1)
    base_shape = UsdGeom.Cube.Define(stage, "/Robot/base/shape")
    base_shape.CreateSizeAttr(0.08)
    sphere = UsdGeom.Sphere.Define(stage, "/Robot/tip/shape")
    sphere.CreateRadiusAttr(0.03)
    for shape in [base_shape, sphere]:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    material = UsdShade.Material.Define(stage, "/Robot/blue")
    shader = UsdShade.Shader.Define(stage, "/Robot/blue/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 1.0))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/fixed")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/base")])
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/base")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/tip")])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.3))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(-0.1, 0.0, 0.0))
    hinge.CreateAxisAttr("Z")
    stage.GetRootLayer().Save()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_joint_pose_write_reaches_kit_capture_without_physics_step(tmp_path, device):
    """A half-turn of a hinge must move its blue tip across the rendered image."""
    path = tmp_path / "hinge.usda"
    _write_hinge_asset(path)
    cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        visualizer_cfgs=[
            KitVisualizerCfg(
                headless=True,
                eye=(0.0, -0.6, 0.8),
                lookat=(0.0, 0.0, 0.15),
                window_width=128,
                window_height=128,
            )
        ],
    )
    with build_simulation_context(sim_cfg=cfg, add_lighting=True) as sim:
        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        scene_cfg.robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=UsdFileCfg(usd_path=str(path)),
            actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=["hinge"], stiffness=0.0, damping=0.0)},
        )
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        sim.reset()
        robot = scene["robot"]
        visualizer = sim.visualizers[0]

        def centroid():
            for _ in range(4):
                sim.forward()
                sim.render()
                scene.update(cfg.dt)
                frame = visualizer.render_rgb_array()
            values = np.asarray(frame).astype(np.int16)
            mask = (values[..., 2] > values[..., 0] + 25) & (values[..., 2] > values[..., 1] + 25)
            assert mask.sum() > 10, "The articulated blue tip is absent from the capture"
            return np.nonzero(mask)[1].mean()

        initial = centroid()
        robot.write_joint_state_to_sim_index(
            position=torch.full((1, 1), math.pi, device=device),
            velocity=torch.zeros((1, 1), device=device),
        )
        sim.forward()
        scene.update(cfg.dt)
        tip_id = robot.find_bodies("tip")[0][0]
        assert robot.data.body_pos_w.torch[0, tip_id, 0] < -0.09
        moved = centroid()
        assert initial - moved > 20, f"The physical hinge moved, but the rendered tip stayed at {initial} -> {moved}"
