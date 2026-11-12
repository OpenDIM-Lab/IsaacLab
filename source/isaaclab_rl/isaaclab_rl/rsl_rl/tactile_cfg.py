# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration contracts for the Tactile Lab RSL-RL 3.1 fork.

These require the perceptive-nav-rl implementation; stock RSL-RL 5 is not compatible.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg
from .rl_cfg import RslRlBaseRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlActorCriticExtrinsicsCfg(RslRlPpoActorCriticCfg):  # stage 1
    """Configuration for PPO actor-critic with privileged extrinsics encoder (Stage 1)."""

    class_name: str = "ActorCriticExtrinsics"
    """Policy class name."""

    # ---- Extrinsics encoder ----
    extrinsics_output_dims: int = MISSING
    """Dimension of the latent extrinsics vector."""

    priv_obs_normalization: bool = MISSING
    """Whether to normalize the privileged observation for the extrinsics encoder."""

    extrinsics_hidden_dims: tuple[int] | list[int] = (256, 256, 256)
    """Hidden layer sizes for the extrinsics encoder MLP."""

    last_activation: str | None = None


@configclass
class RslRlActorCriticCNNCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with perceptual layers."""

    @configclass
    class CNNCfg:
        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

        kernel_size: int | tuple[int] | list[int] = MISSING
        """The kernel size for the CNN."""

        stride: int | tuple[int] | list[int] = 1
        """The stride for the CNN."""

        dilation: int | tuple[int] | list[int] = 1
        """The dilation for the CNN."""

        padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
        """The padding for the CNN."""

        norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = "none"
        """The normalization for the CNN."""

        activation: str = MISSING
        """The activation function for the CNN."""

        max_pool: bool | tuple[bool] | list[bool] = False
        """Whether to use max pooling for the CNN."""

        global_pool: Literal["none", "max", "avg"] = "none"
        """The global pooling for the CNN."""

        flatten: bool = True
        """Whether to flatten the output of the CNN."""

    class_name: str = "ActorCriticCNN"
    """The policy class name. Default is ActorCriticCNN."""

    actor_cnn_cfg: list[CNNCfg] | CNNCfg | None = MISSING
    """The CNN configuration for the actor network."""

    critic_cnn_cfg: list[CNNCfg] | CNNCfg | None = MISSING
    """The CNN configuration for the critic network."""


@configclass
class RslRlOnPolicyCoDesignRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyCoDesignRunner"
    """The runner class name. Default is OnPolicyCoDesignRunner."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    hardware_iteration: int = 0


@configclass
class RslRlDistillationStudentTeacherCNNCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher CNN networks."""

    class_name: str = "StudentTeacherCNN"
    """The policy class name. Default is StudentTeacherCNN."""

    student_cnn_cfg: dict[str, dict] | dict | None = None
    """The CNN configuration for the student network."""


@configclass
class RslRlDistillationStudentTeacherCNNRNNCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher CNN networks."""

    class_name: str = "StudentTeacherCNNRNN"
    """The policy class name. Default is StudentTeacherCNNRNN."""

    student_cnn_cfg: dict[str, dict] | dict | None = None
    """The CNN configuration for the student network."""

    rnn_type: str = MISSING
    """The type of the RNN network. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The hidden dimension of the RNN network."""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN network."""

    teacher_recurrent: bool = MISSING
    """Whether the teacher network is recurrent too."""


#########################
# Policy configurations #
#########################
@configclass
class RslRlExtrinsicsDistillationStudentTeacherCfg:
    """Configuration for extrinsics distillation student-teacher network."""

    class_name: str = "StudentTeacherExtrinsics"

    extrinsics_output_dim: int = MISSING
    history_len: int = MISSING

    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"

    actor_obs_normalization: bool = MISSING
    priv_obs_normalization: bool = MISSING
    student_1d_obs_normalization: bool = MISSING

    actor_hidden_dims: list[int] = MISSING
    teacher_extrinsics_hidden_dims: list[int] = MISSING

    activation: str = MISSING
    # teacher_tanh_extrinsics: bool = False


@configclass
class RslRlExtrinsicsMultiModalDistillationStudentTeacherCfg(RslRlExtrinsicsDistillationStudentTeacherCfg):
    """Configuration for extrinsics distillation student-teacher network."""

    class_name: str = "StudentTeacherMultiModalExtrinsics"

    student_cnn_cfg: dict[str, dict] | dict | None = None


############################
# Algorithm configurations #
############################
@configclass
class RslRlExtrinsicsDistillationAlgorithmCfg:
    """Configuration for extrinsics distillation algorithm."""

    class_name: str = "ExtrinsicsDistillation"

    num_learning_epochs: int = MISSING
    learning_rate: float = MISSING
    gradient_length: int = MISSING
    max_grad_norm: None | float = None

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    loss_type: Literal["mse", "huber"] = "mse"


#########################
# Runner configurations #
#########################


@configclass
class RslRlExtrinsicsDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for extrinsics distillation runner."""

    class_name: str = "ExtrinsicsDistillationRunner"

    policy: RslRlExtrinsicsDistillationStudentTeacherCfg = MISSING
    algorithm: RslRlExtrinsicsDistillationAlgorithmCfg = MISSING


@configclass
class RslRlExtrinsicsMultiModalDistillationRunnerCfg(RslRlExtrinsicsDistillationRunnerCfg):
    """Configuration for extrinsics distillation runner."""

    policy: RslRlExtrinsicsMultiModalDistillationStudentTeacherCfg = MISSING


@configclass
class RslRlTactileDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Legacy distillation configuration including the optional local replay extension."""

    replay_buffer_size: int = 0
    """Number of recent rollout iterations retained for teacher action replay."""
