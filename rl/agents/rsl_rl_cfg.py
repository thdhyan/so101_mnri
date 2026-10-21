"""rsl_rl PPO runner config shared by all SO-101 tasks (Isaac Lab backend).

Referenced from the gym registrations as
``rl.agents.rsl_rl_cfg:SO101PPORunnerCfg``.
"""

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class SO101PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared PPO hyperparameters for the SO-101 manipulation tasks.

    Small-MLP actor/critic on state observations; on-policy PPO with the
    defaults Isaac Lab uses for manipulation tasks, adapted to the SO-101's
    short-horizon pick/lift/reach episodes.
    """

    seed = 42
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "so101"
    run_name = ""
    logger = "tensorboard"
    wandb_project = ""  # set "so101-rl" + logger="wandb" to use WandB
    neptune_project = ""

    vecenv: RslRlVecEnvCfg = RslRlVecEnvCfg(
        clip_observations=100.0,
    )

    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        input_dim=64,  # overridden at runtime by the wrapper (obs size)
        output_dim=12,  # overridden at runtime (action size)
        hidden_dims=[256, 128, 64],
        activation="elu",
    )
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        input_dim=64,
        output_dim=1,
        hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
