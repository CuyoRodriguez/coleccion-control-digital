"""Entrena PPO para el entorno SG90 planar de 2 GDL."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from entorno_rl_2dof import EntornoRL2DOF


TOTAL_TIMESTEPS = 300_000
CHECKPOINT_EVERY = 20_000
USE_TENSORBOARD = False
OUTPUT_DIR = Path(__file__).resolve().parent / "modelos_rl_2dof"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    env = EntornoRL2DOF()
    try:
        checkpoints = sorted(
            CHECKPOINT_DIR.glob("ppo_2dof_*.zip"),
            key=lambda path: int(path.stem.rsplit("_", 2)[-2]),
        )
        if checkpoints:
            model = PPO.load(checkpoints[-1], env=env)
            remaining = max(TOTAL_TIMESTEPS - model.num_timesteps, 0)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                tensorboard_log=str(OUTPUT_DIR / "tensorboard") if USE_TENSORBOARD else None,
                seed=42,
                verbose=1,
            )
            remaining = TOTAL_TIMESTEPS

        if remaining:
            model.learn(
                total_timesteps=remaining,
                tb_log_name="PPO_2DOF",
                reset_num_timesteps=False,
                callback=CheckpointCallback(
                    save_freq=CHECKPOINT_EVERY,
                    save_path=str(CHECKPOINT_DIR),
                    name_prefix="ppo_2dof",
                ),
            )
        model.save(OUTPUT_DIR / "modelo_2dof")
        pesos = {
            nombre: tensor.detach().cpu().numpy()
            for nombre, tensor in model.policy.state_dict().items()
        }
        np.savez(OUTPUT_DIR / "modelo_2dof_pesos.npz", **pesos)
    finally:
        env.close()


if __name__ == "__main__":
    main()
