"""
train_dqn.py
------------
Frames NIDS classification as an MDP and trains a DQN agent
using Stable-Baselines3.

MDP Definition
--------------
  State  : normalised feature vector of one network flow
  Action : Discrete(8)  – predicted traffic class
  Reward :
    +1.0  correct attack detection
    +0.5  correct benign classification
    -1.0  false positive  (predicted attack, true=benign)
    -2.0  missed attack   (predicted benign, true=attack)
    -0.5  wrong attack class
  Episode : one full pass through the training set
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

np.random.seed(42)

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
MODELS_DIR     = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "dqn_checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BENIGN_CLASS     = 0          # integer label for BENIGN after LabelEncoder
TOTAL_TIMESTEPS  = 200_000


# ── Custom Gymnasium Environment ───────────────────────────────────────────────
class NIDSEnv(gym.Env):
    """Single-flow classification environment for NIDS."""

    metadata = {"render_modes": []}

    def __init__(self, X: np.ndarray, y: np.ndarray, benign_class: int = 0):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_samples   = len(X)
        self.n_classes   = int(y.max()) + 1
        self.benign_class = benign_class
        self.current_idx  = 0

        n_features = X.shape[1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n_features,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_classes)

    # ------------------------------------------------------------------
    def _compute_reward(self, action: int, true_label: int) -> float:
        is_benign_true   = (true_label == self.benign_class)
        is_benign_action = (action     == self.benign_class)

        if not is_benign_true and not is_benign_action:
            if action == true_label:
                return 1.0        # correct attack class
            else:
                return -0.5       # wrong attack class
        elif is_benign_true and is_benign_action:
            return 0.5            # correct benign
        elif not is_benign_true and is_benign_action:
            return -2.0           # missed attack
        else:                     # not is_benign_action and is_benign_true
            return -1.0           # false positive

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        obs = self.X[self.current_idx]
        return obs, {}

    def step(self, action: int):
        true_label = int(self.y[self.current_idx])
        reward     = self._compute_reward(int(action), true_label)

        self.current_idx += 1
        terminated = (self.current_idx >= self.n_samples)
        truncated  = False

        if terminated:
            obs = self.X[0]     # dummy obs; episode is done
        else:
            obs = self.X[self.current_idx]

        info = {"true_label": true_label, "predicted": int(action)}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass


# ── Training ───────────────────────────────────────────────────────────────────
def main():
    print("\n=== DQN Agent Training ===")

    print("  Loading training data …")
    X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(DATA_DIR / "y_train.npy").astype(np.int64)
    print(f"  X_train: {X_train.shape}")

    # Stratified 150k subset for CPU-feasible training
    from sklearn.utils import resample
    X_train, y_train = resample(
        X_train, y_train,
        n_samples=150000,
        stratify=y_train,
        random_state=42
    )
    print(f"  Training on {X_train.shape[0]} samples")

    # Determine benign class index from label encoder
    import joblib
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    benign_idx = int(np.where(le.classes_ == "BENIGN")[0][0])
    print(f"  BENIGN class index: {benign_idx}")

    env = NIDSEnv(X_train, y_train, benign_class=benign_idx)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="dqn_nids",
        verbose=1,
    )

    print("  Initialising DQN agent …")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.3,
        gamma=0.95,
        batch_size=64,
        train_freq=4,
        target_update_interval=1_000,
        verbose=1,
        seed=42,
    )

    print(f"  Training for {TOTAL_TIMESTEPS:,} timesteps …")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb)

    out_path = MODELS_DIR / "dqn_agent"
    model.save(str(out_path))
    print(f"  Agent saved → {out_path}.zip")
    print("\n✓ DQN training complete.")


if __name__ == "__main__":
    main()
