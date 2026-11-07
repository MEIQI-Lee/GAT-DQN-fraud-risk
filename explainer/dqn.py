"""DQN agent with replay buffer and target network.

The action set varies per state (frontier edges), so each replay transition
stores the *action feature matrix* of the actions available at the next state;
the TD target is computed as max over those actions (standard DQN with
variable action space).
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """Q(state, edge) -> scalar value."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        buffer_size: int = 5000,
        batch_size: int = 64,
        target_sync: int = 100,
    ):
        self.q_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.updates = 0
        self.last_loss = None

    # ---------------- acting ----------------
    def select(self, state: np.ndarray, action_mat: np.ndarray, eps: float) -> int:
        """Epsilon-greedy over the *available* actions. Returns the row index."""
        if random.random() < eps:
            return random.randrange(len(action_mat))
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).expand(len(action_mat), -1)
            a = torch.from_numpy(action_mat)
            q = self.q_net(s, a)
        return int(torch.argmax(q).item())

    # ---------------- learning ----------------
    def push(self, state, action_vec, reward, next_state, done, next_action_mat):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action_vec, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                None if next_action_mat is None or len(next_action_mat) == 0
                else np.asarray(next_action_mat, dtype=np.float32),
            )
        )

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)

        states = torch.from_numpy(np.stack([b[0] for b in batch]))
        actions = torch.from_numpy(np.stack([b[1] for b in batch]))
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)

        # TD target: r + gamma * max_{a'} Q_target(s', a')  (loop: variable |A|)
        targets = []
        with torch.no_grad():
            for (_, _, r, s2, done, next_mat) in batch:
                if done or next_mat is None:
                    targets.append(r)
                else:
                    s2t = torch.from_numpy(s2).unsqueeze(0).expand(len(next_mat), -1)
                    q_next = self.target_net(s2t, torch.from_numpy(next_mat))
                    targets.append(r + self.gamma * float(q_next.max()))
        targets = torch.tensor(targets, dtype=torch.float32)

        q_pred = self.q_net(states, actions)
        loss = F.smooth_l1_loss(q_pred, targets)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.opt.step()

        self.updates += 1
        self.last_loss = loss.item()
        if self.updates % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return self.last_loss
