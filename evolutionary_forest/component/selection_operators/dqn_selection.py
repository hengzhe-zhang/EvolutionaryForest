import random
from collections import deque, namedtuple

import numpy as np

from evolutionary_forest.component.selection_operators.rl_selection import (
    ParentSelectorRL,
    RLConfig,
)
import torch
import torch.nn as nn


Transition = namedtuple("Transition", ["feats", "reward"])


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size):
        if len(self.buf) == 0:
            return []
        k = min(batch_size, len(self.buf))
        return random.sample(self.buf, k)


class ParentSelectorDQN(ParentSelectorRL):
    """
    DQN-style (contextual bandit) parent selector:
      Q(x_{f,m}) ≈ E[r | x_{f,m}],
      ε-greedy action selection,
      MSE regression to realized (one-step) reward.
    """

    def __init__(
        self,
        sem_dim: int,
        cfg: RLConfig = RLConfig(),
        hidden=(16, 8),
        dropout: float = 0.0,
        buffer_capacity: int = 500,
        batch_size: int = 64,
    ):
        super().__init__(sem_dim=sem_dim, cfg=cfg, hidden=hidden, dropout=dropout)

        self.tau = 1.0  # starting temperature
        self.tau_end = 0.1  # final temperature
        self.tau_decay = 1e-4  # linear
        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size

        # value loss
        self.value_loss = nn.MSELoss()

    def _decay_tau(self):
        self.tau = max(self.tau_end, self.tau - self.tau_decay)

    def select_second_parent(self, i: int, Phi, target, mode: str = "eval"):
        """
        DQN-style selection:
          - mode='train': ε-greedy over Q-values
          - mode='eval' : greedy
        Returns (j, aux) where aux contains 'feats_row' for replay.
        """
        # reuse scoring pipeline from base class
        self.model.eval() if mode == "eval" else self.model.train()
        Phi = self._to_tensor(Phi)
        target = self._to_tensor(target)

        scores, candidate_indices, feats = self.score_candidates(
            i, Phi, target
        )  # scores=Q(x), feats: [N-1,D]
        aux = {}

        if mode == "train":
            # Boltzmann exploration over Q-values
            # stabilize by subtracting max to avoid overflow
            tau = max(1e-6, float(self.tau))
            logits = (scores - scores.max()) / tau  # [N-1]
            probs = torch.softmax(logits, dim=0)  # [N-1]

            # sample according to probs
            m = torch.distributions.Categorical(probs=probs)
            idx = int(m.sample().item())

            self._decay_tau()

            j = int(candidate_indices[idx].item())
            aux["feats_row"] = feats[idx].detach().clone()  # [D]
            aux["probs"] = probs.detach().cpu()  # (optional: for logging)
        else:
            idx = int(torch.argmax(scores).item())
            j = int(candidate_indices[idx].item())
            aux["feats_row"] = None

        aux["scores"] = scores.detach().cpu()
        aux["candidate_indices"] = candidate_indices.detach().cpu()
        return j, aux

    def push_transition(self, aux, reward):
        """
        Push one transition (chosen features, realized reward) into replay.
        Use right after you compute the reward for the chosen second parent.
        """
        if aux is None:
            return
        feats_row = aux.get("feats_row", None)
        if feats_row is None:
            return
        r = float(reward)
        self.replay.push(feats_row.to(self.device), r)

    def learn(self):
        """
        Sample from replay and do one gradient step:
          loss = MSE( Q(feats), reward ).
        Returns the scalar loss (float) or None if not enough data.
        """
        batch = self.replay.sample(self.batch_size)
        if not batch:
            return None

        feats_b = torch.stack([b.feats for b in batch], dim=0).to(self.device)  # [B, D]
        rewards_b = torch.tensor(
            [b.reward for b in batch], dtype=torch.float32, device=self.device
        )  # [B]

        self.model.train()
        q_pred = self.model(feats_b).squeeze(-1)  # [B]

        loss = self.value_loss(q_pred, rewards_b)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm
            )
        self.opt.step()

        return float(loss.detach().cpu())

    def learn_many(self, iters: int = 4):
        """
        Do `iters` independent replay updates.
        Returns a list of losses (floats); empty if no data.
        """
        losses = []
        for _ in range(max(1, iters)):
            loss = self.learn()
            if loss is None:
                break
            losses.append(loss)
        return losses


def update_nn_dqn(
    population,
    model: ParentSelectorDQN,
    reward_clip: float = None,
    standardize: bool = False,
    updates_per_batch: int = 50,
):
    """
    Collect transitions, push to replay, then do multiple replay updates.
    """
    aux_list, rewards = [], []

    for ind in population:
        # Mothers are those with rl_selected == False in your code
        if not getattr(ind, "rl_selected", False):
            aux = getattr(ind, "aux", None)
            if aux is None or aux.get("feats_row", None) is None:
                continue

            # Your reward convention (improvement of offspring1 over parent1)
            r = ind.fitness.wvalues[0] - ind.parent_fitness[0]

            # For DQN-style regression, keep negatives (don’t drop <= 0)
            if reward_clip is not None:
                r = max(-reward_clip, min(reward_clip, r))

            aux_list.append(aux)
            rewards.append(float(r))

    if not aux_list:
        return None

    if standardize and len(rewards) > 1:
        mu = np.mean(rewards)
        sigma = np.std(rewards) + 1e-8
        rewards = [(r - mu) / sigma for r in rewards]

    for aux, r in zip(aux_list, rewards):
        model.push_transition(aux, r)

    losses = model.learn_many(iters=updates_per_batch)
    avg_loss = np.mean(losses)
    print("avg_loss", avg_loss, "losses", losses)
    return avg_loss if losses else None
