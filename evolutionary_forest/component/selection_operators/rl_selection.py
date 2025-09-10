import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast


class PolicyNet(nn.Module):
    def __init__(self, dim_in: int, hidden=(128, 64), dropout: float = 0.0):
        super().__init__()
        layers = []
        last = dim_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class RLConfig:
    lr: float = 1e-3
    gamma: float = 1.0
    temperature: float = 1.0
    baseline_momentum: float = 0.9
    clip_grad_norm: Optional[float] = 5.0
    device: str = "cpu"


class ParentSelectorRL:
    def __init__(
        self,
        sem_dim: int,
        cfg: RLConfig = RLConfig(),
        hidden=(32, 8),
        dropout: float = 0.0,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = PolicyNet(dim_in=sem_dim * 2, hidden=hidden, dropout=dropout).to(
            self.device
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.baseline = None

    def _to_tensor(self, Phi) -> torch.Tensor:
        """Ensure Phi is a torch.Tensor on the right device."""
        if isinstance(Phi, np.ndarray):
            Phi = torch.from_numpy(Phi).float()
        elif not isinstance(Phi, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(Phi)}")
        return Phi.to(self.device)

    @torch.no_grad()
    def _concat_pairs(self, phi_i: torch.Tensor, Phi: torch.Tensor, mask_idx: int):
        N, D = Phi.shape
        candidate_indices = torch.arange(N, device=Phi.device)
        candidate_indices = candidate_indices[candidate_indices != mask_idx]
        phi_i_rep = phi_i.unsqueeze(0).expand(candidate_indices.numel(), D)
        pairs = torch.cat([phi_i_rep, Phi[candidate_indices]], dim=1)
        return pairs, candidate_indices

    def score_candidates(self, i: int, Phi) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi = self._to_tensor(Phi)
        phi_i = Phi[i]
        pairs, candidate_indices = self._concat_pairs(phi_i, Phi, mask_idx=i)
        scores = self.model(pairs)
        return scores, candidate_indices

    def select_second_parent(self, i: int, Phi, mode: str = "eval"):
        self.model.eval() if mode == "eval" else self.model.train()
        Phi = self._to_tensor(Phi)

        scores, candidate_indices = self.score_candidates(i, Phi)
        aux = {}

        if mode == "train":
            logits = scores / max(1e-8, self.cfg.temperature)
            probs = F.softmax(logits, dim=0)
            m = torch.distributions.Categorical(probs=probs)
            idx = m.sample()
            j = int(candidate_indices[idx].item())
            aux["logprob"] = m.log_prob(idx)
            aux["probs"] = probs.detach().cpu()
        else:
            idx = torch.argmax(scores)
            j = int(candidate_indices[idx].item())
            aux["logprob"] = None

        aux["scores"] = scores.detach().cpu()
        aux["candidate_indices"] = candidate_indices.detach().cpu()
        return j, aux

    # ---- modify ParentSelectorRL.update to support batch ----
    def update(self, logprob, reward):
        # Accept single (tensor, float) or batches (list/tuple of tensors & floats)
        if isinstance(logprob, (list, tuple)):
            logprobs = list(logprob)
            rewards = list(reward)
        else:
            logprobs = [logprob]
            rewards = [reward]

        device = self.device
        r = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        # EMA baseline applied sequentially across the batch
        beta = self.cfg.baseline_momentum
        baseline = self.baseline if self.baseline is not None else r[0].detach()

        losses = []
        for lp, ri in zip(logprobs, r):
            adv = ri - baseline
            losses.append(-(lp * adv))
            baseline = beta * baseline + (1.0 - beta) * ri.detach()

        self.baseline = baseline

        total_loss = torch.stack(losses).sum()
        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm
            )
        self.opt.step()

    @staticmethod
    def compute_reward(
        parent1_perf: float, offspring1_perf: float, higher_is_better: bool = True
    ) -> float:
        if higher_is_better:
            return float(offspring1_perf - parent1_perf)
        else:
            return float(parent1_perf - offspring1_perf)


def select_general_reinforcement_learning(
    population, k, model: ParentSelectorRL, target: np.ndarray
):
    selected = []
    Phi = np.stack([ind.predicted_values for ind in population]).astype(np.float32)

    # residuals and L2 row-wise normalization
    residuals = Phi - target.astype(np.float32)  # [N, M]
    l2 = np.linalg.norm(residuals, axis=1, keepdims=True)  # [N, 1]
    residuals = residuals / np.maximum(l2, 1e-12)  # avoid divide-by-zero

    for _ in range(k // 2):
        mother = selAutomaticEpsilonLexicaseFast(population, 1)[0]
        mother.rl_selected = False
        i = population.index(mother)

        j, aux = model.select_second_parent(i, residuals, mode="train")
        father = population[j]
        father.rl_selected = True
        mother.aux = aux

        selected.extend((mother, father))
    return selected


def update_nn(population, model: ParentSelectorRL):
    logprobs, rewards = [], []
    for ind in population:
        if not ind.rl_selected:
            logprobs.append(ind.aux["logprob"])
            rewards.append(ind.fitness.wvalues[0] - ind.parent_fitness[0])

    if logprobs:
        model.update(logprobs, rewards)


if __name__ == "__main__":
    # === 1. Fake population semantics ===
    N, D = 6, 4
    np.random.seed(0)
    Phi = np.random.randn(N, D).astype(np.float32)  # population semantics

    # === 2. Toy performance metric ===
    def performance(vec):
        # negative L2 norm; higher = better
        return -float(np.sum(vec**2))

    # === 3. Simple crossover (mate) ===
    def mate(parent1, parent2):
        # Offspring = average semantics + noise
        child1 = (parent1 + parent2) / 2 + 0.1 * np.random.randn(*parent1.shape)
        child2 = (parent1 - parent2) / 2 + 0.1 * np.random.randn(*parent1.shape)
        return child1.astype(np.float32), child2.astype(np.float32)

    # === 4. Init selector ===
    selector = ParentSelectorRL(sem_dim=D, cfg=RLConfig(device="cpu", temperature=1.0))

    # === 5. Run mock training ===
    for step in range(10):
        # pick first parent
        i = np.random.randint(N)

        # RL chooses second parent
        j, aux = selector.select_second_parent(i, Phi, mode="train")

        parent1, parent2 = Phi[i], Phi[j]
        child1, child2 = mate(parent1, parent2)

        # evaluate
        p1 = performance(parent1)
        c1 = performance(child1)

        # compute reward
        reward = selector.compute_reward(p1, c1, higher_is_better=True)

        # update RL policy
        selector.update(aux["logprob"], reward)

        print(f"Step {step:02d}: Parent1={i}, Parent2={j}, Reward={reward:.3f}")

    # === 6. Switch to eval mode (greedy selection) ===
    i = 0
    j, aux = selector.select_second_parent(i, Phi, mode="eval")
    print("\nGreedy selection for first parent 0 -> second parent", j)
    print("Scores:", aux["scores"].numpy())
