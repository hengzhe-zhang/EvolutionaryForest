import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from matplotlib import pyplot as plt
from typing import Union

from torch.nn import LeakyReLU

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast


class PolicyNet(nn.Module):
    def __init__(self, dim_in: int, hidden=(128, 64), dropout: float = 0.0):
        super().__init__()
        layers = []
        last = dim_in
        for h in hidden:
            layers += [nn.Linear(last, h), LeakyReLU()]
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
    temperature: float = 1.0
    baseline_momentum: float = 0.9
    clip_grad_norm: Optional[float] = 5.0
    device: str = "cpu"


class ParentSelectorRL:
    def __init__(
        self,
        sem_dim: int,
        cfg: RLConfig = RLConfig(),
        hidden=(16, 8),
        dropout: float = 0.0,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # input is element-wise product features -> dimension = sem_dim
        self.model = PolicyNet(dim_in=sem_dim, hidden=hidden, dropout=dropout).to(
            self.device
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.baseline: Optional[torch.Tensor] = None

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")
        return x.to(self.device)

    @torch.no_grad()
    def _pair_features(
        self,
        phi_i: torch.Tensor,  # [D]
        Phi: torch.Tensor,  # [N, D]
        target: Union[np.ndarray, torch.Tensor],  # [D]
        mask_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats_ij = norm2_center( (phi_i + phi_j)/2 ) ⊙ norm2_center(target)
        where norm2_center(x) = (x - mean(x)) / ||x - mean(x)||_2
        """
        N, D = Phi.shape
        candidate_indices = torch.arange(N, device=Phi.device)
        candidate_indices = candidate_indices[candidate_indices != mask_idx]  # [N-1]

        # Build avg per candidate
        phi_i_rep = phi_i.unsqueeze(0).expand(candidate_indices.numel(), D)  # [N-1, D]
        avg = (phi_i_rep + Phi[candidate_indices]) * 0.5  # [N-1, D]

        # Center (per-row) and L2-normalize each row
        avg_centered = avg - avg.mean(dim=1, keepdim=True)  # [N-1, D]
        avg_norm = torch.norm(avg_centered, dim=1, keepdim=True).clamp_min(1e-12)
        avg_unit = avg_centered / avg_norm  # [N-1, D]

        # Target: center once and L2-normalize
        t = self._to_tensor(target).view(-1)  # [D]
        t_centered = t - t.mean()  # [D]
        t_norm = torch.norm(t_centered).clamp_min(1e-12)  # scalar
        t_unit = t_centered / t_norm  # [D]

        # Element-wise product (broadcast over rows)
        feats = avg_unit * t_unit  # [N-1, D]
        return feats, candidate_indices

    def score_candidates(
        self, i: int, Phi, target
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi = self._to_tensor(Phi)  # [N, D]
        phi_i = Phi[i]  # [D]
        feats, candidate_indices = self._pair_features(phi_i, Phi, target, mask_idx=i)
        scores = self.model(feats)  # [N-1]
        return scores, candidate_indices

    def select_second_parent(self, i: int, Phi, target, mode: str = "eval"):
        self.model.eval() if mode == "eval" else self.model.train()
        Phi = self._to_tensor(Phi)
        target = self._to_tensor(target)

        scores, candidate_indices = self.score_candidates(i, Phi, target)
        aux = {}

        if mode == "train":
            logits = scores / max(1e-8, self.cfg.temperature)
            probs = F.softmax(logits, dim=0)
            # print(probs)
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

    # batch-aware REINFORCE update (single optimizer step)
    def update(self, logprob, reward):
        # ---- 1) Normalize inputs & filter Nones in pairs ----
        if isinstance(logprob, (list, tuple)):
            pairs = [(lp, rw) for lp, rw in zip(logprob, reward) if lp is not None]
            if not pairs:
                return
            logprobs, rewards = zip(*pairs)  # tuples
        else:
            if logprob is None:
                return
            logprobs, rewards = (logprob,), (reward,)

        device = self.device
        r = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        # ---- 2) Freeze a baseline for this batch (same for all samples) ----
        beta = self.cfg.baseline_momentum
        # use previous EMA baseline if available; otherwise start with this batch mean
        frozen_baseline = (
            self.baseline.detach()
            if getattr(self, "baseline", None) is not None
            else r.mean().detach()
        )

        # ---- 3) Compute advantages with the frozen baseline ----
        adv = r - frozen_baseline

        # ---- 4) Build REINFORCE loss over the batch ----
        loss = -torch.stack([lp * a for lp, a in zip(logprobs, adv)]).sum()

        # ---- 5) Backprop + (optional) grad clipping ----
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm
            )
        self.opt.step()

        # ---- 6) Offline EMA update of baseline using batch statistic ----
        batch_stat = r.mean().detach()
        new_baseline = beta * frozen_baseline + (1.0 - beta) * batch_stat
        self.baseline = new_baseline

    @staticmethod
    def compute_reward(
        parent1_perf: float, offspring1_perf: float, higher_is_better: bool = True
    ) -> float:
        return (
            float(offspring1_perf - parent1_perf)
            if higher_is_better
            else float(parent1_perf - offspring1_perf)
        )

    def pretrain_with_sum_rule(
        self,
        data,
        epochs: int = 1,
        lr: float = None,
        shuffle: bool = True,
        verbose: bool = True,
    ):
        # Optionally override LR just for this routine
        orig_lrs = [pg["lr"] for pg in self.opt.param_groups]
        if lr is not None:
            for pg in self.opt.param_groups:
                pg["lr"] = lr

        self.model.train()
        history = {"loss": [], "acc": []}

        # Simple helper to turn any input into device tensor
        def _to_dev(x):
            return self._to_tensor(x)

        for ep in range(epochs):
            # Materialize data in memory so we can shuffle if desired
            if not isinstance(data, list):
                dataset = list(data)
            else:
                dataset = data

            if shuffle:
                import random

                random.shuffle(dataset)

            total_loss = 0.0
            total_correct = 0
            total_count = 0

            for i, Phi, target in dataset:
                Phi = _to_dev(Phi)  # [N, D]
                target_t = _to_dev(target)  # [D]

                # Build feats_ij and candidate set (excludes i)
                phi_i = Phi[i]  # [D]
                feats, candidate_indices = self._pair_features(
                    phi_i, Phi, target_t, mask_idx=i
                )  # feats: [N-1, D]

                # Heuristic label: index with largest sum(feats_ij)
                with torch.no_grad():
                    label_local = torch.argmax(feats.sum(dim=1))  # int in [0, N-2]
                # Forward pass
                logits = self.model(feats)  # [N-1]
                # Shape to [1, C] for cross-entropy
                loss = F.cross_entropy(logits.unsqueeze(0), label_local.view(1))

                # Optimize
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad_norm
                    )
                self.opt.step()

                # Stats
                total_loss += float(loss.detach().cpu())
                pred_local = int(torch.argmax(logits).item())
                total_correct += int(pred_local == int(label_local))
                total_count += 1

            mean_loss = total_loss / max(1, total_count)
            acc = total_correct / max(1, total_count)
            history["loss"].append(mean_loss)
            history["acc"].append(acc)
            if verbose:
                print(
                    f"[pretrain epoch {ep + 1}/{epochs}] loss={mean_loss:.4f} acc={acc:.3f}"
                )

        # Restore original LR if we changed it
        if lr is not None:
            for pg, old in zip(self.opt.param_groups, orig_lrs):
                pg["lr"] = old

        return history


def select_general_reinforcement_learning(
    population, k, model: ParentSelectorRL, target: np.ndarray
):
    selected = []
    Phi = np.stack([ind.predicted_values for ind in population]).astype(np.float32)

    for _ in range(k // 2):
        mother = selAutomaticEpsilonLexicaseFast(population, 1)[0]
        mother.rl_selected = False
        i = population.index(mother)

        j, aux = model.select_second_parent(i, Phi, target, mode="train")
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

    # plot_reward_distribution(rewards)


def plot_reward_distribution(rewards):
    plt.hist(rewards, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


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
