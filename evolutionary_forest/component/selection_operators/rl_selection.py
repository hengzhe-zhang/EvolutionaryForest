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
from torch.utils.data import Dataset, DataLoader


class SumRuleDataset(Dataset):
    def __init__(self, data, to_tensor_fn):
        self.samples = list(data)
        self._to_dev = to_tensor_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, Phi, target = self.samples[idx]
        Phi = self._to_dev(Phi)
        target = self._to_dev(target)
        return i, Phi, target


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
        self_learning=False,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # input is element-wise product features -> dimension = sem_dim
        self.model = PolicyNet(dim_in=sem_dim, hidden=hidden, dropout=dropout).to(
            self.device
        )
        self.baseline: Optional[torch.Tensor] = None

        self.fuser = nn.Linear(2 * sem_dim, sem_dim).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fuser.parameters()), lr=cfg.lr
        )

        self.self_learning = self_learning

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")
        return x.to(self.device)

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
        phi_i_rep = (
            phi_i.unsqueeze(0).expand(candidate_indices.numel(), D).detach()
        )  # [N-1, D]

        # Target: center once and L2-normalize
        t = self._to_tensor(target).view(-1).detach()  # [D]

        gating = True
        if gating:
            pair = torch.cat([phi_i_rep, Phi[candidate_indices]], dim=1)  # [N-1, 2D]
            avg = self.fuser(pair)  # [N-1, D]  (trainable linear fusion)
        else:
            avg = (phi_i_rep + Phi[candidate_indices]) * 0.5  # [N-1, D]

        feats = -((avg - t) ** 2)
        return feats, candidate_indices

    def score_candidates(
        self, i: int, Phi, target
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi = self._to_tensor(Phi)  # [N, D]
        phi_i = Phi[i]  # [D]
        feats, candidate_indices = self._pair_features(phi_i, Phi, target, mask_idx=i)
        scores = self.model(feats)  # [N-1]
        return scores, candidate_indices, feats

    def select_second_parent(self, i: int, Phi, target, mode: str = "eval"):
        self.model.eval() if mode == "eval" else self.model.train()
        Phi = self._to_tensor(Phi)
        target = self._to_tensor(target)

        scores, candidate_indices, feats = self.score_candidates(i, Phi, target)
        aux = {}

        logits = scores / max(1e-8, self.cfg.temperature)
        probs = F.softmax(logits, dim=0)
        m = torch.distributions.Categorical(probs=probs)

        if mode == "train":
            idx = m.sample()
            j = int(candidate_indices[idx].item())
            # print(
            #     "Sampled index:",
            #     j,
            #     "Ground Truth:",
            #     torch.argmax(feats.sum(dim=1)).item(),
            # )
            aux["logprob"] = m.log_prob(idx)
            aux["probs"] = probs.detach().cpu()
        else:
            idx = torch.argmax(scores)
            j = int(candidate_indices[idx].item())
            aux["logprob"] = m.log_prob(idx)

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
        adv = r

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
        batch_size: int = 32,  # can be >1 if N fixed
        num_workers: int = 0,
    ):
        # LR override
        orig_lrs = [pg["lr"] for pg in self.opt.param_groups]
        if lr is not None:
            for pg in self.opt.param_groups:
                pg["lr"] = lr

        dataset = SumRuleDataset(data, self._to_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        self.model.train()
        history = {"loss": [], "acc": []}

        for ep in range(epochs):
            total_loss, total_correct, total_count = 0.0, 0, 0

            for batch in loader:
                # batch contains a tuple of batched tensors
                # i: [B], Phi: [B, N, D], target: [B, D]
                i_batch, Phi_batch, target_batch = batch

                batch_losses = []
                batch_correct = 0

                for i, Phi, target in zip(i_batch, Phi_batch, target_batch):
                    Phi = self._to_tensor(Phi)
                    target = self._to_tensor(target)

                    N, D = Phi.shape
                    i = int(i) if isinstance(i, int) else int(i.item())

                    phi_i = Phi[i]  # [D]

                    feats, candidate_indices = self._pair_features(
                        phi_i, Phi, target, mask_idx=i
                    )

                    if self.self_learning:
                        with torch.no_grad():
                            label_local = torch.argmax(feats.sum(dim=1))
                    else:
                        candidate_indices = torch.arange(N, device=Phi.device)
                        candidate_indices = candidate_indices[
                            candidate_indices != i
                        ]  # [N-1]

                        # Repeat phi_i to match candidate count: [N-1, D]
                        phi_i_rep = (
                            phi_i.unsqueeze(0)
                            .expand(candidate_indices.numel(), D)
                            .detach()
                        )

                        # Target as a fixed reference
                        t = target.view(-1).detach()  # [D]

                        # Simple average fusion (replace with your fuser if desired)
                        avg = 0.5 * (phi_i_rep + Phi[candidate_indices])  # [N-1, D]

                        # Ground-truth label: index of the MIN distance, not max
                        with torch.no_grad():
                            d2 = ((avg - t) ** 2).sum(dim=1)  # [N-1]
                            label_local = torch.argmin(d2)  # scalar tensor

                    logits = self.model(feats)
                    loss = F.cross_entropy(
                        logits.unsqueeze(0), label_local.view(1), label_smoothing=0.1
                    )

                    batch_losses.append(loss)

                    pred_local = int(torch.argmax(logits).item())
                    batch_correct += int(pred_local == int(label_local))

                # Average batch loss
                loss = torch.stack(batch_losses).mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad_norm
                    )
                self.opt.step()

                total_loss += float(loss.detach().cpu())
                total_correct += batch_correct
                total_count += len(i_batch)

            mean_loss = total_loss / max(1, total_count)
            acc = total_correct / max(1, total_count)
            history["loss"].append(mean_loss)
            history["acc"].append(acc)
            if verbose:
                print(f"[epoch {ep + 1}/{epochs}] loss={mean_loss:.4f} acc={acc:.3f}")

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
            reward = ind.fitness.wvalues[0] - ind.parent_fitness[0]
            if reward <= 0:
                continue
            rewards.append(reward)

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
