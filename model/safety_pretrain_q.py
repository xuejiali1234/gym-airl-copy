from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.predictive_safety_cost import PredictiveSafetyCostNetwork
from model.predictive_safety_oracle import PredictiveSafetyOracle
from model.safety_q_module import SafetyQNetwork
from model.safety_oracle_q import SafetyOracleQ


def _stack_all_state_action(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    all_states = [torch.as_tensor(traj["state"], dtype=torch.float32) for traj in dataset.trajectories]
    all_actions = [torch.as_tensor(traj["action"], dtype=torch.float32) for traj in dataset.trajectories]
    return torch.cat(all_states, dim=0), torch.cat(all_actions, dim=0)



def _make_generator(seed: Optional[int], device: str | torch.device = "cpu") -> Optional[torch.Generator]:
    if seed is None:
        return None
    device_str = str(device)
    try:
        generator = torch.Generator(device=device_str)
    except RuntimeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _generate_candidate_unsafe_actions(
    states: torch.Tensor,
    *,
    multiplier: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate action proposals that are intentionally more aggressive than the expert.
    The oracle will filter them, so the output only needs to be diverse and slightly biased.
    """
    if states.numel() == 0:
        return torch.empty((0, 2), dtype=states.dtype, device=states.device)

    n = max(1, int(states.shape[0] * multiplier))
    idx = torch.randint(0, states.shape[0], (n,), device=states.device, generator=generator)
    sampled = states[idx]

    actions = torch.empty((n, 2), dtype=sampled.dtype, device=sampled.device)
    actions[:, 0] = torch.empty(n, device=sampled.device).uniform_(-1.0, 1.0, generator=generator)
    # Bias toward large positive longitudinal acceleration to create low-TTC cases.
    actions[:, 1] = torch.empty(n, device=sampled.device).uniform_(0.4, 1.0, generator=generator)

    # Mix in a few hard-braking actions to cover rear-end / oscillation risk too.
    mask_brake = torch.rand(n, device=sampled.device, generator=generator) < 0.25
    actions[mask_brake, 1] = torch.empty(mask_brake.sum(), device=sampled.device).uniform_(
        -1.0,
        -0.4,
        generator=generator,
    )
    return sampled, actions


def _generate_predictive_candidate_actions(
    states: torch.Tensor,
    expert_actions: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if states.numel() == 0:
        empty_states = torch.empty((0, states.shape[-1]), dtype=states.dtype, device=states.device)
        empty_actions = torch.empty((0, 2), dtype=expert_actions.dtype, device=expert_actions.device)
        return empty_states, empty_actions

    templates = torch.tensor(
        [
            [-1.0, 0.0],   # 强行并线
            [-0.8, 0.4],   # 并线+轻加速插入
            [-0.8, -0.4],  # 并线+轻减速等待
            [-0.5, 0.0],   # 温和并线
            [-0.2, -0.5],  # 延迟并线+减速
            [0.0, 0.0],    # 保持车道
            [0.0, 0.4],    # 轻加速插入
            [0.0, -0.4],   # 轻减速等待
            [0.4, 0.0],    # 延迟并线
            [0.8, -0.2],   # 明显延迟并线
        ],
        dtype=expert_actions.dtype,
        device=expert_actions.device,
    )

    n_states = states.shape[0]
    n_templates = templates.shape[0]
    tiled_states = states.repeat_interleave(n_templates, dim=0)
    tiled_actions = templates.repeat(n_states, 1)

    if generator is not None:
        noise = torch.empty_like(tiled_actions).uniform_(-0.05, 0.05, generator=generator)
        tiled_actions = torch.clamp(tiled_actions + noise, min=-1.0, max=1.0)
    return tiled_states, tiled_actions



def build_safety_training_tensors(
    dataset,
    oracle: SafetyOracleQ,
    *,
    max_pairs: int = 50000,
    synthetic_multiplier: float = 1.0,
    device: str | torch.device = "cpu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    states, actions = _stack_all_state_action(dataset)
    sample_generator = _make_generator(seed, device="cpu")
    if states.shape[0] > max_pairs:
        idx = torch.randperm(states.shape[0], generator=sample_generator)[:max_pairs]
        states = states[idx]
        actions = actions[idx]

    states = states.to(device)
    actions = actions.to(device)

    with torch.no_grad():
        labels = oracle.get_labels(states, actions).squeeze(-1)

    safe_mask = labels < 0.5
    safe_states = states[safe_mask]
    safe_actions = actions[safe_mask]

    synth_states, synth_actions = _generate_candidate_unsafe_actions(
        safe_states,
        multiplier=synthetic_multiplier,
        generator=sample_generator,
    )
    with torch.no_grad():
        synth_labels = oracle.get_labels(synth_states, synth_actions).squeeze(-1)
    synth_keep = synth_labels > 0.0

    x_s = torch.cat([states, synth_states[synth_keep]], dim=0)
    x_a = torch.cat([actions, synth_actions[synth_keep]], dim=0)
    y = torch.cat([labels, synth_labels[synth_keep]], dim=0).unsqueeze(1).clamp(0.0, 1.0)

    stats = {
        "n_safe": float((y < 0.5).sum().item()),
        "n_unsafe": float((y >= 0.5).sum().item()),
        "n_warning": float(((y > 0.0) & (y < 0.5)).sum().item()),
    }
    return x_s.cpu(), x_a.cpu(), y.cpu(), stats


def build_predictive_safety_training_tensors(
    dataset,
    oracle: PredictiveSafetyOracle,
    *,
    max_pairs: int = 50000,
    use_candidates: bool = True,
    device: str | torch.device = "cpu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    states, actions = _stack_all_state_action(dataset)
    sample_generator = _make_generator(seed, device="cpu")
    if states.shape[0] > max_pairs:
        idx = torch.randperm(states.shape[0], generator=sample_generator)[:max_pairs]
        states = states[idx]
        actions = actions[idx]

    states = states.to(device)
    actions = actions.to(device)

    with torch.no_grad():
        expert_info = oracle.analyze_batch(states, actions)
    risk = expert_info["risk_score"]
    critical = expert_info["critical_label"]

    x_s = states
    x_a = actions
    y_risk = risk
    y_critical = critical

    if use_candidates:
        candidate_states, candidate_actions = _generate_predictive_candidate_actions(
            states,
            actions,
            generator=sample_generator,
        )
        with torch.no_grad():
            candidate_info = oracle.analyze_batch(candidate_states, candidate_actions)
        x_s = torch.cat([x_s, candidate_states], dim=0)
        x_a = torch.cat([x_a, candidate_actions], dim=0)
        y_risk = torch.cat([y_risk, candidate_info["risk_score"]], dim=0)
        y_critical = torch.cat([y_critical, candidate_info["critical_label"]], dim=0)

    stats = {
        "n_samples": float(x_s.shape[0]),
        "n_critical": float((y_critical >= 0.5).sum().item()),
        "risk_mean": float(y_risk.mean().item()),
        "risk_high": float((y_risk >= 0.7).sum().item()),
    }
    return x_s.cpu(), x_a.cpu(), y_risk.cpu(), y_critical.cpu(), stats



def pretrain_safety_q_network(
    safety_net: SafetyQNetwork,
    dataset,
    oracle: SafetyOracleQ,
    *,
    device: str | torch.device = "cpu",
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    pos_weight: float = 3.0,
    max_pairs: int = 50000,
    synthetic_multiplier: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    safety_net = safety_net.to(device)

    x_s, x_a, y, label_stats = build_safety_training_tensors(
        dataset,
        oracle,
        max_pairs=max_pairs,
        synthetic_multiplier=synthetic_multiplier,
        device=device,
        seed=seed,
    )

    if label_stats["n_safe"] <= 0 or label_stats["n_unsafe"] <= 0:
        raise RuntimeError(
            f"Safety-Q pretraining failed: safe={label_stats['n_safe']}, unsafe={label_stats['n_unsafe']}"
        )

    ds = TensorDataset(x_s, x_a, y)
    loader_generator = _make_generator(seed, device="cpu")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(safety_net.parameters(), lr=lr, weight_decay=weight_decay)

    last_loss = 0.0
    safety_net.train()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for batch_s, batch_a, batch_y in loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_y = batch_y.to(device)

            logits = safety_net(batch_s, batch_a if safety_net.use_action else None)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(safety_net.parameters(), max_norm=1.0)
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        last_loss = running / max(1, n_batches)
        if verbose:
            print(
                f"[Safety-Q Pretrain] epoch={epoch + 1}/{epochs} loss={last_loss:.4f} "
                f"safe={label_stats['n_safe']:.0f} unsafe={label_stats['n_unsafe']:.0f} "
                f"warning={label_stats['n_warning']:.0f}"
            )

    safety_net.eval()
    return {
        "loss": float(last_loss),
        "n_safe": float(label_stats["n_safe"]),
        "n_unsafe": float(label_stats["n_unsafe"]),
        "n_warning": float(label_stats["n_warning"]),
    }


def pretrain_predictive_safety_network(
    safety_net: PredictiveSafetyCostNetwork,
    dataset,
    oracle: PredictiveSafetyOracle,
    *,
    device: str | torch.device = "cpu",
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    max_pairs: int = 50000,
    use_candidates: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    safety_net = safety_net.to(device)

    x_s, x_a, y_risk, y_critical, label_stats = build_predictive_safety_training_tensors(
        dataset,
        oracle,
        max_pairs=max_pairs,
        use_candidates=use_candidates,
        device=device,
        seed=seed,
    )

    ds = TensorDataset(x_s, x_a, y_risk, y_critical)
    loader_generator = _make_generator(seed, device="cpu")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )

    optimizer = torch.optim.Adam(safety_net.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    last_total = 0.0
    last_risk = 0.0
    last_critical = 0.0
    safety_net.train()
    for epoch in range(epochs):
        running_total = 0.0
        running_risk = 0.0
        running_critical = 0.0
        n_batches = 0
        for batch_s, batch_a, batch_risk, batch_critical in loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_risk = batch_risk.to(device)
            batch_critical = batch_critical.to(device)

            risk_logit, critical_logit = safety_net.forward_heads(
                batch_s,
                batch_a if safety_net.use_action else None,
            )
            pred_risk = torch.sigmoid(risk_logit)
            loss_risk = mse_loss(pred_risk, batch_risk)
            loss_crit = bce_loss(critical_logit, batch_critical)
            loss = loss_risk + 2.0 * loss_crit

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(safety_net.parameters(), max_norm=1.0)
            optimizer.step()

            running_total += float(loss.item())
            running_risk += float(loss_risk.item())
            running_critical += float(loss_crit.item())
            n_batches += 1

        last_total = running_total / max(1, n_batches)
        last_risk = running_risk / max(1, n_batches)
        last_critical = running_critical / max(1, n_batches)
        if verbose:
            print(
                f"[Predictive Safety Pretrain] epoch={epoch + 1}/{epochs} "
                f"loss={last_total:.4f} risk={last_risk:.4f} critical={last_critical:.4f} "
                f"samples={label_stats['n_samples']:.0f} critical={label_stats['n_critical']:.0f}"
            )

    safety_net.eval()
    return {
        "loss": float(last_total),
        "risk_loss": float(last_risk),
        "critical_loss": float(last_critical),
        "n_samples": float(label_stats["n_samples"]),
        "n_critical": float(label_stats["n_critical"]),
        "risk_mean": float(label_stats["risk_mean"]),
        "risk_high": float(label_stats["risk_high"]),
    }
