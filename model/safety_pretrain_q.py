from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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



def build_safety_training_tensors(
    dataset,
    oracle: SafetyOracleQ,
    *,
    max_pairs: int = 50000,
    synthetic_multiplier: float = 1.0,
    device: str | torch.device = "cpu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    safe_states = states[labels < 0.5]
    safe_actions = actions[labels < 0.5]
    unsafe_states_from_data = states[labels >= 0.5]
    unsafe_actions_from_data = actions[labels >= 0.5]

    synth_states, synth_actions = _generate_candidate_unsafe_actions(
        safe_states,
        multiplier=synthetic_multiplier,
        generator=sample_generator,
    )
    with torch.no_grad():
        synth_labels = oracle.get_labels(synth_states, synth_actions).squeeze(-1)
    unsafe_states_synth = synth_states[synth_labels >= 0.5]
    unsafe_actions_synth = synth_actions[synth_labels >= 0.5]

    if unsafe_states_from_data.numel() == 0:
        unsafe_states, unsafe_actions = unsafe_states_synth, unsafe_actions_synth
    elif unsafe_states_synth.numel() == 0:
        unsafe_states, unsafe_actions = unsafe_states_from_data, unsafe_actions_from_data
    else:
        unsafe_states = torch.cat([unsafe_states_from_data, unsafe_states_synth], dim=0)
        unsafe_actions = torch.cat([unsafe_actions_from_data, unsafe_actions_synth], dim=0)

    return safe_states.cpu(), safe_actions.cpu(), unsafe_states.cpu(), unsafe_actions.cpu()



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

    safe_s, safe_a, unsafe_s, unsafe_a = build_safety_training_tensors(
        dataset,
        oracle,
        max_pairs=max_pairs,
        synthetic_multiplier=synthetic_multiplier,
        device=device,
        seed=seed,
    )

    if safe_s.numel() == 0 or unsafe_s.numel() == 0:
        raise RuntimeError(
            f"Safety-Q pretraining failed: safe={safe_s.shape[0]}, unsafe={unsafe_s.shape[0]}"
        )

    x_s = torch.cat([safe_s, unsafe_s], dim=0)
    x_a = torch.cat([safe_a, unsafe_a], dim=0)
    y = torch.cat(
        [
            torch.zeros(safe_s.shape[0], 1, dtype=torch.float32),
            torch.ones(unsafe_s.shape[0], 1, dtype=torch.float32),
        ],
        dim=0,
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
                f"safe={safe_s.shape[0]} unsafe={unsafe_s.shape[0]}"
            )

    safety_net.eval()
    return {
        "loss": float(last_loss),
        "n_safe": float(safe_s.shape[0]),
        "n_unsafe": float(unsafe_s.shape[0]),
    }
