# Shield Debug Summary: v6 to v10b

## 1. Scope and baseline

This document summarizes the evaluation-only shield debugging path from `v6` to `v10b`.

Important scope:

- These experiments are **evaluation-only action filters**.
- They do **not** change AIRL training, reward learning, environment reward, or checkpoint saving.
- The base model throughout this phase is:
  - `U220_D230_epoch290`
  - checkpoint:
    `train_log/baseline_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_20260425_224122/checkpoints/baseline_policy_attn_goal_safe_branch_aux_probe_P300_U220_L5e6_D230_epoch_290.zip`

Main goal of this phase:

1. Keep or improve hard-case safety on fixed `hard15`.
2. Avoid introducing new collisions on `full-217`.
3. Improve merge completion without touching the training pipeline.


## 2. Core conclusion

From `v6` to `v10b`, the debugging path can be summarized as:

- `v6-v7`: mainly improved emergency handling and endpoint stability, but still could not eliminate the last hard-case collision.
- `v8`: the shield started to solve the last collision, especially after the `posgap` fix, but merge completion was still weak.
- `v9`: explicit merge recovery was introduced; `v9b` was the first version that made `hard15` fully pass.
- `v10`: the focus shifted from hard15-only success to **full-217 robustness**.
  - `v10a` removed the `vehicle_1085` collision, but hurt `vehicle_434 (traj_index=136)`.
  - `v10b` kept the `1085` fix and recovered `434`, making it the best shield variant in this phase.

Current best evaluation-only shield result:

- Variant: `v10b_leadgap_policy_veto_recovery`
- Directory:
  - `train_log/safety_shield_eval_v10b_leadgap_policy_veto_recovery_full217_20260428`
- Result:
  - `full-217`: `merge=0.8618`, `endpoint=0.9124`, `safety=0.9124`, `collision=0.0000`
  - `hard15`: `merge=0.8000`, `endpoint=1.0000`, `safety=1.0000`, `collision=0.0000`


## 3. Version timeline

### 3.1 v6 stage: emergency follow-side handling

The `v6` stage focused on handling `L5_follow`-type rear risk more aggressively.

Variants:

- `v6a_follow_emergency`
- `v6b_recovery_merge`
- `v6c_margin020`

High-level intent:

- `v6a`: strengthen follow-side emergency handling.
- `v6b`: add recovery-to-merge logic after emergency stabilization.
- `v6c`: keep `v6b` logic and increase the risk-improvement margin to `0.20`.

Hard15 results:

| Variant | Merge | Endpoint | Safety | Collision | Shield rate | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `v6a_follow_emergency` | 0.6667 | 0.8000 | 0.7333 | 0.0667 | 0.5903 | collision still remained |
| `v6b_recovery_merge` | 0.6667 | 0.8000 | 0.7333 | 0.0667 | 0.5884 | recovery logic did not yet change outcome |
| `v6c_margin020` | 0.6667 | 0.8000 | 0.7333 | 0.0667 | 0.5877 | more conservative margin still not enough |

Judgment:

- `v6` proved that simply strengthening emergency follow-side handling was not enough.
- The last hard-case collision was not just a "brake harder" problem.
- Merge completion also remained weak.


### 3.2 v7 stage: better endpoint behavior, but still one collision

Variants:

- `v7a_follow_low_thw`
- `v7b_merge_preserving_warning`

High-level intent:

- `v7a`: make the shield more sensitive to low THW follow-side risk.
- `v7b`: try to preserve merge tendency while still warning / filtering unsafe actions.

Hard15 results:

| Variant | Merge | Endpoint | Safety | Collision | Shield rate | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `v7a_follow_low_thw` | 0.6667 | 0.9333 | 0.8667 | 0.0667 | 0.6181 | endpoint improved, last collision remained |
| `v7b_merge_preserving_warning` | 0.6667 | 0.9333 | 0.8667 | 0.0667 | 0.6231 | warning-based merge preservation still did not remove collision |

Judgment:

- `v7` was better than `v6` on endpoint success.
- But it still did not solve the final hard-case collision.
- This suggested that the remaining issue was becoming more trajectory-specific, not a simple global threshold problem.


### 3.3 v8 stage: first real collision breakthrough

Variants:

- `v8a_follow_burst`
- `v8a_follow_burst_posgap`

High-level intent:

- Introduce a more active follow-side burst response.
- The `posgap` patch further constrained the logic to safer geometric conditions.

Hard15 results:

| Variant | Merge | Endpoint | Safety | Collision | Shield rate | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `v8a_follow_burst` | 0.6000 | 0.8667 | 0.8000 | 0.0667 | 0.6067 | too aggressive, not stable enough |
| `v8a_follow_burst_posgap` | 0.6667 | 0.9333 | 0.9333 | 0.0000 | 0.6291 | first zero-collision hard15 result |

Judgment:

- `v8a_follow_burst_posgap` was a turning point.
- It removed the hard15 collision and lifted safety clearly.
- However, merge completion was still only `0.6667`, so the shield had become safe, but still too willing to end in non-merge transition states.


### 3.4 v9 stage: explicit merge recovery

Variants:

- `v9a_merge_recovery`
- `v9b_near_aux_recovery`

High-level intent:

- Instead of only blocking unsafe actions, actively help the vehicle complete the merge when it is already close to success.
- `v9a` added merge recovery in a narrower transition-zone setting.
- `v9b` expanded recovery earlier and more strongly into the near-aux region.

Hard15 results:

| Variant | Merge | Endpoint | Safety | Collision | Shield rate | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `v9a_merge_recovery` | 0.7333 | 0.9333 | 0.9333 | 0.0000 | 0.6158 | merge improved without reintroducing collision |
| `v9b_near_aux_recovery` | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.6013 | first fully passing hard15 result |

`v9b` full-217 result:

- `merge=0.8618`
- `endpoint=0.9124`
- `safety=0.9078`
- `collision=0.0046` (`1/217`)
- `shield_rate=0.3429`

Judgment:

- `v9b` was the first version that fully solved `hard15`.
- But `full-217` still had one collision, so the merge recovery logic was strong enough to help hard cases, yet sometimes over-committed on a longer trajectory.

This stage is where the problem changed shape:

- hard15 was already good enough;
- the main risk became **over-recovery on certain full trajectories**.


## 4. Why v9b was not yet final

The key failure under `v9b` was:

- `vehicle_1085_trajectory.csv`

Observed behavior:

- Under `v9b`, this trajectory successfully merged into the target lane.
- But it later collided with `L5_follow`.
- The final failure was not because the collision-step action was obviously wrong.
- The deeper issue was that merge recovery pushed the ego vehicle into the target lane too early, creating a bad post-merge follow-side scenario that later became unrecoverable.

Evidence directory:

- `train_log/safety_shield_eval_v9b_near_aux_recovery_full217_20260428/targeted_merge_analysis`
- `train_log/shield_trajectory_diagnosis/vehicle_1085_trajectory`

The diagnostic takeaway was:

- `v9b` solved merge completion,
- but for some trajectories it overrode the policy too aggressively even when the policy was already trying to retreat or stabilize.


## 5. v10 stage: from hard15 success to full-217 robustness

### 5.1 v10a_policy_veto_recovery

Intent:

- Add a **policy retreat veto**.
- If the policy action strongly points back toward the auxiliary side, do not always force merge recovery.

Key parameter:

- `merge_recovery_policy_veto_x = 0.25`

Interpretation:

- If `policy_action_x > 0.25`, treat that as a strong enough retreat / recovery intention and suppress some merge-recovery forcing.

Result:

- `vehicle_1085` was fixed.
- `full-217` collision dropped from `1` to `0`.
- But this global veto was too blunt and damaged another trajectory:
  - `vehicle_434_trajectory.csv`, `traj_index=136`

Key comparison:

| Variant | vehicle_1085 | vehicle_434 (idx136) |
|---|---|---|
| `v9b` | merge yes, endpoint yes, **collision yes** | merge yes, endpoint yes, collision no |
| `v10a` | merge yes, endpoint yes, collision no | **merge no, endpoint no, final lane aux** |

Full-217 summary for `v10a`:

- `merge=0.8479`
- `endpoint=0.9078`
- `safety=0.9078`
- `collision=0.0000`

Judgment:

- `v10a` proved that the `1085` collision could be removed by limiting over-aggressive merge recovery.
- But it also proved that a **global policy-veto threshold** was too coarse.


### 5.2 v10b_leadgap_policy_veto_recovery

This is the key final variant in this debugging chain.

#### 5.2.1 Design motivation

`v10a` answered the question:

- "Can we stop over-recovery?"  
  Yes.

But it created a new problem:

- "Can we stop over-recovery **without** hurting trajectories that genuinely need merge recovery?"  
  `v10a` could not.

So `v10b` changed the logic from:

- "policy wants retreat -> veto merge recovery"

to:

- "policy wants retreat **and** the current situation looks like a follow-side post-merge risk case -> veto merge recovery"

In other words:

- `v10a` used a **global veto**;
- `v10b` used a **context-aware veto**.


#### 5.2.2 Core mechanism

Compared with `v10a`, `v10b` adds two more contextual conditions:

- `merge_recovery_policy_veto_lead_gap = 100.0`
- `merge_recovery_policy_veto_lead_thw = 5.0`

The veto is activated only when all of the following hold:

1. current risk side is `L5_follow`
2. `policy_action_x > 0.25`
3. target-lane lead space is not sufficiently open:
   - `lead_gap < 100.0`, or
   - `lead_thw < 5.0`

Meaning:

- If the policy wants to retreat **and** the target lane ahead is not very open, the shield assumes that forcing merge recovery may be dangerous and blocks it.
- If the policy wants to retreat but the target-lane lead side is extremely open, the shield allows merge recovery to continue.

This is exactly why `v10b` can separate:

- `vehicle_1085`: should be vetoed
- `vehicle_434 (idx136)`: should not be vetoed


#### 5.2.3 Why it works on 1085 and 434 simultaneously

`vehicle_1085` under `v9b`:

- merge recovery pushed it into target lane too early;
- later it collided with `L5_follow`;
- this is a classic case where "policy wants to hold back" should be respected.

`vehicle_434 (traj_index=136)`:

- it still needed merge recovery help to finish the merge;
- a global retreat veto in `v10a` prevented that help and left the vehicle in aux / non-success state;
- in `v10b`, the shield sees that target-lane lead-side space is very open, so it does **not** block recovery there.

This is the most important conceptual gain of `v10b`:

- it is not stronger in a generic sense;
- it is **better targeted**.


#### 5.2.4 Final results of v10b

Directory:

- `train_log/safety_shield_eval_v10b_leadgap_policy_veto_recovery_full217_20260428`

Shield parameters:

```json
{
  "merge_recovery_policy_x_trigger": -0.03,
  "merge_recovery_min_progress": 0.15,
  "merge_recovery_target_x": -0.2,
  "merge_recovery_x_options": [-1.0, -0.75, -0.5, -0.35, -0.25, -0.15, -0.1, -0.05],
  "merge_recovery_risk_slack": 1.5,
  "merge_recovery_aux_slack": 2.0,
  "merge_recovery_policy_veto_x": 0.25,
  "merge_recovery_policy_veto_lead_gap": 100.0,
  "merge_recovery_policy_veto_lead_thw": 5.0
}
```

Results:

| Split | Merge | Endpoint | Safety | Collision | Collisions | Shield rate |
|---|---:|---:|---:|---:|---:|---:|
| `full-217` | 0.8618 | 0.9124 | 0.9124 | 0.0000 | 0 | 0.3332 |
| `hard15` | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.6023 |

Direct comparison:

| Variant | Full merge | Full endpoint | Full safety | Full collision | Shield rate | Judgment |
|---|---:|---:|---:|---:|---:|---|
| `v9b` | 0.8618 | 0.9124 | 0.9078 | 0.0046 | 0.3429 | hard15 excellent, but one full collision remained |
| `v10a` | 0.8479 | 0.9078 | 0.9078 | 0.0000 | 0.3337 | removed collision, but over-conservative and hurt success |
| `v10b` | 0.8618 | 0.9124 | 0.9124 | 0.0000 | 0.3332 | kept the `v10a` safety gain without sacrificing `v9b` success |

So `v10b` is the best trade-off in this line:

- zero collision on `full-217`
- zero collision and full pass on `hard15`
- no regression relative to `v9b` on full merge / endpoint
- slightly lower shield intervention rate than `v9b`


#### 5.2.5 Remaining issues after v10b

`v10b` is not perfect yet. Its remaining issue is no longer collision, but merge completion.

Current full-217 merge-false summary:

- `merge_false_count = 30`
- `endpoint_true_merge_false_count = 14`
- `endpoint_false_merge_false_count = 16`

Current hard15 remaining merge-false trajectories:

- `vehicle_757_trajectory.csv`
- `vehicle_303_trajectory.csv`
- `vehicle_905_trajectory.csv`

These three are all:

- `endpoint_success = True`
- `safety_success = True`
- `collision = False`
- but `merge_success = False`

That means the next stage should be:

- not collision-oriented;
- not global hyperparameter sweeping;
- but **targeted merge finishing** on near-threshold transition cases.


## 6. Final interpretation

The progression from `v6` to `v10b` is not random tuning. It has a clear logic:

1. `v6-v7`: learn that stronger emergency protection alone is insufficient.
2. `v8`: solve the last hard-case collision, but merge completion still lags.
3. `v9`: add active merge recovery and fully solve hard15.
4. `v10a`: discover that the full-set remaining collision comes from over-aggressive recovery.
5. `v10b`: make the veto context-aware, preserving both safety and success.

In short:

- `v8` solved "can we make it safe on hard15?"
- `v9` solved "can we also complete the merge on hard15?"
- `v10b` solved "can we keep that improvement without reintroducing full-set collisions?"

That is why `v10b_leadgap_policy_veto_recovery` should be treated as the current best shield variant in this debugging chain.


## 7. Key evidence files

Main result directories:

- `train_log/safety_shield_eval_v6a_follow_emergency_hard15`
- `train_log/safety_shield_eval_v6b_recovery_merge_hard15`
- `train_log/safety_shield_eval_v6c_margin020_hard15`
- `train_log/safety_shield_eval_v7a_follow_low_thw_hard15`
- `train_log/safety_shield_eval_v7b_merge_preserving_warning_hard15`
- `train_log/safety_shield_eval_v8a_follow_burst_hard15`
- `train_log/safety_shield_eval_v8a_follow_burst_posgap_hard15`
- `train_log/safety_shield_eval_v9a_merge_recovery_hard15`
- `train_log/safety_shield_eval_v9b_near_aux_recovery_full217_20260428`
- `train_log/safety_shield_eval_v10a_policy_veto_recovery_full217_20260428`
- `train_log/safety_shield_eval_v10b_leadgap_policy_veto_recovery_full217_20260428`

Diagnostic evidence:

- `train_log/shield_trajectory_diagnosis/vehicle_1085_trajectory`
- `train_log/shield_trajectory_diagnosis/vehicle_434_trajectory`
- `train_log/safety_shield_eval_v9b_near_aux_recovery_full217_20260428/targeted_merge_analysis`

Supporting code:

- `evaluation/safety_shield_evaluate.py`
- `evaluation/diagnose_shield_trajectory.py`

