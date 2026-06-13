# Archive

Historical artifacts kept for provenance, not used by any script.

- **`evaluation_results.RANDOMBUG.bak.json`** — evaluation output produced *before*
  the 2026-06-10 inference-determinism fix. At that time every DRL agent's
  `select_action` short-circuited to a random action whenever the replay buffer was
  smaller than one batch; a freshly loaded agent has an empty buffer, so these numbers
  reflect **random routing, not the trained policy**. Superseded by the corrected
  `results/evaluation_results.json`. The regression that caused this is now guarded by
  `tests/test_agents.py::test_*_eval_mode_is_deterministic_not_random`.
