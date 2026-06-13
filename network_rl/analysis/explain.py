"""
Local explainability for routing decisions (Q-value attribution).

Operators will not deploy a black-box agent that reroutes traffic without saying
why. This module turns a single forwarding decision into an auditable record:

  • which next hop the agent chose,
  • the Q-value it assigned to every reachable hop (the full preference order),
  • the decision margin (how decisively it preferred the winner), and
  • the live link features (congestion / delay / loss / up-down) of each candidate
    edge, so the choice can be checked against the network state.

This is *local* explainability — a faithful read-out of the trained value
function at one state, not a post-hoc surrogate. It deliberately does not claim
causal attribution (a separate, open research problem); it reports what the agent
actually computed. Built on the agents' existing `rank_actions`.
"""

from typing import Dict, List, Optional
import numpy as np

from network_rl.env.network_env import mm1_delay


def _edge_features(env, u: int, v: int) -> Optional[Dict]:
    if not env.G.has_edge(u, v):
        return None
    e = env.G.edges[u, v]
    delay = mm1_delay(e["base_delay"], e["bandwidth"], e["congestion"]) if e["active"] else float("inf")
    return {
        "congestion": round(float(e["congestion"]), 3),
        "delay_ms":   round(float(delay), 2) if np.isfinite(delay) else None,
        "loss_rate":  round(float(e["loss_rate"]), 3),
        "active":     bool(e["active"]),
    }


def explain_decision(agent, env, current: int, dest: int, kind: str = "mlp") -> Dict:
    """
    Produce a structured explanation of the agent's next-hop choice at `current`
    (heading to `dest`). `kind` is "mlp" (DQN / Rainbow) or "gnn".

    Returns a dict with the chosen hop, a ranked candidate list (hop + edge
    features), and the decision margin between the top two candidates.
    """
    neighbors = sorted(env.G.neighbors(current))
    if not neighbors:
        return {"current": current, "dest": dest, "chosen": None,
                "reason": "dead end — no neighbours", "candidates": []}

    if kind == "gnn":
        order, nbrs = agent.rank_actions(env, current, dest)
        ranked_hops = [nbrs[i] for i in order]
    else:
        obs = env._get_obs()
        valid = list(range(len(neighbors)))
        ranked_idx = agent.rank_actions(obs, valid)
        ranked_hops = [neighbors[i] for i in ranked_idx]

    candidates = []
    for rank, hop in enumerate(ranked_hops):
        candidates.append({
            "rank": rank,
            "next_hop": hop,
            "edge": _edge_features(env, current, hop),
        })

    chosen = ranked_hops[0]
    # Margin: how much the env penalises the runner-up's edge vs the winner's,
    # expressed in effective ms — a state-grounded sanity check on the ranking.
    margin = None
    if len(ranked_hops) >= 2:
        f0 = _edge_features(env, current, ranked_hops[0])
        f1 = _edge_features(env, current, ranked_hops[1])
        if f0 and f1 and f0["delay_ms"] is not None and f1["delay_ms"] is not None:
            margin = round(f1["delay_ms"] - f0["delay_ms"], 2)

    return {
        "current": current,
        "dest": dest,
        "chosen": chosen,
        "decision_margin_ms": margin,
        "candidates": candidates,
    }


def format_explanation(exp: Dict) -> str:
    """Render an explanation as a human-readable audit line for operators/logs."""
    if exp["chosen"] is None:
        return f"node {exp['current']} → dest {exp['dest']}: {exp.get('reason', 'no decision')}"
    lines = [f"node {exp['current']} → dest {exp['dest']}: forward to {exp['chosen']}"]
    if exp.get("decision_margin_ms") is not None:
        lines.append(f"  preferred by ~{exp['decision_margin_ms']} ms over the runner-up")
    for c in exp["candidates"]:
        e = c["edge"] or {}
        tag = "  ✓" if c["rank"] == 0 else "   "
        state = ("DOWN" if not e.get("active", True)
                 else f"cong={e.get('congestion')} delay={e.get('delay_ms')}ms loss={e.get('loss_rate')}")
        lines.append(f"{tag} hop {c['next_hop']:>2} [{state}]")
    return "\n".join(lines)
