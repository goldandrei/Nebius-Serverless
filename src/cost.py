def endpoint_cost(t_up: float, t_down: float, rate_hr: float) -> float:
    """Total serving cost for one endpoint lifetime at rate_hr $/hour (per-second billing)."""
    return (t_down - t_up) * (rate_hr / 3600.0)


def cost_per_1k_tokens(serving_cost: float, completion_tokens: int) -> float:
    if completion_tokens == 0:
        return 0.0
    return serving_cost / completion_tokens * 1000.0
