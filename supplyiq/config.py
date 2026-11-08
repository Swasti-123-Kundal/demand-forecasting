"""All ASSUMED plant-side parameters live here (not real HUL / plant data)."""
from dataclasses import dataclass, asdict


@dataclass
class PlantParams:
    # ---- ASSUMED parameters: say so in CV / README -------------------------
    lead_time_days: int = 7            # planned replenishment lead time (L)
    lead_time_sd_days: float = 1.0     # real lead time wobbles around L (disturbance)
    holding_rate_per_year: float = 0.25  # holding cost = 25 % of unit cost per year
    unit_cost_ratio: float = 0.70      # unit cost = 70 % of selling price
    order_cost_frac: float = 0.05      # fixed cost / order = 5 % of one day's sales value
    shortage_cost_mult: float = 0.50   # lost sale costs 0.5 x price (margin + goodwill)
    fixed_cover_days: float = 3.0      # Policy A: fixed safety stock = 3 days of demand
    service_level: float = 0.95        # target cycle service level (z = norm.ppf)

    def to_dict(self):
        return asdict(self)


# Forecast horizon: every forecast for day d only uses information up to d - MIN_LAG.
# Keep MIN_LAG >= lead time so forecasts used in ROP at time t are truly available at t.
MIN_LAG = 7
