import math
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

INDUSTRY_OUTPUT_PER_WORKER = {
    1: 8500.0,
    2: 20000.0,
    3: 15000.0,
    4: 10000.0,
    5: 9500.0,
    7: 12000.0,
    8: 8500.0,
    9: 8800.0,
    10: 18000.0,
    11: 15000.0,
    12: 9500.0,
    13: 12000.0,
    14: 10500.0,
    15: 11500.0,
    16: 7500.0,
    17: 9000.0,
    18: 10000.0,
}
SKILL_MULTIPLIER = {0: 0.9, 1: 0.95, 2: 1.0, 3: 1.1, 4: 1.25}
_HOURS_PER_MONTH = 173.2


@dataclass
class FirmRewardConfig:
    vacancy_posting_cost: float = 2.5
    profit_buffer: float = 100.0
    match_quality_floor: float = 0.6


def compute_firm_reward(
    firm_state: Dict[str, Any],
    action: Dict[str, Any],
    outcome: Dict[str, Any],
    config: Optional[Any] = None,
    verbose: bool = False,
    firm_id: int = 0,
    return_decomposition: bool = False,
    return_components: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]], Tuple[float, float]]:
    employees = firm_state.get("employees", [])
    n_vacancies = len(firm_state.get("vacancies", []))
    firm_industry = firm_state.get("industry", 8)
    vac_cost_param = getattr(config, "vacancy_posting_cost", 2.5)
    buffer = getattr(config, "profit_buffer", 100.0)
    quality_floor = getattr(config, "match_quality_floor", 0.6)
    trade_demand_multiplier = float(firm_state.get("trade_demand_multiplier", 1.0))
    trade_productivity_multiplier = float(firm_state.get("trade_productivity_multiplier", 1.0))
    trade_input_cost_rate = float(firm_state.get("trade_input_cost_rate", 0.0))
    counterfactual_revenue = 0.0
    wage_bill = 0.0
    for emp in employees:
        wage = emp.get("hourly_wage", 20.0)
        industry = emp.get("industry", firm_industry)
        skill = emp.get("skill_level", 2)
        quality = max(emp.get("match_quality", 0.5), quality_floor)
        base_hourly_product = INDUSTRY_OUTPUT_PER_WORKER.get(industry, 10000.0) / _HOURS_PER_MONTH
        effective_product = base_hourly_product * SKILL_MULTIPLIER.get(skill, 1.0) * quality
        counterfactual_revenue += effective_product
        wage_bill += wage
    revenue = counterfactual_revenue * trade_demand_multiplier * trade_productivity_multiplier
    trade_input_costs = counterfactual_revenue * trade_input_cost_rate
    vacancy_costs = n_vacancies * vac_cost_param
    profit = revenue - wage_bill - vacancy_costs - trade_input_costs
    scaled_reward = math.log(max(profit + buffer, 1.0)) - math.log(buffer)
    if return_decomposition:
        decomp = {
            "revenue": revenue,
            "counterfactual_revenue": counterfactual_revenue,
            "trade_revenue_loss": counterfactual_revenue - revenue,
            "trade_input_costs": trade_input_costs,
            "wage_bill": wage_bill,
            "vacancy_costs": vacancy_costs,
            "profit": profit,
            "scaled_reward": scaled_reward,
            "employee_value": revenue,
            "vacancy_cost": -vacancy_costs,
            "excess_vacancy": 0.0,
            "hiring": 0.0,
            "separation": 0.0,
            "growth": 0.0,
            "staffing_penalty": 0.0,
            "adjustment_cost": 0.0,
            "baseline_proximity": 0.0,
        }
        return (scaled_reward, decomp)
    if return_components:
        return (scaled_reward, profit)
    return scaled_reward
