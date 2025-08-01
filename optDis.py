"""
A drop‑in replacement module that lets Pyomo solve the **year‑long dispatch**
problem *and* maximise Net‑Present‑Value (NPV) internally.

How to integrate
----------------
1. `import hydrogen_dispatch_npv ` in **pso_pyomo.py** (after the other imports).
2. Replace the call that used to invoke `hydrogen_8760_dispatch` with:

    ```python
    result = optD,s.optimize_dispatch_with_hydrogen_pyomo_npv(
        x,                      # PSO position vector
        hourly_production,
        electricity_demand * cap_obj,
        electricity_price,
        SYSTEMS,
        discount_rate,
        project_lifetime,
        hydrogen_price,
        electricity_to_hydrogen,
        args.rel
    )
    ```

3. For battery-only case, replace call to `battery_8760_dispatch(...)` with:

    ```python
    result = optDis.optimize_dispatch_with_battery_pyomo_npv(
        x,
        hourly_production,
        electricity_demand * cap_obj,
        electricity_price,
        SYSTEMS,
        discount_rate,
        project_lifetime,
        args.rel
    )
    ```

Both functions return a dictionary containing:
- `npv`,
- `electricity_sold`,
- `battery_soc`,
- `battery_charge`,
- `battery_discharge`,
- (plus `hydrogen_production` if hydrogen)
"""


import numpy as np
import pyomo.environ as pyo
from utils import calculate_costs, replacement_cost

__all__ = [
    "optimize_dispatch_with_hydrogen_pyomo_npv",
]


def optimize_dispatch_with_hydrogen_pyomo_npv(
    x,
    hourly_production,
    hourly_demand,
    hourly_price,
    SYSTEMS,
    discount_rate,
    project_lifetime,
    hydrogen_price,
    electricity_to_hydrogen,
    rel,
):
    """Year‑long Pyomo dispatch that **maximises NPV**.

    Parameters
    ----------
    x : array‑like
        PSO position `[num_SMR, num_wind, num_solar, num_battery, num_electrolyzer]`.
    hourly_production, hourly_demand, hourly_price : np.ndarray (length 8760)
    SYSTEMS : dict – technology parameters (as in your original script)
    discount_rate : float
    project_lifetime : int – analysis horizon (years)
    hydrogen_price : float (USD/kg)
    electricity_to_hydrogen : float (kWh/kg) – conversion factor
    rel : float – ± tolerance for demand matching (0.5 → ±50 %)

    Returns
    -------
    dict  – keys: "electricity_sold", "hydrogen_production", "battery_soc",
             "battery_charge", "battery_discharge", "npv", "hydrogen_storage"
             All monetary figures are in **million USD**.
    """

    # -----  unpack subsystem counts  ---------------------------------------
    num_battery      = int(round(x[3]))
    num_electrolyzer = int(round(x[4]))

    # -----  fixed‑cost side (CAPEX, O&M, replacements)  --------------------
    annual_om_cost, annual_capex, total_capital_cost = calculate_costs(x, SYSTEMS, discount_rate)
    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)

    # discount factors by *year*
    df_year = [(1 + discount_rate) ** -y for y in range(1, project_lifetime + 1)]
    df_sum  = sum(df_year)  # used later for revenue PV

    discounted_replacements = sum(r * d for r, d in zip(replacement_capex_by_year, df_year))

    npv_fixed_costs = (
        total_capital_cost +  # M$
        annual_om_cost * df_sum +
        discounted_replacements
    )

    # ----------------------------------------------------------------------
    hours = 8760

    # --------------  build Pyomo model  ------------------------------------
    model = pyo.ConcreteModel()
    model.T      = pyo.RangeSet(0, hours)
    model.T_idx  = pyo.RangeSet(0, hours - 1)

    # decision vars ---------------------------------------------------------
    model.soc           = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.charge        = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.discharge     = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.elec_to_h2    = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.h2_prod       = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.h2_storage    = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.grid_export   = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.slack         = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)

    # initial states --------------------------------------------------------
    batt_cap   = SYSTEMS["battery"]["capacity"]
    hydr_cap   = SYSTEMS["hydrogen_storage"]["capacity"]

    model.soc[0].fix(num_battery * batt_cap / 2)  # start @ 50 %
    model.h2_storage[0].fix(0)

    # parameters ------------------------------------------------------------
    batt_eta   = SYSTEMS["battery"]["efficiency"]
    batt_crate = SYSTEMS["battery"]["charge_rate"]
    batt_drate = SYSTEMS["battery"]["discharge_rate"]

    ely_cap    = SYSTEMS["electrolyzer"]["capacity"]
    ely_eff    = SYSTEMS["electrolyzer"]["efficiency"]

    # constraints -----------------------------------------------------------
    def soc_update(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.soc[t] == m.soc[t - 1] + batt_eta * m.charge[t - 1] - m.discharge[t - 1] / batt_eta
    model.soc_update = pyo.Constraint(model.T, rule=soc_update)

    model.soc_bounds     = pyo.Constraint(model.T, rule=lambda m, t: (0, m.soc[t], num_battery * batt_cap))
    model.charge_limit   = pyo.Constraint(model.T_idx, rule=lambda m, t: m.charge[t]   <= num_battery * batt_crate)
    model.discharge_lim  = pyo.Constraint(model.T_idx, rule=lambda m, t: m.discharge[t] <= num_battery * batt_drate)
    model.discharge_soc  = pyo.Constraint(model.T_idx, rule=lambda m, t: m.discharge[t] <= m.soc[t])

    model.elec_to_h2_lim = pyo.Constraint(model.T_idx, rule=lambda m, t: m.elec_to_h2[t] <= num_electrolyzer * ely_cap)
    model.h2_prod_def    = pyo.Constraint(model.T_idx, rule=lambda m, t: m.h2_prod[t] == m.elec_to_h2[t] * ely_eff / electricity_to_hydrogen)

    def h2_stock_bal(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.h2_storage[t] == m.h2_storage[t - 1] + m.h2_prod[t - 1]
    model.h2_stock_bal = pyo.Constraint(model.T, rule=h2_stock_bal)

    model.h2_cap = pyo.Constraint(model.T, rule=lambda m, t: m.h2_storage[t] <= hydr_cap)

    model.grid_balance = pyo.Constraint(model.T_idx, rule=lambda m, t: m.grid_export[t] == hourly_production[t] + m.discharge[t] - m.charge[t] - m.elec_to_h2[t])
    #model.over_demand  = pyo.Constraint(model.T_idx, rule=lambda m, t: m.grid_export[t] <= hourly_demand[t] * (1 + rel))
    #model.under_demand = pyo.Constraint(model.T_idx, rule=lambda m, t: m.grid_export[t] >= hourly_demand[t] * (1 - rel))

    model.slack_pos = pyo.Constraint(model.T_idx, rule=lambda m, t: m.slack[t] >= m.grid_export[t] - hourly_demand[t])
    model.slack_neg = pyo.Constraint(model.T_idx, rule=lambda m, t: m.slack[t] >= -m.grid_export[t] + hourly_demand[t])

    def revenue_expr(m):
        elec_rev = sum(m.grid_export[t] / 1e3 * hourly_price[t]  - 5e-2*model.slack[t] for t in m.T_idx)
        h2_rev = sum(m.h2_prod[t] * hydrogen_price for t in m.T_idx)
        return elec_rev + h2_rev

    model.revenue = pyo.Expression(rule=revenue_expr)

    # --- Compute discounted NPV ---
    annual_om_cost, annual_capex, total_capital_cost = calculate_costs(x, SYSTEMS, discount_rate)
    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)
    discount_factors = [(1 + discount_rate) ** -(year + 1) for year in range(project_lifetime)]

    def npv_obj_rule(m):
        discounted_revenue = sum((m.revenue / 1e6) * df for df in discount_factors)
        discounted_om = sum(annual_om_cost * df for df in discount_factors)
        discounted_replacement = sum(replacement_capex_by_year[y] * discount_factors[y] for y in range(project_lifetime))
        return discounted_revenue - total_capital_cost - discounted_om - discounted_replacement

    model.obj = pyo.Objective(rule=npv_obj_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory('glpk')
    result = solver.solve(model, tee=False)

    if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
        battery_charge = np.array([pyo.value(model.charge[t]) for t in model.T_idx])
        battery_discharge = np.array([pyo.value(model.discharge[t]) for t in model.T_idx])
        battery_soc = np.array([pyo.value(model.soc[t]) for t in model.T])
        electricity_sold = np.array([pyo.value(model.grid_export[t]) for t in model.T_idx])
        hydrogen_production = np.array([pyo.value(model.h2_prod[t]) for t in model.T_idx])
        npv = pyo.value(model.obj)

        return {
            "npv": npv,
            "electricity_sold": electricity_sold,
            "hydrogen_production": hydrogen_production,
            "battery_soc": battery_soc,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge
        }
    else:
        return None


def optimize_dispatch_with_battery_pyomo_npv(
    x,
    total_gen,
    demand,
    hourly_price,
    SYSTEMS,
    discount_rate,
    project_lifetime,
    rel
):
    horizon = 8760
    num_battery = int(round(x[3]))

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, horizon)
    model.T_idx = pyo.RangeSet(0, horizon - 1)

    model.soc = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.grid_export = pyo.Var(model.T_idx, domain=pyo.Reals)
    model.slack = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)

    model.soc[0].fix(SYSTEMS["battery"]["capacity"] * num_battery / 2)

    def soc_update_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.soc[t] == m.soc[t - 1] + SYSTEMS["battery"]["efficiency"] * m.charge[t - 1] - m.discharge[t - 1] / SYSTEMS["battery"]["efficiency"]
    model.soc_update = pyo.Constraint(model.T, rule=soc_update_rule)

    def soc_bounds_rule(m, t):
        return (0, m.soc[t], num_battery * SYSTEMS["battery"]["capacity"])
    model.soc_bounds = pyo.Constraint(model.T, rule=soc_bounds_rule)

    def charge_rate_rule(m, t):
        return m.charge[t] <= num_battery * SYSTEMS["battery"]["charge_rate"]
    model.charge_rate = pyo.Constraint(model.T_idx, rule=charge_rate_rule)

    def discharge_rate_rule(m, t):
        return m.discharge[t] <= num_battery * SYSTEMS["battery"]["discharge_rate"]
    model.discharge_rate = pyo.Constraint(model.T_idx, rule=discharge_rate_rule)

    def discharge_soc_rule(m, t):
        return m.discharge[t] <= m.soc[t]
    model.discharge_soc = pyo.Constraint(model.T_idx, rule=discharge_soc_rule)

    def grid_export_rule(m, t):
        return m.grid_export[t] == total_gen[t] + m.discharge[t] - m.charge[t]
    model.grid_export_rule = pyo.Constraint(model.T_idx, rule=grid_export_rule)

    model.slack_pos = pyo.Constraint(model.T_idx, rule=lambda m, t: m.slack[t] >= m.grid_export[t] - demand[t])
    model.slack_neg = pyo.Constraint(model.T_idx, rule=lambda m, t: m.slack[t] >= -m.grid_export[t] + demand[t])


    # def over_demand_rule(m, t):
    #     return m.grid_export[t] <= demand[t] * (1 + rel)
    # model.over_demand_rule = pyo.Constraint(model.T_idx, rule=over_demand_rule)

    # def under_demand_rule(m, t):
    #     return m.grid_export[t] >= demand[t] * (1 - rel)
    # model.under_demand_rule = pyo.Constraint(model.T_idx, rule=under_demand_rule)

    # --- Capture total revenue ---  
    def revenue_expr(m):
        return sum(m.grid_export[t] / 1e3 * hourly_price[t] - 5e-2 * model.slack[t] for t in m.T_idx)

    model.revenue = pyo.Expression(rule=revenue_expr)

    # --- Compute discounted NPV in objective ---
    annual_om_cost, annual_capex, total_capital_cost = calculate_costs(x, SYSTEMS, discount_rate)
    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)

    discount_factors = [(1 + discount_rate) ** -(year + 1) for year in range(project_lifetime)]

    def npv_obj_rule(m):
        discounted_revenue = sum((m.revenue / 1e6) * df for df in discount_factors)
        discounted_om = sum(annual_om_cost * df for df in discount_factors)
        discounted_replacement = sum(replacement_capex_by_year[y] * discount_factors[y] for y in range(project_lifetime))
        return discounted_revenue - total_capital_cost - discounted_om - discounted_replacement

    model.obj = pyo.Objective(rule=npv_obj_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory('glpk')
    result = solver.solve(model, tee=False)

    if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
        battery_charge = np.array([pyo.value(model.charge[t]) for t in model.T_idx])
        battery_discharge = np.array([pyo.value(model.discharge[t]) for t in model.T_idx])
        battery_soc = np.array([pyo.value(model.soc[t]) for t in model.T])
        electricity_sold = np.array([pyo.value(model.grid_export[t]) for t in model.T_idx])
        npv = pyo.value(model.obj)

        return {
            "npv": npv,
            "electricity_sold": electricity_sold,
            "battery_soc": battery_soc,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge
        }
    else:
        return None