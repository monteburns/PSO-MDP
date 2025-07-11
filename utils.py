import numpy as np
from system_config import SYSTEMS, discount_rate, project_lifetime, hydrogen_price, cap_obj

# --- Utility Functions ---
def crf(rate, n):
    return (rate * (1 + rate)**n) / ((1 + rate)**n - 1)

# ---- LCOH Calculation ----
def lcoh_system_level(total_capex, annual_om_cost, replacement_capex_by_year,
                      hydrogen_produced, electricity_revenue_by_year, discount_factors):
    """
    Calculates LCOH at system level, subtracting electricity sales from total system cost.
    """

    # Total discounted hydrogen production (in kg)
    total_discounted_h2 = sum(hydrogen_produced.sum() * df for df in discount_factors)

    # Total discounted system cost
    total_discounted_cost = total_capex
    for year in range(project_lifetime):
        total_discounted_cost += annual_om_cost * discount_factors[year]
        total_discounted_cost += replacement_capex_by_year[year] * discount_factors[year]

    # Discounted electricity revenue (USD)
    discounted_electricity_revenue = sum(electricity_revenue_by_year * df for df in discount_factors)

    net_cost_allocated_to_h2 = total_discounted_cost - discounted_electricity_revenue
    #print(discounted_electricity_revenue, total_discounted_cost, net_cost_allocated_to_h2, total_discounted_h2)
    return net_cost_allocated_to_h2*1e6 / total_discounted_h2 if total_discounted_h2 > 0 else np.inf


def lcoh_cal(num_electrolyzer, hydrogen_produced, discount_factors):
    # Discounted hydrogen production (in kg)
    total_discounted_h2 = sum(hydrogen_produced.sum() * df for df in discount_factors)

    # Initial CAPEX
    hydrogen_capex = (
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["capital"]
    )

    # O&M
    hydrogen_om = (
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["O&M"]
    )

    # Total discounted cost
    total_discounted_h2_cost = hydrogen_capex
    for year in range(1, project_lifetime + 1):
        total_discounted_h2_cost += hydrogen_om * discount_factors[year - 1]

        for system_name in ["electrolyzer", "hydrogen_storage"]:
            lifetime = SYSTEMS[system_name]["lifetime"]
            if year % lifetime == 0:
                units = num_electrolyzer
                unit_cap = SYSTEMS[system_name]["capacity"]
                capital_cost = SYSTEMS[system_name]["capital"]
                repl_cost = units * unit_cap * capital_cost
                total_discounted_h2_cost += repl_cost * discount_factors[year - 1] 
    print(total_discounted_h2_cost)
    print(total_discounted_h2)
    return total_discounted_h2_cost / total_discounted_h2 if total_discounted_h2 > 0 else np.inf


def replacement_cost(best_pos, systems, project_lifetime):
    
    if len(best_pos) > 4:
        num_SMR, num_wind, num_solar, num_battery, num_electrolyzer = map(round, best_pos)
    else:
        num_SMR, num_wind, num_solar, num_battery = map(round, best_pos)
        num_electrolyzer = 0
    
    replacement_capex_by_year = []

    for year in range(1, project_lifetime + 1):
        replacement_cost_year = 0
        for system_name, system_info in systems.items():
            lifetime = system_info.get("lifetime")
            if lifetime is None:
                continue
            if year % lifetime == 0:
                if system_name == "SMR":
                    units = num_SMR
                elif system_name == "wind":
                    units = num_wind
                elif system_name == "solar":
                    units = num_solar
                elif system_name == "battery":
                    units = num_battery
                elif system_name == "electrolyzer":
                    units = num_electrolyzer
                else:
                    continue
                unit_cap = system_info["capacity"]
                replacement_cost_year += units * unit_cap * system_info["capital"] / 1e6
        
        replacement_capex_by_year.append(replacement_cost_year)

    return replacement_capex_by_year

def npv_cal(total_capital_cost, annual_revenue, annual_om_cost,
            electricity_sold, discount_rate, project_lifetime, replacement_capex_by_year):
    
    #total_annual_generation = np.sum(electricity_sold / 1e3)  # MWh
    discount_factors = [(1 + discount_rate)**-i for i in range(1, project_lifetime + 1)]
    annual_net_revenue = annual_revenue - annual_om_cost

    npv = -total_capital_cost
    for year in range(project_lifetime):
        npv += annual_net_revenue * discount_factors[year]
        npv -= replacement_capex_by_year[year] * discount_factors[year]

    return npv

def compute_lcoe(project_lifetime, discount_rate, annual_om_cost,
                 total_capital_cost, replacement_capex_by_year, electricity_sold):
    
    # Convert annual electricity sold to MWh (if in kWh)
    total_annual_generation = np.sum(electricity_sold) / 1e3  # MWh

    discount_factors = [(1 + discount_rate) ** -i for i in range(1, project_lifetime + 1)]

    total_discounted_cost = total_capital_cost

    for year in range(project_lifetime):
        total_discounted_cost += annual_om_cost * discount_factors[year]
        total_discounted_cost += replacement_capex_by_year[year]  * discount_factors[year]

    total_discounted_generation = total_annual_generation * sum(discount_factors)

    lcoe = 1e6 * total_discounted_cost / total_discounted_generation # $/MWh

    return lcoe




