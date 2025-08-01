import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
import multiprocessing as mp
import multiprocessing.pool
import argparse
import os, sys
import pyomo.environ as pyo
from itertools import product
from system_config import SYSTEMS, discount_rate, project_lifetime, cap_obj, electricity_to_hydrogen
from utils import *
from postprocess import *
import optDis

wind_generation = None
solar_generation = None
electricity_demand = None
electricity_price = None
hydrogen_price = None

hours_in_year = 8760

# Safe way to get available CPUs
def get_available_cpus(max_limit=None):
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count()

    if max_limit is not None:
        return min(available, max_limit)
    return available

# Custom non-daemonic process for nested multiprocessing
class NoDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

#class NonDaemonPool(mp.pool.Pool):
#    Process = NoDaemonProcess

class NoDaemonContext(type(mp.get_context("fork"))):
    Process = NoDaemonProcess

class NonDaemonPool(mp.pool.Pool):
    """Custom Pool that allows nested multiprocessing with 'fork' context."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, context=NoDaemonContext(), **kwargs)

def optimize_dispatch_with_battery_pyomo(
    demand, 
    total_gen, 
    battery_soc_init, 
    battery_capacity,
    num_battery, 
    battery_charge_rate, 
    battery_discharge_rate, 
    battery_efficiency, 
    daily_price, 
    rel
):
    horizon = 24

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, horizon)
    model.T_idx = pyo.RangeSet(0, horizon-1)

    model.soc = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.grid_export = pyo.Var(model.T_idx, domain=pyo.Reals)

    model.soc[0].fix(battery_soc_init)

    def soc_update_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.soc[t] == m.soc[t-1] + battery_efficiency * m.charge[t-1] - m.discharge[t-1] / battery_efficiency
    model.soc_update = pyo.Constraint(model.T, rule=soc_update_rule)

    def soc_bounds_rule(m, t):
        return (0, m.soc[t], num_battery * battery_capacity)
    model.soc_bounds = pyo.Constraint(model.T, rule=soc_bounds_rule)

    def charge_rate_rule(m, t):
        return m.charge[t] <= num_battery * battery_charge_rate
    model.charge_rate = pyo.Constraint(model.T_idx, rule=charge_rate_rule)

    def discharge_rate_rule(m, t):
        return m.discharge[t] <= num_battery * battery_discharge_rate
    model.discharge_rate = pyo.Constraint(model.T_idx, rule=discharge_rate_rule)

    def discharge_soc_rule(m, t):
        return m.discharge[t] <= m.soc[t]
    model.discharge_soc = pyo.Constraint(model.T_idx, rule=discharge_soc_rule)

    def grid_export_rule(m, t):
        return m.grid_export[t] == total_gen[t] + m.discharge[t] - m.charge[t]
    model.grid_export_rule = pyo.Constraint(model.T_idx, rule=grid_export_rule)

    def over_demand_rule(m, t):
        return m.grid_export[t] <= demand[t] * (1 + rel)
    model.over_demand_rule = pyo.Constraint(model.T_idx, rule=over_demand_rule)

    def under_demand_rule(m, t):
        return m.grid_export[t] >= demand[t] * (1 - rel)
    model.under_demand_rule = pyo.Constraint(model.T_idx, rule=under_demand_rule)

    def obj_rule(m):
        return sum(m.grid_export[t] / 1e3 * daily_price[t] for t in m.T_idx)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory('glpk')
    result = solver.solve(model, tee=False)
    if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
        battery_charge = [pyo.value(model.charge[t]) for t in model.T_idx]
        battery_discharge = [pyo.value(model.discharge[t]) for t in model.T_idx]
        battery_soc = [pyo.value(model.soc[t]) for t in model.T]
        grid_export = [pyo.value(model.grid_export[t]) for t in model.T_idx]
        return {
            "battery_charge": np.array(battery_charge),
            "battery_discharge": np.array(battery_discharge),
            "battery_soc": np.array(battery_soc),
            "electricity_sold_daily": np.array(grid_export)
        }
    else:
        return None

def optimize_dispatch_with_hydrogen_pyomo(
    demand, 
    total_gen, 
    battery_soc_init, 
    hydrogen_storage_init,
    battery_capacity,
    num_battery, 
    battery_charge_rate, 
    battery_discharge_rate, 
    battery_efficiency, 
    hydrogen_capacity,
    num_electrolyzer,
    electricity_to_hydrogen,
    electrolyzer_capacity,
    electrolyzer_efficiency, 
    daily_price, 
    hydrogen_price,
    rel
):
    horizon = 24


    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, horizon)
    model.T_idx = pyo.RangeSet(0, horizon-1)

    model.soc = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.h2_prod = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.elec_to_h2 = pyo.Var(model.T_idx, domain=pyo.NonNegativeReals)
    model.h2_storage = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.grid_export = pyo.Var(model.T_idx, domain=pyo.Reals)

    model.soc[0].fix(battery_soc_init)
    model.h2_storage[0].fix(hydrogen_storage_init)

    def soc_update_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.soc[t] == m.soc[t-1] + battery_efficiency * m.charge[t-1] - m.discharge[t-1] / battery_efficiency
    model.soc_update = pyo.Constraint(model.T, rule=soc_update_rule)

    def soc_bounds_rule(m, t):
        return (0, m.soc[t], num_battery * battery_capacity)
    model.soc_bounds = pyo.Constraint(model.T, rule=soc_bounds_rule)

    def charge_rate_rule(m, t):
        return m.charge[t] <= num_battery * battery_charge_rate
    model.charge_rate = pyo.Constraint(model.T_idx, rule=charge_rate_rule)

    def discharge_rate_rule(m, t):
        return m.discharge[t] <= num_battery * battery_discharge_rate
    model.discharge_rate = pyo.Constraint(model.T_idx, rule=discharge_rate_rule)

    def discharge_soc_rule(m, t):
        return m.discharge[t] <= m.soc[t]
    model.discharge_soc = pyo.Constraint(model.T_idx, rule=discharge_soc_rule)

    def elec_to_h2_rule(m, t):
        return m.elec_to_h2[t] <= num_electrolyzer * electrolyzer_capacity
    model.elec_to_h2_limit = pyo.Constraint(model.T_idx, rule=elec_to_h2_rule)

    def h2_prod_rule(m, t):
        return m.h2_prod[t] == m.elec_to_h2[t] * electrolyzer_efficiency / electricity_to_hydrogen
    model.h2_prod_con = pyo.Constraint(model.T_idx, rule=h2_prod_rule)

    def h2_storage_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.h2_storage[t] == m.h2_storage[t-1] + m.h2_prod[t-1]
    model.h2_storage_update = pyo.Constraint(model.T, rule=h2_storage_rule)

    def h2_storage_capacity_rule(m, t):
        return m.h2_storage[t] <= hydrogen_capacity
    model.h2_storage_cap = pyo.Constraint(model.T, rule=h2_storage_capacity_rule)

    def grid_export_rule(m, t):
        return m.grid_export[t] == total_gen[t] + m.discharge[t] - m.charge[t] - m.elec_to_h2[t]
    model.grid_export_rule = pyo.Constraint(model.T_idx, rule=grid_export_rule)

    def grid_export_nonneg_rule(m, t):
        return m.grid_export[t] >= 0
    model.grid_export_nonneg = pyo.Constraint(model.T_idx, rule=grid_export_nonneg_rule)

    def over_demand_rule(m, t):
        return m.grid_export[t] <= demand[t] * (1 + rel)
    model.over_demand_rule = pyo.Constraint(model.T_idx, rule=over_demand_rule)

    def under_demand_rule(m, t):
        return m.grid_export[t] >= demand[t] * (1 - rel)
    model.under_demand_rule = pyo.Constraint(model.T_idx, rule=under_demand_rule)

    def obj_rule(m):
        return sum(m.grid_export[t] / 1e3 * daily_price[t] for t in m.T_idx) + \
               sum(m.h2_prod[t] * hydrogen_price for t in m.T_idx)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    solver = pyo.SolverFactory('glpk')
    result = solver.solve(model, tee=False)
    if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
        battery_charge = [pyo.value(model.charge[t]) for t in model.T_idx]
        battery_discharge = [pyo.value(model.discharge[t]) for t in model.T_idx]
        battery_soc = [pyo.value(model.soc[t]) for t in model.T]
        h2_prod = [pyo.value(model.h2_prod[t]) for t in model.T_idx]
        grid_export = [pyo.value(model.grid_export[t]) for t in model.T_idx]
        h2_storage = [pyo.value(model.h2_storage[t]) for t in model.T]
        return {
            "battery_charge": np.array(battery_charge),
            "battery_discharge": np.array(battery_discharge),
            "battery_soc": np.array(battery_soc),
            "hydrogen_production": np.array(h2_prod),
            "electricity_sold_daily": np.array(grid_export),
            "hydrogen_storage": np.array(h2_storage)
        }
    else:
        return None

def battery_8760_dispatch(
        x,
        hourly_production, 
        hourly_demand, 
        electricity_price,
        num_battery,
        battery_capacity,
        battery_c_rate,
        battery_dc_rate,
        battery_eff
        ):
    hours_in_year = 8760
    electricity_sold = np.zeros(8760)
    battery_soc_history = np.zeros(8760)
    battery_charge = np.zeros(8760)
    battery_discharge = np.zeros(8760)

    soc = num_battery * battery_capacity / 2

    for day in range(365):
        start = day * 24
        end = start + 24

        result = optimize_dispatch_with_battery_pyomo(
            x,
            hourly_demand[start:end],
            hourly_production[start:end],
            soc,
            battery_capacity,
            num_battery,
            battery_c_rate,
            battery_dc_rate,
            battery_eff,
            electricity_price[start:end],
            args.rel
        )

        if result is not None:
            electricity_sold[start:end] = result["electricity_sold_daily"]
            battery_soc_history[start:end] = result["battery_soc"][1:]
            battery_charge[start:end] = result["battery_charge"]
            battery_discharge[start:end] = result["battery_discharge"]
            soc = result["battery_soc"][-1]
        else:
            return None

    return electricity_sold, battery_soc_history, battery_charge, battery_discharge

def hydrogen_8760_dispatch(
        hourly_production, 
        hourly_demand, 
        electricity_price,
        num_battery,
        battery_capacity,
        battery_c_rate,
        battery_dc_rate,
        battery_eff,
        hydrogen_capacity,
        electrolyzer_capacity,
        electrolyzer_eff,
        hydrogen_price,
        num_electrolyzer, 
        electricity_to_hydrogen
        ):

    electricity_sold = np.zeros(8760)
    hydrogen_production = np.zeros(8760)
    battery_soc_history = np.zeros(8760)
    battery_charge = np.zeros(8760)
    battery_discharge = np.zeros(8760)
    h2_history = np.zeros(8760)

    battery_soc_history[0] = num_battery * battery_capacity / 2
    h2_history[0] = 0

    soc = num_battery * battery_capacity / 2
    h2 = 0

    for day in range(365):
        start = day * 24
        end = start + 24

        result = optimize_dispatch_with_hydrogen_pyomo(
            hourly_demand[start:end],
            hourly_production[start:end],
            soc,
            h2,
            battery_capacity,
            num_battery,
            battery_c_rate,
            battery_dc_rate,
            battery_eff,
            hydrogen_capacity,
            num_electrolyzer,
            electricity_to_hydrogen,
            electrolyzer_capacity,
            electrolyzer_eff,
            electricity_price[start:end],
            hydrogen_price,
            args.rel
        )
                
        if result is not None:
            electricity_sold[start:end] = result["electricity_sold_daily"]
            hydrogen_production[start:end] = result["hydrogen_production"]
            battery_soc_history[start:end] = result["battery_soc"][1:]
            battery_charge[start:end] = result["battery_charge"]
            battery_discharge[start:end] = result["battery_discharge"]
            soc = result["battery_soc"][-1]
            h2 = result["hydrogen_storage"][-1]
        else:
            return None

    return electricity_sold, hydrogen_production, battery_soc_history, battery_charge, battery_discharge

# (Rule based dispatch and objective functions unchanged, except for removing multiprocessing)

def rule_based_dispatch_battery(hourly_production, hourly_demand, num_battery, battery_capacity,
                                     battery_efficiency, battery_c_rate, battery_dc_rate, hours=8760):
    electricity_sold = np.zeros(hours)
    battery_soc = np.zeros(hours + 1)
    battery_charge = np.zeros(hours)
    battery_discharge = np.zeros(hours)

    battery_soc[0] = num_battery * battery_capacity / 2  # start at 50%

    for t in range(hours):
        net = hourly_production[t] - hourly_demand[t]

        if net > 0:
            charge_possible = min(
                num_battery * battery_c_rate,
                (num_battery * battery_capacity - battery_soc[t])
            )
            charge = min(charge_possible, net * battery_efficiency)
            battery_charge[t] = charge
            battery_soc[t + 1] = battery_soc[t] + charge * battery_efficiency
            electricity_sold[t] = hourly_production[t] - charge

        else:
            deficit = -net
            discharge_possible = min(
                num_battery * battery_dc_rate,
                battery_soc[t] * battery_efficiency
            )
            discharge = min(discharge_possible, deficit)
            battery_discharge[t] = discharge
            battery_soc[t + 1] = battery_soc[t] - discharge / battery_efficiency
            electricity_sold[t] = hourly_production[t] + discharge

        battery_soc[t + 1] = np.clip(battery_soc[t + 1], 0, num_battery * battery_capacity)

    return electricity_sold, battery_soc[:-1], battery_charge, battery_discharge

def rule_based_dispatch_hydrogen(
    hourly_production,
    hourly_demand,
    num_battery,
    battery_capacity,
    battery_efficiency,
    battery_c_rate,
    battery_dc_rate,
    hydrogen_capacity,
    electrolyzer_capacity,
    electrolyzer_efficiency,
    num_electrolyzer,
    electricity_to_hydrogen,
    hours
):
    electricity_sold = np.zeros(hours)
    battery_soc = np.zeros(hours + 1)
    hydrogen_storage = np.zeros(hours + 1)
    battery_charge = np.zeros(hours)
    battery_discharge = np.zeros(hours)
    hydrogen_production = np.zeros(hours)

    battery_soc[0] = num_battery * battery_capacity / 2
    hydrogen_storage[0] = 0

    for t in range(hours):
        net = hourly_production[t] - hourly_demand[t]

        if net > 0:
            charge_possible = min(
                num_battery * battery_c_rate,
                (num_battery * battery_capacity - battery_soc[t])
            )
            charge = min(charge_possible, net * battery_efficiency)
            battery_charge[t] = charge
            battery_soc[t + 1] = battery_soc[t] + charge * battery_efficiency
            net = net - charge

            elec_for_h2 = min(num_electrolyzer * electrolyzer_capacity, net)
            h2_produced = elec_for_h2 * electrolyzer_efficiency / electricity_to_hydrogen
            h2_space = hydrogen_capacity - hydrogen_storage[t]
            h2_produced = min(h2_produced, h2_space)
            hydrogen_production[t] = h2_produced
            hydrogen_storage[t + 1] = hydrogen_storage[t] + h2_produced
            net -= h2_produced * electricity_to_hydrogen / electrolyzer_efficiency

            electricity_sold[t] = hourly_production[t] - charge - h2_produced * electricity_to_hydrogen / electrolyzer_efficiency

        else:
            deficit = -net

            discharge_possible = min(
                num_battery * battery_dc_rate,
                battery_soc[t] * battery_efficiency
            )
            discharge = min(discharge_possible, deficit)
            battery_discharge[t] = discharge
            battery_soc[t + 1] = battery_soc[t] - discharge / battery_efficiency
            deficit -= discharge

            electricity_sold[t] = hourly_production[t] + discharge

        battery_soc[t + 1] = np.clip(battery_soc[t + 1], 0, num_battery * battery_capacity)
        hydrogen_storage[t + 1] = np.clip(hydrogen_storage[t + 1], 0, num_electrolyzer * hydrogen_capacity)

    return electricity_sold,battery_soc[:-1], hydrogen_storage[:-1],battery_charge,battery_discharge,hydrogen_production

# Objective functions (unchanged except for naming the dispatch functions)
def objective_function_hydrogen_opt(x, return_details=False):
    num_SMR = int(round(x[0]))
    num_wind = int(round(x[1]))
    num_solar = int(round(x[2]))
    num_battery = int(round(x[3]))
    num_electrolyzer = int(round(x[4]))

    hourly_production = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["availability"] +
        num_wind * SYSTEMS["wind"]["capacity"] * wind_generation +
        num_solar * SYSTEMS["solar"]["capacity"] * solar_generation
    )

    hourly_demand = electricity_demand * cap_obj

    # result = hydrogen_8760_dispatch(
    #     hourly_production, 
    #     hourly_demand, 
    #     electricity_price,
    #     num_battery,
    #     SYSTEMS["battery"]["capacity"],
    #     SYSTEMS["battery"]["charge_rate"],
    #     SYSTEMS["battery"]["discharge_rate"],
    #     SYSTEMS["battery"]["efficiency"],
    #     SYSTEMS["hydrogen_storage"]["capacity"],
    #     num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"],
    #     SYSTEMS["electrolyzer"]["efficiency"],
    #     hydrogen_price,
    #     num_electrolyzer, 
    #     electricity_to_hydrogen
    # )

    result = optDis.optimize_dispatch_with_hydrogen_pyomo_npv(
        x,                      # full PSO position vector
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

    if result is not None:
        #electricity_sold, hydrogen_produced, battery_soc_history, battery_charge, battery_discharge = result
        npv               = result['npv']             # already discounted-lifetime value
        electricity_sold  = result['electricity_sold']
        hydrogen_produced = result['hydrogen_production']
        battery_soc_hist  = result['battery_soc']
        battery_charge    = result['battery_charge']
        battery_discharge = result['battery_discharge']
    else:
        return 1e9

    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    if penalty > args.rel and return_details is not True:
        return 1e9

    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6
    hydrogen_revenue = np.sum(hydrogen_produced * hydrogen_price) / 1e6

    annual_om_cost, annual_capex, total_capital_cost = calculate_costs(x, SYSTEMS, discount_rate)

    # replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)
    # npv = npv_cal(
    #     total_capital_cost,
    #     annual_revenue + hydrogen_revenue,
    #     annual_om_cost,
    #     electricity_sold,
    #     discount_rate,
    #     project_lifetime,
    #     replacement_capex_by_year
    # )
    profit = annual_revenue + hydrogen_revenue - (annual_om_cost + annual_capex)

    return -npv if not return_details else (
        profit,
        battery_soc_hist,
        hourly_demand,
        electricity_sold,
        annual_revenue,
        hydrogen_produced,
        hydrogen_revenue,
        annual_om_cost,
        penalty,
        battery_charge,
        battery_discharge
    )

def objective_function_hydrogen_rb(x, return_details=False):
    num_SMR = int(round(x[0]))
    num_wind = int(round(x[1]))
    num_solar = int(round(x[2]))
    num_battery = int(round(x[3]))
    num_electrolyzer = int(round(x[4]))

    hourly_production = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["availability"] +
        num_wind * SYSTEMS["wind"]["capacity"] * wind_generation +
        num_solar * SYSTEMS["solar"]["capacity"] * solar_generation
    )

    hourly_demand = electricity_demand * cap_obj

    electricity_sold, battery_soc_history, hydrogen_soc, battery_charge, battery_discharge, hydrogen_produced = rule_based_dispatch_hydrogen(
        hourly_production,
        hourly_demand,
        num_battery,
        SYSTEMS["battery"]["capacity"],
        SYSTEMS["battery"]["efficiency"],
        SYSTEMS["battery"]["charge_rate"],
        SYSTEMS["battery"]["discharge_rate"],
        SYSTEMS["hydrogen_storage"]["capacity"],
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"],
        SYSTEMS["electrolyzer"]["efficiency"],
        num_electrolyzer, 
        electricity_to_hydrogen,
        hours_in_year
    )

    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    if penalty > args.rel and return_details is not True:
        return 1e9

    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6
    hydrogen_revenue = np.sum(hydrogen_produced * hydrogen_price) / 1e6
  
    annual_om_cost, annual_capex, total_capital_cost = calculate_costs(x, SYSTEMS, discount_rate)

    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)
    npv = npv_cal(
        total_capital_cost,
        annual_revenue + hydrogen_revenue,
        annual_om_cost,
        electricity_sold,
        discount_rate,
        project_lifetime,
        replacement_capex_by_year
    )
    profit = annual_revenue + hydrogen_revenue - (annual_om_cost + annual_capex)

    return -npv if not return_details else (
        profit,
        battery_soc_history,
        hourly_demand,
        electricity_sold,
        annual_revenue,
        hydrogen_produced,
        hydrogen_revenue,
        annual_om_cost,
        penalty,
        battery_charge,
        battery_discharge
    )

def objective_function_battery_rb(x, return_details=False):
    num_SMR = int(round(x[0]))
    num_wind = int(round(x[1]))
    num_solar = int(round(x[2]))
    num_battery = int(round(x[3]))

    hourly_production = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["availability"] +
        num_wind * SYSTEMS["wind"]["capacity"] * wind_generation +
        num_solar * SYSTEMS["solar"]["capacity"] * solar_generation
    )

    hourly_demand = electricity_demand * cap_obj

    electricity_sold, battery_soc_history, battery_charge, battery_discharge = rule_based_dispatch_battery(
        hourly_production,
        hourly_demand,
        num_battery,
        SYSTEMS["battery"]["capacity"],
        SYSTEMS["battery"]["efficiency"],
        SYSTEMS["battery"]["charge_rate"],
        SYSTEMS["battery"]["discharge_rate"],
        hours_in_year
    )

    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    if penalty > args.rel and not return_details:
        return 1e9

    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6
    annual_om_cost, annual_capex, total_capital_cost = calculate_costs(x, SYSTEMS, discount_rate)
    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)

    npv = npv_cal(
        total_capital_cost,
        annual_revenue,
        annual_om_cost,
        electricity_sold,
        discount_rate,
        project_lifetime,
        replacement_capex_by_year
    )

    profit = annual_revenue - (annual_om_cost + annual_capex)

    return -npv if not return_details else (
        profit,
        battery_soc_history,
        hourly_demand,
        electricity_sold,
        annual_revenue,
        annual_om_cost,
        penalty,
        battery_charge,
        battery_discharge
    )


def objective_function_battery_opt(x, return_details=False):
    num_SMR = int(round(x[0]))
    num_wind = int(round(x[1]))
    num_solar = int(round(x[2]))
    num_battery = int(round(x[3]))

    hourly_production = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["availability"] +
        num_wind * SYSTEMS["wind"]["capacity"] * wind_generation +
        num_solar * SYSTEMS["solar"]["capacity"] * solar_generation
    )

    hourly_demand = electricity_demand * cap_obj

    # result = battery_8760_dispatch(
    #     hourly_production,
    #     hourly_demand,
    #     electricity_price,
    #     num_battery,
    #     SYSTEMS["battery"]["capacity"],
    #     SYSTEMS["battery"]["charge_rate"],
    #     SYSTEMS["battery"]["discharge_rate"],
    #     SYSTEMS["battery"]["efficiency"]
    # )
    result = optDis.optimize_dispatch_with_battery_pyomo_npv(
        x,
        hourly_production,
        hourly_demand,
        electricity_price,
        SYSTEMS,
        discount_rate,
        project_lifetime,
        args.rel
    )

    if result is not None:
        #electricity_sold, battery_soc_history, battery_charge, battery_discharge = result
        npv               = result['npv']           
        electricity_sold  = result['electricity_sold']
        battery_soc_history  = result['battery_soc']
        battery_charge    = result['battery_charge']
        battery_discharge = result['battery_discharge']
    else:
        return 1e9

    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / (hourly_demand))
    if penalty > args.rel and not return_details:
        return 1e9

    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6

    annual_om_cost, annual_capex, total_capital_cost = calculate_costs(x, SYSTEMS, discount_rate)

    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)
    npv = npv_cal(
        total_capital_cost,
        annual_revenue,
        annual_om_cost,
        electricity_sold,
        discount_rate,
        project_lifetime,
        replacement_capex_by_year
    )

    profit = annual_revenue - (annual_om_cost + annual_capex)

    return -npv if not return_details else (
        profit,
        battery_soc_history,
        hourly_demand,
        electricity_sold,
        annual_revenue,
        annual_om_cost,
        penalty,
        battery_charge, 
        battery_discharge
    )

def print_results(best_pos, optdis, rbdis):

    if len(best_pos) > 4:
        h2 = True
        if optdis:
            profit, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, hydrogen_produced, hydrogen_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function_hydrogen_opt(best_pos, return_details=True)
        elif rbdis:
            profit, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, hydrogen_produced, hydrogen_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function_hydrogen_rb(best_pos, return_details=True)
        else:
            print("Please provide dispatch type!")
            exit()

        num_SMR, num_wind, num_solar, num_battery, num_electrolyzer = map(round, best_pos)

    else:
        h2=False
        num_electrolyzer, hydrogen_revenue = 0, 0

        if optdis:
            profit, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function_battery_opt(best_pos, return_details=True)
        elif rbdis:
            profit, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function_battery_rb(best_pos, return_details=True)
        else:
            print("Please provide dispatch type!")
            exit()

        num_SMR, num_wind, num_solar, num_battery = map(round, best_pos)
    
    print(f"\nOptimal Configuration:")
    print(f"SMRs: {num_SMR}")
    print(f"Wind Turbines: {num_wind}")
    print(f"Solar Panels: {num_solar}")
    print(f"Battery Units: {num_battery}")
    print(f"Hydrogen Units: {num_electrolyzer}")
    print(f"Demand Matching Efficiency: {1-penalty}")

    total_capital_cost = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"] +
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["capital"]
    ) / 1e6  # Convert to million USD

    replacement_capex_by_year = replacement_cost(best_pos, SYSTEMS, project_lifetime)
    npv = npv_cal(
        total_capital_cost,
        annual_revenue + hydrogen_revenue, 
        annual_om_cost,
        electricity_sold,
        discount_rate,
        project_lifetime,
        replacement_capex_by_year
    )

    lcoe = compute_lcoe(
        project_lifetime,
        discount_rate,
        annual_om_cost,
        total_capital_cost,
        replacement_capex_by_year,
        electricity_sold
    )

    discount_factors = [(1 + discount_rate) ** -i for i in range(1, project_lifetime + 1)]
    #lcoh = lcoh_cal(num_electrolyzer, hydrogen_produced, discount_factors)
    if h2:
        lcoh = lcoh_system_level(total_capital_cost, annual_om_cost, replacement_capex_by_year, hydrogen_produced, annual_revenue, discount_factors)
    else:
        lcoh = 0
    print(f"\nFinancial Summary (Million $):")
    print(f"PPA: ${np.mean(electricity_price):.2f}M") if args.ppa else print(f"Mean Electricity price: ${np.mean(electricity_price):.2f}M")
    print(f"Capital Cost: ${total_capital_cost:.2f}M")
    print(f"Annual O&M Cost: ${annual_om_cost:.2f}M")
    print(f"Annual Electricity Revenue: ${annual_revenue:.2f}M")
    print(f"Annual Hydrogen Revenue: ${hydrogen_revenue:.2f}M")
    print(f"Annual Net Revenue: ${(annual_revenue + hydrogen_revenue - annual_om_cost):.2f}M")
    # Ensure denominator is not zero
    net_annual_profit = (annual_revenue + hydrogen_revenue  - annual_om_cost)
    if net_annual_profit > 0:
        print(f"Simple Payback Period: {total_capital_cost/net_annual_profit:.1f} years")
    else:
        print("Simple Payback Period: Not applicable (no positive annual net revenue)")

    print(f"\nLevelized Costs:")
    print(f"LCOE: ${lcoe:.2f}/MWh")
    print(f"LCOH: ${lcoh:.2f}/kg")
    print(f"NPV: ${npv:.2f}M")

    smr_gen = np.full(hours_in_year, num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["availability"])
    wind_gen = num_wind * SYSTEMS["wind"]["capacity"] * wind_generation
    solar_gen = num_solar * SYSTEMS["solar"]["capacity"] * solar_generation
    hourly_production = smr_gen + wind_gen + solar_gen
    if h2:
        elect_to_h2 = hydrogen_produced *  electricity_to_hydrogen / SYSTEMS["electrolyzer"]["efficiency"] 
    else: 
        elect_to_h2 = np.zeros(hours_in_year)
    plot_sample_day_grid([5, 40, 200, 300], smr_gen, wind_gen, solar_gen, hourly_production, electricity_sold, battery_soc_history, hourly_demand, electricity_price, elect_to_h2)

def run_pso_with_params(c1, c2, w, hydrogen, battery):
    if hydrogen and not battery:
        bounds = ([4, 0, 0, 0, 0], [12, 200, 200, 200, 200])
        options = {"c1": c1, "c2": c2, "w": w}
        optimizer = ps.single.GlobalBestPSO(n_particles=250, dimensions=5, options=options, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(pso_objective_function_hydrogen_opt, iters=1, verbose=False)
    elif battery and not hydrogen:
        bounds = ([4, 0, 0, 0], [12, 200, 200, 550])
        options = {"c1": c1, "c2": c2, "w": w}
        optimizer = ps.single.GlobalBestPSO(n_particles=400, dimensions=4, options=options, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(pso_objective_function_battery_opt, iters=1, verbose=False)
    else:
        print("Please indicate the type of NHES! (battery/hydrogen)")
        exit()

    return {
        "c1": c1,
        "c2": c2,
        "w": w,
        "best_cost": best_cost,
        "best_pos": best_pos
    }

def grid_search_pso_hyperparameters(hydrogen, battery):
    # Define ranges to test
    c1_vals = [1.5, 1.8, 2.0]
    c2_vals = [1.5, 2.0, 2.5]
    w_vals = [0.3, 0.5, 0.7]

    grid = list(product(c1_vals, c2_vals, w_vals))

    print(f"🔍 Starting grid search over {len(grid)} PSO configurations...\n")

    results = []
    for c1, c2, w in grid:
        result = run_pso_with_params(c1, c2, w, hydrogen, battery)
        print(f"* Finished: c1={c1}, c2={c2}, w={w} -> Cost: {result['best_cost']:.3f}")
        results.append(result)

    results = sorted(results, key=lambda x: x["best_cost"])

    print("\n--- Grid Search Results ---")
    for r in results:
        print(f"c1={r['c1']}, c2={r['c2']}, w={r['w']} -> Cost: {r['best_cost']:.3f}")

    best = results[0]
    print(f"\nBest Params: c1={best['c1']}, c2={best['c2']}, w={best['w']}")
    print(f"Best Cost: {best['best_cost']}, Best Position: {best['best_pos']}")


def pso_objective_function_hydrogen_opt(x):
    with NonDaemonPool(get_available_cpus(250)) as pool:
        return pool.map(objective_function_hydrogen_opt, x)

def pso_objective_function_hydrogen_rb(x):
    with NonDaemonPool(get_available_cpus(250)) as pool:
        return pool.map(objective_function_hydrogen_rb, x)

def pso_objective_function_battery_opt(x):
    with NonDaemonPool(get_available_cpus(250)) as pool:
        return pool.map(objective_function_battery_opt, x)
    
def pso_objective_function_battery_rb(x):
    with NonDaemonPool(get_available_cpus(250)) as pool:
        return pool.map(objective_function_battery_rb, x)


def main(args):

    global electricity_price, electricity_demand, wind_generation, solar_generation, hydrogen_price

    hydrogen_price = args.h2_price

    df = pd.read_csv('demand_forecast_2025.csv', sep=';', parse_dates=['DateTime'])
    electricity_demand = (df['Forecast_Demand_MWh'].values[:hours_in_year]) # Take first 8760 hours
    electricity_demand = electricity_demand / np.max(electricity_demand) # Normalize, ensure this is correct

    # Load wind generation data
    df_wind = pd.read_csv('wind_power_output.csv', parse_dates=[0])
    wind_generation = (df_wind['Wind_Power_Output'].values[:hours_in_year] / 2000.0)  # Take first 8760 hours and normalize

    # Load solar generation data
    df_solar = pd.read_csv('solar_power_output_tmy.csv', parse_dates=[0])
    df_solar['Power_Output'] = df_solar['Power_Output'].apply(lambda x: 0 if x < 0 else x)
    solar_data = df_solar['Power_Output'].values[:hours_in_year]  # Take first 8760 hours
    solar_generation = solar_data*10 / SYSTEMS["solar"]["capacity"] # multiplied by 10 for 10 panel capacity

    # Load electricity price data
    if args.ppa is not None:
        electricity_price = np.full(hours_in_year, args.ppa)
        print(f" Using fixed electricity price: {args.ppa} USD/MWh")
    else:
        # Load from CSV
        print(f" No PPA is present, using day-ahead market prices")
        df = pd.read_csv('price_forecast_2025.csv', sep=';', parse_dates=['DateTime'])
        electricity_price = df['Forecast_Price_USD'].values[:hours_in_year]
        electricity_price = electricity_price * args.dar

    if args.post_process:
        if args.battery:
            # Note: You'll need to define num_SMR, num_wind, etc. for post-processing if you use a fixed best_pos
            best_pos_battery = [6, 13, 5, 126] # Example values
            print_results(best_pos_battery, args.optdis, args.rbdis)
        elif args.hydrogen:
            best_pos_h2_pp = [7, 30, 50, 1, 14] # Adjust example values for hydrogen
            print_results(best_pos_h2_pp, args.optdis, args.rbdis)
        else:
            print("Please provide options for post process!")
            exit()

    elif args.grid_search:
        grid_search_pso_hyperparameters(args.hydrogen, args.battery)
        exit()

    elif args.hydrogen: #optimization the system with hydrogen
        bounds = ([4, 0, 0, 0, 0], [12, 200, 200, 200, 200])
        options = {"c1": 1.5, "c2": 2.5, "w": 0.3}
        if args.optdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=250, dimensions=5, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_hydrogen_opt, iters=args.iter)
            #profit, battery_soc_history, hourly_demand, electricity_sold, hydrogen_produced, annual_revenue, hydrogen_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function_hydrogen(best_pos, return_details=True)
        elif args.rbdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=5, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_hydrogen_rb, iters=args.iter)
            #profit, battery_soc_history, hourly_demand, electricity_sold, hydrogen_produced, annual_revenue, hydrogen_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function_hydrogen_rb(best_pos, return_details=True)
        else:
            print("Please indicate the dispatch type!")
            exit()
        print_results(best_pos, args.optdis, args.rbdis)
        #cost_history = optimizer.cost_history
        # Create the convergence plot
        #plt.plot(cost_history)
        #plt.title('Convergence Plot')
        #plt.xlabel('Iteration')
        #plt.ylabel('Cost')
        #plt.savefig("Convergence.png", dpi=300)
        #plt.close()
    elif args.battery:
        bounds = ([4, 0, 0, 0], [12, 200, 200, 200])
        options = {"c1": 2.0, "c2": 1.5, "w": 0.7}
        if args.optdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=250, dimensions=4, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_battery_opt, iters=args.iter)
        elif args.rbdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=4, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_battery_rb, iters=args.iter)
        else:
            print("Please indicate the dispatch type!")
            exit()
        print_results(best_pos, args.optdis, args.rbdis)

    else:
        print("Please provide options!")



if __name__ == '__main__':
    workingDirectory  = os.path.realpath(sys.argv[0])

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--post_process", action="store_true", help="Create graphics")
    parser.add_argument("--grid_search", action="store_true", help="Run PSO hyperparameter grid search")
    parser.add_argument("--ppa", type=float, help="Single electricity price value in USD/MWh")
    parser.add_argument("--dar", type=float, default=1.0, help="Day a head price ratio")
    parser.add_argument("--iter", type=int, help="Number of iterations")
    parser.add_argument("--battery", action="store_true", help="Rule based dispatch")
    parser.add_argument("--hydrogen", action="store_true", help="Adds Hydrogen to the NHES")
    parser.add_argument("--optdis", action="store_true", help="Optimize dispatch")
    parser.add_argument("--rbdis", action="store_true", help="Rule based dispatch")
    parser.add_argument("--rel", type=float, default=0.5, help="Demand percentage not covered")
    parser.add_argument("--h2_price", type=float, default=6, help="Hydrogen price USD/kg")

    args = parser.parse_args()

    main(args) # your main function definition can re-use the objective/dispatch logic above as needed.