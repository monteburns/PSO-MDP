
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
import argparse
import os, sys
import numpy as np
import cvxpy as cp
import multiprocessing as mp
from utils import *
from postprocess import *
import multiprocessing.pool
from itertools import product
from tqdm import tqdm
from multiprocessing import Manager
from functools import partial
from system_config import SYSTEMS, discount_rate, project_lifetime, hydrogen_price, cap_obj, electricity_to_hydrogen

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy.problems.problem")

wind_generation = None
solar_generation = None
electricity_demand = None
electricity_price = None

# General parameters
hours_in_year = 8760
days_in_year=365


# Safe way to get available CPUs
def get_available_cpus(max_limit=None):
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count()

    if max_limit is not None:
        return min(available, max_limit)
    return available

# global scope (still needed for shared data in PSO's inner multiprocessing for objective function)
_shared_hourly_production = None
_shared_hourly_demand = None
_shared_price = None

def optimize_dispatch_with_battery(
    demand, 
    total_gen, 
    battery_soc_init, 
    battery_capacity,
    num_battery, 
    battery_charge_rate, 
    battery_discharge_rate, 
    battery_efficiency, 
    daily_price, 
    horizon=24
):
    soc = cp.Variable(horizon + 1)
    charge = cp.Variable(horizon)
    discharge = cp.Variable(horizon)
    grid_export = cp.Variable(horizon)

    constraints = [soc[0] == battery_soc_init]

    for t in range(horizon):
        constraints += [
            soc[t+1] == soc[t] + battery_efficiency * charge[t] - discharge[t] / battery_efficiency,
            soc[t+1] >= 0,
            soc[t+1] <= num_battery * battery_capacity,
            charge[t] >= 0,
            charge[t] <= num_battery * battery_charge_rate,
            discharge[t] >= 0,
            discharge[t] <= num_battery * battery_discharge_rate,
            discharge[t] <= soc[t]
        ]

        constraints += [
            grid_export[t] == total_gen[t] + discharge[t] - charge[t]
        ]
    
    revenue = cp.sum(cp.multiply(grid_export / 1e3, daily_price))

    prob = cp.Problem(cp.Maximize(revenue), constraints)

    try:
        prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            max_iter=1000,
            eps_abs=1e-3,
            eps_rel=1e-3
        )
    except Exception as e:
        print("Solver failed with error:", e)
        return None

    # Return results
    if prob.status not in ["infeasible", "unbounded"]:
        return {
            "battery_charge": charge.value,
            "battery_discharge": discharge.value,
            "battery_soc": soc.value,
            "electricity_sold_daily": grid_export.value
        }
    else:
        print(prob.status)
        return None

def setup_dispatch_problem_battery(
    num_battery,
    battery_capacity,
    battery_c_rate,
    battery_dc_rate,
    battery_eff,
    horizon=24):

    # Define parameters
    demand = cp.Parameter(horizon)
    gen = cp.Parameter(horizon)
    price = cp.Parameter(horizon)
    soc_init = cp.Parameter()

    # Define decision variables
    soc = cp.Variable(horizon + 1)
    charge = cp.Variable(horizon)
    discharge = cp.Variable(horizon)
    grid_export = cp.Variable(horizon)

    constraints = [
        soc[0] == soc_init,
    ]

    for t in range(horizon):
        constraints += [
            soc[t+1] == soc[t] + battery_eff * charge[t] - discharge[t] / battery_eff,
            soc[t+1] >= 0,
            soc[t+1] <= num_battery * battery_capacity,
            charge[t] >= 0,
            charge[t] <= num_battery * battery_c_rate,
            discharge[t] >= 0,
            discharge[t] <= num_battery * battery_dc_rate,
            discharge[t] <= soc[t],

            grid_export[t] == gen[t] + discharge[t] - charge[t],
            grid_export[t] >= 0
        ]

    revenue = cp.sum(cp.multiply(grid_export / 1e3, price))
    prob = cp.Problem(cp.Maximize(revenue), constraints)

    param_dict = {
        "demand": demand,
        "gen": gen,
        "price": price,
        "soc_init": soc_init
    }
    var_dict = {
        "soc": soc,
        "charge": charge,
        "discharge": discharge,
        "grid_export": grid_export
    }

    return prob, param_dict, var_dict




def setup_dispatch_problem_hydrogen(
    num_battery,
    battery_capacity,
    battery_c_rate,
    battery_dc_rate,
    battery_eff,
    hydrogen_capacity,
    num_electrolyzer,
    electrolyzer_capacity,
    electrolyzer_eff,
    electricity_to_hydrogen,
    hydrogen_price,
    horizon=24):

    # Define parameters
    demand = cp.Parameter(horizon)
    gen = cp.Parameter(horizon)
    price = cp.Parameter(horizon)
    soc_init = cp.Parameter()
    h2_init = cp.Parameter()

    # Define decision variables
    soc = cp.Variable(horizon + 1)
    charge = cp.Variable(horizon)
    discharge = cp.Variable(horizon)
    h2_prod = cp.Variable(horizon)
    elec_to_h2 = cp.Variable(horizon)
    h2_storage = cp.Variable(horizon + 1)
    grid_export = cp.Variable(horizon)

    
    constraints = [
        soc[0] == soc_init,
        h2_storage[0] == h2_init
    ]

    for t in range(horizon):
        constraints += [
            soc[t+1] == soc[t] + battery_eff * charge[t] - discharge[t] / battery_eff,
            soc[t+1] >= 0,
            soc[t+1] <= num_battery * battery_capacity,
            charge[t] >= 0,
            charge[t] <= num_battery * battery_c_rate,
            discharge[t] >= 0,
            discharge[t] <= num_battery * battery_dc_rate,
            discharge[t] <= soc[t],

            h2_prod[t] == elec_to_h2[t] * electrolyzer_eff / electricity_to_hydrogen,
            h2_prod[t] >= 0,
            h2_prod[t] <= hydrogen_capacity, # - h2_storage[t],
            elec_to_h2[t] >= 0,
            elec_to_h2[t] <= num_electrolyzer * electrolyzer_capacity,
            h2_storage[t+1] == h2_storage[t] + h2_prod[t],
            h2_storage[t+1] <= hydrogen_capacity,

            grid_export[t] == gen[t] + discharge[t] - charge[t] - h2_prod[t] * electricity_to_hydrogen / electrolyzer_eff
            #grid_export[t] >= 0
        ]

    revenue = cp.sum(cp.multiply(grid_export / 1e3, price)) + cp.sum(cp.multiply(h2_prod, hydrogen_price))
    prob = cp.Problem(cp.Maximize(revenue), constraints)

    param_dict = {
        "demand": demand,
        "gen": gen,
        "price": price,
        "soc_init": soc_init,
        "h2_init": h2_init
    }
    var_dict = {
        "soc": soc,
        "charge": charge,
        "discharge": discharge,
        "h2_prod": h2_prod,
        "elec_to_h2": elec_to_h2,
        "h2_storage": h2_storage,
        "grid_export": grid_export
    }

    return prob, param_dict, var_dict

def solve_battery_dispatch_daily(prob, param_dict, var_dict, daily_inputs):
    # Assign inputs
    param_dict["demand"].value = daily_inputs["demand"]
    param_dict["gen"].value = daily_inputs["gen"]
    param_dict["price"].value = daily_inputs["price"]
    param_dict["soc_init"].value = daily_inputs["soc_init"]

    # Solve with warm-start
    try:
        prob.solve(solver=cp.ECOS, warm_start=True, reltol=1e-5, feastol=1e-5)
        #prob.solve(solver=cp.SCS, eps=1e-3, max_iters=5000)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return {
                "battery_soc": var_dict["soc"].value,
                "battery_charge": var_dict["charge"].value,
                "battery_discharge": var_dict["discharge"].value,
                "electricity_sold_daily": var_dict["grid_export"].value
            }
        else:
            return None
    except Exception as e:
        #print("Solver error:", e)
        return None


def solve_dispatch_daily(prob, param_dict, var_dict, daily_inputs):
    # Assign inputs
    param_dict["demand"].value = daily_inputs["demand"]
    param_dict["gen"].value = daily_inputs["gen"]
    param_dict["price"].value = daily_inputs["price"]
    param_dict["soc_init"].value = daily_inputs["soc_init"]
    param_dict["h2_init"].value = daily_inputs["h2_init"]

    # Solve with warm-start
    try:
        prob.solve(solver=cp.SCS, warm_start=True)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return {
                "battery_soc": var_dict["soc"].value,
                "battery_charge": var_dict["charge"].value,
                "battery_discharge": var_dict["discharge"].value,
                "hydrogen_production": var_dict["h2_prod"].value,
                "electricity_sold_daily": var_dict["grid_export"].value,
                "hydrogen_storage": var_dict["h2_storage"].value
            }
        else:
            return None
    except Exception as e:
        #print("Solver error:", e)
        return None
    
def battery_8760_dispatch(
        hourly_production, 
        hourly_demand, 
        electricity_price,
        num_battery,
        battery_capacity,
        battery_c_rate,
        battery_dc_rate,
        battery_eff
        ):
    
    prob, param_dict, var_dict = setup_dispatch_problem_battery(
        num_battery,
        battery_capacity,
        battery_c_rate,
        battery_dc_rate,
        battery_eff,
        horizon=24)

    hours_in_year = 8760
    electricity_sold = np.zeros(8760)
    battery_soc_history = np.zeros(8760)
    battery_charge = np.zeros(8760)
    battery_discharge = np.zeros(8760)

    #battery_soc_history[0] = num_battery * battery_capacity / 2

    soc = num_battery * battery_capacity / 2
    #h2 = 0

    for day in range(365):
        start = day * 24
        end = start + 24

        daily_inputs = {
            "demand": hourly_demand[start:end],
            "gen": hourly_production[start:end],
            "price": electricity_price[start:end],
             "soc_init": soc,
            # "h2_init": h2
            #"soc_init" : battery_soc_history[start],
        }
    
        result = solve_battery_dispatch_daily(prob, param_dict, var_dict, daily_inputs)


        if result is not None:
            electricity_sold[start:end] = result["electricity_sold_daily"]
            battery_soc_history[start:end] = result["battery_soc"][1:]
            battery_charge[start:end] = result["battery_charge"]
            battery_discharge[start:end] = result["battery_discharge"]
            soc = result["battery_soc"][-1]
            #h2 = result["hydrogen_storage"][-1]

        else:
            #print(f"Optimization failed on day {day}")
            return None 

    return electricity_sold, battery_soc_history, battery_charge, battery_discharge



def run_8760_dispatch(
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
    
    prob, param_dict, var_dict = setup_dispatch_problem_hydrogen(
        num_battery,
        battery_capacity,
        battery_c_rate,
        battery_dc_rate,
        battery_eff,
        hydrogen_capacity,
        num_electrolyzer,
        electrolyzer_capacity,
        electrolyzer_eff,
        electricity_to_hydrogen,
        hydrogen_price,
        horizon=24)

    hours_in_year = 8760
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

        daily_inputs = {
            "demand": hourly_demand[start:end],
            "gen": hourly_production[start:end],
            "price": electricity_price[start:end],
            # "soc_init": soc,
            # "h2_init": h2
            "soc_init" : battery_soc_history[start],
            "h2_init" : h2_history[start]
        }
    
        result = solve_dispatch_daily(prob, param_dict, var_dict, daily_inputs)


        if result is not None:
            electricity_sold[start:end] = result["electricity_sold_daily"]
            hydrogen_production[start:end] = result["hydrogen_production"]
            battery_soc_history[start:end] = result["battery_soc"][1:]
            battery_charge[start:end] = result["battery_charge"]
            battery_discharge[start:end] = result["battery_discharge"]
            #soc = result["battery_soc"][-1]
            #h2 = result["hydrogen_storage"][-1]

        else:
            #print(f"Optimization failed on day {day}")
            return None 

    return electricity_sold, hydrogen_production, battery_soc_history, battery_charge, battery_discharge

def optimize_dispatch_with_hydrogen(
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
    horizon=24
):

    battery_soc = cp.Variable(horizon + 1)
    battery_charge = cp.Variable(horizon)
    battery_discharge = cp.Variable(horizon)
    hydrogen_prod = cp.Variable(horizon)
    electricity_h2 = cp.Variable(horizon)
    hydrogen_storage = cp.Variable(horizon + 1)
    grid_export = cp.Variable(horizon)

    constraints = [
        battery_soc[0] == battery_soc_init,
        hydrogen_storage[0] == hydrogen_storage_init,
    ]

    for t in range(horizon):
        constraints += [
            # Battery constraints
            battery_charge[t] >= 0,
            battery_discharge[t] >= 0,
            battery_soc[t] >= 0,
            battery_soc[t+1] == battery_soc[t] + battery_charge[t]*battery_efficiency - battery_discharge[t]/battery_efficiency,
            battery_soc[t+1] <= num_battery * battery_capacity,
            battery_charge[t] <= num_battery * battery_charge_rate,
            battery_discharge[t] <= num_battery * battery_discharge_rate,
            battery_discharge[t] <= battery_soc[t],

            # Hydrogen constraints
            hydrogen_prod[t] >= 0,
            electricity_h2[t] >= 0,
            electricity_h2[t] <= num_electrolyzer * electrolyzer_capacity,
            hydrogen_prod[t] == electricity_h2[t] * electrolyzer_efficiency / electricity_to_hydrogen,
            hydrogen_storage[t+1] == hydrogen_storage[t] + hydrogen_prod[t],
            hydrogen_storage[t+1] <= hydrogen_capacity,

            # Power balance
            grid_export[t] == total_gen[t] + battery_discharge[t] - battery_charge[t] - electricity_h2[t],
            grid_export[t] >= 0,
        ]

    revenue = cp.sum(cp.multiply(grid_export / 1e3, daily_price)) + cp.sum(cp.multiply(hydrogen_prod, hydrogen_price))
    prob = cp.Problem(cp.Maximize(revenue), constraints)

    solvers_to_try = [
        #("OSQP", {"max_iter": 1000, "eps_abs": 1e-3, "eps_rel": 1e-3, "verbose": False}),
        ("ECOS", {"abstol": 1e-5, "reltol": 1e-5, "feastol": 1e-5, "verbose": False}),
        ("SCS",  {"eps": 1e-3, "max_iters": 5000, "verbose": False})
    ]

    for solver_name, options in solvers_to_try:
        try:
            print(f"Trying solver: {solver_name}")
            prob.solve(solver=getattr(cp, solver_name), **options)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                break
        except Exception as e:
            print(f"{solver_name} failed: {e}")
            continue

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Final status: {prob.status}")
        return None

    return {
        "battery_charge": battery_charge.value,
        "battery_discharge": battery_discharge.value,
        "battery_soc": battery_soc.value,
        "hydrogen_production": hydrogen_prod.value,
        "electricity_sold_daily": grid_export.value,
        "hydrogen_storage": hydrogen_storage.value
    }




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


def rule_based_dispatch(
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
            #net -= charge
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
    
def sequential_daily_dispatch_battery(hourly_production, hourly_demand, electricity_price, num_battery,
                                 battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency):


    hours_in_year = 8760
    electricity_sold = np.zeros(hours_in_year)
    battery_soc_history = np.zeros(hours_in_year)
    battery_charge = np.zeros(hours_in_year)
    battery_discharge = np.zeros(hours_in_year)

    soc = num_battery * battery_capacity / 2

    for day in range(365):
        start = day * 24
        end = start + 24
        daily_gen = hourly_production[start:end]
        daily_demand = hourly_demand[start:end]
        daily_price = electricity_price[start:end]

          
        result = optimize_dispatch_with_battery(
            daily_demand, daily_gen,
            soc, battery_capacity, num_battery, battery_c_rate, battery_dc_rate, battery_efficiency,
            daily_price)

        if result is not None:
            electricity_sold[start:end] = result["electricity_sold_daily"]
            battery_soc_history[start:end] = result["battery_soc"][1:]
            battery_charge[start:end] = result["battery_charge"]
            battery_discharge[start:end] = result["battery_discharge"]
            soc = result["battery_soc"][-1]
        else:
            electricity_sold[start:end] = 0
            battery_soc_history[start:end] = soc
            battery_charge[start:end] = 0
            battery_discharge[start:end] = 0

    return electricity_sold, battery_soc_history, battery_charge, battery_discharge



def sequential_daily_dispatch_h2(hourly_production, hourly_demand, electricity_price, num_battery,
                                 battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency,
                                 hydrogen_capacity, electrolyzer_capacity, electrolyzer_efficiency,
                                 hydrogen_price, num_electrolyzer, electricity_to_hydrogen):
    """
    Sequential daily dispatch with battery and hydrogen storage continuity.
    """

    hours_in_year = 8760
    electricity_sold = np.zeros(hours_in_year)
    hydrogen_production = np.zeros(hours_in_year)
    battery_soc_history = np.zeros(hours_in_year)
    battery_charge = np.zeros(hours_in_year)
    battery_discharge = np.zeros(hours_in_year)

    soc = num_battery * battery_capacity / 2
    h2_init = hydrogen_capacity / 2

    for day in range(365):
        period = 24
        start = day * period
        end = start + period
        daily_gen = hourly_production[start:end]
        daily_demand = hourly_demand[start:end]
        daily_price = electricity_price[start:end]

        if day == 3:
            print(daily_gen)
            print(daily_price)
            print(electricity_sold)
            print(battery_soc_history)
            print(hydrogen_production)


        result = optimize_dispatch_with_hydrogen(
            daily_demand, daily_gen,
            soc, h2_init,
            battery_capacity, num_battery, battery_c_rate, battery_dc_rate, battery_efficiency,
            hydrogen_capacity, num_electrolyzer, electricity_to_hydrogen, electrolyzer_capacity, 
            electrolyzer_efficiency, daily_price, hydrogen_price
        )

        if result is not None:
            electricity_sold[start:end] = result["electricity_sold_daily"]
            battery_soc_history[start:end] = result["battery_soc"][1:]
            hydrogen_production[start:end] = result["hydrogen_production"]
            battery_charge[start:end] = result["battery_charge"]
            battery_discharge[start:end] = result["battery_discharge"]

            soc = result["battery_soc"][-1]
            h2_init = result["hydrogen_storage"][-1]
        else:
            print(f"Day {day}: Optimization failed, using rule-based fallback")
            electricity_sold_day, battery_soc_day, hydrogen_storage, battery_charge_day, battery_discharge_day, hydrogen_production_day = rule_based_dispatch(
                daily_gen, daily_demand, num_battery, battery_capacity, battery_efficiency,
                battery_c_rate, battery_dc_rate, hydrogen_capacity, electrolyzer_capacity,
                electrolyzer_efficiency, num_electrolyzer, electricity_to_hydrogen, hours=period
            )
            electricity_sold[start:end] = electricity_sold_day
            battery_soc_history[start:end] = battery_soc_day
            hydrogen_production[start:end] = hydrogen_production_day
            battery_charge[start:end] = battery_charge_day
            battery_discharge[start:end] = battery_discharge_day


    return electricity_sold, hydrogen_production, battery_soc_history, battery_charge, battery_discharge


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


    # electricity_sold, hydrogen_produced, battery_soc_history, battery_charge, battery_discharge = sequential_daily_dispatch_h2(
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

    result = run_8760_dispatch(
        hourly_production, 
        hourly_demand, 
        electricity_price,
        num_battery,
        SYSTEMS["battery"]["capacity"],
        SYSTEMS["battery"]["charge_rate"],
        SYSTEMS["battery"]["discharge_rate"],
        SYSTEMS["battery"]["efficiency"],
        SYSTEMS["hydrogen_storage"]["capacity"],
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"],
        SYSTEMS["electrolyzer"]["efficiency"],
        hydrogen_price,
        num_electrolyzer, 
        electricity_to_hydrogen
        )

    if result is not None:
        electricity_sold, hydrogen_produced, battery_soc_history, battery_charge, battery_discharge = result
    else:
        return 1e9

    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    if penalty > args.rel and return_details is not True:
        return 1e9

    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6
    hydrogen_revenue = np.sum(hydrogen_produced * hydrogen_price) / 1e6

    annual_om_cost = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["O&M"] +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["O&M"] +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["O&M"] +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["O&M"] +
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["O&M"]
    ) / 1e6

    annual_capex = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] * crf(discount_rate, SYSTEMS["SMR"]["lifetime"]) +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] * crf(discount_rate, SYSTEMS["wind"]["lifetime"]) +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] * crf(discount_rate, SYSTEMS["solar"]["lifetime"]) +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"] * crf(discount_rate, SYSTEMS["battery"]["lifetime"]) +
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["capital"] * crf(discount_rate, SYSTEMS["electrolyzer"]["lifetime"])
    ) / 1e6

    total_capital_cost = (
            num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] +
            num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] +
            num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] +
            num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"] +
            num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["capital"]
    ) / 1e6  # Convert to million USD

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

    electricity_sold, battery_soc_history, hydrogen_soc, battery_charge, battery_discharge, hydrogen_produced = rule_based_dispatch(
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

    annual_om_cost = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["O&M"] +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["O&M"] +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["O&M"] +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["O&M"] +
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["O&M"]
    ) / 1e6

    annual_capex = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] * crf(discount_rate, SYSTEMS["SMR"]["lifetime"]) +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] * crf(discount_rate, SYSTEMS["wind"]["lifetime"]) +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] * crf(discount_rate, SYSTEMS["solar"]["lifetime"]) +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"] * crf(discount_rate, SYSTEMS["battery"]["lifetime"]) +
        num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["capital"] * crf(discount_rate, SYSTEMS["electrolyzer"]["lifetime"])
    ) / 1e6

    total_capital_cost = (
            num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] +
            num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] +
            num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] +
            num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"] +
            num_electrolyzer * SYSTEMS["electrolyzer"]["capacity"] * SYSTEMS["electrolyzer"]["capital"]
        ) / 1e6  # Convert to million USD

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

    # Hourly power production (in kW)
    hourly_production = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["availability"] +
        num_wind * SYSTEMS["wind"]["capacity"] * wind_generation +
        num_solar * SYSTEMS["solar"]["capacity"] * solar_generation
    )

    # Hourly electricity demand scaled to system objective capacity
    hourly_demand = electricity_demand * cap_obj

    # Rule-based battery dispatch (no hydrogen)
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

    # Demand matching penalty (optional cutoff)
    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    if penalty > args.rel and not return_details:
        return 1e9

    # Revenue (USD)
    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6  # in million $

    # Annual O&M cost
    annual_om_cost = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["O&M"] +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["O&M"] +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["O&M"] +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["O&M"]
    ) / 1e6

    # Annualized CAPEX
    annual_capex = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] * crf(discount_rate, SYSTEMS["SMR"]["lifetime"]) +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] * crf(discount_rate, SYSTEMS["wind"]["lifetime"]) +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] * crf(discount_rate, SYSTEMS["solar"]["lifetime"]) +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"] * crf(discount_rate, SYSTEMS["battery"]["lifetime"])
    ) / 1e6

    total_capital_cost = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"]
    ) / 1e6  # million USD

    # Replacement CAPEX by year (array)
    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)

    # Net Present Value
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

    return -profit if not return_details else (
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

    # electricity_sold, battery_soc_history, battery_charge, battery_discharge = sequential_daily_dispatch_battery(
    #     hourly_production,
    #     hourly_demand,
    #     electricity_price,
    #     num_battery,
    #     SYSTEMS["battery"]["capacity"],
    #     SYSTEMS["battery"]["charge_rate"],
    #     SYSTEMS["battery"]["discharge_rate"],
    #     SYSTEMS["battery"]["efficiency"]
    # )

    result = battery_8760_dispatch(
        hourly_production,
        hourly_demand,
        electricity_price,
        num_battery,
        SYSTEMS["battery"]["capacity"],
        SYSTEMS["battery"]["charge_rate"],
        SYSTEMS["battery"]["discharge_rate"],
        SYSTEMS["battery"]["efficiency"]
    )

    if result is not None:
        electricity_sold, battery_soc_history, battery_charge, battery_discharge = result
    else:
        return 1e9

    #penalty = np.sum(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / (hourly_demand))
    if penalty > args.rel and not return_details:
        return 1e9

    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6

    annual_om_cost = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["O&M"] +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["O&M"] +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["O&M"] +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["O&M"]
    ) / 1e6  # in million USD

    annual_capex = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] * crf(discount_rate, SYSTEMS["SMR"]["lifetime"]) +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] * crf(discount_rate, SYSTEMS["wind"]["lifetime"]) +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] * crf(discount_rate, SYSTEMS["solar"]["lifetime"]) +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"] * crf(discount_rate, SYSTEMS["battery"]["lifetime"])
    ) / 1e6  # in million USD

    total_capital_cost = (
        num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] +
        num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] +
        num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] +
        num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"]
    ) / 1e6  # million USD

    # Replacement CAPEX by year (array)
    replacement_capex_by_year = replacement_cost(x, SYSTEMS, project_lifetime)

    # Net Present Value
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

    return -profit if not return_details else (
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


def run_pso_with_params(c1, c2, w, hydrogen, battery):
    if hydrogen and not battery:
        bounds = ([4, 0, 0, 0, 0], [8, 200, 200, 200, 200])
        options = {"c1": c1, "c2": c2, "w": w}
        optimizer = ps.single.GlobalBestPSO(n_particles=300, dimensions=5, options=options, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(pso_objective_function_hydrogen_opt, iters=1, verbose=False)
    elif battery and not hydrogen:
        bounds = ([4, 0, 0, 0], [8, 200, 200, 200])
        options = {"c1": c1, "c2": c2, "w": w}
        optimizer = ps.single.GlobalBestPSO(n_particles=600, dimensions=4, options=options, bounds=bounds)
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


def pso_objective_function_hydrogen_opt(x):
    # Same as above, but for hydrogen case
    with mp.Pool(get_available_cpus(206)) as pool:
        return pool.map(objective_function_hydrogen_opt, x)

def pso_objective_function_hydrogen_rb(x):
    # Same as above, but for hydrogen case
    with mp.Pool(get_available_cpus(206)) as pool:
        return pool.map(objective_function_hydrogen_rb, x)

def pso_objective_function_battery_opt(x):
    with mp.Pool(get_available_cpus(206)) as pool:
        return pool.map(objective_function_battery_opt, x)

def pso_objective_function_battery_rb(x):
    with mp.Pool(get_available_cpus(206)) as pool:
        return pool.map(objective_function_battery_rb, x)    


def main(args):

    global wind_generation, solar_generation, electricity_demand, electricity_price

    # Load electricity price data
    if args.ppa is not None:
        electricity_price = np.full(hours_in_year, args.ppa)
        print(f" Using fixed electricity price: {args.ppa} USD/MWh")
    else:
        # Load from CSV
        print(f" No PPA is present, using day-ahead market prices")
        df = pd.read_csv('price_forecast_2025.csv', sep=';', parse_dates=['DateTime'])
        electricity_price = df['Forecast_Price_USD'].values[:hours_in_year]
        #electricity_price = electricity_price * 1.2

    if args.iter is not None:
        iters = args.iter
    else:
        iters = 1


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


    if args.post_process:
        if args.battery:
            # Note: You'll need to define num_SMR, num_wind, etc. for post-processing if you use a fixed best_pos
            best_pos_battery = [5, 18, 1, 30] # Example values
            print_results(best_pos_battery, args.optdis, args.rbdis)
        elif args.hydrogen:
            best_pos_h2_pp = [6, 12, 12, 27, 15] # Adjust example values for hydrogen
            print_results(best_pos_h2_pp, args.optdis, args.rbdis)
        else:
            print("Please provide options for post process!")
            exit()


    elif args.grid_search:
        grid_search_pso_hyperparameters(args.hydrogen, args.battery)
        exit()

    elif args.hydrogen: #optimization the system with hydrogen
        bounds = ([4, 0, 0, 0, 0], [12, 200, 200, 200, 200])
        options = {"c1": 2.0, "c2": 1.5, "w": 0.5}
        if args.optdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=300, dimensions=5, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_hydrogen_opt, iters=iters)
            #profit, battery_soc_history, hourly_demand, electricity_sold, hydrogen_produced, annual_revenue, hydrogen_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function_hydrogen(best_pos, return_details=True)
        elif args.rbdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=300, dimensions=5, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_hydrogen_rb, iters=iters)
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
        bounds = ([4, 0, 0, 0], [8, 200, 200, 200])
        options = {"c1": 2.0, "c2": 1.5, "w": 0.7}
        if args.optdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=400, dimensions=4, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_battery_opt, iters=iters)
        elif args.rbdis:
            optimizer = ps.single.GlobalBestPSO(n_particles=400, dimensions=4, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(pso_objective_function_battery_rb, iters=iters)
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
    parser.add_argument("--iter", type=int, help="Number of iterations")
    parser.add_argument("--battery", action="store_true", help="Rule based dispatch")
    parser.add_argument("--hydrogen", action="store_true", help="Adds Hydrogen to the NHES")
    parser.add_argument("--optdis", action="store_true", help="Optimize dispatch")
    parser.add_argument("--rbdis", action="store_true", help="Rule based dispatch")
    parser.add_argument("--rel", type=float, default=0.15, help="Demand percentage not covered")

    args = parser.parse_args()

    main(args)