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
from system_config import SYSTEMS, discount_rate, project_lifetime, hydrogen_price, cap_obj

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

# Custom non-daemonic process for nested multiprocessing
class NoDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NonDaemonPool(mp.pool.Pool):
    Process = NoDaemonProcess

# global scope
_shared_hourly_production = None
_shared_hourly_demand = None
_shared_price = None


def mpc_dispatch_with_hydrogen(
    demand, 
    smr_gen, 
    wind_gen, 
    solar_gen, 
    battery_soc_init, 
    hydrogen_storage_init,
    battery_capacity, 
    battery_charge_rate, 
    battery_discharge_rate, 
    battery_efficiency, 
    hydrogen_capacity, 
    electrolyzer_capacity,
    electrolyzer_efficiency, 
    electricity_price, 
    hydrogen_price,
    horizon=24
):
    """
    CVXPY-based MPC optimization for energy dispatch with hydrogen integration.
    
    Returns:
        dict: Dispatch decisions and updated storage values.
    """
    # Inputs
    total_gen = smr_gen + wind_gen + solar_gen

    # Decision variables
    battery_soc = cp.Variable(horizon + 1)
    battery_charge = cp.Variable(horizon)
    battery_discharge = cp.Variable(horizon)
    hydrogen_prod = cp.Variable(horizon)
    hydrogen_storage = cp.Variable(horizon + 1)
    grid_export = cp.Variable(horizon)

    constraints = []

    # Initial conditions
    constraints += [battery_soc[0] == battery_soc_init]
    constraints += [hydrogen_storage[0] == hydrogen_storage_init]

    for t in range(horizon):
        net_gen = total_gen[t] - demand[t]

        # Battery dynamics
        constraints += [
            battery_charge[t] >= 0,
            battery_discharge[t] >= 0,
            battery_soc[t+1] == battery_soc[t] + battery_charge[t]*battery_efficiency - battery_discharge[t]/battery_efficiency,
            battery_soc[t+1] <= battery_capacity,
            battery_charge[t] <= battery_charge_rate,
            battery_discharge[t] <= battery_discharge_rate,
        ]

        # Hydrogen dynamics
        constraints += [
            hydrogen_prod[t] >= 0,
            hydrogen_prod[t] <= electrolyzer_capacity,
            hydrogen_storage[t+1] == hydrogen_storage[t] + hydrogen_prod[t] * electrolyzer_efficiency,
            hydrogen_storage[t+1] <= hydrogen_capacity,
        ]

        # Power balance: use generation to meet demand, charge battery, produce H2, sell to grid
        constraints += [
            net_gen == battery_charge[t] - battery_discharge[t] + hydrogen_prod[t] + grid_export[t],
            grid_export[t] >= 0,
        ]

    # Objective: Maximize revenue from electricity sold + hydrogen produced
    revenue = cp.sum(cp.multiply(grid_export, electricity_price)) + cp.sum(cp.multiply(hydrogen_prod, hydrogen_price))
    objective = cp.Maximize(revenue)

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, max_iter=1000, eps_abs=1e-3, eps_rel=1e-3, warm_start=True)

    # Return results
    if prob.status not in ["infeasible", "unbounded"]:
        return {
            "battery_charge": battery_charge.value,
            "battery_discharge": battery_discharge.value,
            "battery_soc": battery_soc.value,
            "hydrogen_production": hydrogen_prod.value,
            "hydrogen_storage": hydrogen_storage.value,
            "grid_export": grid_export.value,
            "revenue": revenue.value
        }
    else:
        return None

def run_mpc_dispatch(daily_gen, daily_demand, daily_price, battery_capacity, battery_c_rate, battery_dc_rate, battery_eff, initial_soc):
    hours = 24
    soc = cp.Variable(hours + 1)
    charge = cp.Variable(hours)
    discharge = cp.Variable(hours)

    constraints = [soc[0] == initial_soc]

    for t in range(hours):
        constraints += [
            soc[t+1] == soc[t] + battery_eff * charge[t] - discharge[t] / battery_eff,
            soc[t+1] >= 0,
            soc[t+1] <= battery_capacity,
            charge[t] >= 0,
            charge[t] <= battery_c_rate,
            discharge[t] >= 0,
            discharge[t] <= battery_dc_rate
        ]

    net_export = daily_gen - daily_demand + discharge - charge
    revenue = cp.sum(cp.multiply(net_export / 1e3, daily_price))

    prob = cp.Problem(cp.Maximize(revenue), constraints)

    try:
        prob.solve(solver=cp.OSQP, warm_start=True, solver_opts={
        "max_iter": 1000,
        "eps_abs": 1e-3,
        "eps_rel": 1e-3
    })
    except Exception:
        return np.zeros(24), np.zeros(24), np.full(25, initial_soc)

    return charge.value, discharge.value, soc.value


def parallel_dispatch_worker(day_idx, num_battery, battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency, soc):
    #day_idx, hourly_production, hourly_demand, electricity_price, num_battery, battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency, soc = args
    global _shared_hourly_production, _shared_hourly_demand, _shared_price

    start = day_idx * 24
    end = start + 24

    daily_gen = _shared_hourly_production[start:end]
    daily_demand = _shared_hourly_demand[start:end]
    daily_price = _shared_price[start:end]

    try:
        c, d, socs = run_mpc_dispatch(
            daily_gen, daily_demand, daily_price,
            num_battery * battery_capacity,
            num_battery * battery_c_rate,
            num_battery * battery_dc_rate,
            battery_efficiency,
            soc
        )
    except Exception:
        c = np.zeros(24)
        d = np.zeros(24)
        socs = np.full(25, soc)

    return (start, end, c, d, socs[1:], socs[-1])


def parallel_daily_dispatch(hourly_production, hourly_demand, electricity_price, num_battery, battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency):
    global _shared_hourly_production, _shared_hourly_demand, _shared_price

    _shared_hourly_production = hourly_production
    _shared_hourly_demand = hourly_demand
    _shared_price = electricity_price
    
    electricity_sold = np.zeros_like(hourly_production)
    battery_soc_history = np.zeros_like(hourly_production)

    battery_charge = np.zeros_like(hourly_production)
    battery_discharge = np.zeros_like(hourly_production)

    soc = num_battery * battery_capacity / 2

    args_list = [
        (day, num_battery, battery_capacity, battery_c_rate,
         battery_dc_rate, battery_efficiency, soc)
        for day in range(365)
    ]
    
    with mp.get_context("fork").Pool(get_available_cpus(64)) as pool:
        results = pool.starmap(parallel_dispatch_worker, args_list)

    for start, end, c, d, soc_day, soc in results:
        electricity_sold[start:end] = _shared_hourly_production[start:end] - c + d #hourly_production
        battery_soc_history[start:end] = soc_day
        battery_charge[start:end] = c
        battery_discharge[start:end] = d

    return electricity_sold, battery_soc_history, battery_charge, battery_discharge

# --- Objective Function Without Hydrogen ---
def objective_function(x, return_details=False):
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

    electricity_sold, battery_soc_history, battery_charge, battery_discharge = parallel_daily_dispatch(
        hourly_production,
        hourly_demand,
        electricity_price,
        num_battery,
        SYSTEMS["battery"]["capacity"],
        SYSTEMS["battery"]["charge_rate"],
        SYSTEMS["battery"]["discharge_rate"],
        SYSTEMS["battery"]["efficiency"]
    )

    #penalty = np.sum(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / (hourly_demand))
    if penalty > 0.10:
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

def pso_objective_function(x):
    with NonDaemonPool(get_available_cpus(8)) as pool:
        return pool.map(objective_function, x)

def run_pso_with_params(c1, c2, w):

    bounds = ([4, 0, 0, 50], [7, 200, 200, 250])
    options = {"c1": c1, "c2": c2, "w": w}

    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=4, options=options, bounds=bounds)

    best_cost, best_pos = optimizer.optimize(pso_objective_function, iters=10, verbose=False)

    return {
        "c1": c1,
        "c2": c2,
        "w": w,
        "best_cost": best_cost,
        "best_pos": best_pos
    }
def trackable_run(c1, c2, w, progress_list):
    result = run_pso_with_params(c1, c2, w)
    progress_list.append((c1, c2, w))  # Mark this config as completed
    print(f"* Finished: c1={c1}, c2={c2}, w={w} -> Cost: {result['best_cost']:.3f}")
    return result

def grid_search_pso_hyperparameters():

    # Define ranges to test
    c1_vals = [1.5, 1.8, 2.0]
    c2_vals = [1.5, 2.0, 2.5]
    w_vals = [0.3, 0.5, 0.7]

    grid = list(product(c1_vals, c2_vals, w_vals))

    print(f"🔍 Starting grid search over {len(grid)} PSO configurations...\n")

    manager = Manager()
    progress_list = manager.list()

    with NonDaemonPool(get_available_cpus(2)) as pool:
        func = partial(trackable_run, progress_list=progress_list)
        results = list(pool.starmap(func, grid))

    results = sorted(results, key=lambda x: x["best_cost"])
    
    print("\n--- Grid Search Results ---")
    for r in results:
        print(f"c1={r['c1']}, c2={r['c2']}, w={r['w']} -> Cost: {r['best_cost']:.3f}")
    
    best = results[0]
    print(f"\nBest Params: c1={best['c1']}, c2={best['c2']}, w={best['w']}")
    print(f"Best Cost: {best['best_cost']}, Best Position: {best['best_pos']}")

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


    df = pd.read_csv('demand_forecast_2025.csv', sep=';', parse_dates=['DateTime'])
    electricity_demand = (df['Forecast_Demand_MWh'].values[:hours_in_year]) # Take first 8760 hours
    electricity_demand = electricity_demand / np.max(electricity_demand)

    # Load wind generation data
    df_wind = pd.read_csv('wind_power_output.csv', parse_dates=[0])
    wind_generation = (df_wind['Wind_Power_Output'].values[:hours_in_year] / 2000.0)  # Take first 8760 hours and normalize

    # Load solar generation data
    df_solar = pd.read_csv('solar_power_output_tmy.csv', parse_dates=[0])
    df_solar['Power_Output'] = df_solar['Power_Output'].apply(lambda x: 0 if x < 0 else x)
    solar_data = df_solar['Power_Output'].values[:hours_in_year]  # Take first 8760 hours
    solar_generation = solar_data*10 / SYSTEMS["solar"]["capacity"] # multiplied by 10 for 10 panel capacity

    if args.post_process:
        best_pos = [5, 3, 48, 136]
        profit, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function(best_pos, return_details=True)
        smr_gen = np.full(hours_in_year, num_SMR *  SYSTEMS["SMR"]["capacity"] *  SYSTEMS["SMR"]["availability"])
        wind_gen = num_wind * SYSTEMS["wind"]["capacity"] * wind_generation
        solar_gen = num_solar * SYSTEMS["solar"]["capacity"] * solar_generation
        hourly_production = smr_gen + wind_gen + solar_gen
        plot_sample_day_grid([15, 100, 172, 280], smr_gen, wind_gen, solar_gen, hourly_production, electricity_sold, battery_soc_history, hourly_demand, electricity_price)
        plot_battery_charge_discharge_days([15, 100, 172, 280], battery_charge, battery_discharge, battery_soc_history)
        plot_battery_week_profile(45, battery_charge, battery_discharge, battery_soc_history)
        plot_energy_week_profile(100, battery_charge, battery_discharge, battery_soc_history,
                                hourly_demand, electricity_sold, electricity_price, hourly_production)
    elif args.grid_search:
        grid_search_pso_hyperparameters()
        exit()

    else: #optimization 
        
        bounds = ([3, 0, 0, 0], [8, 200, 200, 350])
        options = {"c1": 1.5, "c2": 1.5, "w": 0.5}
        optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=4, options=options, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(pso_objective_function, iters=150)
        profit, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, annual_om_cost, penalty, battery_charge, battery_discharge = objective_function(best_pos, return_details=True)
        cost_history = optimizer.cost_history
        # Create the convergence plot
        plt.plot(cost_history)
        plt.title('Convergence Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.savefig("Convergence.png", dpi=300)
        plt.close()
        num_SMR, num_wind, num_solar, num_battery = map(round, best_pos)
        print(f"\nOptimal Configuration:")
        print(f"SMRs: {num_SMR}")
        print(f"Wind Turbines: {num_wind}")
        print(f"Solar Panels: {num_solar}")
        print(f"Battery Units: {num_battery}")
        print(f"Demand Matching Efficiency: {1-penalty}")

        total_capital_cost = (
            num_SMR * SYSTEMS["SMR"]["capacity"] * SYSTEMS["SMR"]["capital"] +
            num_wind * SYSTEMS["wind"]["capacity"] * SYSTEMS["wind"]["capital"] +
            num_solar * SYSTEMS["solar"]["capacity"] * SYSTEMS["solar"]["capital"] +
            num_battery * SYSTEMS["battery"]["capacity"] * SYSTEMS["battery"]["capital"]
        ) / 1e6  # Convert to million USD

        print(f"\nFinancial Summary (Million $):")
        print(f"PPA: ${np.mean(electricity_price):.2f}M") if args.ppa else print(f"Mean Electricity price: ${np.mean(electricity_price):.2f}M")
        print(f"Capital Cost: ${total_capital_cost:.2f}M")
        print(f"Annual O&M Cost: ${annual_om_cost:.2f}M")
        print(f"Annual Electricity Revenue: ${annual_revenue:.2f}M")
        print(f"Annual Net Revenue: ${(annual_revenue  - annual_om_cost):.2f}M")
        print(f"Simple Payback Period: {total_capital_cost/(annual_revenue - annual_om_cost):.1f} years")

    
    replacement_capex_by_year = replacement_cost(best_pos, SYSTEMS, project_lifetime)
    npv = npv_cal(
        total_capital_cost,
        annual_revenue,
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

    print(f"LCOE: {lcoe:.2f} USD/MWh")
    print(f"NPV: ${npv:.2f}M")



if __name__ == '__main__':

    workingDirectory  = os.path.realpath(sys.argv[0])

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--post_process", action="store_true", help="Create graphics")
    parser.add_argument("--grid_search", action="store_true", help="Run PSO hyperparameter grid search")
    parser.add_argument("--ppa", type=float, help="Single electricity price value in USD/MWh")

    args = parser.parse_args()

    main(args)





    
