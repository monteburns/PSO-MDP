import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
import argparse
import os, sys

# --- Setup Parameters ---
hours_in_year = 8760
SMR_capacity = 77000
SMR_availability = 0.95
wind_capacity = 2000
solar_capacity = 2250
battery_capacity = 1000
battery_efficiency = 0.9
battery_dc_rate = battery_capacity / 4
battery_c_rate = battery_capacity / 4
electrolyzer_capacity = 1  # MW
hydrogen_storage_capacity = 100e6  # kg
electrolyzer_efficiency = 0.57
hydrogen_price = 6 #usd/kg
electricity_to_hydrogen = 55
cap_obj = 600000  # 600 MW
discount_rate = 0.05
project_lifetime = 60

system_lifetimes = {
    "SMR": 60, "wind": 25, "solar": 25, "battery": 15,
    "electrolyzer": 20, "hydrogen_storage": 10
}
costs = {
    "SMR": {"capital": 6000, "O&M": 25e-3 * hours_in_year * SMR_availability},
    "wind": {"capital": 1600, "O&M": 30},
    "solar": {"capital": 1500, "O&M": 20},
    "battery": {"capital": 400, "O&M": 10},
    "electrolyzer": {"capital": 1800, "O&M": 0.05*1800},
    "hydrogen_storage": {"capital": 0, "O&M": 0}
}

# Load electricity price data
df = pd.read_csv('price_forecast_2025.csv', sep=';', parse_dates=['DateTime']) # $/MWh
electricity_price = (df['Forecast_Price_USD'].values[:hours_in_year]) # Take first 8760 hours
#electricity_price =  np.full(hours_in_year, 73) #PPA

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
solar_generation = solar_data*10 / solar_capacity # multiplied by 10 for 10 panel capacity

# --- Utility Functions ---
def crf(rate, n):
    return (rate * (1 + rate)**n) / ((1 + rate)**n - 1)

# ---- LCOH Calculation ----
def lcoh_cal(num_electrolyzer, hydrogen_produced, discount_factors):
    # Discounted hydrogen production (in kg)
    total_discounted_h2 = sum(hydrogen_produced.sum() * df for df in discount_factors)

    # Hydrogen system capital cost (initial)
    hydrogen_capex = (
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["capital"] +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["capital"]
    )

    # Hydrogen system O&M
    hydrogen_om = (
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["O&M"] +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["O&M"]
    )  
    # Total discounted hydrogen costs
    total_discounted_h2_cost = hydrogen_capex

    for year in range(1, project_lifetime + 1):
        total_discounted_h2_cost += hydrogen_om * discount_factors[year - 1]

        for system in ["electrolyzer", "hydrogen_storage"]:
            lifetime = system_lifetimes[system]
            if year % lifetime == 0:
                if system == "electrolyzer":
                    units = num_electrolyzer
                    unit_cap = electrolyzer_capacity * 1000
                else:  # hydrogen_storage
                    units = num_electrolyzer
                    unit_cap = hydrogen_storage_capacity

                repl_cost = units * unit_cap * costs[system]["capital"] / 1e6
                total_discounted_h2_cost += repl_cost * discount_factors[year - 1]

    # Avoid division by zero
    if total_discounted_h2 > 0:
        lcoh = total_discounted_h2_cost / total_discounted_h2
    else:
        lcoh = np.inf

    return lcoh

def compute_lcoe(
    project_lifetime,
    discount_rate,
    annual_om_cost,
    initial_capex,
    replacement_capex_by_year,  # list of len n with CAPEX replacements (in million USD)
    electricity_sold  # np.array of 8760 hourly kWh values for one typical year
):
    # Calculate total energy produced per year in MWh
    E_t = np.sum(electricity_sold) / 1e3  # Convert kWh to MWh

    numerator = initial_capex
    denominator = 0

    for t in range(1, project_lifetime + 1):
        discount_factor = (1 + discount_rate) ** -t

        I_t = replacement_capex_by_year[t - 1]  # Already in million USD
        M_t = annual_om_cost  # in million USD
        F_t = 0  # Assume zero for renewables unless stated otherwise

        numerator += (I_t + M_t + F_t) * discount_factor
        denominator += E_t * discount_factor  # MWh
    numerator = numerator * 1e6 #Convert MUSD to USD
    lcoe = numerator / denominator  # USD per MWh

    return lcoe


# --- Objective Function With Hydrogen ---
def objective_function(x, return_details=False):
    num_SMR = float(x[0])
    num_wind = float(x[1])
    num_solar = float(x[2])
    num_battery = float(x[3])
    num_electrolyzer = float(x[4])

    hourly_production = (
        num_SMR * SMR_capacity * SMR_availability +
        num_wind * wind_capacity * wind_generation +
        num_solar * solar_capacity * solar_generation
    )
    hourly_demand = electricity_demand * cap_obj
    electricity_sold = np.zeros_like(hourly_production)
    battery_soc = num_battery * battery_capacity / 2
    hydrogen_storage = 0
    hydrogen_produced = np.zeros_like(hourly_production)
    battery_soc_history = np.zeros_like(hourly_production)

    rel = 0

    for i in range(hours_in_year):
        net = hourly_production[i] - hourly_demand[i]

        if net <= 0:
            # Discharge from battery to meet unmet demand
            discharge = min(battery_soc * battery_efficiency, num_battery * battery_dc_rate * battery_efficiency)
            battery_soc -= discharge

            electricity_sold[i] = hourly_production[i] + discharge
        else:
            # Charge battery first
            charge = min(net * battery_efficiency, num_battery * battery_c_rate * battery_efficiency)
            battery_soc += charge
            battery_soc = min(battery_soc, num_battery * battery_capacity)

            remaining = max(0, net - charge)

            # Calculate max hydrogen energy intake (kWh)
            electrolyzer_energy_limit = num_electrolyzer * electrolyzer_capacity * 1000  # kWh

            # Apply energy limit
            h2_energy = min(remaining, electrolyzer_energy_limit)

            # Convert to kg of hydrogen
            h2_kg = h2_energy * electrolyzer_efficiency / electricity_to_hydrogen

            # Apply storage capacity
            storable_h2 = min(h2_kg, hydrogen_storage_capacity - hydrogen_storage)

            hydrogen_storage += storable_h2
            hydrogen_produced[i] = storable_h2


            # Reduce sold electricity by hydrogen conversion energy
            electricity_sold[i] = max(0, hourly_production[i] - charge - (storable_h2 * electricity_to_hydrogen / electrolyzer_efficiency))
        if abs(electricity_sold[i] - hourly_demand[i])/ hourly_demand[i]> 0.01:
            rel = rel + 1
        battery_soc_history[i] = battery_soc

#    if rel/hours_in_year > 0.01:
#        return 1e11

    # Calculate hourly production for all particles at once
    #penalty = np.sum(1e-6 * np.abs(electricity_sold - hourly_demand)) #  1000000 kW
    penalty = np.sum(np.abs(electricity_sold - hourly_demand)/ hourly_demand)


    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6
    hydrogen_revenue = np.sum(hydrogen_produced) * hydrogen_price / 1e6

    annual_om_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["O&M"] +
        num_wind * wind_capacity * costs["wind"]["O&M"] +
        num_solar * solar_capacity * costs["solar"]["O&M"] +
        num_battery * battery_capacity * costs["battery"]["O&M"] +
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["O&M"] +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["O&M"]
    ) / 1e6

    annual_capex = (
        num_SMR * SMR_capacity * costs["SMR"]["capital"] * crf(discount_rate, system_lifetimes["SMR"]) +
        num_wind * wind_capacity * costs["wind"]["capital"] * crf(discount_rate, system_lifetimes["wind"]) +
        num_solar * solar_capacity * costs["solar"]["capital"] * crf(discount_rate, system_lifetimes["solar"]) +
        num_battery * battery_capacity * costs["battery"]["capital"] * crf(discount_rate, system_lifetimes["battery"]) +
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["capital"] * crf(discount_rate, system_lifetimes["electrolyzer"]) +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["capital"] * crf(discount_rate, system_lifetimes["hydrogen_storage"])
    ) / 1e6

    total_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["capital"] + num_SMR * SMR_capacity * costs["SMR"]["O&M"] +
        num_wind * wind_capacity * costs["wind"]["capital"] + num_wind * wind_capacity * costs["wind"]["O&M"]  +
        num_solar * solar_capacity * costs["solar"]["capital"] + num_solar * solar_capacity * costs["solar"]["O&M"]  +
        num_battery * battery_capacity * costs["battery"]["capital"] + num_battery * battery_capacity * costs["battery"]["O&M"] +
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["capital"] + num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["O&M"] +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["capital"] + num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["O&M"]
    ) / 1e6  # Convert to millions


    profit = annual_revenue + hydrogen_revenue - (annual_capex + annual_om_cost)
    return -0.8*profit + 0.2*penalty  if not return_details else (profit, penalty, hydrogen_produced, electricity_sold, annual_revenue, hydrogen_revenue, hourly_demand, battery_soc_history, rel)

# --- Objective Function Without Hydrogen ---
def objective_function_no_hydrogen(x, return_details=False):
    num_SMR = float(x[0])
    num_wind = float(x[1])
    num_solar = float(x[2])
    num_battery = float(x[3])

    hourly_production = (
        num_SMR * SMR_capacity * SMR_availability +
        num_wind * wind_capacity * wind_generation +
        num_solar * solar_capacity * solar_generation
    )
    hourly_demand = electricity_demand * cap_obj
    electricity_sold = np.zeros_like(hourly_production)

    battery_soc = num_battery * battery_capacity / 2
    battery_soc_history = np.zeros_like(hourly_production)
    
    rel = 0

    for i in range(hours_in_year):
        net = hourly_production[i] - hourly_demand[i]
        if net > 0:
            charge_amount = min(net * battery_efficiency,
                                num_battery * battery_c_rate * battery_efficiency)
            battery_soc += charge_amount
            battery_soc = min(battery_soc, num_battery * battery_capacity)
            electricity_sold[i] = hourly_production[i] - charge_amount
        else:
            discharge_amount = min(battery_soc * battery_efficiency,
                                   num_battery * battery_dc_rate * battery_efficiency)
            battery_soc -= discharge_amount
            electricity_sold[i] = hourly_production[i] + discharge_amount
        battery_soc_history[i] = battery_soc
        if abs(electricity_sold[i] - hourly_demand[i])/ hourly_demand[i]> 0.01:
            rel = rel + 1

    # Calculate hourly production for all particles at once
    #penalty = np.sum(1e-6*np.abs(electricity_sold - hourly_demand)) #  1000000 kW
    penalty = np.sum(np.abs(electricity_sold - hourly_demand)/ hourly_demand)


    annual_revenue = np.sum(electricity_sold / 1e3 * electricity_price) / 1e6
    annual_om_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["O&M"] +
        num_wind * wind_capacity * costs["wind"]["O&M"] +
        num_solar * solar_capacity * costs["solar"]["O&M"] +
        num_battery * battery_capacity * costs["battery"]["O&M"]
    ) / 1e6
    annual_capex = (
        num_SMR * SMR_capacity * costs["SMR"]["capital"] * crf(discount_rate, system_lifetimes["SMR"]) +
        num_wind * wind_capacity * costs["wind"]["capital"] * crf(discount_rate, system_lifetimes["wind"]) +
        num_solar * solar_capacity * costs["solar"]["capital"] * crf(discount_rate, system_lifetimes["solar"]) +
        num_battery * battery_capacity * costs["battery"]["capital"] * crf(discount_rate, system_lifetimes["battery"])
    ) / 1e6

    total_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["capital"] + num_SMR * SMR_capacity * costs["SMR"]["O&M"] +
        num_wind * wind_capacity * costs["wind"]["capital"] + num_wind * wind_capacity * costs["wind"]["O&M"]  +
        num_solar * solar_capacity * costs["solar"]["capital"] + num_solar * solar_capacity * costs["solar"]["O&M"]  +
        num_battery * battery_capacity * costs["battery"]["capital"] + num_battery * battery_capacity * costs["battery"]["O&M"]
    ) / 1e6  # Convert to millions

    profit = annual_revenue - (annual_om_cost + annual_capex)
    return -0.9*profit + 0.1*penalty if not return_details else (profit, penalty, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, rel)

def pso_objective_function_hydrogen(x):
    # x shape: (n_particles, 4)
    return np.array([objective_function(particle) for particle in x])

def pso_objective_function_no_hydrogen(x):
    # x shape: (n_particles, 4)
    return np.array([objective_function_no_hydrogen(particle) for particle in x])

def plot_sample_day(hourly_production, wind_gen, solar_gen, smr_gen, electricity_sold, battery_soc, demand, day=172):
    """
    Plots generation and demand for a selected day.
    `day=172` corresponds to June 21.
    """
    start = day * 24
    end = start + 24
    hours = np.arange(24)

    plt.figure()
    hours = np.arange(24)
    plt.plot(hours, smr_gen[start:end], label="SMR Generation")
    plt.plot(hours, wind_gen[start:end], label="Wind Generation")
    plt.plot(hours, solar_gen[start:end], label="Solar Generation")
    plt.plot(hours, battery_soc[start:end], label="Battery SOC")
    plt.plot(hours, hourly_production[start:end], label="Total Generation", linestyle='-.')
    plt.plot(hours, electricity_sold[start:end], label="Electricty Sold", linestyle='--')
    plt.plot(hours, demand[start:end], label="Demand", linestyle=':')
    plt.title("Power Generation and Demand on June 21")
    plt.xlabel("Hour of Day")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sample_day_grid(days, smr_gen, wind_gen, solar_gen, total_gen, electricity_sold, battery_soc, demand, electricity_price):
    """
    Plots power generation and demand for selected days in a 2x2 grid.

    Parameters:
        days (list): List of 4 day indices (0-based, e.g., 15 = Jan 16).
        smr_gen (np.ndarray): Hourly SMR generation array (8760,).
        wind_gen (np.ndarray): Hourly wind generation array (8760,).
        solar_gen (np.ndarray): Hourly solar generation array (8760,).
        total_gen (np.ndarray): Hourly total generation array (8760,).
        demand (np.ndarray): Hourly demand array (8760,).
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    ax2 = [None] * 4
    

    for idx, day in enumerate(days):
        start = day * 24
        end = start + 24
        hours = np.arange(24)

        axs[idx].plot(hours, smr_gen[start:end], label="SMR Generation", linewidth=2)
        axs[idx].plot(hours, wind_gen[start:end], label="Wind Generation", linewidth=2)
        axs[idx].plot(hours, solar_gen[start:end], label="Solar Generation", linewidth=2)
        axs[idx].plot(hours, battery_soc[start:end], label="Battery SOC", linewidth=2)
        axs[idx].plot(hours, total_gen[start:end], label="Total Generation", linestyle='--', linewidth=2)
        axs[idx].plot(hours, electricity_sold[start:end], label="Electricity Sold", linestyle='-.', linewidth=2)
        axs[idx].plot(hours, demand[start:end], label="Demand", linestyle=':', linewidth=2)

        ax2[idx] = axs[idx].twinx()
        ax2[idx].plot(hours, electricity_price[start:end], label="Electricity Price", linewidth=2)
        ax2[idx].set_ylabel("USD/MWh")

        axs[idx].set_title(f"Day {day + 1}", fontsize=12, weight='bold')
        axs[idx].set_xlabel("Hour of Day")
        axs[idx].set_ylabel("Power (kW)")
        axs[idx].grid(True, linestyle='--', alpha=0.6)
        axs[idx].legend(fontsize=8)

    plt.suptitle("Power Generation and Demand on Sample Days", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("sampledays_rb.png", dpi=300)
    plt.close()

def main(hydrogen):
    
    num_SMR, num_solar, num_wind, num_battery, num_electrolyzer = 0, 0, 0, 0, 0
    hydrogen_revenue = 0

    if hydrogen:
        # PSO setup
        #bounds = [min, max] for SMR, wind, solar, battery, hydrogen
        bounds = ([0, 0, 0, 0,0], [12, 200, 200, 200, 200])
        options = {"c1": 1.811, "c2": 2.456, "w": 0.406}

        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=5, options=options, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(pso_objective_function_hydrogen, iters=100)
        profit, penalty, hydrogen_produced, electricity_sold, annual_revenue, hydrogen_revenue, hourly_demand, battery_soc_history, rel = objective_function(best_pos, return_details=True)
        num_SMR, num_wind, num_solar, num_battery, num_electrolyzer = map(round, best_pos)
        print(f"\nOptimal Configuration:")
        print(f"SMRs: {num_SMR}")
        print(f"Wind Turbines: {num_wind}")
        print(f"Solar Panels: {num_solar}")
        print(f"Battery Units: {num_battery}")
        print(f"Electrolyzer Units: {num_electrolyzer}")
        plt.figure()
        plt.plot(electricity_sold, label="electricity_sold")
        plt.plot(hourly_demand, label="hourly_demand")
        plt.plot(battery_soc_history, label="Battery")
        plt.xlabel("Hour of Day")
        plt.ylabel("Hydrogen[kg])")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    else:
        # PSO setup
        #bounds = [min, max] for SMR, wind, solar, battery
        bounds = ([0, 0, 0, 0], [12, 200, 200, 200])
        options = {"c1": 1.811, "c2": 2.456, "w": 0.406}

        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=4, options=options, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(pso_objective_function_no_hydrogen, iters=100)
        profit, penalty, battery_soc_history, hourly_demand, electricity_sold, annual_revenue, rel = objective_function_no_hydrogen(best_pos, return_details=True)
        num_SMR, num_wind, num_solar, num_battery = map(round, best_pos)
        print(f"\nOptimal Configuration:")
        print(f"SMRs: {num_SMR}")
        print(f"Wind Turbines: {num_wind}")
        print(f"Solar Panels: {num_solar}")
        print(f"Battery Units: {num_battery}")
        plt.figure()
        plt.plot(battery_soc_history, label="Battery")
        plt.plot(hourly_demand, label="hourly_demand")
        plt.plot(electricity_sold, label="electricity_sold")
        plt.xlabel("Hour of Day")
        plt.ylabel(" [kWh]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    #print(f"Demand Matching Efficiency: {1 - (rel/hours_in_year)}")
    print(f"Demand Matching Efficiency: {1 - (penalty/hours_in_year)}")
    cost_history = optimizer.cost_history
    # Create the convergence plot
    plt.plot(cost_history)
    plt.title('Convergence Plot')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

    annual_om_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["O&M"] +
        num_wind * wind_capacity * costs["wind"]["O&M"] +
        num_solar * solar_capacity * costs["solar"]["O&M"] +
        num_battery * battery_capacity * costs["battery"]["O&M"] +
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["O&M"] +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["O&M"]
    ) / 1e6

    annual_capex = (
        num_SMR * SMR_capacity * costs["SMR"]["capital"] * crf(discount_rate, system_lifetimes["SMR"]) +
        num_wind * wind_capacity * costs["wind"]["capital"] * crf(discount_rate, system_lifetimes["wind"]) +
        num_solar * solar_capacity * costs["solar"]["capital"] * crf(discount_rate, system_lifetimes["solar"]) +
        num_battery * battery_capacity * costs["battery"]["capital"] * crf(discount_rate, system_lifetimes["battery"]) +
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["capital"] * crf(discount_rate, system_lifetimes["electrolyzer"]) +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["capital"] * crf(discount_rate, system_lifetimes["hydrogen_storage"])
    ) / 1e6

    total_capital_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["capital"] +
        num_wind * wind_capacity * costs["wind"]["capital"] +
        num_solar * solar_capacity * costs["solar"]["capital"] +
        num_battery * battery_capacity * costs["battery"]["capital"] +
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["capital"] +  # Electrolyzer cost
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["capital"]  # Hydrogen storage cost
    ) / 1e6  # Convert to millions

    print(f"\nFinancial Summary (Million $):")
    print(f"Capital Cost: ${total_capital_cost:.2f}M")
    print(f"Annual O&M Cost: ${annual_om_cost:.2f}M")
    print(f"Annual Electricity Revenue: ${annual_revenue:.2f}M")
    print(f"Annual Hyrdogen Revenue: ${hydrogen_revenue:.2f}M")
    print(f"Annual Net Revenue: ${(annual_revenue + hydrogen_revenue - annual_om_cost):.2f}M")
    print(f"Simple Payback Period: {total_capital_cost/((annual_revenue + hydrogen_revenue) - annual_om_cost):.1f} years")

    total_annual_generation = np.sum(electricity_sold/1e3) #MWh

    # Discount factors
    discount_factors = [(1 + discount_rate)**-i for i in range(1, project_lifetime + 1)]

    # Annual net revenue
    annual_net_revenue = annual_revenue + hydrogen_revenue - annual_om_cost

    # ---- NPV Calculation ----
    npv = -total_capital_cost

    for year in range(1, project_lifetime + 1):
        npv += annual_net_revenue * discount_factors[year - 1]

        for system, lifetime in system_lifetimes.items():
            if year % lifetime == 0:
                if system == "SMR":
                    units = num_SMR
                    unit_cap = SMR_capacity
                elif system == "wind":
                    units = num_wind
                    unit_cap = wind_capacity
                elif system == "solar":
                    units = num_solar
                    unit_cap = solar_capacity
                elif system == "battery":
                    units = num_battery
                    unit_cap = battery_capacity
                elif system == "electrolyzer":
                    units = num_electrolyzer
                    unit_cap = electrolyzer_capacity * 1000
                else:
                    continue

                replacement_cost = units * unit_cap * costs[system]["capital"] / 1e6
                npv -= replacement_cost * discount_factors[year - 1]

    # ---- LCOE Calculation ----
    total_discounted_generation = sum(total_annual_generation * df for df in discount_factors)
    total_discounted_cost = total_capital_cost

    replacement_capex_by_year = []

    for year in range(1, project_lifetime + 1):
        total_discounted_cost += annual_om_cost * discount_factors[year - 1]
        replacement_cost = 0

        for system, lifetime in system_lifetimes.items():
            if year % lifetime == 0:
                if system == "SMR":
                    units = num_SMR
                    unit_cap = SMR_capacity
                elif system == "wind":
                    units = num_wind
                    unit_cap = wind_capacity
                elif system == "solar":
                    units = num_solar
                    unit_cap = solar_capacity
                elif system == "battery":
                    units = num_battery
                    unit_cap = battery_capacity
                elif system == "electrolyzer":
                    units = num_electrolyzer
                    unit_cap = electrolyzer_capacity * 1000
                else:
                    continue

                replacement_cost = units * unit_cap * costs[system]["capital"] / 1e6
                total_discounted_cost += replacement_cost * discount_factors[year - 1]

        replacement_capex_by_year.append(replacement_cost)

    lcoe = total_discounted_cost / (total_discounted_generation / 1e6)

    lcoh = lcoh_cal(num_electrolyzer, hydrogen_produced, discount_factors) if hydrogen else 0

    print(f"LCOE: {lcoe:.2f} USD/MWh")
    print(f"LCOH: {lcoh:.2f} USD/kg")
    print(f"NPV: ${npv:.2f}M")

    slcoe = compute_lcoe(
        project_lifetime,
        discount_rate,
        annual_om_cost,
        total_capital_cost,
        replacement_capex_by_year,
        electricity_sold
    )

    print(f"Sahsi LCOE: {slcoe:.2f} USD/MWh")
    print(profit, penalty)

    smr_gen = np.full(hours_in_year, num_SMR * SMR_capacity * SMR_availability)
    wind_gen = num_wind * wind_capacity * wind_generation
    solar_gen = num_solar * solar_capacity * solar_generation
    hourly_production = smr_gen + wind_gen + solar_gen
    plot_sample_day(hourly_production, wind_gen, solar_gen, smr_gen, electricity_sold, battery_soc_history, hourly_demand)
    plot_sample_day_grid([15, 100, 172, 280], smr_gen, wind_gen, solar_gen, hourly_production, electricity_sold, battery_soc_history, hourly_demand, electricity_price)

if __name__ == '__main__':

    hydrogen = False

    main(hydrogen)










    
