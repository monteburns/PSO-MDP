import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

# CHANGE LOG
# Instead of 1 panel, 10 panel capacity taken for reducing steps in optimization
# 4 Hour battery means that battery can discharge its whole capacity in 4 hours

# Cost and technical data
SMR_capacity = 77000  # kWe
SMR_availability = 0.95
wind_capacity = 2000   # kWe per turbine
solar_capacity = 225e1  # kWe per 10 panel


# System lifetimes (in years)
system_lifetimes = {
    "SMR": 60,
    "wind": 25,
    "solar": 25,
    "battery": 15,  # Assuming battery needs replacement every 15 years,
    "electrolyzer": 10  # Assuming hydrogen needs replacement every 10 years
}

# Add battery system parameters
battery_capacity = 1000  # kWh per unit
battery_efficiency = 0.90  # 90% round-trip efficiency
battery_dc_rate = battery_capacity / 4 # 4 hour battery
battery_c_rate = battery_capacity / 4 # 4 hour battery

# Load and process data to ensure consistent length
hours_in_year = 8760  # Standard year length

# Add hydrogen system parameters
electrolyzer_capacity = 1  # MW per unit
hydrogen_storage_capacity = 1000  # kg per unit
electrolyzer_efficiency = 0.7  # 70% efficiency
hydrogen_price = 3  # $/kg
electricity_to_hydrogen = 50  # kWh/kg H2 production rate

#"SMR": {"capital": 4000, "O&M": 136 + (11 + 2.6)*1e-3*hours_in_year*SMR_availability}  META-ANALYSIS OF ADVANCED NUCLEAR REACTOR COST ESTIMATES
#"SMR": {"capital": 5000, "O&M": 25e-3*hours_in_year*SMR_availability} ORGINAL REPORT

costs = {
    "SMR": {"capital": 6000, "O&M": 25e-3*hours_in_year*SMR_availability},      # $5000/kW capital, $25/MWh O&M ($90 /kw-year)
    "wind": {"capital": 1600, "O&M": 30},     # $1600/kW capital, $30/kW-year   O&M
    "solar": {"capital": 1000, "O&M": 10},    # $1000/kW capital, $10/kW-year O&M
    "battery": {"capital": 400, "O&M": 10},    # $500/kWh capital, $12.5/kWh O&M
    "electrolyzer": {"capital": 1000, "O&M": 0},  # $1000/kW capital, assume no O&M for now
    "hydrogen_storage": {"capital": 50, "O&M": 0}  # $50/kWh capital, assume no O&M for now
}

# Add these parameters after the costs dictionary
project_lifetime = 60  # years
discount_rate = 0.05  # 5% discount rate

# Load electricity price data
df = pd.read_csv('/home/yuce/phd/price_forecast_2025.csv', sep=';', parse_dates=['DateTime']) # $/MWh
electricity_price = (df['Forecast_Price_USD'].values[:hours_in_year]) # Take first 8760 hours

df = pd.read_csv('/home/yuce/phd/demand_forecast_2025.csv', sep=';', parse_dates=['DateTime'])
electricity_demand = (df['Forecast_Demand_MWh'].values[:hours_in_year]) # Take first 8760 hours
electricity_demand = electricity_demand / np.max(electricity_demand)

# Load wind generation data
df_wind = pd.read_csv('/home/yuce/phd/wind_power_output.csv', parse_dates=[0])
wind_generation = (df_wind['Wind_Power_Output'].values[:hours_in_year] / 2000.0)  # Take first 8760 hours and normalize

# Load solar generation data
df_solar = pd.read_csv('/home/yuce/phd/solar_power_output_tmy.csv', parse_dates=[0])
df_solar['Power_Output'] = df_solar['Power_Output'].apply(lambda x: 0 if x < 0 else x)
solar_data = df_solar['Power_Output'].values[:hours_in_year]  # Take first 8760 hours
solar_generation = solar_data*10 / solar_capacity # multiplied by 10 for 10 panel capacity

num_SMR, num_solar, num_wind, num_battery, num_electrolyzer = 0, 0, 0, 0, 0
hydrogen_revenue = 0
p_threshold = np.median(electricity_price)

def objective_function_hydrogen(x, return_details=False):
    num_SMR, num_wind, num_solar, num_hydrogen, ratio = map(int, x)


    hourly_production = (
        num_SMR * SMR_capacity * SMR_availability +
        num_wind * wind_capacity * wind_generation +
        num_solar * solar_capacity * solar_generation
    )

    hydrogen_produced = np.zeros_like(hourly_production)
    hydrogen_soc = num_hydrogen * hydrogen_storage_capacity / 2
    battery_activity = np.zeros(hours_in_year)

    for i in range(hours_in_year):
        if hourly_production[i] < p_threshold:
            battery_activity[i] = -1  # charging
            charge_limit = num_battery * battery_c_rate
            available_energy = hourly_production[i]
            charge_energy = min(available_energy, charge_limit)
            
            battery_soc += charge_energy * battery_efficiency
            battery_soc = min(battery_soc, num_battery * battery_capacity)
            
            electricity_sold[i] = hourly_production[i] - charge_energy

        else:
            battery_activity[i] = 1  # discharging
            discharge_limit = num_battery * battery_dc_rate
            available_discharge = min(battery_soc, discharge_limit)
            
            discharged_energy = available_discharge * battery_efficiency
            battery_soc -= available_discharge
            
            electricity_sold[i] = hourly_production[i] + discharged_energy


cap_obj = 500000 # 600 MWe Capacity 
avail_obj = cap_obj * 0.0


# --- Objective Function ---
def objective_function(x, return_details=False):
    num_SMR, num_wind, num_solar, num_battery = map(int, x)

    #p_threshold = np.percentile(electricity_price, ratio)

    hourly_demand = electricity_demand * cap_obj

    hourly_production = (
        num_SMR * SMR_capacity * SMR_availability +
        num_wind * wind_capacity * wind_generation +
        num_solar * solar_capacity * solar_generation
    )

    """
battery_soc = num_battery * battery_capacity / 2  # Start at 50% charge
electricity_sold = np.zeros(hours_in_year)
rel = 0  # relative unmet demand occurrences

for i in range(hours_in_year):
    net_production = hourly_production[i] - hourly_demand[i]

    if net_production < 0:
        # Not enough production, try to discharge battery
        possible_discharge = min(
            battery_soc,
            num_battery * battery_dc_rate
        )
        actual_discharge = min(-net_production / battery_efficiency, possible_discharge)
        battery_soc -= actual_discharge
        electricity_sold[i] = hourly_production[i] + actual_discharge * battery_efficiency
    else:
        # Surplus production, try to charge battery
        possible_charge = min(
            net_production * battery_efficiency,
            num_battery * battery_c_rate * battery_efficiency
        )
        battery_soc += possible_charge
        battery_soc = min(battery_soc, num_battery * battery_capacity)
        electricity_sold[i] = hourly_production[i] - possible_charge / battery_efficiency

    # Do not oversell beyond demand
    electricity_sold[i] = min(electricity_sold[i], hourly_demand[i])

    if electricity_sold[i] < hourly_demand[i]:
        rel += 1

if rel / hours_in_year > 0.01:
    return 1e9

    """


    electricity_sold = np.zeros_like(hourly_production)
    battery_soc = num_battery * battery_capacity / 2
    battery_activity = np.zeros(hours_in_year)

    rel = 0

    for i in range(hours_in_year):
        net_production = hourly_production[i] - hourly_demand[i]

        if net_production < 0:
            # Not enough production, try to discharge battery
            possible_discharge = min(
                battery_soc,
                num_battery * battery_dc_rate
            )
            actual_discharge = min(-net_production / battery_efficiency, possible_discharge)
            battery_soc -= actual_discharge
            electricity_sold[i] = hourly_production[i] + actual_discharge * battery_efficiency
        else:
            # Surplus production, try to charge battery
            possible_charge = min(
                net_production * battery_efficiency,
                num_battery * battery_c_rate * battery_efficiency
            )
            battery_soc += possible_charge
            battery_soc = min(battery_soc, num_battery * battery_capacity)
            electricity_sold[i] = hourly_production[i] - possible_charge / battery_efficiency

        # No constraint on selling more than demand
        if electricity_sold[i] < hourly_demand[i]:
            rel += 1

    # Optionally penalize high unmet demand hours
    if rel / hours_in_year > 0.01:
        return 1e9


#    if np.sum(electricity_sold) - cap_obj) > 100:
#        return 1e9  # Large penalty if capacity != 1000 MWe


    #if np.min(electricity_sold) < avail_obj:
    #    return 1e9
    
#    if np.sum(electricity_sold) < total_capacity*0.85*hours_in_year:
#        return 1e9

    annual_revenue = np.sum(np.multiply(electricity_sold/1e3,electricity_price)) / 1e6

    annual_om_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["O&M"] +
        num_wind * wind_capacity * costs["wind"]["O&M"] +
        num_solar * solar_capacity * costs["solar"]["O&M"] +
        num_battery * battery_capacity * costs["battery"]["O&M"] +
        num_electrolyzer * electrolyzer_capacity * 1000 * costs["electrolyzer"]["O&M"] +
        num_electrolyzer * hydrogen_storage_capacity * costs["hydrogen_storage"]["O&M"]
    ) / 1e6

    def crf(discount_rate, lifetime):
        return (discount_rate * (1 + discount_rate)**lifetime ) / ((1+discount_rate)**lifetime -1)

    annual_capital_cost = (
        num_SMR * SMR_capacity * costs["SMR"]["capital"] * crf(discount_rate, system_lifetimes["SMR"]) +
        num_wind * wind_capacity * costs["wind"]["capital"] * crf(discount_rate, system_lifetimes["wind"]) +
        num_solar * solar_capacity * costs["solar"]["capital"] * crf(discount_rate, system_lifetimes["solar"])+
        num_battery * battery_capacity * costs["battery"]["capital"] * crf(discount_rate, system_lifetimes["battery"]) 
    )/ 1e6

    net_annual_profit = annual_revenue - (annual_capital_cost + annual_om_cost)
    #net_annual_profit = annual_revenue - annual_om_cost

    if return_details:
        return -net_annual_profit, electricity_sold, battery_activity, hourly_production, rel
    


    return -net_annual_profit  # Negative for maximization

# Problem-specific bounds
lb = np.array([0, 0, 0, 0])
ub = np.array([12, 200, 200, 100])
DIM = len(lb)

def run_pso(W, C1, C2, return_dict, key):
    NUM_PARTICLES = 100
    MAX_ITER =  100

    positions = np.random.randint(lb, ub + 1, size=(NUM_PARTICLES, DIM)).astype(float)
    velocities = np.random.uniform(-1, 1, size=(NUM_PARTICLES, DIM))
    personal_best_positions = positions.copy()
    personal_best_scores = np.array([objective_function(np.round(p)) for p in positions])
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]

    for _ in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
            velocities[i] = (
                W * velocities[i] +
                C1 * r1 * (personal_best_positions[i] - positions[i]) +
                C2 * r2 * (global_best_position - positions[i])
            )
            positions[i] += velocities[i]
            positions[i] = np.clip(np.round(positions[i]), lb, ub)
            score = objective_function(positions[i])

            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

    return_dict[key] = (global_best_position.astype(int), -global_best_score)

# Define search grids
W_vals = [0.4, 0.5, 0.6, 0.7, 0.8]
C1_vals = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
C2_vals = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

# Start grid search in parallel
if __name__ == "__main__":
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    key = 0
    for w in range(len(W_vals)):
        for c1 in range(len(C1_vals)):
            for c2 in range(len(C2_vals)):
                W, C1, C2 = W_vals[w], C1_vals[c1], C2_vals[c2]
                p = mp.Process(target=run_pso, args=(W, C1, C2, return_dict, key))
                p.start()
                processes.append((p, (W, C1, C2)))
                key += 1

    for p, _ in processes:
        p.join()

    best_score = -np.inf
    best_config = None
    best_params = ()

    for idx, (p, (W, C1, C2)) in enumerate(processes):
        position, score = return_dict[idx]
        print(f"W={W:.2f}, C1={C1:.2f}, C2={C2:.2f} => Net Annual Profit={score:.2f} MUSD")
        if score > best_score:
            best_score = score
            best_config = position
            best_params = (W, C1, C2)

    result = objective_function(best_config, return_details=True)
    _, electricity_sold, battery_activity, hourly_production, rel = result

    print("\n Best PSO configuration found:")
    print(f"  Parameters: W={best_params[0]:.2f}, C1={best_params[1]:.2f}, C2={best_params[2]:.2f}")
    print(f"  Decision x: {best_config}")
    print(f"  Max Net Annual Profit: {best_score:.2f} MUSD")

    print(f"  Reliability: {rel/hours_in_year}")

    plt.figure(figsize=(15, 6))

    # Electricity sold
    plt.plot(electricity_sold / 1e3, label="Electricity Sold (MW)", color="blue", linewidth=1)

    # Price threshold line
    plt.axhline(y=best_config[3], color="red", linestyle="--", label=f"Price Threshold (${best_config[3]})")

    # Electricity price overlay (scaled down for visual clarity)
    scaled_price = electricity_price / max(electricity_price) * max(electricity_sold / 1e3)
    plt.plot(scaled_price, color="orange", alpha=0.5, label="Scaled Electricity Price")

    # Battery activity shading
    charging = battery_activity == -1
    discharging = battery_activity == 1
    plt.fill_between(np.arange(hours_in_year), 0, max(electricity_sold / 1e3), where=charging,
                 color="green", alpha=0.1, label="Charging")
    plt.fill_between(np.arange(hours_in_year), 0, max(electricity_sold / 1e3), where=discharging,
                 color="purple", alpha=0.1, label="Discharging")

    plt.title("Electricity Sold, Battery Operation & Price Threshold")
    plt.xlabel("Hour")
    plt.ylabel("Electricity Sold (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("electricity_sold_detailed.png", dpi=300)
    plt.close()

