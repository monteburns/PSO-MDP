def optimize_dispatch_with_hydrogen(
    demand, 
    total_gen, 
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
    CVXPY-based convex optimization for energy dispatch with hydrogen integration.
    
    Returns:
        dict: Dispatch decisions and updated storage values.
    """
    # Inputs

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
# Hydrogen dispatch worker function

def parallel_dispatch_worker_h2(day_idx, num_battery, battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency,
                                 hydrogen_capacity, electrolyzer_capacity, electrolyzer_efficiency, soc, h2_init):
    global _shared_hourly_production, _shared_hourly_demand, _shared_price

    start = day_idx * 24
    end = start + 24

    daily_gen = _shared_hourly_production[start:end]
    daily_demand = _shared_hourly_demand[start:end]
    daily_price = _shared_price[start:end]

    try:
        result = optimize_dispatch_with_hydrogen(
            daily_demand, daily_gen,  # Pass full gen as solar only
            soc, h2_init,
            battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency,
            hydrogen_capacity, electrolyzer_capacity, electrolyzer_efficiency,
            daily_price, hydrogen_price
        )
    except Exception:
        result = None

    if result is not None:
        return (start, end, result["grid_export"], result["battery_soc"][1:], result["hydrogen_production"], result["battery_charge"], result["battery_discharge"], result["hydrogen_storage"][-1], result["battery_soc"][-1])
    else:
        return (start, end, np.zeros(24), np.full(24, soc), np.zeros(24), np.zeros(24), np.zeros(24), h2_init, soc)


def parallel_daily_dispatch_h2(hourly_production, hourly_demand, electricity_price, num_battery,
                                battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency,
                                hydrogen_capacity, electrolyzer_capacity, electrolyzer_efficiency):

    global _shared_hourly_production, _shared_hourly_demand, _shared_price

    _shared_hourly_production = hourly_production
    _shared_hourly_demand = hourly_demand
    _shared_price = electricity_price

    electricity_sold = np.zeros_like(hourly_production)
    hydrogen_production = np.zeros_like(hourly_production)
    battery_soc_history = np.zeros_like(hourly_production)
    battery_charge = np.zeros_like(hourly_production)
    battery_discharge = np.zeros_like(hourly_production)

    soc = num_battery * battery_capacity / 2
    h2_init = hydrogen_capacity / 2

    args_list = [
        (day, num_battery, battery_capacity, battery_c_rate, battery_dc_rate, battery_efficiency,
         hydrogen_capacity, electrolyzer_capacity, electrolyzer_efficiency, soc, h2_init)
        for day in range(365)
    ]

    with mp.get_context("fork").Pool(get_available_cpus(64)) as pool:
        results = pool.starmap(parallel_dispatch_worker_h2, args_list)

    for start, end, s, soc_day, h2_prod, c, d, h2_last, soc in results:
        electricity_sold[start:end] = s
        battery_soc_history[start:end] = soc_day
        hydrogen_production[start:end] = h2_prod
        battery_charge[start:end] = c
        battery_discharge[start:end] = d

    return electricity_sold, hydrogen_production, battery_soc_history, battery_charge, battery_discharge


def objective_function_hydrogen(x, return_details=False):
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

    electricity_sold, hydrogen_produced, battery_soc_history, battery_charge, battery_discharge = parallel_daily_dispatch_h2(
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
        SYSTEMS["electrolyzer"]["efficiency"]
    )

    penalty = np.mean(np.abs(electricity_sold - hourly_demand) / hourly_demand)
    #if penalty > 0.15:
    #    return 1e9

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

    profit = annual_revenue + hydrogen_revenue - (annual_om_cost + annual_capex)

    return -profit + 1000*penalty if not return_details else (
        profit,
        battery_soc_history,
        hourly_demand,
        electricity_sold,
        hydrogen_produced,
        annual_revenue,
        hydrogen_revenue,
        annual_om_cost,
        penalty,
        battery_charge,
        battery_discharge
    )

def optimized_dispatch(daily_gen, daily_demand, daily_price, battery_capacity, battery_c_rate, battery_dc_rate, battery_eff, initial_soc):
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
        c, d, socs = optimized_dispatch(
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
    if penalty > 0.15:
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
