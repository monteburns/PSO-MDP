import matplotlib.pyplot as plt
import numpy as np

def plot_sample_day_grid(days, smr_gen, wind_gen, solar_gen, total_gen, electricity_sold, battery_soc, demand, electricity_price, elect_to_h2):
    """
    Plots power generation and demand for selected days in a 2x2 grid.

    Parameters:
        days (list): List of 4 day indices (0-based, e.g., 15 = Jan 16).
        smr_gen (np.ndarray): Hourly SMR generation array (8760,).
        wind_gen (np.ndarray): Hourly wind generation array (8760,).
        solar_gen (np.ndarray): Hourly solar generation array (8760,).
        total_gen (np.ndarray): Hourly total generation array (8760,).
        demand (np.ndarray): Hourly demand array (8760,).
        electricity price(np.ndarray):
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
        axs[idx].plot(hours, elect_to_h2[start:end], label="Electricty to H2", linewidth=2)
        axs[idx].plot(hours, total_gen[start:end], label="Total Generation", linestyle='--', linewidth=2)
        axs[idx].plot(hours, electricity_sold[start:end], label="Electricity Sold", linestyle='-.', linewidth=2)
        axs[idx].plot(hours, demand[start:end], label="Demand", linestyle=':', linewidth=2)

        ax2[idx] = axs[idx].twinx()
        ax2[idx].plot(hours, electricity_price[start:end], label="Electricity Price", linewidth=1)
        ax2[idx].set_ylabel("USD/MWh")

        axs[idx].set_title(f"Day {day + 1}", fontsize=12, weight='bold')
        axs[idx].set_xlabel("Hour of Day")
        axs[idx].set_ylabel("Power (kW)")
        axs[idx].grid(True, linestyle='--', alpha=0.6)
        axs[idx].legend(fontsize=8)

    plt.suptitle("Power Generation and Demand on Sample Days", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("sampledays.png", dpi=300)
    plt.close()

def plot_battery_charge_discharge_days(days, battery_charge, battery_discharge, battery_soc):
    """
    Plots battery charge, discharge, and SOC for selected days in a 2x2 grid.

    Parameters:
        days (list): List of 4 day indices (0-based).
        battery_charge (np.ndarray): Hourly charge values (8760,).
        battery_discharge (np.ndarray): Hourly discharge values (8760,).
        battery_soc (np.ndarray): Hourly state of charge (8760,).
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
    axs = axs.flatten()

    for idx, day in enumerate(days):
        start = day * 24
        end = start + 24
        hours = np.arange(24)

        ax1 = axs[idx]
        ax2 = ax1.twinx()

        ax1.plot(hours, battery_charge[start:end], label='Charge', color='tab:blue', alpha=0.6)
        ax1.plot(hours, battery_discharge[start:end], label='Discharge', color='tab:orange', alpha=0.6)
        ax2.plot(hours, battery_soc[start:end], label='SOC', color='tab:green', linestyle='--', linewidth=2)

        ax1.set_title(f"Day {day + 1}")
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Charge/Discharge [kWh]")
        ax2.set_ylabel("SOC [kWh]")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[idx].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.grid(True)

    plt.tight_layout()
    plt.savefig("Battery_Charge_Discharge_SOC_4Days.png", dpi=300)
    plt.close()


def plot_battery_week_profile(start_day, battery_charge, battery_discharge, battery_soc):
    """
    Plots battery charge (positive), discharge (negative), and SOC on the same Y axis for 1 week.

    Parameters:
        start_day (int): Index of the starting day (0-based).
        battery_charge (np.ndarray): Hourly charge values (8760,).
        battery_discharge (np.ndarray): Hourly discharge values (8760,).
        battery_soc (np.ndarray): Hourly state of charge (8760,).
    """
    start = start_day * 24
    end = start + 168
    hours = np.arange(168)

    charge = battery_charge[start:end]
    discharge = -battery_discharge[start:end]  # Negatif değerler
    soc = battery_soc[start:end]

    fig, ax = plt.subplots(figsize=(16, 5))

    # Dolgu alanları: şarj (+), deşarj (-)
    ax.fill_between(hours, 0, charge, where=charge > 0, 
                    color='tab:blue', alpha=0.4, label='Charge [+]')
    ax.fill_between(hours, 0, discharge, where=discharge < 0, 
                    color='tab:orange', alpha=0.4, label='Discharge [-]')

    # SOC çizgisi
    ax.plot(hours, soc, color='tab:green', linestyle='--', linewidth=2, label='SOC')

    ax.set_xlabel("Hour (from Day {})".format(start_day + 1))
    ax.set_ylabel("Energy [kWh]")
    ax.set_title("Battery Operation Profile (1 Week, Day {}–{})".format(start_day + 1, start_day + 7))

    ax.legend(loc='upper left')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("Battery_Week_Profile_NegativeDischarge.png", dpi=300)
    plt.close()


def plot_energy_week_profile(start_day, battery_charge, battery_discharge, battery_soc,
                              hourly_demand, electricity_sold, electricity_price, hourly_production):
    """
    Plots battery charge, discharge, SOC, demand, electricity sold, production, and electricity price over 1 week.

    Parameters:
        start_day (int): Starting day index (0-based).
        battery_charge (np.ndarray): Hourly charge values (8760,).
        battery_discharge (np.ndarray): Hourly discharge values (8760,).
        battery_soc (np.ndarray): SOC values (8760,).
        hourly_demand (np.ndarray): Hourly electricity demand (8760,).
        electricity_sold (np.ndarray): Hourly electricity sold to grid (8760,).
        electricity_price (np.ndarray): Hourly electricity price (8760,).
        hourly_production (np.ndarray): Hourly electricity production (8760,).
    """
    start = start_day * 24
    end = start + 24*5
    hours = np.arange(24*5)

    charge = battery_charge[start:end]
    discharge = -battery_discharge[start:end]
    soc = battery_soc[start:end]
    demand = hourly_demand[start:end]
    sold = electricity_sold[start:end]
    price = electricity_price[start:end]
    production = hourly_production[start:end]

    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax2 = ax1.twinx()

    # Fill charge/discharge
    ax1.fill_between(hours, 0, charge, where=charge > 0, color='tab:blue', alpha=0.3, label='Charge [+]')
    ax1.fill_between(hours, 0, discharge, where=discharge < 0, color='tab:orange', alpha=0.3, label='Discharge [-]')

    # Line plots on primary axis
    ax1.plot(hours, soc, color='tab:green', linestyle='--', linewidth=2, label='SOC [kWh]')
    ax1.plot(hours, demand, color='black', linewidth=1.2, label='Demand [kWh]')
    ax1.plot(hours, sold, color='tab:purple', linewidth=1.2, label='Electricity Sold [kWh]')
    ax1.plot(hours, production, color='tab:gray', linewidth=1.2, label='Production [kWh]')

    # Price on secondary axis
    ax2.plot(hours, price, color='red', linestyle=':', linewidth=2, label='Electricity Price [$/MWh]')

    # Axes labels
    ax1.set_xlabel(f"Hour (from Day {start_day + 1})")
    ax1.set_ylabel("Energy [kWh]")
    ax2.set_ylabel("Electricity Price [$/MWh]")
    ax1.set_title(f"Battery and System Profile (1 Week, Day {start_day + 1}–{start_day + 5})")

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig("Week_Profile_All_Layers.png", dpi=300)
    plt.close()


    
    