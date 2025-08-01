# system_config.py

hours_in_year = 8760

SYSTEMS = {
    "SMR": {
        "capacity": 77000,  # kW
        "availability": 0.95,
        "lifetime": 60,
        "capital": 6000,  # $/kW
        "O&M": 25e-3 * hours_in_year * 0.95  # $/kW-year (hourly * availability)
    },
    "wind": {
        "capacity": 2000,
        "lifetime": 25,
        "capital": 1600,
        "O&M": 30
    },
    "solar": {
        "capacity": 2250,
        "lifetime": 25,
        "capital": 1500,
        "O&M": 20
    },
    "battery": {
        "capacity": 1000,
        "lifetime": 15,
        "capital": 400,
        "O&M": 10,
        "efficiency": 0.9,
        "charge_rate": 1000 / 4,
        "discharge_rate": 1000 / 4
    },
    "electrolyzer": {
        "capacity": 1000,  # kW
        "lifetime": 20,
        "capital": 1800,
        "O&M": 0.05 * 1800,
        "efficiency": 0.57
    },
    "hydrogen_storage": {
        "capacity": 100e6,  # kg
        "lifetime": 10,
        "capital": 0,
        "O&M": 0
    }
}

# Other constants
discount_rate = 0.05
project_lifetime = 60
electricity_to_hydrogen = 55 
hydrogen_price = 7  # USD/kg
cap_obj = 600000  # Target capacity in kW
