
from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

app = Flask(__name__)

# =====================================================
# 1. AI MODEL TRAINING
# =====================================================

def build_models():

    rng = np.random.default_rng(42)
    n = 7000

    # ---------------- LOAD PREDICTION MODEL ----------------

    hour = rng.uniform(0, 24, n)
    weekend = rng.integers(0, 2, n)
    temperature = rng.uniform(15, 42, n)
    previous_load = rng.uniform(0.25, 4.0, n)

    morning_peak = np.exp(-0.5 * ((hour - 8) / 1.8) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 20) / 2.2) ** 2)

    load_target = (
        0.55
        + 0.9 * morning_peak
        + 1.45 * evening_peak
        + 0.035 * np.maximum(temperature - 25, 0)
        + 0.16 * weekend
        + 0.18 * previous_load
        + rng.normal(0, 0.12, n)
    )

    load_target = np.clip(load_target, 0.25, 5.5)

    X_load = np.column_stack([
        hour,
        weekend,
        temperature,
        previous_load
    ])

    load_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    load_model.fit(X_load, load_target)

    # ---------------- SOLAR PV PREDICTION MODEL ----------------

    pv_hour = rng.uniform(0, 24, n)
    panel_temperature = rng.uniform(10, 65, n)

    # Generate daylight-dependent irradiance
    daylight = np.clip(
        np.sin(np.pi * (pv_hour - 6) / 12),
        0,
        None
    )

    clear_sky_irradiance = rng.uniform(200, 1000, n)
    irradiance = clear_sky_irradiance * daylight

    pv_target = (
        (irradiance / 1000)
        * (1 - 0.004 * (panel_temperature - 25))
        + rng.normal(0, 0.02, n)
    )

    pv_target = np.clip(pv_target, 0, 1.05)

    X_pv = np.column_stack([
        pv_hour,
        irradiance,
        panel_temperature
    ])

    pv_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=43,
        n_jobs=-1
    )

    pv_model.fit(X_pv, pv_target)

    return load_model, pv_model


LOAD_MODEL, PV_MODEL = build_models()


# =====================================================
# 2. INPUT VALIDATION
# =====================================================

def number(form, key, default, minimum=None, maximum=None):

    try:
        value = float(form.get(key, default))
    except (TypeError, ValueError):
        value = float(default)

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


# =====================================================
# 3. HOURLY SOLAR IRRADIANCE PROFILE
# =====================================================

def get_hourly_irradiance(hour, peak_irradiance):

    # No solar generation during nighttime
    if hour < 6 or hour >= 18:
        return 0.0

    daylight_factor = np.sin(
        np.pi * (hour - 6) / 12
    )

    return peak_irradiance * max(0, daylight_factor)


# =====================================================
# 4. P2P TRADING AND BATTERY MANAGEMENT
# =====================================================

def manage_one_hour(
    houses,
    battery_energy,
    battery_capacity,
    max_battery_kw,
    grid_tariff,
    export_tariff,
    p2p_tariff
):

    # Calculate total surplus and deficit
    sellers = [
        {"idx": i, "left": house["surplus"]}
        for i, house in enumerate(houses)
    ]

    buyers = [
        {"idx": i, "left": house["deficit"]}
        for i, house in enumerate(houses)
    ]

    trades = []

    # ---------------- P2P ENERGY MATCHING ----------------

    for buyer in buyers:

        for seller in sellers:

            if buyer["left"] <= 1e-9:
                break

            if seller["left"] <= 1e-9:
                continue

            if seller["idx"] == buyer["idx"]:
                continue

            energy = min(
                buyer["left"],
                seller["left"]
            )

            if energy > 1e-9:

                trades.append({
                    "from": f"House {seller['idx'] + 1}",
                    "to": f"House {buyer['idx'] + 1}",
                    "energy": energy
                })

                buyer["left"] -= energy
                seller["left"] -= energy

    p2p_total = sum(
        trade["energy"] for trade in trades
    )

    remaining_surplus = sum(
        seller["left"] for seller in sellers
    )

    remaining_deficit = sum(
        buyer["left"] for buyer in buyers
    )

    # ---------------- BATTERY MANAGEMENT ----------------

    # Assumed battery efficiencies
    charge_efficiency = 0.95
    discharge_efficiency = 0.95

    old_soc = (
        100 * battery_energy / battery_capacity
    )

    # Maximum energy that can be accepted by battery
    max_charge = min(
        max_battery_kw,
        (battery_capacity - battery_energy)
        / charge_efficiency
    )

    # Maximum energy that can be supplied by battery
    max_discharge = min(
        max_battery_kw,
        battery_energy * discharge_efficiency
    )

    battery_charge = min(
        remaining_surplus,
        max_charge
    )

    battery_discharge = min(
        remaining_deficit,
        max_discharge
    )

    # Update stored battery energy
    new_battery_energy = (
        battery_energy
        + battery_charge * charge_efficiency
        - battery_discharge / discharge_efficiency
    )

    new_battery_energy = np.clip(
        new_battery_energy,
        0,
        battery_capacity
    )

    new_soc = (
        100 * new_battery_energy / battery_capacity
    )

    # ---------------- GRID INTERACTION ----------------

    grid_export = max(
        0,
        remaining_surplus - battery_charge
    )

    grid_import = max(
        0,
        remaining_deficit - battery_discharge
    )

    # ---------------- COST CALCULATIONS ----------------

    grid_cost = grid_import * grid_tariff

    export_revenue = (
        grid_export * export_tariff
    )

    p2p_payment = p2p_total * p2p_tariff

    net_grid_cost = (
        grid_cost - export_revenue
    )

    return {

        "trades": trades,

        "p2p_total": p2p_total,

        "battery_charge": battery_charge,

        "battery_discharge": battery_discharge,

        "grid_import": grid_import,

        "grid_export": grid_export,

        "old_soc": old_soc,

        "new_soc": new_soc,

        "battery_energy": new_battery_energy,

        "grid_cost": grid_cost,

        "export_revenue": export_revenue,

        "p2p_payment": p2p_payment,

        "net_grid_cost": net_grid_cost,

        "new_battery_energy": new_battery_energy
    }


# =====================================================
# 5. 24-HOUR AI ENERGY FORECAST AND MANAGEMENT
# =====================================================

def run_energy_management(form):

    # ---------------- GENERAL INPUTS ----------------

    india_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    start_hour = int(number(
        form,
        "hour",
        india_time.hour,
        0,
        23
    ))

    weekend = int(number(
        form, "weekend", 0, 0, 1
    ))

    ambient = number(
        form, "ambient_temp", 30, 0, 50
    )

    peak_irradiance = number(
        form, "irradiance", 600, 0, 1200
    )

    panel_temp = number(
        form, "panel_temp", 40, -10, 90
    )

    # ---------------- BATTERY INPUTS ----------------

    battery_capacity = number(
        form, "battery_capacity", 10, 0.1, 100
    )

    initial_soc = number(
        form, "soc", 50, 0, 100
    )

    max_battery_kw = number(
        form, "battery_power", 3, 0, 50
    )

    # ---------------- ELECTRICITY TARIFFS ----------------

    grid_tariff = number(
        form, "grid_tariff", 8, 0, 100
    )

    export_tariff = number(
        form, "export_tariff", 3, 0, 100
    )

    p2p_tariff = number(
        form, "p2p_tariff", 7, 0, 100
    )

    # ---------------- THREE HOUSE SETTINGS ----------------

    house_settings = []

    for i in range(1, 4):

        previous_load = number(
            form,
            f"h{i}_prev_load",
            [1.2, 1.0, 1.4][i - 1],
            0,
            10
        )

        pv_capacity = number(
            form,
            f"h{i}_pv_capacity",
            [3, 2, 0][i - 1],
            0,
            20
        )

        house_settings.append({

            "previous_load": previous_load,

            "pv_capacity": pv_capacity

        })

    # ---------------- INITIAL BATTERY ENERGY ----------------

    battery_energy = (
        battery_capacity * initial_soc / 100
    )

    # ---------------- DAILY SUMMARY ----------------

    daily = {

        "total_load": 0.0,

        "total_pv": 0.0,

        "p2p_total": 0.0,

        "battery_charge": 0.0,

        "battery_discharge": 0.0,

        "grid_import": 0.0,

        "grid_export": 0.0,

        "grid_cost": 0.0,

        "export_revenue": 0.0,

        "p2p_payment": 0.0
    }

    hourly_results = []

    all_trades = []

    # Starting date and time in India
    forecast_start = india_time.replace(
        hour=start_hour,
        minute=0,
        second=0,
        microsecond=0
    )

    # ---------------- 24-HOUR FORECAST LOOP ----------------

    for step in range(24):

        forecast_datetime = (
            forecast_start + timedelta(hours=step)
        )

        forecast_hour = forecast_datetime.hour

        # Calculate hourly solar irradiance
        hourly_irradiance = get_hourly_irradiance(
            forecast_hour,
            peak_irradiance
        )

        houses = []

        # ---------------- PREDICT EACH HOUSE ----------------

        for i, settings in enumerate(house_settings):

            previous_load = settings["previous_load"]

            pv_capacity = settings["pv_capacity"]

            # AI load prediction
            load_features = np.array([[

                forecast_hour,

                weekend,

                ambient,

                previous_load

            ]])

            predicted_load = max(
                0,
                float(
                    LOAD_MODEL.predict(load_features)[0]
                )
            )

            # AI solar PV prediction
            pv_features = np.array([[

                forecast_hour,

                hourly_irradiance,

                panel_temp

            ]])

            predicted_pv_per_kw = max(
                0,
                float(
                    PV_MODEL.predict(pv_features)[0]
                )
            )

            predicted_pv = min(
                pv_capacity,
                predicted_pv_per_kw * pv_capacity
            )

            # Ensure zero solar generation at night
            if forecast_hour < 6 or forecast_hour >= 18:
                predicted_pv = 0.0

            # Energy balance for this house
            balance = predicted_pv - predicted_load

            house = {

                "name": f"House {i + 1}",

                "load": predicted_load,

                "pv": predicted_pv,

                "balance": balance,

                "surplus": max(0, balance),

                "deficit": max(0, -balance),

                "pv_capacity": pv_capacity

            }

            houses.append(house)

            # Use the predicted load for the next hour's
            # previous-load input in this simulation
            settings["previous_load"] = predicted_load

        # ---------------- ENERGY MANAGEMENT ----------------

        management = manage_one_hour(

            houses,

            battery_energy,

            battery_capacity,

            max_battery_kw,

            grid_tariff,

            export_tariff,

            p2p_tariff

        )

        # Carry battery energy into the next hour
        battery_energy = management["new_battery_energy"]

        # ---------------- TIME LABEL ----------------

        hour_label = forecast_datetime.strftime("%I %p").lstrip("0")

        date_label = forecast_datetime.strftime("%d %b %Y")

        # ---------------- SAVE HOURLY RESULTS ----------------

        row = {

            "hour": forecast_hour,

            "time": hour_label,

            "date": date_label,

            "datetime": forecast_datetime.strftime(
                "%Y-%m-%d %H:%M"
            ),

            "irradiance": hourly_irradiance,

            "houses": houses,

            **management

        }

        hourly_results.append(row)

        # Save individual P2P transactions
        for trade in management["trades"]:

            all_trades.append({

                "time": hour_label,

                "date": date_label,

                **trade

            })

        # ---------------- DAILY ENERGY TOTALS ----------------

        daily["total_load"] += sum(
            house["load"] for house in houses
        )

        daily["total_pv"] += sum(
            house["pv"] for house in houses
        )

        for key in [

            "p2p_total",

            "battery_charge",

            "battery_discharge",

            "grid_import",

            "grid_export",

            "grid_cost",

            "export_revenue",

            "p2p_payment"

        ]:

            daily[key] += management[key]

    # ---------------- FINAL DAILY SUMMARY ----------------

    daily["net_grid_cost"] = (
        daily["grid_cost"] - daily["export_revenue"]
    )

    # ---------------- FINAL RESULT ----------------

    return {

        "hourly_results": hourly_results,

        "houses": hourly_results[-1]["houses"],

        "trades": all_trades,

        "total_load": daily["total_load"],

        "total_pv": daily["total_pv"],

        "p2p_total": daily["p2p_total"],

        "battery_charge": daily["battery_charge"],

        "battery_discharge": daily["battery_discharge"],

        "grid_import": daily["grid_import"],

        "grid_export": daily["grid_export"],

        "old_soc": initial_soc,

        "new_soc": (
            100 * battery_energy / battery_capacity
        ),

        "final_battery_energy": battery_energy,

        "grid_cost": daily["grid_cost"],

        "export_revenue": daily["export_revenue"],

        "p2p_payment": daily["p2p_payment"],

        "net_grid_cost": daily["net_grid_cost"],

        "grid_tariff": grid_tariff,

        "export_tariff": export_tariff,

        "p2p_tariff": p2p_tariff,

        "hour": start_hour,

        "irradiance": peak_irradiance,

        "forecast_start": forecast_start.strftime(
            "%d %b %Y, %I %p"
        ),

        "forecast_end": (
            forecast_start + timedelta(hours=23)
        ).strftime("%d %b %Y, %I %p")

    }


# =====================================================
# 6. WEBSITE ROUTES
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    india_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    if request.method == "POST":

        result = run_energy_management(request.form)

    return render_template(

        "index.html",

        result=result,

        now=india_time

    )


# =====================================================
# 7. APPLICATION HEALTH CHECK
# =====================================================

@app.route("/health")
def health():

    return {

        "status": "ok",

        "models": [

            "load_forecast",

            "pv_forecast"

        ],

        "houses": 3,

        "forecast_hours": 24,

        "timezone": "Asia/Kolkata"

    }


# =====================================================
# 8. RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)
