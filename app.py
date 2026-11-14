
from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

app = Flask(__name__)

# ---------------- AI MODEL TRAINING ----------------

def build_models():
    rng = np.random.default_rng(42)
    n = 7000

    hour = rng.uniform(0, 24, n)
    weekend = rng.integers(0, 2, n)
    temp = rng.uniform(15, 42, n)
    previous_load = rng.uniform(0.25, 4.0, n)

    morning = np.exp(-0.5 * ((hour - 8) / 1.8) ** 2)
    evening = np.exp(-0.5 * ((hour - 20) / 2.2) ** 2)

    load_target = (
        0.55 + 0.9 * morning + 1.45 * evening
        + 0.035 * np.maximum(temp - 25, 0)
        + 0.16 * weekend + 0.18 * previous_load
        + rng.normal(0, 0.12, n)
    )

    load_target = np.clip(load_target, 0.25, 5.5)

    X_load = np.column_stack([
        hour, weekend, temp, previous_load
    ])

    load_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    load_model.fit(X_load, load_target)

    irradiance = rng.uniform(0, 1000, n)
    panel_temp = rng.uniform(10, 65, n)
    prev_pv_per_kw = rng.uniform(0, 1.0, n)

    daylight = np.clip(
        np.sin(np.pi * (hour - 6) / 12), 0, None
    )

    pv_target = (
        (irradiance / 1000) * daylight
        * (1 - 0.004 * (panel_temp - 25))
    )

    pv_target = np.clip(
        pv_target + 0.08 * prev_pv_per_kw
        + rng.normal(0, 0.025, n),
        0, 1.05
    )

    X_pv = np.column_stack([
        hour, irradiance, panel_temp, prev_pv_per_kw
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


# ---------------- INPUT VALIDATION ----------------

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


# ---------------- ONE-HOUR ENERGY MANAGEMENT ----------------

def manage_one_hour(houses, battery_energy, battery_capacity,
                    max_battery_kw, grid_tariff,
                    export_tariff, p2p_tariff):

    total_surplus = sum(h["surplus"] for h in houses)
    total_deficit = sum(h["deficit"] for h in houses)

    sellers = [
        {"idx": i, "left": h["surplus"]}
        for i, h in enumerate(houses)
    ]

    buyers = [
        {"idx": i, "left": h["deficit"]}
        for i, h in enumerate(houses)
    ]

    trades = []

    # Match houses for P2P energy sharing
    for buyer in buyers:
        for seller in sellers:

            if buyer["left"] <= 1e-9:
                break

            if seller["left"] <= 1e-9:
                continue

            if seller["idx"] == buyer["idx"]:
                continue

            amount = min(buyer["left"], seller["left"])

            if amount > 1e-9:
                trades.append({
                    "from": f"House {seller['idx'] + 1}",
                    "to": f"House {buyer['idx'] + 1}",
                    "energy": amount
                })

                buyer["left"] -= amount
                seller["left"] -= amount

    p2p_total = sum(t["energy"] for t in trades)

    remaining_surplus = sum(s["left"] for s in sellers)
    remaining_deficit = sum(b["left"] for b in buyers)

    # Shared battery management
    max_charge = min(
        max_battery_kw,
        battery_capacity - battery_energy
    )

    max_discharge = min(
        max_battery_kw,
        battery_energy
    )

    battery_charge = min(remaining_surplus, max_charge)
    battery_discharge = min(remaining_deficit, max_discharge)

    grid_export = max(0, remaining_surplus - battery_charge)
    grid_import = max(0, remaining_deficit - battery_discharge)

    new_battery_energy = (
        battery_energy + battery_charge - battery_discharge
    )

    new_soc = 100 * new_battery_energy / battery_capacity

    # Hourly cost calculations
    grid_cost = grid_import * grid_tariff
    export_revenue = grid_export * export_tariff
    p2p_payment = p2p_total * p2p_tariff

    return {
        "trades": trades,
        "p2p_total": p2p_total,
        "battery_charge": battery_charge,
        "battery_discharge": battery_discharge,
        "grid_import": grid_import,
        "grid_export": grid_export,
        "old_soc": 100 * battery_energy / battery_capacity,
        "new_soc": new_soc,
        "grid_cost": grid_cost,
        "export_revenue": export_revenue,
        "p2p_payment": p2p_payment,
        "net_grid_cost": grid_cost - export_revenue,
        "new_battery_energy": new_battery_energy
    }


# ---------------- 24-HOUR FORECAST ----------------

def run_energy_management(form):

    start_hour = int(number(form, "hour", 18, 0, 23))

    weekend = int(number(form, "weekend", 0, 0, 1))
    ambient = number(form, "ambient_temp", 30, 0, 50)

    # This is treated as the assumed peak/daylight irradiance
    peak_irradiance = number(form, "irradiance", 600, 0, 1200)

    panel_temp = number(form, "panel_temp", 40, -10, 90)
    previous_pv = number(form, "previous_pv", 0.5, 0, 1.2)

    battery_capacity = number(
        form, "battery_capacity", 10, 0.1, 100
    )

    initial_soc = number(form, "soc", 50, 0, 100)

    max_battery_kw = number(
        form, "battery_power", 3, 0, 50
    )

    grid_tariff = number(form, "grid_tariff", 8, 0, 100)
    export_tariff = number(form, "export_tariff", 3, 0, 100)
    p2p_tariff = number(form, "p2p_tariff", 7, 0, 100)

    # House-specific settings
    house_settings = []

    for i in range(1, 4):

        prev_load = number(
            form, f"h{i}_prev_load",
            [1.2, 1.0, 1.4][i-1], 0, 10
        )

        pv_capacity = number(
            form, f"h{i}_pv_capacity",
            [3, 2, 0][i-1], 0, 20
        )

        house_settings.append({
            "previous_load": prev_load,
            "pv_capacity": pv_capacity,
            "previous_pv": previous_pv
        })

    # Initial shared battery energy
    battery_energy = battery_capacity * initial_soc / 100

    hourly_results = []
    all_trades = []

    daily = {
        "total_load": 0,
        "total_pv": 0,
        "p2p_total": 0,
        "battery_charge": 0,
        "battery_discharge": 0,
        "grid_import": 0,
        "grid_export": 0,
        "grid_cost": 0,
        "export_revenue": 0,
        "p2p_payment": 0
    }

    # Generate 24 hourly predictions
    for step in range(24):

        forecast_hour = (start_hour + step) % 24

        houses = []

        for i, settings in enumerate(house_settings):

            prev_load = settings["previous_load"]
            pv_capacity = settings["pv_capacity"]
            prev_pv = settings["previous_pv"]

            # AI load prediction
            load_features = np.array([[
                forecast_hour,
                weekend,
                ambient,
                prev_load
            ]])

            predicted_load = max(
                0,
                float(LOAD_MODEL.predict(load_features)[0])
            )

            # AI solar prediction
            pv_features = np.array([[
                forecast_hour,
                peak_irradiance,
                panel_temp,
                prev_pv
            ]])

            predicted_pv_per_kw = max(
                0,
                float(PV_MODEL.predict(pv_features)[0])
            )

            predicted_pv = min(
                pv_capacity,
                predicted_pv_per_kw * pv_capacity
            )

            balance = predicted_pv - predicted_load

            houses.append({
                "name": f"House {i + 1}",
                "load": predicted_load,
                "pv": predicted_pv,
                "balance": balance,
                "surplus": max(0, balance),
                "deficit": max(0, -balance),
                "pv_capacity": pv_capacity
            })

            # Use the predicted value as the next hour's previous value
            settings["previous_load"] = predicted_load
            settings["previous_pv"] = predicted_pv_per_kw

        # Manage energy for this hour
        management = manage_one_hour(
            houses,
            battery_energy,
            battery_capacity,
            max_battery_kw,
            grid_tariff,
            export_tariff,
            p2p_tariff
        )

        battery_energy = management["new_battery_energy"]

        # Hour label
        hour_label = datetime.strptime(
            f"{forecast_hour:02d}", "%H"
        ).strftime("%I %p").lstrip("0")

        # Save hourly results
        row = {
            "hour": forecast_hour,
            "time": hour_label,
            "houses": houses,
            **management
        }

        hourly_results.append(row)

        for trade in management["trades"]:
            all_trades.append({
                "time": hour_label,
                **trade
            })

        # Add to daily totals
        daily["total_load"] += sum(h["load"] for h in houses)
        daily["total_pv"] += sum(h["pv"] for h in houses)

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

    # Final daily summary
    daily["net_grid_cost"] = (
        daily["grid_cost"] - daily["export_revenue"]
    )

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
        "new_soc": 100 * battery_energy / battery_capacity,
        "grid_cost": daily["grid_cost"],
        "export_revenue": daily["export_revenue"],
        "p2p_payment": daily["p2p_payment"],
        "net_grid_cost": daily["net_grid_cost"],
        "grid_tariff": grid_tariff,
        "export_tariff": export_tariff,
        "p2p_tariff": p2p_tariff,
        "hour": start_hour,
        "irradiance": peak_irradiance
    }


# ---------------- WEBSITE ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":
        result = run_energy_management(request.form)

    return render_template(
        "index.html",
        result=result,
        now=datetime.now()
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "models": ["load_forecast", "pv_forecast"],
        "forecast_hours": 24
    }


if __name__ == "__main__":
    app.run(debug=True)
