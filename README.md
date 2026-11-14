# Three-House Smart PV-Battery AI Energy Management Website

A beginner-friendly Flask simulation for three houses. It uses two machine-learning models:
1. Household load prediction (Random Forest Regression).
2. Solar PV output prediction (Random Forest Regression).

The predicted values are passed to a simple energy-management calculation that:
- identifies surplus and deficit houses;
- matches available surplus with deficit through simulated P2P trading;
- charges/discharges a shared battery within its power and capacity limits;
- calculates remaining grid import/export and indicative energy costs.

## Important limitations
- The website trains on synthetic demonstration data generated in `app.py`. These are not measured results and are not validated forecasts.
- The simulation uses a one-hour interval. Therefore, kW values are numerically treated as kWh for that interval.
- The battery is modelled as one shared community battery, not three separate batteries.
- The P2P price is an assumed simulation input. This is not a real electricity-market settlement.
- This is an academic simulation only. Do not connect it to real electrical equipment or use it for protection/control.

## Requirements
- Windows 10/11
- Python 3.11, 3.12, or 3.13 (64-bit recommended)

## Windows setup (PowerShell)
1. Extract the ZIP file.
2. Open the extracted `three_house_ai_energy_website` folder.
3. Click the folder address bar, type `powershell`, and press Enter.
4. Run these commands one at a time:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

Using the virtual-environment Python directly avoids the PowerShell script-activation restriction. You do not need to run `venv\Scripts\activate`.

5. Open this address in Chrome: http://127.0.0.1:5000

To stop the server, click the PowerShell window and press Ctrl+C.

## Inputs
- Hour and weekday/weekend flag
- Ambient and PV panel temperature
- Solar irradiance
- Previous load for each house
- PV capacity for each house
- Shared battery capacity, initial state of charge, and charge/discharge power
- Grid, export, and P2P tariffs

Default example: House 1 has 3 kWp PV, House 2 has 2 kWp PV, House 3 has no PV. You can change these capacities.

## Connecting Simulink data later
The next development step is to export time-series data from MATLAB/Simulink to CSV. A useful dataset should include timestamp/hour, house ID, measured load (kW), PV output (kW), irradiance (W/m²), panel temperature (°C), ambient temperature (°C), and previous-step values. Train and test the models on separate time periods and report MAE/RMSE before claiming prediction accuracy. The current prototype does not yet load a Simulink CSV automatically.
