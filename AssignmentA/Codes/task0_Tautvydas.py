# Import necessary libraries
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
from data import get_fixed_data
from PriceProcess import price_model  # Import the price model
from WindProcess import wind_model 

# Retrieve Data
data = get_fixed_data()
T = data['num_timeslots']
D = data['demand_schedule']
C = data['hydrogen_capacity']
P2H = data['p2h_max_rate']
H2P = data['h2p_max_rate']
R_p2h = data['conversion_p2h']
R_h2p = data['conversion_h2p']
C_elzr = data['electrolyzer_cost']

## Initialize Wind Power Dynamically
wind_power = np.zeros(T)
wind_power[0] = data['target_mean_wind']  # Start with target mean wind
if T > 1:
    wind_power[1] = wind_power[0]  # Initialize second value

# Generate wind power using wind_model
for t in range(2, T):
    wind_power[t] = wind_model(wind_power[t-1], wind_power[t-2], data)

# Initialize Electricity Price Dynamically
price = np.zeros(T)
price[0] = data['mean_price']  # Start with the mean price
if T > 1:
    price[1] = price[0]  # Initialize the second value as well
for t in range(2, T):
    price[t] = price_model(price[t-1], price[t-2], wind_power[t], data)

# Create Pyomo Model
model = ConcreteModel()

# Define Variables
model.grid = Var(range(T), domain=NonNegativeReals)  # Power from the grid
model.p2h = Var(range(T), domain=NonNegativeReals)  # Power-to-Hydrogen
model.h2p = Var(range(T), domain=NonNegativeReals, bounds=(0, H2P))  # Hydrogen-to-Power
model.h = Var(range(T+1), domain=NonNegativeReals, bounds=(0, C))  # Hydrogen storage level
model.E_ON = Var(range(T), domain=Binary)  # Electrolyzer ON status
model.E_SW = Var(range(T), domain=Binary)  # State change detection

# Define Objective Function (Minimizing Cost)
def cost_function(model):
    return sum(price[t] * model.grid[t] + C_elzr * model.E_ON[t] for t in range(T))

model.objective = Objective(rule=cost_function, sense=minimize)

# Define Constraints
def power_balance(model, t):
    return model.grid[t] + wind_power[t] + model.h2p[t] == D[t] + model.p2h[t]
model.power_balance_constr = Constraint(range(T), rule=power_balance)

# Modify electrolyzer control to consider decision from t-1
def electrolyzer_control(model, t):
    return model.p2h[t] <= P2H * model.E_ON[t-1] if t > 0 else model.p2h[t] <= P2H * model.E_ON[t]
model.electrolyzer_constr = Constraint(range(T), rule=electrolyzer_control)

# Define hydrogen storage dynamics
def hydrogen_storage(model, t):
    return model.h[t+1] == model.h[t] + R_p2h * model.p2h[t] - model.h2p[t] / R_h2p
model.hydrogen_storage_constr = Constraint(range(T), rule=hydrogen_storage)

def electrolyzer_switch_on(model, t):
    if t > 0:
        # Introduce new variable to represent the absolute difference
        model.z = Var(within=NonNegativeReals)
        
        # Linearize the absolute value term |E_ON[t] - E_ON[t-1]|
        model.electrolyzer_switch_on_constr_1 = Constraint(expr=model.z >= model.E_ON[t] - model.E_ON[t-1])
        model.electrolyzer_switch_on_constr_2 = Constraint(expr=model.z >= -(model.E_ON[t] - model.E_ON[t-1]))
        
        # Ensure E_SW[t] is at least the absolute difference
        return model.E_SW[t] >= model.z
    return model.E_SW[t] == 0  # Initial state when t=0 is off


# Solve Optimization Problem (Using Gurobi)
solver = SolverFactory('gurobi')
solver.solve(model, tee=True)

# Organize Results for Visualization
results = {
    "electrolyzer_status": [model.E_ON[t]() for t in range(T)],
    "hydrogen_storage_level": [model.h[t]() for t in range(T)],
    "power_to_hydrogen": [model.p2h[t]() for t in range(T)],
    "hydrogen_to_power": [model.h2p[t]() for t in range(T)],
    "grid_power": [model.grid[t]() for t in range(T)]
}

wind_trajectory = list(wind_power)  # Wind power trajectory
price_trajectory = list(price)  # Electricity price trajectory

# Plot Results
plt.figure(figsize=(14, 10))

times = range(T)

for i in range(1, 9):
    plt.subplot(8, 1, i)
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))

plt.subplot(8, 1, 1)
plt.plot(times, wind_trajectory, label="Wind Power", color="blue")
plt.ylabel("Wind Power")
plt.legend()

plt.subplot(8, 1, 2)
plt.plot(times, data['demand_schedule'], label="Demand Schedule", color="orange")
plt.ylabel("Demand")
plt.legend()

plt.subplot(8, 1, 3)
plt.step(times, results['electrolyzer_status'], label="Electrolyzer Status", color="red", where="post")
plt.ylabel("El. Status")
plt.legend()

plt.subplot(8, 1, 4)
plt.plot(times, results['hydrogen_storage_level'], label="Hydrogen Level", color="green")
plt.ylabel("Hydr. Level")
plt.legend()

plt.subplot(8, 1, 5)
plt.plot(times, results['power_to_hydrogen'], label="p2h", color="orange")
plt.ylabel("p2h")
plt.legend()

plt.subplot(8, 1, 6)
plt.plot(times, results['hydrogen_to_power'], label="h2p", color="blue")
plt.ylabel("h2p")
plt.legend()

plt.subplot(8, 1, 7)
plt.plot(times, results['grid_power'], label="Grid Power", color="green")
plt.ylabel("Grid Power")
plt.legend()

plt.subplot(8, 1, 8)
plt.plot(times, price_trajectory, label="price", color="red")
plt.ylabel("Price")
plt.xlabel("Time")
plt.legend()

plt.tight_layout()
plt.show()
