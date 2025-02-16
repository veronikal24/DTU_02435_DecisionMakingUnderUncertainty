import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
import WindProcess
import PriceProcess
from data import get_fixed_data

def create_model():
        # Get fixed data
    data = get_fixed_data()
    T = data['num_timeslots']
    D = data['demand_schedule']
    P2H = data['p2h_rate']
    H2P = data['h2p_rate']
    C = data['hydrogen_capacity']
    R_p2h = data['conversion_p2h']
    R_h2p = data['conversion_h2p']
    C_elzr = data['electrolyzer_cost']

    # Generate wind and price time series
    p_wind = np.random.normal(data['target_mean_wind'], 1, T)
    lambda_grid = np.clip(np.random.normal(data['mean_price'], 5, T), data['price_floor'], data['price_cap'])

    # Define the optimization model
    model = ConcreteModel()

    # Sets
    model.T = RangeSet(0, T-1)

    # Decision Variables
    model.x = Var(model.T, within=Binary)
    model.p2h = Var(model.T, within=NonNegativeReals)
    model.h2p = Var(model.T, within=NonNegativeReals)
    model.g = Var(model.T, within=NonNegativeReals)
    model.s = Var(model.T, within=NonNegativeReals, bounds=(0, C))

    # Objective function: Minimize cost
    def objective_rule(model):
        return sum(lambda_grid[t] * model.g[t] + C_elzr * model.x[t] for t in model.T)
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    def power_balance_rule(model, t):
        return D[t] == model.g[t] + p_wind[t] + model.h2p[t] - model.p2h[t]
    model.power_balance = Constraint(model.T, rule=power_balance_rule)

    def storage_dynamics_rule(model, t):
        if t > 0:
            return model.s[t] == model.s[t-1] + R_p2h * model.p2h[t] - model.h2p[t] / R_h2p
        return Constraint.Skip
    model.storage_dynamics = Constraint(model.T, rule=storage_dynamics_rule)

    def electrolyzer_operation_rule(model, t):
        return model.p2h[t] <= P2H * model.x[t]
    model.electrolyzer_operation = Constraint(model.T, rule=electrolyzer_operation_rule)

    def hydrogen_to_power_rule(model, t):
        return model.h2p[t] <= H2P
    model.hydrogen_to_power = Constraint(model.T, rule=hydrogen_to_power_rule)

    def electrolyzer_switching_rule_lower(model, t):
        if t > 0:
            return -1 <= model.x[t] - model.x[t-1]
        return Constraint.Skip
    model.electrolyzer_switching_lower = Constraint(model.T, rule=electrolyzer_switching_rule_lower)

    def electrolyzer_switching_rule_upper(model, t):
        if t > 0:
            return model.x[t] - model.x[t-1] <= 1
        return Constraint.Skip
    model.electrolyzer_switching_upper = Constraint(model.T, rule=electrolyzer_switching_rule_upper)

    # Solve the model
    solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured
    solver.solve(model)

    return solver, model, T, p_wind

