#!/usr/bin/env python3
# test_solver_terminal.py

import logging
from ortools.linear_solver import pywraplp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting OR-Tools terminal test...")

    # Create solver using CBC (safe for terminal)
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        logger.error("Solver could not be created.")
        return

    # Create simple decision variables
    x = solver.NumVar(0, 10, 'x')  # 0 <= x <= 10
    y = solver.NumVar(0, 10, 'y')  # 0 <= y <= 10

    # Simple constraint
    solver.Add(x + y <= 5)

    # Objective: maximize x + y
    solver.Maximize(x + y)

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        logger.info(f"Optimal solution found: x={x.solution_value()}, y={y.solution_value()}")
        logger.info(f"Objective value: {solver.Objective().Value()}")
    elif status == pywraplp.Solver.FEASIBLE:
        logger.warning(f"Feasible solution found: x={x.solution_value()}, y={y.solution_value()}")
    else:
        logger.error("No solution found.")

if __name__ == "__main__":
    main()
