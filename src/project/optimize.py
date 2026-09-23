
import pandas as pd
import pulp

from config.settings import POSITION_CONSTRAINTS, TEAM_CONSTRAINTS
from src.data import data_store


def columns_in_df(df: pd.DataFrame, columns: list[str]) -> None:
    """Check if all specified columns are present in the pd.DataFrame.
    Args:
        columns: The list of columns names to check.
    Raises:
        KeyError: If any columns are missing from self.pd.DataFrame."""

    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"The following columns were not found in the DataFrame: {sorted(missing)}")

def select_team(df: pd.DataFrame, metric: str, store: bool = True) -> pd.DataFrame:
    """Select the optimal team using Integer Linear Programming.
    Args:
        df: The pd.DataFrame with the players to choose from.
        metric: The metric on which optimization is performed.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A pd.DataFrame containing the selected optimal players."""

    columns_in_df(df=df, columns=["now_cost", "team_name", "position_name"])

    problem_team = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)

    df = df.reset_index(drop=True)

    # Decision Variables (1 if player selected, 0 otherwise)
    player_vars = [pulp.LpVariable(f"player_{i}", cat=pulp.LpBinary)
        for i in df.index]

    # Objective Function: Maximize expected points
    problem_team += (
        pulp.lpSum(player_vars[i]*df.loc[i, metric]
            for i in df.index
        ),
        "Total_Expected_Points")

    # Constraints
    # Total team size = 15
    problem_team += (pulp.lpSum(player_vars) == TEAM_CONSTRAINTS['total'],
        "Team_Size_Constraint")

    # Total cost <= Budget
    problem_team += (pulp.lpSum(player_vars[i]*df.loc[i, 'now_cost']
            for i in df.index
        ) <= TEAM_CONSTRAINTS['budget'],
        "Budget_Constraint")

    # Position limits
    for pos, count in POSITION_CONSTRAINTS.items():
        pos_indices = df[df["position_name"] == pos].index
        problem_team += (pulp.lpSum(player_vars[i] for i in pos_indices) == count['total'],
            f"Position_{pos}_Constraint")

    # Max 3 players per team
    for team in df["team_name"].unique():
        team_indices = df[df["team_name"] == team].index
        problem_team += (
            pulp.lpSum(player_vars[i] for i in team_indices) <= TEAM_CONSTRAINTS['max_players_per_team'],
            f"Team_{team}_Constraint")

    # Solve using CBC solver
    problem_team.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract and return selected team
    selected_indices = [i for i in df.index if player_vars[i].varValue == 1.0]

    df_team: pd.DataFrame = df.loc[selected_indices].copy()
    df_team = df_team.sort_values(by=['position_name', metric], ascending=[True, False])

    if store:
        file_path = "fpl_data/predict/optimal_team_data.csv"
        data_store.save_df_to_csv(df=df_team, file_path=file_path, overwrite=True)

    return df_team

def select_starters(df: pd.DataFrame, metric: str, store: bool = True) -> pd.DataFrame:
    """Select the optimal starter players using Integer Linear Programming.
    Args:
        df: The pd.DataFrame with the players data to choose from (optimal team or all FPL players).
        metric: The metric on which optimization is performed.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A pd.DataFrame containing the selected optimal starter players."""

    columns_in_df(df=df, columns=["now_cost", "team_name", "position_name"])

    problem_starters = pulp.LpProblem("FPL_Starters_Optimization", pulp.LpMaximize)

    df = df.reset_index(drop=True)

    # Decision Variables (1 if player selected, 0 otherwise)
    player_vars = [pulp.LpVariable(f"player_{i}", cat=pulp.LpBinary)
        for i in df.index]

    # Objective Function: Maximize expected points
    problem_starters += (
        pulp.lpSum(player_vars[i]*df.loc[i, metric]
            for i in df.index
        ),
        "Starters_Expected_Points")

    # Constraints
    # Starter team size = 11
    problem_starters += (pulp.lpSum(player_vars) == TEAM_CONSTRAINTS['starters'],
        "Starter_Team_Size_Constraint")

    # Total cost <= Budget
    gkp_min_val = df[df['position_name'] == 'GKP']['now_cost'].min()
    def_min_val = df[df['position_name'] == 'DEF']['now_cost'].min()
    mid_min_val = df[df['position_name'] == 'MID']['now_cost'].min()
    fwd_min_val = df[df['position_name'] == 'FWD']['now_cost'].min()
    budget_bench = gkp_min_val + 3*max(def_min_val, mid_min_val, fwd_min_val)
    budget_left = TEAM_CONSTRAINTS['budget'] - budget_bench
    problem_starters += (pulp.lpSum(player_vars[i]*df.loc[i, 'now_cost']
            for i in df.index
        ) <= budget_left,
        "Budget_Constraint")

    # Position lower limits
    for pos, count in POSITION_CONSTRAINTS.items():
        pos_indices = df[df["position_name"] == pos].index
        problem_starters += (pulp.lpSum(player_vars[i] for i in pos_indices) >= count['min_start'],
            f"Position_{pos}_Lower_Constraint")

    # Position upper limits
    for pos, count in POSITION_CONSTRAINTS.items():
        pos_indices = df[df["position_name"] == pos].index
        problem_starters += (pulp.lpSum(player_vars[i] for i in pos_indices) <= count['max_start'],
            f"Position_{pos}_Upper_Constraint")

    # Max 3 players per team
    for team in df["team_name"].unique():
        team_indices = df[df["team_name"] == team].index
        problem_starters += (
            pulp.lpSum(player_vars[i] for i in team_indices) <= TEAM_CONSTRAINTS['max_players_per_team'],
            f"Team_{team}_Constraint")

    # Solve using CBC solver
    problem_starters.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract and return selected team
    selected_indices = [i for i in df.index if player_vars[i].varValue == 1.0]

    df_starters: pd.DataFrame = df.loc[selected_indices].copy()
    df_starters = df_starters.sort_values(by=['position_name', metric], ascending=[True, False])

    if store:
        file_path = "fpl_data/predict/optimal_starters_data.csv"
        data_store.save_df_to_csv(df=df_starters, file_path=file_path, overwrite=True)

    return df_starters
