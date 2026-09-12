
import pandas as pd
import pulp

from config.settings import FPLSettings, POSITION_MAP


def select_team(df: pd.DataFrame, settings: FPLSettings) -> pd.DataFrame:
    """Select the optimal FPL team using Integer Linear Programming.
    Args:
        df: The pd.DataFrame with the FPL players data to choose from.
        settings: The FPLSettings instance specifying constraints.
    Returns:
        A pd.DataFrame containing the selected optimal players."""

    if "now_cost" not in df.columns:
        raise ValueError("DataFrame does not contain the now_cost column!")
    if "team_name" not in df.columns:
        raise ValueError("DataFrame does not contain the team_name column!")
    if "position_name" not in df.columns:
        raise ValueError("DataFrame does not contain the position_name column!")

    problem_team = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)

    df = df.reset_index(drop=True)

    # Decision Variables (1 if player selected, 0 otherwise)
    player_vars = [pulp.LpVariable(f"player_{i}", cat=pulp.LpBinary)
        for i in df.index]

    # Objective Function: Maximize expected points
    problem_team += (
        pulp.lpSum(player_vars[i]*df.loc[i, settings.metric]
            for i in df.index
        ),
        "Total_Expected_Points")

    # Constraints
    # Total team size = 15
    problem_team += (pulp.lpSum(player_vars) == settings.total_team_size,
        "Team_Size_Constraint")

    # Total cost <= Budget
    problem_team += (pulp.lpSum(player_vars[i]*df.loc[i, 'now_cost']
            for i in df.index
        ) <= settings.total_budget,
        "Budget_Constraint")

    # Position limits
    limits: list[int] = [settings.n_gkp, settings.n_def, settings.n_mid, settings.n_fwd]
    pos_limits: dict[str, int] = dict(zip(POSITION_MAP.values(), limits))
    for pos, count in pos_limits.items():
        pos_indices = df[df["position_name"] == pos].index
        problem_team += (pulp.lpSum(player_vars[i] for i in pos_indices) == count,
            f"Position_{pos}_Constraint")

    # Max 3 players per team
    for team in df["team_name"].unique():
        team_indices = df[df["team_name"] == team].index
        problem_team += (
            pulp.lpSum(player_vars[i] for i in team_indices) <= settings.max_players_per_team,
            f"Team_{team}_Constraint")

    # Solve using CBC solver
    problem_team.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract and return selected team
    selected_indices = [i for i in df.index if player_vars[i].varValue == 1.0]

    df_team: pd.DataFrame = df.loc[selected_indices].copy()
    df_team = df_team.sort_values(by=['position_name', settings.metric], ascending=[True, False])

    return df_team

def select_starters(df: pd.DataFrame, settings: FPLSettings) -> pd.DataFrame:
    """Select the optimal starter FPL players using Integer Linear Programming.
    Args:
        df: The pd.DataFrame with the FPL players data to choose from
            (optimal team or all FPL players).
        settings: The FPLSettings instance specifying constraints.
    Returns:
        A pd.DataFrame containing the selected optimal starter players."""

    if "now_cost" not in df.columns:
        raise ValueError("DataFrame does not contain the now_cost column!")
    if "position_name" not in df.columns:
        raise ValueError("DataFrame does not contain the position_name column!")
    if "team_name" not in df.columns:
        raise ValueError("DataFrame does not contain the team_name column!")

    problem_starters = pulp.LpProblem("FPL_Starters_Optimization", pulp.LpMaximize)

    df = df.reset_index(drop=True)

    # Decision Variables (1 if player selected, 0 otherwise)
    player_vars = [pulp.LpVariable(f"player_{i}", cat=pulp.LpBinary)
        for i in df.index]

    # Objective Function: Maximize expected points
    problem_starters += (
        pulp.lpSum(player_vars[i]*df.loc[i, settings.metric]
            for i in df.index
        ),
        "Starters_Expected_Points")

    # Constraints
    # Starter team size = 11
    problem_starters += (pulp.lpSum(player_vars) == settings.starting_team_size,
        "Starter_Team_Size_Constraint")

    # Total cost <= Budget
    gkp_min_val: float = df[df['position_name'] == 'GKP']['now_cost'].min()
    def_min_val: float = df[df['position_name'] == 'DEF']['now_cost'].min()
    mid_min_val: float = df[df['position_name'] == 'MID']['now_cost'].min()
    fwd_min_val: float = df[df['position_name'] == 'FWD']['now_cost'].min()
    budget_bench: float = gkp_min_val + 3*max(def_min_val, mid_min_val, fwd_min_val)
    budget_left: float = settings.total_budget - budget_bench
    problem_starters += (pulp.lpSum(player_vars[i]*df.loc[i, 'now_cost']
            for i in df.index
        ) <= budget_left,
        "Budget_Constraint")

    # Position lower limits
    low_limits: list[int] = [settings.n_gkp_start_min, settings.n_def_start_min,
                         settings.n_mid_start_min, settings.n_fwd_start_min]
    pos_low_limits: dict[str, int] = dict(zip(POSITION_MAP.values(), low_limits))
    for pos, count in pos_low_limits.items():
        pos_indices = df[df["position_name"] == pos].index
        problem_starters += (pulp.lpSum(player_vars[i] for i in pos_indices) >= count,
            f"Position_{pos}_Lower_Constraint")

    # Position upper limits
    high_limits: list[int] = [settings.n_gkp_start_max, settings.n_def_start_max,
                         settings.n_mid_start_max, settings.n_fwd_start_max]
    pos_high_limits: dict[str, int] = dict(zip(POSITION_MAP.values(), high_limits))
    for pos, count in pos_high_limits.items():
        pos_indices = df[df["position_name"] == pos].index
        problem_starters += (pulp.lpSum(player_vars[i] for i in pos_indices) <= count,
            f"Position_{pos}_Upper_Constraint")

    # Max 3 players per team
    for team in df["team_name"].unique():
        team_indices = df[df["team_name"] == team].index
        problem_starters += (
            pulp.lpSum(player_vars[i] for i in team_indices) <= settings.max_players_per_team,
            f"Team_{team}_Constraint")

    # Solve using CBC solver
    problem_starters.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract and return selected team
    selected_indices = [i for i in df.index if player_vars[i].varValue == 1.0]

    df_starters: pd.DataFrame = df.loc[selected_indices].copy()
    df_starters = df_starters.sort_values(by=['position_name', settings.metric], ascending=[True, False])

    return df_starters
