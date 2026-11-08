
from typing import Any
import logging

import pandas as pd

from config import settings
from config.naming import Names
from config.settings import ColumnsRaw, Columns
from src.data import data_store, data_cleaner


logger = logging.getLogger(__name__)

def prepare_live_data(
    live_data: dict[str, Any], n_fixtures: int, store: bool = True
    ) -> dict[str, pd.DataFrame]:
    """Prepare players live data for analysis.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data.
        n_fixtures: The number of upcoming fixtures to consider.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of pd.DataFrames of players live data prepared for analysis.
    Raises:
        ValueError: If the pd.DataFrame contains not-allowed negative entries."""

    df_players = pd.DataFrame(data=live_data[Names.PLAYERS])

    df_players = keep_players_live_data(df_players)

    df_players.rename(columns=settings.PLAYERS_COLUMNS_RENAME, inplace=True)

    df_players[Columns.POSITION_NAME] = (df_players[Columns.POSITION_ID].map(settings.POSITION_MAP))

    # GW, opp_id, is_home
    df_players = add_fixtures_live_data(df_players, live_data[Names.FIXTURES], n_fixtures)

    # team_name, team_id_opp, team_name_opp
    df_players = add_teams_live_data(df_players, live_data[Names.TEAMS])
    df_players.drop(
        columns=[Columns.TEAM_ID, Columns.OPPONENT_ID, Columns.TEAM_ID + Names.OPPONENT_SUFFIX], inplace=True
        )

    # Features form
    df_players = add_gw_live_data(df_players, live_data[Names.GW_HISTORY])

    df_players.drop(columns=[Columns.POSITION_ID, Columns.PLAYER_ID], inplace=True)
    df_players = df_players.sort_values(
        by=[Columns.GAMEWEEK, Columns.TEAM_NAME, Columns.POSITION_NAME],
        ascending=[True, True, True]
    )

    cleaner = data_cleaner.DataCleaner(df_players)
    cleaner.clean_data()
    cleaner.raise_negative_values(settings.EXCLUDED_FROM_NEGATIVE_CHECK)

    df = cleaner.df

    if store:
        data_store.save_df_to_csv(df, settings.STORE_FILE_PATH[Names.CLEAN][Names.LIVE], overwrite=True)

    df_dict = {settings.CURRENT_SEASON: df}

    return df_dict

def keep_players_live_data(df_players: pd.DataFrame) -> pd.DataFrame:
    """Keep only necessary players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
    Returns:
        A pd.DataFrame of necessary players live data."""

    # 'u': unavailable, 'i': injured, 's': suspended, 'd': doubtful, 'a': available
    is_available = ((df_players[ColumnsRaw.CAN_BE_SELECTED]) &
                    (~df_players[ColumnsRaw.PLAYER_STATUS].isin(['u', 'i', 's'])))

    is_active = (df_players[ColumnsRaw.PLAYER_STATUS] == 'a')

    df = df_players.loc[is_available & is_active, settings.PLAYERS_COLUMNS_RENAME.keys()].copy()

    if ColumnsRaw.AVAILABILITY in df.columns:
        df[ColumnsRaw.AVAILABILITY] = 100

    return df

def add_fixtures_live_data(
        df_players: pd.DataFrame, fixtures_data: list[dict[str, Any]], n_fixtures: int
        ) -> pd.DataFrame:
    """Add the required fixtures live data to the players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
        fixtures_data: The list of dictionaries of fixtures live data.
        n_fixtures: The number of upcoming fixtures to consider.
    Returns:
        A pd.DataFrame of players live data augmented with the fixtures live data."""

    df_fixtures = (
        pd.DataFrame(data=fixtures_data, columns=settings.FIXTURES_COLUMNS_RENAME.keys())
        .rename(columns=settings.FIXTURES_COLUMNS_RENAME)
        )

    df_home = df_fixtures.rename(
        columns={Columns.HOME_TEAM_ID: Columns.TEAM_ID, Columns.AWAY_TEAM_ID: Columns.OPPONENT_ID}
        )
    df_home[Columns.IS_HOME] = True

    df_away = df_fixtures.rename(
        columns={Columns.AWAY_TEAM_ID: Columns.TEAM_ID, Columns.HOME_TEAM_ID: Columns.OPPONENT_ID}
        )
    df_away[Columns.IS_HOME] = False

    df_home_away = pd.concat([df_home, df_away], ignore_index=True)

    df_home_away.sort_values(by=Columns.GAMEWEEK, ascending=True, inplace=True)

    is_finished = (df_home_away[Columns.IS_FINISHED])
    df_finished = df_home_away[is_finished]
    df_upcoming = df_home_away[~is_finished].groupby(Columns.TEAM_ID).head(n_fixtures)

    df_fixtures = pd.concat([df_finished, df_upcoming])

    l_before = len(df_players)
    df = (df_players.merge(df_fixtures, on=Columns.TEAM_ID, how="inner")) # GW, opp_id, is_home
    check_number_of_rows(df, l_before, msg='merge live fixtures')

    return df

def add_teams_live_data(df_players: pd.DataFrame, teams_data: list[dict[str, Any]]) -> pd.DataFrame:
    """Add the required teams live data to the players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
        teams_data: The list of dictionaries of teams live data.
    Returns:
        A pd.DataFrame of players live data augmented with the teams live data."""

    df_teams = (
        pd.DataFrame(data=teams_data, columns=settings.TEAMS_COLUMNS_RENAME.keys())
        .rename(columns=settings.TEAMS_COLUMNS_RENAME)
    )

    l_before = len(df_players)
    df = (
        df_players
        .merge(df_teams, on=Columns.TEAM_ID, how="inner")
        .merge(df_teams, left_on=Columns.OPPONENT_ID, right_on=Columns.TEAM_ID,
               how="inner", suffixes=("", Names.OPPONENT_SUFFIX))
    )  # team_name, team_id_opp, team_name_opp
    check_number_of_rows(df, l_before, msg='merge live teams')

    return df

def add_gw_live_data(df_players: pd.DataFrame, gw_data: list[list[dict[str, Any]]]) -> pd.DataFrame:
    """Add the required GW live data to the players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
        gw_data: The list of lists of dictionaries of GW live data.
    Returns:
        A pd.DataFrame of players live data augmented with the GW live data."""

    df_players_gw = pd.concat([pd.DataFrame(data=player) for player in gw_data], ignore_index=True)

    df_players_gw = (
        df_players_gw[settings.PLAYERS_GW_COLUMNS_RENAME.keys()]
        .rename(columns=settings.PLAYERS_GW_COLUMNS_RENAME)
    )

    cleaner = data_cleaner.DataCleaner(df_players_gw)
    cleaner.clean_data()
    df_players_gw = cleaner.df

    l_before = len(df_players)
    df = df_players.merge(df_players_gw, on=[Columns.PLAYER_ID, Columns.GAMEWEEK], how="left")
    check_number_of_rows(df, l_before, msg='merge live gameweek')

    return df

def add_past_data(
        players_data: dict[str, pd.DataFrame],
        past_data: dict[str, dict[str, pd.DataFrame]], store: bool = True
        ) -> dict[str, pd.DataFrame]:
    """Add players past data to players live data.
    Args:
        players_data: The dictionary of pd.DataFrames of players live data.
        past_data: The dictionary of players, teams past data.
        store (optional): Boolean indicating whether to store past data or not.
    Returns:
        A dictionary of pd.DataFrames of players live and past data prepared for analysis.
    Raises:
        ValueError: If a pd.DataFrame contains not-allowed negative entries."""

    players_dict = players_data.copy()

    players_past_data = prepare_past_data(past_data, store=True)

    for s, data in players_past_data.items():
        players_dict[s] = data

    return players_dict

def prepare_past_data(
    past_data: dict[str, dict[str, pd.DataFrame]], store: bool = True
    ) -> dict[str, pd.DataFrame]:
    """Prepare players past data for analysis.
    Args:
        past_data: The dictionary of players, teams past data.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of pd.DataFrames of players past data prepared for analysis.
    Raises:
        ValueError: If a pd.DataFrame contains not-allowed negative entries."""

    players_past_data: dict[str, pd.DataFrame] = {}

    for s in past_data[Names.PLAYERS].keys():

        df_players = past_data[Names.PLAYERS][s]

        df_players = df_players[settings.SEASONS_COLUMNS_RENAME.keys()].copy()
        df_players.rename(columns=settings.SEASONS_COLUMNS_RENAME, inplace=True)

        rows = (df_players[Columns.POSITION_NAME] == 'GK')
        df_players.loc[rows, Columns.POSITION_NAME] = Names.GOALKEEPER

        df_teams = past_data[Names.TEAMS][s]
        df_teams = df_teams[settings.SEASONS_TEAMS_COLUMNS_RENAME.keys()].copy()
        df_teams.rename(columns=settings.SEASONS_TEAMS_COLUMNS_RENAME, inplace=True)

        # Merge by team name (no team_id in df_players)
        l_before = len(df_players)
        df_players = df_players.merge(df_teams, on=Columns.TEAM_NAME_FULL, how="inner") # team_id, team_name
        check_number_of_rows(df_players, l_before, msg='merge past team')

        df_players.drop(columns=[Columns.TEAM_NAME_FULL], inplace=True)
        df_players = df_players.sort_values(
            by=[Columns.GAMEWEEK, Columns.TEAM_NAME, Columns.POSITION_NAME],
            ascending=[True, True, True]
        )

        # Add opponent
        df_teams.drop(columns=[Columns.TEAM_NAME_FULL], inplace=True)
        df_opponent = df_teams.drop_duplicates(subset=[Columns.TEAM_ID])

        l_before = len(df_players)
        df_players = df_players.merge(
            df_opponent, left_on=Columns.OPPONENT_ID, right_on=Columns.TEAM_ID, how='inner',
            suffixes=('', Names.OPPONENT_SUFFIX)
            ) # team_id_opp, team_name_opp
        check_number_of_rows(df_players, l_before, msg='merge past opponent')

        df_players.drop(
            columns=[
                Columns.TEAM_ID, Columns.OPPONENT_ID, Columns.TEAM_ID + Names.OPPONENT_SUFFIX
            ], inplace=True
        )

        df_players[Columns.IS_FINISHED] = True

        cleaner = data_cleaner.DataCleaner(df_players)
        cleaner.clean_data()
        cleaner.raise_negative_values(settings.EXCLUDED_FROM_NEGATIVE_CHECK)

        players_past_data[s] = cleaner.df

    if store:
        for s in players_past_data.keys():
            file_path = settings.STORE_FILE_PATH[Names.CLEAN][Names.PAST].format(season=s)
            data_store.save_df_to_csv(players_past_data[s], file_path, overwrite=True)

    return players_past_data

def check_number_of_rows(df: pd.DataFrame, l: int, msg: str = None) -> None:
    """Check if rows were dropped in a dataFrame.
    Args:
        df: The pd.DataFrame to check.
        l: The previous length of df.
        msg: The display message in case of dropped columns."""

    l_new = len(df)
    if l != l_new:
        logger.warning(f'Number of rows changed in df during {msg}! {l_new-l}')
