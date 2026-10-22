
from typing import Any
import logging

import pandas as pd

from config import settings
from config.naming import Names
from config.settings import FPLSettings, ColumnsRaw, Columns
from src.data import data_store, data_cleaner


logger = logging.getLogger(__name__)

def prepare_live_data(
    live_data: dict[str, Any], fpl_settings: FPLSettings, store: bool = True
    ) -> pd.DataFrame:
    """Prepare players live data for analysis.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data.
        fpl_settings: The FPLSettings instance specifying constraints and settings.
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A pd.DataFrame of players live data prepared for analysis.
    Raises:
        ValueError: If the pd.DataFrame contains not-allowed negative entries."""

    df_players = pd.DataFrame(data=live_data[Names.PLAYERS])

    df_players = keep_players_live_data(df_players)

    df_players.rename(columns=settings.PLAYERS_COLUMNS_RENAME, inplace=True)

    df_players[Columns.POSITION_NAME] = (df_players[Columns.POSITION_ID].map(settings.POSITION_MAP))

    # GW, opp_id, is_home
    df_players = add_fixtures_live_data(df_players, live_data[Names.FIXTURES], fpl_settings.n_fixtures)

    # team_name, team_id_opp, team_name_opp
    df_players = add_teams_live_data(df_players, live_data[Names.TEAMS])
    df_players.drop(
        columns=[Columns.TEAM_ID, Columns.OPPONENT_ID, Columns.TEAM_ID + Names.OPPONENT_SUFFIX], inplace=True
        )

    # Features form
    df_players = add_gw_live_data(df_players, live_data[Names.GW_HISTORY], fpl_settings.n_form)

    df_players.sort_values(
        by=[Columns.PLAYER_ID, Columns.GW], ascending=[True, True], inplace=True
        )
    df_players.drop(columns=[Columns.POSITION_ID, Columns.PLAYER_ID], inplace=True)

    cleaner = data_cleaner.DataCleaner(df_players)
    cleaner.clean_data()
    cleaner.raise_negative_values(settings.EXCLUDED_FROM_NEGATIVE_CHECK)

    df_players = cleaner.df

    if store:
        data_store.save_df_to_csv(df_players, settings.STORE_FILE_PATH[Names.CLEAN][Names.LIVE], overwrite=True)

    return df_players

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

    is_upcoming = (~df_fixtures[Columns.IS_FINISHED])
    cols = [c for c in settings.FIXTURES_COLUMNS_RENAME.values() if c != Columns.IS_FINISHED]
    df_fixtures = df_fixtures.loc[is_upcoming, cols]

    df_home = df_fixtures.rename(
        columns={Columns.HOME_TEAM_ID: Columns.TEAM_ID, Columns.AWAY_TEAM_ID: Columns.OPPONENT_ID}
        )
    df_home[Columns.IS_HOME] = True

    df_away = df_fixtures.rename(
        columns={Columns.AWAY_TEAM_ID: Columns.TEAM_ID, Columns.HOME_TEAM_ID: Columns.OPPONENT_ID}
        )
    df_away[Columns.IS_HOME] = False

    df_home_away = pd.concat([df_home, df_away], ignore_index=True)

    df_home_away.sort_values(by=Columns.GW, ascending=True, inplace=True)

    df_home_away = df_home_away.groupby(Columns.TEAM_ID).head(n_fixtures)

    df = (df_players.merge(df_home_away, on=Columns.TEAM_ID, how="outer")) # GW, opp_id, is_home

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

    df = (
        df_players
        .merge(df_teams, on=Columns.TEAM_ID, how="outer")
        .merge(df_teams, left_on=Columns.OPPONENT_ID, right_on=Columns.TEAM_ID,
               how="outer", suffixes=("", Names.OPPONENT_SUFFIX))
    )  # team_name, team_id_opp, team_name_opp

    return df

def add_gw_live_data(
    df_players: pd.DataFrame, gw_data: list[list[dict[str, Any]]], n_form: int
    ) -> pd.DataFrame:
    """Add the required GW live data to the players live data.
    Args:
        df_players: The pd.DataFrame of players live data.
        gw_data: The list of lists of dictionaries of GW live data.
        n_form: The number of past GWs to consider for players' form.
    Returns:
        A pd.DataFrame of players live data augmented with the GW live data."""

    df_players_gw = pd.concat([pd.DataFrame(data=player) for player in gw_data], ignore_index=True)

    df_players_gw = (
        df_players_gw[settings.PLAYERS_GW_COLUMNS_RENAME.keys()]
        .rename(columns=settings.PLAYERS_GW_COLUMNS_RENAME)
        .sort_values(by=[Columns.GW, Columns.PLAYER_ID], ascending=True)
        .groupby(Columns.PLAYER_ID)
        .tail(n_form)
    )

    cleaner = data_cleaner.DataCleaner(df_players_gw)
    cleaner.clean_data()
    df_players_gw = cleaner.df

    features = settings.FEATURES.copy()
    if Columns.DEFENSIVE_CONTRIBUTIONS not in features:
        features.append(Columns.DEFENSIVE_CONTRIBUTIONS)

    # Drop original columns
    df_form = (
        df_players_gw.groupby(Columns.PLAYER_ID)[features]
        .mean()
        .add_suffix(Names.FORM_SUFFIX)
        .reset_index()
    )

    df = df_players.merge(df_form, on=Columns.PLAYER_ID, how="left")

    return df

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
        l_after = len(df_players)
        if l_before != l_after:
            logger.warning(f'Rows lost during past seasons team merge! {l_before-l_after}')

        df_players.drop(columns=[Columns.TEAM_NAME_FULL], inplace=True)
        df_players = df_players.sort_values(
            by=[Columns.GW, Columns.TEAM_NAME, Columns.POSITION_NAME],
            ascending=[True, True, True]
        )

        # Add opponent
        df_teams.drop(columns=[Columns.TEAM_NAME_FULL], inplace=True)
        df_opponent = df_teams.drop_duplicates(subset=[Columns.TEAM_ID])
        df_players = df_players.merge(
            df_opponent, left_on=Columns.OPPONENT_ID, right_on=Columns.TEAM_ID, how='left',
            suffixes=('', Names.OPPONENT_SUFFIX)
            ) # team_id_opp, team_name_opp
        df_players.drop(
            columns=[
                Columns.TEAM_ID, Columns.OPPONENT_ID, Columns.TEAM_ID + Names.OPPONENT_SUFFIX
            ], inplace=True
        )

        cleaner = data_cleaner.DataCleaner(df_players)
        cleaner.clean_data()
        cleaner.raise_negative_values(settings.EXCLUDED_FROM_NEGATIVE_CHECK)

        players_past_data[s] = cleaner.df

    if store:
        for s in players_past_data.keys():
            file_path = settings.STORE_FILE_PATH[Names.CLEAN][Names.PAST].format(season=s)
            data_store.save_df_to_csv(players_past_data[s], file_path, overwrite=True)

    return players_past_data
