
from typing import Any, cast
import time
import logging

import pandas as pd

from config import settings
from config.naming import NamesRaw, Names
from src.data import data_fetch, data_store


logger = logging.getLogger(__name__)

def fetch_live_data(store: bool = True) -> dict[str, Any]:
    """Fetch live data (players, teams, fixtures, GW history) from API or existing directory.
    Args:
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of players, teams, fixtures, GW history live data."""

    # Players, Teams
    live_data = data_fetch.fetch_json_data(settings.URL[Names.LIVE], settings.STORE_FILE_PATH[Names.RAW][Names.LIVE])

    # rename columns
    key_map = {NamesRaw.PLAYERS: Names.PLAYERS, NamesRaw.TEAMS: Names.TEAMS}
    live_data = cast(dict[str, Any], {key_map.get(k, k): v for k, v in live_data.items()})

    key = Names.FIXTURES
    if key not in live_data.keys():
        live_data[key] = data_fetch.fetch_json_data(settings.URL[Names.FIXTURES])

    key = Names.GW_HISTORY
    if key not in live_data.keys():
        player_ids = [row[NamesRaw.PLAYER_ID] for row in live_data[Names.PLAYERS]]
        live_data[key] = fetch_live_gw_data(player_ids)

    if store:
        store_live_data_raw(live_data)

    return live_data

def fetch_live_gw_data(player_ids: list[int]) -> list[list[dict[str, Any]]]:
    """Fetch players GW history live data.
    Args:
        player_ids: The list of live player ids.
    Returns:
        A list of lists of dictionaries of players GW history live data.
        Players whose fetch fails are skipped and logged as a warning."""

    gw_history: list[list[dict[str, Any]]] = []

    for pid in sorted(player_ids):
        try:
            url = settings.URL[Names.GW_HISTORY] + str(pid)
            player_history = cast(dict[str, Any], data_fetch.fetch_json_data(url))
            player_gw_history = cast(list[dict[str, Any]], player_history[NamesRaw.PLAYER_HISTORY])
            time.sleep(0.2)
            gw_history.append(player_gw_history)
        except Exception as e:
            logger.warning(f"Failed to fetch data for player {pid}. Exception: {e}")

    return gw_history

def store_live_data_raw(live_data: dict[str, Any]) -> None:
    """Store live raw data.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data."""

    data_store.store_data_to_json(live_data, settings.STORE_FILE_PATH[Names.RAW][Names.LIVE])

    data_store.save_list_dict_to_csv(live_data[Names.PLAYERS], settings.STORE_FILE_PATH[Names.RAW][Names.PLAYERS])

    data_store.save_list_dict_to_csv(live_data[Names.TEAMS], settings.STORE_FILE_PATH[Names.RAW][Names.TEAMS])

    data_store.save_list_dict_to_csv(live_data[Names.FIXTURES], settings.STORE_FILE_PATH[Names.RAW][Names.FIXTURES])

    df_gw_list = [pd.DataFrame(data=player) for player in live_data[Names.GW_HISTORY]]
    df_players_gw = pd.concat(df_gw_list, ignore_index=True)
    data_store.save_df_to_csv(df_players_gw, settings.STORE_FILE_PATH[Names.RAW][Names.GW_HISTORY])

def fetch_past_data(store: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch past data (players, teams) from API or existing directory.
    Args:
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of pd.DataFrames of players, teams past data."""

    file_paths = [
        settings.STORE_FILE_PATH[Names.RAW][Names.PAST_PLAYERS].format(season=s) for s in settings.PAST_SEASONS
        ]
    file_paths_teams = [
        settings.STORE_FILE_PATH[Names.RAW][Names.PAST_TEAMS].format(season=s) for s in settings.PAST_SEASONS
        ]

    prefix = settings.URL[Names.PAST]
    suffix = settings.PAST_PLAYERS_SUFFIX
    past_players_data = data_fetch.fetch_multiple_csv_data(prefix, suffix, settings.PAST_SEASONS, file_paths)

    suffix = settings.PAST_TEAMS_SUFFIX
    past_teams_data = data_fetch.fetch_multiple_csv_data(prefix, suffix, settings.PAST_SEASONS, file_paths_teams)

    past_data = {Names.PLAYERS: past_players_data, Names.TEAMS: past_teams_data}

    if store:
        store_past_data_raw(past_data)

    return past_data

def store_past_data_raw(past_data: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Store past raw data.
    Args:
        past_data: The dictionary of players, teams past data."""

    for s, df in past_data[Names.PLAYERS].items():
        file_path = settings.STORE_FILE_PATH[Names.RAW][Names.PAST_PLAYERS].format(season=s)
        data_store.save_df_to_csv(df, file_path)
    for s, df in past_data[Names.TEAMS].items():
        file_path = settings.STORE_FILE_PATH[Names.RAW][Names.PAST_TEAMS].format(season=s)
        data_store.save_df_to_csv(df, file_path)
