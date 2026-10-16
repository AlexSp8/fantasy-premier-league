
from typing import Any, cast
import time
import logging

import pandas as pd

from config import settings, naming
from config.settings import ColumnsRaw, Columns
from src.data import data_fetch, data_store


logger = logging.getLogger(__name__)

def fetch_live_data(store: bool = True) -> dict[str, Any]:
    """Fetch live data (players, teams, fixtures, GW history) from API or existing directory.
    Args:
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of players, teams, fixtures, GW history live data."""

    # Players, Teams
    live_data = cast(dict[str, Any], data_fetch.fetch_json_data(settings.LIVE_URL, settings.LIVE_STORE_RAW))

    # rename columns
    key_map = {ColumnsRaw.PLAYERS: naming.PLAYERS, ColumnsRaw.TEAMS: naming.TEAMS}
    live_data = {key_map.get(k, k): v for k, v in live_data.items()}

    key = naming.FIXTURES
    if key not in live_data.keys():
        live_data[key] = data_fetch.fetch_json_data(settings.FIXTURES_URL)

    key = Columns.GW_HISTORY
    if key not in live_data.keys():
        player_ids = [row[ColumnsRaw.PLAYER_ID] for row in live_data[naming.PLAYERS]]
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
            url = settings.GW_URL + str(pid)
            player_history = cast(dict[str, Any], data_fetch.fetch_json_data(url))
            player_gw_history = cast(list[dict[str, Any]], player_history[ColumnsRaw.PLAYER_HISTORY])
            time.sleep(0.5)
            gw_history.append(player_gw_history)
        except Exception as e:
            logger.warning(f"Failed to fetch data for player {pid}. Exception: {e}")

    return gw_history

def store_live_data_raw(live_data: dict[str, Any]) -> None:
    """Store live raw data.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data."""

    data_store.store_data_to_json(live_data, settings.LIVE_STORE_RAW)

    data_store.save_list_dict_to_csv(live_data[naming.PLAYERS], settings.LIVE_PLAYERS_STORE_RAW)

    data_store.save_list_dict_to_csv(live_data[naming.TEAMS], settings.LIVE_TEAMS_STORE_RAW)

    data_store.save_list_dict_to_csv(live_data[naming.FIXTURES], settings.LIVE_FIXTURES_STORE_RAW)

    df_gw_list = [pd.DataFrame(data=player) for player in live_data[Columns.GW_HISTORY]]
    df_players_gw = pd.concat(df_gw_list, ignore_index=True)
    data_store.save_df_to_csv(df_players_gw, settings.LIVE_GW_STORE_RAW)

def fetch_past_data(store: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch past data (players, teams) from API or existing directory.
    Args:
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of pd.DataFrames of players, teams past data."""

    file_paths = [settings.PAST_STORE_RAW.format(season=s) for s in settings.PAST_SEASONS]
    file_paths_teams = [settings.PAST_TEAMS_STORE_RAW.format(season=s) for s in settings.PAST_SEASONS]

    prefix = settings.PAST_URL
    suffix = settings.PAST_PLAYERS_SUFFIX
    past_players_data = data_fetch.fetch_multiple_csv_data(prefix, suffix, settings.PAST_SEASONS, file_paths)

    suffix = settings.PAST_TEAMS_SUFFIX
    past_teams_data = data_fetch.fetch_multiple_csv_data(prefix, suffix, settings.PAST_SEASONS, file_paths_teams)

    past_data = {naming.PLAYERS: past_players_data, naming.TEAMS: past_teams_data}

    if store:
        store_past_data_raw(past_data)

    return past_data

def store_past_data_raw(past_data: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Store past raw data.
    Args:
        past_data: The dictionary of players, teams past data."""

    for s, df in past_data[naming.PLAYERS].items():
        file_path = settings.PAST_STORE_RAW.format(season=s)
        data_store.save_df_to_csv(df, file_path)
    for s, df in past_data[naming.TEAMS].items():
        file_path = settings.PAST_TEAMS_STORE_RAW.format(season=s)
        data_store.save_df_to_csv(df, file_path)
