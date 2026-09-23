
from typing import Any, cast
import time
import logging

import pandas as pd

from config import settings
from src.data import data_fetch, data_store


logger = logging.getLogger(__name__)

def fetch_live_data(store: bool = True) -> dict[str, Any]:
    """Fetch live data (players, teams, fixtures, GW history) from API or existing directory.
    Args:
        store (optional): Boolean indicating whether to store data or not.
    Returns:
        A dictionary of players, teams, fixtures, GW history live data."""

    # Players, Teams
    live_data = cast(dict[str, Any], data_fetch.fetch_json_data(
        url=settings.LIVE_URL, file_path=settings.LIVE_STORE_RAW)
        )

    key = "fixtures"
    if key not in live_data.keys():
        live_data[key] = data_fetch.fetch_json_data(url=settings.FIXTURES_URL)

    key = "gw_history"
    if key not in live_data.keys():
        player_ids = [row['id'] for row in live_data["elements"]]
        live_data[key] = fetch_live_gw_data(player_ids=player_ids)

    if store:
        store_live_data_raw(live_data=live_data)

    return live_data

def fetch_live_gw_data(player_ids: list[int]) -> list[list[dict[str, Any]]]:
    """Fetch players GW history live data.
    Args:
        player_ids: The list of live player ids.
    Returns:
        A list of lists of dictionaries of players GW history live data.
        **Warning**: Log a warning if a request for a player ID fails."""

    gw_history: list[list[dict[str, Any]]] = []

    for pid in sorted(player_ids):
        try:
            url = settings.GW_URL + str(pid)
            player_history = cast(dict[str, Any], data_fetch.fetch_json_data(url=url))
            player_gw_history = cast(list[dict[str, Any]], player_history['history'])
            time.sleep(0.5)
            gw_history.append(player_gw_history)
        except Exception as e:
            logger.warning(f"Failed to fetch data for player {pid}. Exception: {e}")

    return gw_history

def store_live_data_raw(live_data: dict[str, Any]) -> None:
    """Store live raw data.
    Args:
        live_data: The dictionary of players, teams, fixtures, GW history live data."""

    data_store.store_data_to_json(data=live_data, file_path=settings.LIVE_STORE_RAW)

    data_store.save_list_dict_to_csv(
        data_list_dict=live_data["elements"],file_path=settings.LIVE_PLAYERS_STORE_RAW
        )

    data_store.save_list_dict_to_csv(
        data_list_dict=live_data["teams"], file_path=settings.LIVE_TEAMS_STORE_RAW
        )

    data_store.save_list_dict_to_csv(
        data_list_dict=live_data["fixtures"], file_path=settings.LIVE_FIXTURES_STORE_RAW
        )

    df_gw_list = [pd.DataFrame(data=player) for player in live_data["gw_history"]]
    df_players_gw = pd.concat(df_gw_list, ignore_index=True)
    data_store.save_df_to_csv(df=df_players_gw, file_path=settings.LIVE_GW_STORE_RAW)

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
    past_players_data = data_fetch.fetch_multiple_csv_data(
        prefix=prefix, suffix=suffix, identifiers=settings.PAST_SEASONS, file_paths=file_paths
        )

    suffix = settings.PAST_TEAMS_SUFFIX
    past_teams_data = data_fetch.fetch_multiple_csv_data(
        prefix=prefix, suffix=suffix, identifiers=settings.PAST_SEASONS, file_paths=file_paths_teams
        )

    past_data = {'players': past_players_data, 'teams': past_teams_data}

    if store:
        store_past_data_raw(past_data=past_data)

    return past_data

def store_past_data_raw(past_data: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Store past raw data.
    Args:
        past_data: The dictionary of players, teams past data."""

    for s, df in past_data['players'].items():
        file_path = settings.PAST_STORE_RAW.format(season=s)
        data_store.save_df_to_csv(df=df, file_path=file_path)
    for s, df in past_data['teams'].items():
        file_path = settings.PAST_TEAMS_STORE_RAW.format(season=s)
        data_store.save_df_to_csv(df=df, file_path=file_path)
