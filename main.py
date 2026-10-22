
import logging

import pandas as pd

from config.naming import Names
from config.settings import FPLSettings, Columns
from src.project import fetch, prepare, features, predictions, optimize


def main() -> None:
    """Run the FPL pipeline to predict the players that will yield the most points in upcoming fixtures."""

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 500)

    # logging prints nothing below "level" of severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_format = "%(asctime)s [%(levelname)s] %(name)s/%(filename)s (Line %(lineno)d): %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_format, datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[logging.FileHandler(filename="info.log", mode="w")]
        # handlers=[logging.StreamHandler(), logging.FileHandler("info.log", mode="w")]
        )

    fpl_settings = FPLSettings(prediction_model=Names.XGBOOST)

    live_data = fetch.fetch_live_data(store=True)

    df_players = prepare.prepare_live_data(live_data, fpl_settings, store=True)

    df_features = features.add_features_live(df_players, fpl_settings, store=True)

    df_features_past = None
    if fpl_settings.prediction_model != Names.FORMULA:
        past_data = fetch.fetch_past_data(store=True)
        players_past_data = prepare.prepare_past_data(past_data, store=True)
        df_features_past = features.add_features_past(players_past_data, fpl_settings, store=True)

    df_predict = predictions.set_predictions(df_features, df_features_past, fpl_settings, store=True)

    df_team = optimize.select_team(df_predict, fpl_settings.metric, store=True)
    # df_starters = optimize.select_starters(df_team, fpl_settings.metric)
    df_starters = optimize.select_starters(df_predict, fpl_settings.metric, store=True)

    for df in [df_team, df_starters]:
        print(f"\nPlayers:\n{df.drop(columns=Columns.PLAYER_CODE)}")
        print(f"Total cost: {df[Columns.PLAYER_COST].sum()}, Total PTS: {df[fpl_settings.metric].sum()}")

if __name__ == '__main__':
    main()
