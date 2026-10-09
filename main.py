
import logging

import pandas as pd

from config.settings import FPLSettings
from src.project import fetch, prepare, features, predictions, optimize


def main() -> None:
    """Run the FPL pipeline to predict the optimal FPL players that will yield
    the most points in the upcoming fixtures, following FPL rules and constraints."""

    # logging prints nothing below "level" of severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_format = "%(asctime)s [%(levelname)s] %(name)s/%(filename)s (Line %(lineno)d): %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_format, datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[logging.FileHandler(filename="info.log", mode="w")]
        # handlers=[logging.StreamHandler(), logging.FileHandler("info.log", mode="w")]
        )

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 500)

    metric, prediction_model = 'xP_predicted_sum', 'xgboost'
    fpl_settings = FPLSettings(metric=metric, prediction_model=prediction_model)

    # Fetch players, teams, fixtures, past seasons
    live_data = fetch.fetch_live_data(store=True)

    # Prepare players live data
    df_players = prepare.prepare_players_live_data(live_data=live_data, fpl_settings=fpl_settings, store=True)

    if fpl_settings.prediction_model != "formula":
        past_data = fetch.fetch_past_data(store=True)
        players_past_data = prepare.prepare_players_past_data(past_data=past_data, store=True)

    # Add Features players, seasons
    df_features, df_features_past = features.add_features(
        df_players=df_players, players_past_data=players_past_data,
        fpl_settings=fpl_settings, store=True
        )

    # Predict
    df_predict = predictions.set_predictions(
        df_features=df_features, df_features_past=df_features_past,
        fpl_settings=fpl_settings, store=True
        )

    # Optimize
    df_team = optimize.select_team(df=df_predict, metric=fpl_settings.metric, store=True)
    # df_starters = optimize.select_starters(df=df_team, metric=fpl_settings.metric)
    df_starters = optimize.select_starters(df=df_predict, metric=fpl_settings.metric, store=True)

    for df in [df_team, df_starters]:
        print(f"\nPlayers:\n{df}")
        print(f"Total cost: {df['now_cost'].sum()}")
        print(f"Total points: {df[fpl_settings.metric].sum()}")

if __name__ == '__main__':
    main()
