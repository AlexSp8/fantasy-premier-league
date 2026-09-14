
import pandas as pd

from config.settings import FPLSettings
from src.data import project_data
from src.optimization import fpl_optimize


def main() -> None:
    """Run the FPL pipeline to predict the optimal FPL players that will yield
    the most points in the upcoming fixtures, following FPL rules and constraints."""

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 500)

    metric = 'xP_predicted_sum'
    fpl_settings = FPLSettings(metric=metric)

    # Fetch players, teams, fixtures, past seasons
    live_data, past_data = project_data.fetch_data(fpl_settings=fpl_settings, store=True)

    # Prepare players (live, past) data
    df_players, players_past_data = project_data.prepare_players_data(
        live_data=live_data, past_data=past_data, fpl_settings=fpl_settings, store=True
        )

    # Add Features players, seasons
    df_features, df_features_past = project_data.add_features(
        df_players=df_players, players_past_data=players_past_data, n_form=fpl_settings.n_form, store=True
        )

    # Predict
    df_predict = project_data.set_predictions(
        df_features=df_features, df_features_past=df_features_past, fpl_settings=fpl_settings, store=True
        )

    # Optimize
    df_team = fpl_optimize.select_team(df=df_predict, fpl_settings=fpl_settings, store=True)
    # df_starters = fpl_optimize.select_starters(df=df_team, fpl_settings=fpl_settings)
    df_starters = fpl_optimize.select_starters(df=df_predict, fpl_settings=fpl_settings, store=True)

    for df in [df_team, df_starters]:
        print(f"\nPlayers:\n{df}")
        print(f"Total cost: {df['now_cost'].sum()}")
        print(f"Total points: {df[fpl_settings.metric].sum()}")

if __name__ == '__main__':
    main()
