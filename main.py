
import pandas as pd

from config import settings
from src.data import fpl_players_data, fpl_seasons_data, data_store
from src.models import fpl_model
from src.optimization import fpl_optimize


def main() -> None:
    """Run the FPL pipeline to predict the optimal FPL players that will yield
    the most points in the upcoming fixtures, following FPL rules and constraints."""

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 500)

    metric: str = 'xP_predicted_sum'
    fpl_settings = settings.FPLSettings(metric=metric)

    # Fetch players, teams, fixtures
    fpl_data, fixtures_data = fpl_players_data.fetch_fpl_data()
    fpl_players_data.store_fpl_data(fpl_data=fpl_data, fixtures_data=fixtures_data)

    # Clean players
    df_players_raw: pd.DataFrame = pd.DataFrame(data=fpl_data['elements'])
    df_players_clean: pd.DataFrame = fpl_players_data.clean_players_data(
        df_players_raw=df_players_raw, teams_data=fpl_data['teams'], fixtures_data=fixtures_data,
        n_fixtures=fpl_settings.n_fixtures
        )
    # print(df_players_clean.info())
    data_store.save_df_to_csv(df=df_players_clean, file_path="fpl_data/clean/players_data_clean.csv")

    # Features players
    df_players_features: pd.DataFrame = fpl_players_data.set_players_features(
        df_players_clean=df_players_clean, settings=fpl_settings)
    # print(df_players_features.info())
    data_store.save_df_to_csv(df=df_players_features, file_path="fpl_data/clean/players_data_features.csv")

    # Predict (calculate)
    df_predict: pd.DataFrame =  fpl_players_data.set_fpl_predictions(
        df_players_features=df_players_features, settings=fpl_settings)
    # print(df_predict.info())
    data_store.save_df_to_csv(df=df_predict, file_path="fpl_data/predict/players_data_xP_formula.csv")

    df_team_opt: pd.DataFrame = fpl_optimize.select_team(df=df_predict, settings=fpl_settings)
    file_path: str = "fpl_data/predict/optimal_team_data_formula.csv"
    data_store.save_df_to_csv(df=df_team_opt, file_path=file_path)

    df_starters_opt: pd.DataFrame = fpl_optimize.select_starters(df=df_team_opt, settings=fpl_settings)
    file_path: str = "fpl_data/predict/optimal_starters_data_formula.csv"
    data_store.save_df_to_csv(df=df_starters_opt, file_path=file_path)

    # Model
    # Fetch seasons
    seasons_data, seasons_teams_data = fpl_seasons_data.fetch_seasons_data()
    fpl_seasons_data.store_seasons_data(seasons_data=seasons_data,
                                        seasons_teams_data=seasons_teams_data)

    # Clean seasons
    seasons_clean: dict[str, pd.DataFrame] = fpl_seasons_data.clean_seasons_data(
        seasons_data=seasons_data, seasons_teams_data=seasons_teams_data
    )
    # Features seasons
    seasons_features: dict[str, pd.DataFrame] = fpl_seasons_data.set_seasons_features(
        seasons_clean=seasons_clean, group_key='player_id', n_window=fpl_settings.n_form
    )

    # Merge seasons
    df_seasons_merged: pd.DataFrame = fpl_seasons_data.merge_seasons(seasons_features=seasons_features)
    df_seasons_merged = fpl_seasons_data.clean_df_seasons_merged(df_seasons_merged=df_seasons_merged)
    file_path = f"fpl_data/past/clean/past_seasons_merged.csv"
    data_store.save_df_to_csv(df=df_seasons_merged, file_path=file_path)
    print(df_seasons_merged.info())

    # Position models
    models_dict = fpl_model.set_position_models(df_seasons_merged=df_seasons_merged, test_season='2025-26')
    metrics_dict = fpl_model.evaluate_position_models(df_seasons_merged=df_seasons_merged,
                                                      models_dict=models_dict, test_season='2025-26')
    for key, value in metrics_dict.items():
        print(key, value)

    # all data for training
    models_dict = fpl_model.set_position_models(df_seasons_merged=df_seasons_merged, test_season=None)
    importances: dict[str, pd.Series[float]] = fpl_model.get_features_importances(models_dict=models_dict)
    for pos, imp in importances.items():
        print(f"\n{pos} features importance\n{imp.sort_values(ascending=False)}")

    # ML predictions
    df_predict_ml: pd.DataFrame = fpl_model.set_fpl_ml_predictions(df_players_features=df_players_features,
                                                                models_dict=models_dict)
    # print(df_predict_ml.info())
    data_store.save_df_to_csv(df=df_predict_ml, file_path="fpl_data/predict/players_data_xP_ML.csv")

    # Optimize
    df_team_opt: pd.DataFrame = fpl_optimize.select_team(df=df_predict_ml, settings=fpl_settings)
    file_path: str = "fpl_data/predict/optimal_team_data_ML.csv"
    data_store.save_df_to_csv(df=df_team_opt, file_path=file_path)

    print(f"\nTeam:\n{df_team_opt}")
    print(f"Total cost: {df_team_opt['now_cost'].sum()}")
    print(f"Total points: {df_team_opt[metric].sum()}")

    # df_starters_opt: pd.DataFrame = fpl_optimize.select_starters(df=df_team_opt, settings=fpl_settings)
    df_starters_opt: pd.DataFrame = fpl_optimize.select_starters(df=df_predict_ml, settings=fpl_settings)
    file_path: str = "fpl_data/predict/optimal_starters_data_ML.csv"
    data_store.save_df_to_csv(df=df_starters_opt, file_path=file_path)

    print(f"\nStarters:\n{df_starters_opt}")
    print(f"Total cost: {df_starters_opt['now_cost'].sum()}")
    print(f"Total points: {df_starters_opt[metric].sum()}")


if __name__ == '__main__':
    main()
