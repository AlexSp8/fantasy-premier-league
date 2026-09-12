
from typing import Any

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from config import settings
from config.settings import FPLSettings
from src.data import data_fetch, data_store
from src.data.data_cleaner import DataCleaner
from src.data.data_features import DataFeatures


def fetch_fpl_data():

    # Players, Teams
    fpl_data = data_fetch.fetch_json_data(
        url="https://fantasy.premierleague.com/api/bootstrap-static/", file_path="fpl_data/raw/bootstrap_raw.json")
    # print(f"\n{list(fpl_data.keys())}")

    # gw_data = data_fetch.fetch_json_data(
    #     url="https://fantasy.premierleague.com/api/element-summary/1/", file_path="fpl_data/raw/player_1_raw.json")
    # data_store.store_data_to_json(data=gw_data, file_path="fpl_data/raw/player_1_raw.json")
    # data_store.save_list_dict_to_csv(data_list_dict=gw_data['fixtures'],
    #                                 file_path= "fpl_data/raw/player_1_fixtures_raw.csv")
    # data_store.save_list_dict_to_csv(data_list_dict=gw_data['history'],
    #                                 file_path= "fpl_data/raw/player_1_history_raw.csv")
    # data_store.save_list_dict_to_csv(data_list_dict=gw_data['history_past'],
    #                                 file_path= "fpl_data/raw/player_1_history_past_raw.csv")

    # Season fixtures
    fixtures_data = data_fetch.fetch_json_data(
        url="https://fantasy.premierleague.com/api/fixtures/", file_path='fpl_data/raw/fixtures_data_raw.json')

    return fpl_data, fixtures_data

def store_fpl_data(fpl_data: dict[str, Any], fixtures_data: list[dict[str, Any]]) -> None:

    # Players, Teams
    data_store.store_data_to_json(data=fpl_data, file_path="fpl_data/raw/bootstrap_raw.json")
    data_store.save_list_dict_to_csv(data_list_dict=fpl_data['elements'], file_path="fpl_data/raw/players_data_raw.csv")
    data_store.save_list_dict_to_csv(data_list_dict=fpl_data['teams'], file_path="fpl_data/raw/teams_data_raw.csv")

    # Season fixtures
    data_store.store_data_to_json(data=fixtures_data, file_path="fpl_data/raw/fixtures_data_raw.json")
    data_store.save_list_dict_to_csv(data_list_dict=fixtures_data, file_path= "fpl_data/raw/fixtures_data_raw.csv")

def clean_players_data(df_players_raw: pd.DataFrame, teams_data: list[dict[str, Any]],
    fixtures_data: list[dict[str, Any]], n_fixtures: int = 1) -> pd.DataFrame:
    """Perform the cleaning process for the FPL players data on the raw pd.DataFrame.
    Args:
        df_players_raw: The raw players pd.DataFrame.
        teams_data: The FPL teams list of data dictionaries.
        fixtures_data: The FPL fixtures list of data dictionaries.
        n_fixtures (optional): The number of future fixtures to consider. Defaults to 1.
    Returns:
        The cleaned players pd.DataFrame."""

    df_clean: pd.DataFrame = df_players_raw.copy()

    df_clean = df_clean[settings.COLUMNS_KEEP]

    available_players: pd.Series[bool] = ((df_clean['can_select']) & (df_clean['status'] != 'u') &
                                          (df_clean['status'] != 'i') & (df_clean['status'] != 's'))

    df_clean = df_clean[available_players].rename(columns=settings.COLUMNS_RENAME)

    rows: pd.Series[bool] = (df_clean['status'] == 'a')
    column: str = 'availability'
    df_clean.loc[rows, column] = 100

    df_clean = df_clean[df_clean['availability'] == 100].copy()

    df_clean['position_name'] = df_clean['element_type'].map(settings.POSITION_MAP)

    df_teams: pd.DataFrame = pd.DataFrame(data=teams_data)[[
        'id', 'short_name',
        # 'strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away'
        ]]
    df_teams = df_teams.rename(
        columns={"id": "team", "short_name": "team_name",
        # 'strength_attack_home': 'OFF_home', 'strength_attack_away': 'OFF_away',
        # 'strength_defence_home': 'DEF_home', 'strength_defence_away': 'DEF_away'
        })

    df_clean = df_clean.merge(df_teams, on='team', how="inner")

    df_fixtures: pd.DataFrame = pd.DataFrame(data=fixtures_data)[['finished', 'event', 'team_a', 'team_h']]
    df_fixtures = df_fixtures.rename(columns={'event': 'GW'})
    df_fixtures = df_fixtures[~df_fixtures['finished']]
    df_fixtures = df_fixtures.drop(columns=['finished'])

    df_home = df_fixtures.rename(columns={'team_a': 'opponent', 'team_h': 'team'})
    df_home['is_home'] = True

    df_away = df_fixtures.rename(columns={'team_h': 'opponent', 'team_a': 'team'})
    df_away['is_home'] = False

    df_home_away = pd.concat([df_home, df_away], ignore_index=True)
    df_home_away = df_home_away.sort_values(by='GW', ascending=True).groupby('team').head(n_fixtures)

    df_clean = df_clean.merge(df_home_away, on='team', how="inner")

    df_clean = df_clean.merge(df_teams, left_on="opponent", right_on='team', how="inner",
                              suffixes=("", "_opponent"))

    df_clean = df_clean.sort_values(by=['GW', 'team_name', 'element_type'], ascending=[True, True, True])

    df_clean = df_clean.drop(columns=['can_select', 'status', 'team', 'element_type',
                                      'opponent', 'team_opponent',
                                    #   'OFF_home', 'OFF_away', 'DEF_home', 'DEF_away'
                                      ])

    cleaner = DataCleaner(df=df_clean)
    cleaner.clean_data()

    df_negative: pd.DataFrame = cleaner.flag_negative_values(excluded_keywords=settings.EXCLUDED_FROM_NEGATIVE)
    if not df_negative.empty:
        raise ValueError("DataFrame contains incorrect negative values!")

    return cleaner.df.copy()

def set_players_features(df_players_clean: pd.DataFrame, settings: FPLSettings) -> pd.DataFrame:
    """Set FPL players features for future use on predicting optimal FPL stats.
    Args:
        df_players_clean: The cleaned pd.DataFrame containing FPL players data.
        settings: The FPLSettings instance specifying FPL constraints.
    Returns:
        A pd.DataFrame containing additional FPL features."""

    fpl_features = DataFeatures(df=df_players_clean)

    # Average first (form)
    columns: list[str] = ['MIN', 'YC', 'RC', 'PM', 'PS', 'OG', 'BP', 'CS', 'DC', 'SA',
                          'PTS_GW'
                          ]
    fpl_features.convert_to_expected_from_total(columns=columns, n=settings.n_form)

    # Convert to per 90
    columns.remove('MIN')
    columns = ['x'+c for c in columns]
    fpl_features.convert_to_per_x(columns=columns, x_col='xMIN', x=90.0)

    bounds_dict: dict[str, list[float]] = {
        'xYC_per_90': [0.0, 2.0], 'xRC_per_90': [0.0, 1.0],
        'xBP_per_90': [0.0, 3.0], 'xCS_per_90': [0.0, 1.0]}
    fpl_features.set_features_bounds(bounds_dict)

    # Expected next
    columns = ['xG_per_90', 'xA_per_90', 'xGC_per_90', 'xCS_per_90', 'xDC_per_90', 'xSA_per_90',
               'xYC_per_90', 'xRC_per_90', 'xPM_per_90', 'xPS_per_90', 'xOG_per_90', 'xBP_per_90',
               'xPTS_GW_per_90'
               ]
    fpl_features.convert_from_per_x(columns=columns, x_col='xMIN', x=90.0)

    df: pd.DataFrame = fpl_features.df.copy()

    if 'OFF_home_opponent' not in df.columns or 'OFF_away_opponent' not in df.columns:
        df['OFF_home_opponent'], df['OFF_away_opponent'] = set_opponent_strength(df=df, metric='xG')
    if 'DEF_home_opponent' not in df.columns or 'DEF_away_opponent' not in df.columns:
        df['DEF_home_opponent'], df['DEF_away_opponent'] = set_opponent_strength(df=df, metric='xGC')
        df['DEF_home_opponent'], df['DEF_away_opponent'] = 1.0/df['DEF_home_opponent'], 1.0/df['DEF_away_opponent']

    df['FDR_off'], df['FDR_def'] = set_fixture_difficulty(df)
    df['FDR_off'] = df['FDR_off'].fillna(1.0)
    df['FDR_def'] = df['FDR_def'].fillna(1.0)
    df['FDR_off'] = df['FDR_off'].clip(lower=0.0, upper=2.0)
    df['FDR_def'] = df['FDR_def'].clip(lower=0.0, upper=2.0)

    # columns = ['xG', 'xA', 'xGC', 'xCS', 'xDC', 'xSA',
    #            'xYC', 'xRC', 'xPM', 'xPS', 'xOG', 'xBP', 'xMIN']
    # print(df[columns].isna().sum())

    return df.copy()

def set_opponent_strength(df: pd.DataFrame, metric: str
                          ) -> tuple[pd.Series[float], pd.Series[float]]:
    """Calculate opponent strength based on the opponent's specified metric.
    Args:
        df: The pd.DataFrame containing the FPL players data.
        metric: The metric upon which strength is measured.
    Returns:
        A tuple (pd.Series) for the opponent's strength."""

    df_home = df[df['is_home']]
    sum_min_h: pd.Series[float] = df_home.groupby('team_name')['xMIN'].sum().replace(0, np.nan)
    sum_strength_h: pd.Series[float] = df_home.groupby('team_name')[metric].sum()

    df_away = df[~df['is_home']]
    sum_min_a: pd.Series[float] = df_away.groupby('team_name')['xMIN'].sum().replace(0, np.nan)
    sum_strength_a: pd.Series[float] = df_away.groupby('team_name')[metric].sum()

    #divide xGC by active players
    if 'xGC' in metric:
        count_played_h: pd.Series[int] = df_home[df_home['xMIN'] > 0].groupby('team_name')['xMIN'].count()
        count_played_a: pd.Series[int] = df_away[df_away['xMIN'] > 0].groupby('team_name')['xMIN'].count()
        sum_strength_h /= count_played_h
        sum_strength_a /= count_played_a

    # print(metric, sum_min_a['COV'], sum_strength_a['COV'])
    # print(metric, sum_min_h['COV'], sum_strength_h['COV'])

    strength_h_series = 90*(sum_strength_h/sum_min_h)
    strength_a_series = 90*(sum_strength_a/sum_min_a)

    strength_h = df['team_name_opponent'].map(strength_h_series)
    strength_a = df['team_name_opponent'].map(strength_a_series)

    return strength_h, strength_a

def set_fixture_difficulty(df: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """Calculate fixture difficulty (offensive and defensive)
    based on the opponent's offensive/defensive strength.
    Args:
        df: The pd.DataFrame containing the FPL players data.
    Returns:
        A tuple (NDArray) for the offensive/defensive fixture difficulty ratings."""

    def_away_mean: float = df['DEF_away_opponent'].mean()
    def_home_mean: float = df['DEF_home_opponent'].mean()
    off_away_mean: float = df['OFF_away_opponent'].mean()
    off_home_mean: float = df['OFF_home_opponent'].mean()

    fdr_off: NDArray = np.where(df['is_home'], df['DEF_away_opponent']/def_away_mean,
                                df['DEF_home_opponent']/def_home_mean)
    fdr_def: NDArray = np.where(df['is_home'], df['OFF_away_opponent']/off_away_mean,
                                df['OFF_home_opponent']/off_home_mean)

    return fdr_off, fdr_def

def set_fpl_predictions(df_players_features: pd.DataFrame,
                                 settings: FPLSettings) -> pd.DataFrame:
    """Set FPL predictions for each player based on the analytical points formulas.
    Args:
        df_players_features: The pd.DataFrame containing the FPL features data.
        settings: The FPLSettings instance specifying FPL constraints.
    Returns:
        A pd.DataFrame containing the predictions for FPL points for each player."""

    df = df_players_features.copy()

    df['xP_predicted'] = calculate_expected_points(df=df, settings=settings)

    df = df.groupby('code', as_index=False).agg(
        player_name=('player_name', 'first'), position_name=('position_name', 'first'),
        team_name=('team_name', 'first'), availability=('availability', 'first'),
        now_cost=('now_cost', 'first'), xP_predicted_sum=('xP_predicted', 'sum'),
        # xP_FPL_sum=('xP_FPL', 'sum'),
        )
    df = df.sort_values(by=['team_name', 'position_name'], ascending=[True, True])

    return df.copy()

def calculate_expected_points(df: pd.DataFrame, settings: FPLSettings) -> pd.Series[float]:
    """Calculate the expected FPL points for each player based on
    expected values of the related quantities.
    Args:
        df: the pd.DataFrame containing the FPL players data.
        settings: The FPLSettings instance specifying FPL constraints.
    Returns:
        The pd.Series containing the 'xP_predicted' column."""

    fdr_off: pd.Series[float] = df['FDR_off']
    fdr_def: pd.Series[float] = df['FDR_def']

    xMIN: pd.Series[float] = df['xMIN'] # minutes

    xA: pd.Series[float] = df['xA']/fdr_off # assists
    xPS: pd.Series[float] = df['xPS'] # penalties saved
    xBP: pd.Series[float] = df['xBP'] # bonus points
    xYC: pd.Series[float] = df['xYC'] # yellow cards
    xRC: pd.Series[float] = df['xRC'] # red cards
    xOG: pd.Series[float] = df['xOG'] # own goals
    xPM: pd.Series[float] = df['xPM'] # penalties missed

    cxA = settings.xA_coefficient
    cxPS = settings.xPS_coefficient
    cxYC = settings.xYC_coefficient
    cxYRC = settings.xRC_coefficient
    cxOG = settings.xOG_coefficient
    cxPM = settings.xPM_coefficient

    cxMIN_low = settings.xMIN_low_coefficient
    cxMIN_high = settings.xMIN_high_coefficient
    cxMIN_barrier = settings.xMIN_barrier
    cxMIN_threshold = settings.xMIN_threshold
    cxMIN_penalty = settings.xMIN_penalty

    cxSA = settings.xSA_coefficient
    cxGC = settings.xGC_coefficient

    cxCS = settings.xCS_coefficient
    cxCS_mid = settings.xCS_coefficient_mid

    cxDC = settings.xDC_coefficient
    cxDC_barrier_def = settings.xDC_barrier_def
    cxDC_barrier_mid = settings.xDC_barrier_mid

    cxG_gkp = settings.xG_coefficient_gkp
    cxG_def = settings.xG_coefficient_def
    cxG_mid = settings.xG_coefficient_mid
    cxG_fwd = settings.xG_coefficient_fwd

    xC: pd.Series[float] = cxA*xA + cxPS*xPS + xBP - cxYC*xYC - cxYRC*xRC - cxOG*xOG - cxPM*xPM # common

    xMIN_less_than_60: pd.Series[bool] = xMIN < cxMIN_barrier
    xC[xMIN_less_than_60] += cxMIN_low
    min_minutes_threshold: float = cxMIN_threshold
    xMIN_0: pd.Series[bool] = xMIN < min_minutes_threshold
    xC[xMIN_0] -= cxMIN_penalty
    xMIN_more_than_60: pd.Series[bool] = xMIN >= cxMIN_barrier
    xC[xMIN_more_than_60] += cxMIN_high

    xG: pd.Series[float] = df['xG']/fdr_off
    xGC: pd.Series[float] = df['xGC']*fdr_def # goals conceded
    xSA: pd.Series[float] = df['xSA']*fdr_def # saves
    xCS: pd.Series[float] = df['xCS']/fdr_def # clean sheets
    xDC: pd.Series[float] = df['xDC']*fdr_def # defensive contributions

    xP: pd.Series[float] = pd.Series(index=df.index, dtype=float)

    gkp_mask = df["position_name"] == "GKP"
    xP[gkp_mask] = cxG_gkp*xG - cxGC*xGC + cxSA*xSA + cxCS*xCS + xC

    def_mask = df["position_name"] == "DEF"
    xP[def_mask] = cxG_def*xG - cxGC*xGC + cxCS*xCS + xC
    xDC10: pd.Series[bool] = xDC >= cxDC_barrier_def
    xP[def_mask & xDC10] += cxDC

    mid_mask = df["position_name"] == "MID"
    xP[mid_mask] = cxG_mid*xG + cxCS_mid*xCS + xC
    xDC12: pd.Series[bool] = xDC >= cxDC_barrier_mid
    xP[mid_mask & xDC12] += cxDC

    fwd_mask = df["position_name"] == "FWD"
    xP[fwd_mask] = cxG_fwd*xG + xC
    xP[fwd_mask & xDC12] += cxDC

    return xP
