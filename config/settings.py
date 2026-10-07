
from typing import Literal
from enum import StrEnum

from pydantic import BaseModel, Field

from config import naming


class ColumnsRaw(StrEnum):

    PLAYERS = 'elements'
    PLAYER_ID = 'id'
    POSITION_ID = 'element_type'
    AVAILABILITY = 'chance_of_playing_next_round'
    CAN_BE_SELECTED = 'can_select'
    PLAYER_STATUS = 'status'
    PLAYER_HISTORY = 'history'
    TEAMS = 'teams'
    IS_FINISHED = 'finished'
    TEAM_H = 'team_h'
    TEAM_A = 'team_a'


class Columns(StrEnum):

    PLAYERS = naming.PLAYERS
    TEAMS = naming.TEAMS

    PLAYER_ID = naming.PLAYER + naming.SEPARATOR + naming.ID
    PLAYER_NAME = naming.PLAYER + naming.SEPARATOR + naming.NAME

    POSITION_ID = naming.POSITION + naming.SEPARATOR + naming.ID
    POSITION_NAME = naming.POSITION + naming.SEPARATOR + naming.NAME

    TEAM_ID = naming.TEAM + naming.SEPARATOR + naming.ID
    TEAM_NAME = naming.TEAM + naming.SEPARATOR + naming.NAME
    TEAM_NAME_FULL = 'team_name_full'

    FIXTURES = naming.FIXTURES

    IS_HOME = 'is_home'
    OPP_ID = 'opp_id'
    GW = naming.GAMEWEEK
    GW_HISTORY = naming.GAMEWEEK + naming.SEPARATOR + naming.HISTORY

    EXPECTED_GOALS = naming.EXPECTED + naming.GOAL
    EXPECTED_GOALS_CONCEDED = naming.EXPECTED + naming.GOAL_CONCEDED
    EXPECTED_ASSISTS = naming.EXPECTED + naming.ASSIST
    MINUTES = naming.MINUTES
    POINTS = naming.POINTS
    YELLOW_CARD = naming.YELLOW_CARD
    RED_CARD = naming.RED_CARD
    BONUS_POINTS = naming.BONUS_POINTS
    CLEAN_SHEET = naming.CLEAN_SHEET
    SAVES = naming.SAVES
    DEFENSIVE_CONTRIBUTIONS = naming.DEFENSIVE_CONTRIBUTIONS
    PENALTIES_SAVED = naming.PENALTY + naming.SEPARATOR + naming.SAVES
    PENALTIES_MISSED = naming.PENALTY + naming.SEPARATOR + naming.MISSES
    OWN_GOALS = naming.OWN + naming.SEPARATOR + naming.GOAL
    FDR = naming.FIXTURE_DIFFICULTY
    OFFENSE = naming.OFFENSE
    DEFENSE = naming.DEFENSE
    OPPONENT = naming.OPPONENT
    FEATURES_SUFFIX = '_form'


MINUTES_PER_GAME = 90

POSITION_MAP = {
    1: naming.GOALKEEPER, 2: naming.DEFENDER, 3: naming.MIDFIELDER, 4: naming.FORWARD
    }

LIVE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

LIVE_STORE_RAW = "fpl_data/raw/fpl_data_raw.json"
LIVE_PLAYERS_STORE_RAW = "fpl_data/raw/players_data_raw.csv"
LIVE_TEAMS_STORE_RAW = "fpl_data/raw/teams_data_raw.csv"
LIVE_FIXTURES_STORE_RAW = "fpl_data/raw/fixtures_data_raw.csv"
LIVE_GW_STORE_RAW = "fpl_data/raw/players_gw_data_raw.csv"

LIVE_PLAYERS_STORE_CLEAN = "fpl_data/clean/players_data_clean.csv"
LIVE_PLAYERS_STORE_FEATURES = "fpl_data/clean/players_data_features.csv"

PAST_STORE_RAW = "fpl_data/past/raw/season_{season}_raw.csv"
PAST_TEAMS_STORE_RAW = "fpl_data/past/raw/season_{season}_teams_raw.csv"

PAST_STORE_CLEAN = "fpl_data/past/clean/season_{season}_clean.csv"
PAST_STORE_FEATURES = "fpl_data/past/clean/season_{season}_features.csv"
PAST_STORE_FEATURES_MERGED = "fpl_data/past/clean/past_seasons_features_merged.csv"

PLAYERS_COLUMNS_KEEP = [
    'web_name', 'code', # player identifier
    'element_type', # map position name
    'team', # merge fixtures: GW, opp_id, is_home / teams: team_name, team_id_opp, team_name_opp
    'id', # merge GW features form (xG etc.)
    'now_cost', # optimization
    'event_points',
    # 'chance_of_playing_next_round',
    # 'ep_next'
    ]
PLAYERS_COLUMNS_RENAME = {
    'web_name': Columns.PLAYER_NAME,
    'id': Columns.PLAYER_ID, 'team': Columns.TEAM_ID,
    'event_points': Columns.POINTS, #'PTS_previous',
    'element_type': Columns.POSITION_ID,
    # 'chance_of_playing_next_round': 'availability',
    # 'ep_next': 'xP_FPL'
    }

FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FIXTURES_COLUMNS_KEEP = [
    'finished', # keep only upcoming
    'event', # GW identifier
    'team_a', # merge to players
    'team_h' # merge to players
    ]
FIXTURES_COLUMNS_RENAME = {'event': Columns.GW}

TEAMS_COLUMNS_KEEP = [
    'id', # merge to players
    'short_name', # team identifier
    # 'strength_attack_home', 'strength_attack_away',
    # 'strength_defence_home', 'strength_defence_away'
    ]
TEAMS_COLUMNS_RENAME = {
    "id": Columns.TEAM_ID, "short_name": Columns.TEAM_NAME,
    # 'strength_attack_home': 'off_home', 'strength_attack_away': 'off_away',
    # 'strength_defence_home': 'def_home', 'strength_defence_away': 'def_away'
    }

GW_URL = "https://fantasy.premierleague.com/api/element-summary/"
PLAYERS_GW_COLUMNS_RENAME = {
    'element': Columns.PLAYER_ID, 'total_points': Columns.POINTS,
    'round': Columns.GW, 'minutes': Columns.MINUTES,
    'expected_goals': Columns.EXPECTED_GOALS, 'expected_assists': Columns.EXPECTED_ASSISTS,
    'expected_goals_conceded': Columns.EXPECTED_GOALS_CONCEDED,
    'clean_sheets': Columns.CLEAN_SHEET, 'saves': Columns.SAVES,
    'defensive_contribution': Columns.DEFENSIVE_CONTRIBUTIONS,
    'yellow_cards': Columns.YELLOW_CARD, 'red_cards': Columns.RED_CARD,
    'penalties_missed': Columns.PENALTIES_MISSED, 'penalties_saved': Columns.PENALTIES_SAVED,
    'own_goals': Columns.OWN_GOALS, 'bonus': Columns.BONUS_POINTS
    }

# expected values from 2022-23, DC from 2025-26
PAST_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
PAST_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
PAST_PLAYERS_SUFFIX = "/gws/merged_gw.csv"
PAST_TEAMS_SUFFIX = "/teams.csv"

# missing 'code', 'team_id'
SEASONS_COLUMNS_KEEP = [
    'name', 'position', 'element', # identifiers
    'team', # merge teams: team_id, team_name
    'opponent_team', # merge team_name_opp
    'bonus', 'clean_sheets', 'expected_assists',
    'expected_goals', 'expected_goals_conceded', 'minutes', 'own_goals', 'penalties_missed',
    'penalties_saved', 'red_cards', 'yellow_cards', 'saves', 'total_points',
    'GW', 'was_home',
    # 'defensive_contribution'
    ]
SEASONS_COLUMNS_RENAME = {
    'name': Columns.PLAYER_NAME, 'position': Columns.POSITION_NAME,
    'element': Columns.PLAYER_ID, 'minutes': Columns.MINUTES,
    'expected_goals': Columns.EXPECTED_GOALS,
    'expected_assists': Columns.EXPECTED_ASSISTS,
    'expected_goals_conceded': Columns.EXPECTED_GOALS_CONCEDED,
    'clean_sheets': Columns.CLEAN_SHEET, 'saves': Columns.SAVES,
    'yellow_cards': Columns.YELLOW_CARD, 'red_cards': Columns.RED_CARD,
    'penalties_missed': Columns.PENALTIES_MISSED, 'penalties_saved': Columns.PENALTIES_SAVED,
    'own_goals': Columns.OWN_GOALS, 'bonus': Columns.BONUS_POINTS,
    'total_points': Columns.POINTS,
    'opponent_team': 'opp_id', 'was_home': 'is_home',
    'team': 'team_name_full'
    # 'defensive_contribution': 'DC'
    }

SEASONS_TEAMS_COLUMNS_KEEP = [
    'name', # merge to players
    'id', 'short_name', # team identifier
    # 'strength_attack_home', 'strength_attack_away',
    # 'strength_defence_home', 'strength_defence_away'
    ]
SEASONS_TEAMS_COLUMNS_RENAME = {
    'id': Columns.TEAM_ID, 'name': 'team_name_full', 'short_name': Columns.TEAM_NAME,
    # 'strength_attack_home': 'off_home', 'strength_attack_away': 'off_away',
    # 'strength_defence_home': 'def_home', 'strength_defence_away': 'def_away'
    }

EXCLUDED_FROM_NEGATIVE_CHECK = ['change', 'balance', 'xP', 'bps', 'points', 'form', 'value', 'PTS']

FEATURES_FORM = [
    'MIN_form', 'CS_form', 'BP_form', 'SA_form', 'YC_form', 'RC_form', 'PTS_form',
    'PM_form', 'PS_form', 'OG_form', 'xG_form', 'xA_form', 'xGC_form',
    # 'DC_form'
    ]

ML_FEATURES_POS = {
    'GKP': ['MIN_form', 'xGC_form', 'CS_form', 'BP_form', 'SA_form', 'fdr_def',
            # 'YC_form', 'RC_form', 'PS_form', 'OG_form',
            'PTS_form'
            ],
    'DEF': ['MIN_form', 'xG_form', 'xA_form', 'xGC_form', 'CS_form', 'BP_form', 'fdr_def', 'fdr_off',
            # 'YC_form', 'RC_form', 'PM_form', 'OG_form',
            'PTS_form'
            ],
    'MID': ['MIN_form', 'xG_form', 'xA_form', 'CS_form', 'BP_form', 'fdr_def', 'fdr_off',
            # 'YC_form', 'RC_form', 'PM_form', 'OG_form',
            'PTS_form'
            ],
    'FWD': ['MIN_form', 'xG_form', 'xA_form', 'BP_form', 'fdr_off',
            # 'YC_form', 'RC_form', 'PM_form', 'OG_form',
            'PTS_form'
            ]
}

POINTS_COEFFICIENTS = {
    'xA': 3, 'xPS': 5, 'xPM': 2, 'xOG': 2, 'xYC': 1, 'xRC': 3,
    'xMIN_low': 1, 'xMIN_high': 2, 'xMIN_bar': 60, 'xGC': 0.5, 'xDC': 2
}

POINTS_COEFFICIENTS_POS = {
    'GKP': {
        'xG': 10, 'xCS': 4, 'xSA': 1.0/3.0
    },
    'DEF': {
        'xG': 6, 'xCS': 4, 'xDC_bar': 10
    },
    'MID': {
        'xG': 5, 'xCS': 1, 'xDC_bar': 12
    },
    'FWD': {
        'xG': 4
    },
}

POSITION_CONSTRAINTS = {
    'GKP': {'total': 2, 'min_start': 1, 'max_start': 1},
    'DEF': {'total': 5, 'min_start': 3, 'max_start': 5},
    'MID': {'total': 5, 'min_start': 2, 'max_start': 5},
    'FWD': {'total': 3, 'min_start': 1, 'max_start': 3},
}

TEAM_CONSTRAINTS = {
    'starters': 11, 'max_players_per_team': 3, 'budget': 1000,
    'total': sum(pos['total'] for pos in POSITION_CONSTRAINTS.values())
}


class FPLSettings(BaseModel):
    """Class for FPL-specific settings that can be tuned."""

    n_form: int = Field(default=3, gt=0)

    n_fixtures: int = Field(default=3, ge=1)

    metric: Literal['xP_predicted_sum'] = Field(default="xP_predicted_sum")

    prediction_model: Literal['formula', 'xgboost'] = Field(default='xgboost')

    ml_target: Literal['PTS'] = Field(default='PTS')

    features_scale: Literal['form', 'form_per_90'] = Field(default='form')


class DataURLs:
    """Class-container of URLs for fetching raw data."""

    LIVE = "https://fantasy.premierleague.com/api/bootstrap-static/"
    FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
    GW = "https://fantasy.premierleague.com/api/element-summary/"
    PAST = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
    PAST_PLAYERS_SUFFIX = "/gws/merged_gw.csv"
    PAST_TEAMS_SUFFIX = "/teams.csv"


class FilePathsRaw:
    """Class-container of file paths for fetching or storing raw data."""

    LIVE_STORE = "fpl_data/raw/fpl_data_raw.json"
    LIVE_PLAYERS_STORE = "fpl_data/raw/players_data_raw.csv"
    LIVE_TEAMS_STORE = "fpl_data/raw/teams_data_raw.csv"
    LIVE_FIXTURES_STORE = "fpl_data/raw/fixtures_data_raw.csv"
    LIVE_GW_STORE = "fpl_data/raw/players_gw_data_raw.csv"

    PAST_STORE = "fpl_data/past/raw/season_{season}_raw.csv"
    PAST_TEAMS_STORE = "fpl_data/past/raw/season_{season}_teams_raw.csv"


class FilePathsClean:
    """Class-container of file paths for fetching or storing of clean data."""

    LIVE_PLAYERS_STORE_CLEAN = "fpl_data/clean/players_data_clean.csv"
    PAST_STORE_CLEAN = "fpl_data/past/clean/season_{season}_clean.csv"
