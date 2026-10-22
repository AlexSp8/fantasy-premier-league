
from typing import Literal
from enum import StrEnum

from pydantic import BaseModel, Field

from config.naming import Names


class ColumnsRaw(StrEnum):

    AVAILABILITY = 'chance_of_playing_next_round'
    CAN_BE_SELECTED = 'can_select'
    PLAYER_STATUS = 'status'


class Columns(StrEnum):

    PLAYER_ID = Names.PLAYER + Names.ID_SUFFIX
    PLAYER_NAME = Names.PLAYER + Names.SEPARATOR + Names.NAME
    PLAYER_CODE = Names.PLAYER + Names.SEPARATOR + Names.CODE
    PLAYER_COST = Names.PLAYER + Names.SEPARATOR + Names.COST

    POSITION_ID = Names.POSITION + Names.ID_SUFFIX
    POSITION_NAME = Names.POSITION + Names.SEPARATOR + Names.NAME

    TEAM_ID = Names.TEAM + Names.ID_SUFFIX
    TEAM_NAME = Names.TEAM + Names.SEPARATOR + Names.NAME
    TEAM_NAME_FULL = TEAM_NAME + Names.SEPARATOR + 'full'

    SEASON = Names.SEASON

    IS_HOME = 'is' + Names.HOME_SUFFIX
    IS_FINISHED = 'is' + Names.SEPARATOR + Names.FINISHED
    OPPONENT_ID = Names.OPPONENT + Names.ID_SUFFIX
    GW = Names.GAMEWEEK
    HOME_TEAM_ID = Names.TEAM + Names.HOME_SUFFIX + Names.ID_SUFFIX
    AWAY_TEAM_ID = Names.TEAM + Names.AWAY_SUFFIX + Names.ID_SUFFIX
    OFFENSE_HOME = Names.OFFENSE + Names.HOME_SUFFIX
    DEFENSE_HOME = Names.DEFENSE + Names.HOME_SUFFIX
    OFFENSE_AWAY = Names.OFFENSE + Names.AWAY_SUFFIX
    DEFENSE_AWAY = Names.DEFENSE + Names.AWAY_SUFFIX

    EXPECTED_POINTS = Names.EXPECTED + Names.POINTS
    EXPECTED_POINTS_PREDICTED = Names.EXPECTED + Names.POINTS + Names.PREDICTED_SUFFIX
    EXPECTED_POINTS_PREDICTED_SUM = Names.EXPECTED + Names.POINTS + Names.PREDICTED_SUFFIX + Names.SUM_SUFFIX

    EXPECTED_GOALS = Names.EXPECTED + Names.GOAL
    EXPECTED_GOALS_FORM = EXPECTED_GOALS + Names.FORM_SUFFIX

    EXPECTED_GOALS_CONCEDED = Names.EXPECTED + Names.GOAL_CONCEDED
    EXPECTED_GOALS_CONCEDED_FORM = EXPECTED_GOALS_CONCEDED + Names.FORM_SUFFIX

    EXPECTED_ASSISTS = Names.EXPECTED + Names.ASSIST
    EXPECTED_ASSISTS_FORM = EXPECTED_ASSISTS + Names.FORM_SUFFIX

    MINUTES = Names.MINUTES
    MINUTES_FORM = MINUTES + Names.FORM_SUFFIX

    POINTS = Names.POINTS

    YELLOW_CARDS = Names.YELLOW_CARD
    YELLOW_CARDS_FORM = YELLOW_CARDS + Names.FORM_SUFFIX

    RED_CARDS = Names.RED_CARD
    RED_CARDS_FORM = RED_CARDS + Names.FORM_SUFFIX

    BONUS_POINTS = Names.BONUS_POINTS
    BONUS_POINTS_FORM = BONUS_POINTS + Names.FORM_SUFFIX

    CLEAN_SHEETS = Names.CLEAN_SHEET
    CLEAN_SHEETS_FORM = CLEAN_SHEETS + Names.FORM_SUFFIX

    SAVES = Names.SAVES
    SAVES_FORM = SAVES + Names.FORM_SUFFIX

    DEFENSIVE_CONTRIBUTIONS = Names.DEFENSIVE_CONTRIBUTIONS
    DEFENSIVE_CONTRIBUTIONS_FORM = DEFENSIVE_CONTRIBUTIONS + Names.FORM_SUFFIX

    PENALTIES_SAVED = Names.PENALTY + Names.SEPARATOR + Names.SAVES
    PENALTIES_SAVED_FORM = PENALTIES_SAVED + Names.FORM_SUFFIX

    PENALTIES_MISSED = Names.PENALTY + Names.SEPARATOR + Names.MISSES
    PENALTIES_MISSED_FORM = PENALTIES_MISSED + Names.FORM_SUFFIX

    OWN_GOALS = Names.OWN_GOALS
    OWN_GOALS_FORM = OWN_GOALS + Names.FORM_SUFFIX

    FDR_OFF = Names.FIXTURE_DIFFICULTY + Names.SEPARATOR + Names.OFFENSE
    FDR_DEF = Names.FIXTURE_DIFFICULTY + Names.SEPARATOR + Names.DEFENSE


MINUTES_PER_GAME = 90

POSITION_MAP = {
    1: Names.GOALKEEPER, 2: Names.DEFENDER, 3: Names.MIDFIELDER, 4: Names.FORWARD
    }

URL = {
    Names.LIVE: "https://fantasy.premierleague.com/api/bootstrap-static/",
    Names.FIXTURES: "https://fantasy.premierleague.com/api/fixtures/",
    Names.GW_HISTORY: "https://fantasy.premierleague.com/api/element-summary/",
    Names.PAST: "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
}
PAST_PLAYERS_SUFFIX = "/gws/merged_gw.csv"
PAST_TEAMS_SUFFIX = "/teams.csv"

STORE_FILE_PATH = {
    Names.RAW: {
        Names.LIVE: "fpl_data/raw/fpl_data_raw.json",
        Names.PLAYERS: "fpl_data/raw/players_data_raw.csv",
        Names.TEAMS: "fpl_data/raw/teams_data_raw.csv",
        Names.FIXTURES: "fpl_data/raw/fixtures_data_raw.csv",
        Names.GW_HISTORY: "fpl_data/raw/players_gw_data_raw.csv",
        Names.PAST_PLAYERS: "fpl_data/raw/past/season_{season}_raw.csv",
        Names.PAST_TEAMS: "fpl_data/raw/past/season_{season}_teams_raw.csv"
    },
    Names.CLEAN: {
        Names.LIVE: "fpl_data/clean/players_data_clean.csv",
        Names.PAST: "fpl_data/clean/past/season_{season}_clean.csv"
    },
    'features': {
        Names.LIVE: "fpl_data/clean/players_data_features.csv",
        Names.PAST: "fpl_data/clean/past/season_{season}_features.csv",
        'past_merged': "fpl_data/clean/past/past_seasons_features_merged.csv"
    }
}

PLAYERS_COLUMNS_RENAME = {
    'web_name': Columns.PLAYER_NAME, 'code': Columns.PLAYER_CODE,
    'id': Columns.PLAYER_ID, 'now_cost': Columns.PLAYER_COST,
    'event_points': Columns.POINTS,
    'element_type': Columns.POSITION_ID, 'team': Columns.TEAM_ID,
    # 'chance_of_playing_next_round': 'availability',
    # 'ep_next': Names.EXPECTED + Names.POINTS + '_fpl'
    }

FIXTURES_COLUMNS_RENAME = {
    'event': Columns.GW, 'finished': Columns.IS_FINISHED,
    'team_h': Columns.HOME_TEAM_ID, 'team_a': Columns.AWAY_TEAM_ID,
    }

TEAMS_COLUMNS_RENAME = {
    'id': Columns.TEAM_ID, 'short_name': Columns.TEAM_NAME,
    # 'strength_attack_home': Columns.OFFENSE_HOME, 'strength_attack_away': Columns.OFFENSE_AWAY,
    # 'strength_defence_home': Columns.DEFENSE_HOME, 'strength_defence_away': Columns.DEFENSE_AWAY
    }

PLAYERS_GW_COLUMNS_RENAME = {
    'element': Columns.PLAYER_ID, 'total_points': Columns.POINTS,
    'round': Columns.GW, 'minutes': Columns.MINUTES,
    'expected_goals': Columns.EXPECTED_GOALS, 'expected_assists': Columns.EXPECTED_ASSISTS,
    'expected_goals_conceded': Columns.EXPECTED_GOALS_CONCEDED,
    'clean_sheets': Columns.CLEAN_SHEETS, 'saves': Columns.SAVES,
    'defensive_contribution': Columns.DEFENSIVE_CONTRIBUTIONS,
    'yellow_cards': Columns.YELLOW_CARDS, 'red_cards': Columns.RED_CARDS,
    'penalties_missed': Columns.PENALTIES_MISSED, 'penalties_saved': Columns.PENALTIES_SAVED,
    'own_goals': Columns.OWN_GOALS, 'bonus': Columns.BONUS_POINTS
    }

# expected values from 2022-23, DC from 2025-26
PAST_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']

# missing 'code', 'team_id'
SEASONS_COLUMNS_RENAME = {
    'name': Columns.PLAYER_NAME, 'position': Columns.POSITION_NAME,
    'element': Columns.PLAYER_ID, 'minutes': Columns.MINUTES,
    'expected_goals': Columns.EXPECTED_GOALS,
    'expected_assists': Columns.EXPECTED_ASSISTS,
    'expected_goals_conceded': Columns.EXPECTED_GOALS_CONCEDED,
    'clean_sheets': Columns.CLEAN_SHEETS, 'saves': Columns.SAVES,
    'yellow_cards': Columns.YELLOW_CARDS, 'red_cards': Columns.RED_CARDS,
    'penalties_missed': Columns.PENALTIES_MISSED, 'penalties_saved': Columns.PENALTIES_SAVED,
    'own_goals': Columns.OWN_GOALS, 'bonus': Columns.BONUS_POINTS,
    'total_points': Columns.POINTS, 'GW': Columns.GW,
    'opponent_team': Columns.OPPONENT_ID, 'was_home': Columns.IS_HOME,
    'team': Columns.TEAM_NAME_FULL,
    # 'defensive_contribution': Columns.DEFENSIVE_CONTRIBUTIONS
    }

SEASONS_TEAMS_COLUMNS_RENAME = {
    'id': Columns.TEAM_ID, 'name': Columns.TEAM_NAME_FULL, 'short_name': Columns.TEAM_NAME,
    # 'strength_attack_home': Columns.OFFENSE_HOME, 'strength_attack_away': Columns.OFFENSE_AWAY,
    # 'strength_defence_home': Columns.DEFENSE_HOME, 'strength_defence_away': Columns.DEFENSE_AWAY
    }

EXCLUDED_FROM_NEGATIVE_CHECK = [
    'change', 'balance', 'value', Columns.EXPECTED_POINTS, Names.BONUS_POINTS, Names.POINTS, Names.FORM
    ]

FEATURES = [
    Columns.MINUTES, Columns.CLEAN_SHEETS, Columns.BONUS_POINTS, Columns.SAVES,
    Columns.YELLOW_CARDS, Columns.RED_CARDS, Columns.POINTS, Columns.PENALTIES_MISSED,
    Columns.PENALTIES_SAVED, Columns.OWN_GOALS, Columns.EXPECTED_GOALS,
    Columns.EXPECTED_ASSISTS, Columns.EXPECTED_GOALS_CONCEDED,
    # Columns.DEFENSIVE_CONTRIBUTIONS
    ]
FEATURES_FORM = [c + Names.FORM_SUFFIX for c in FEATURES]

ML_FEATURES_POS = {
    Names.GOALKEEPER: [
        Columns.MINUTES, Columns.EXPECTED_GOALS_CONCEDED, Columns.CLEAN_SHEETS,
        Columns.BONUS_POINTS, Columns.SAVES, Columns.FDR_DEF,
        # Columns.YELLOW_CARDS, Columns.RED_CARDS, Columns.PENALTIES_SAVED, Columns.OWN_GOALS,
        Columns.POINTS
        ],
    Names.DEFENDER: [
        Columns.MINUTES, Columns.EXPECTED_GOALS, Columns.EXPECTED_ASSISTS, Columns.EXPECTED_GOALS_CONCEDED,
        Columns.CLEAN_SHEETS, Columns.BONUS_POINTS, Columns.FDR_DEF, Columns.FDR_OFF,
        # Columns.YELLOW_CARDS, Columns.RED_CARDS, Columns.PENALTIES_MISSED, Columns.OWN_GOALS,
        Columns.POINTS
        ],
    Names.MIDFIELDER: [
        Columns.MINUTES, Columns.EXPECTED_GOALS, Columns.EXPECTED_ASSISTS, Columns.CLEAN_SHEETS,
        Columns.BONUS_POINTS, Columns.FDR_DEF, Columns.FDR_OFF,
        # Columns.YELLOW_CARDS, Columns.RED_CARDS, Columns.PENALTIES_MISSED, Columns.OWN_GOALS,
        Columns.POINTS
        ],
    Names.FORWARD: [
        Columns.MINUTES, Columns.EXPECTED_GOALS, Columns.EXPECTED_ASSISTS, Columns.BONUS_POINTS,
        Columns.FDR_OFF,
        # Columns.YELLOW_CARDS, Columns.RED_CARDS, Columns.PENALTIES_MISSED, Columns.OWN_GOALS,
        Columns.POINTS
        ]
}
ML_FEATURES_POS_FORM = {}
for pos, ft in ML_FEATURES_POS.items():
    ML_FEATURES_POS_FORM[pos] = [c + Names.FORM_SUFFIX for c in ft if Names.FIXTURE_DIFFICULTY not in c]

POINTS_COEFFICIENTS = {
    Names.ASSIST: 3, Columns.PENALTIES_SAVED: 5, Columns.PENALTIES_MISSED: 2,
    Names.OWN_GOALS: 2, Names.YELLOW_CARD: 1, Names.RED_CARD: 3,
    'xMIN_low': 1, 'xMIN_high': 2, 'xMIN_bar': 60, Names.GOAL_CONCEDED: 0.5,
    Names.DEFENSIVE_CONTRIBUTIONS: 2
}

POINTS_COEFFICIENTS_POS = {
    Names.GOALKEEPER: {Names.GOAL: 10, Names.CLEAN_SHEET: 4, Names.SAVES: 1.0/3.0},
    Names.DEFENDER: {Names.GOAL: 6, Names.CLEAN_SHEET: 4, 'xDC_bar': 10},
    Names.MIDFIELDER: {Names.GOAL: 5, Names.CLEAN_SHEET: 1, 'xDC_bar': 12},
    Names.FORWARD: {Names.GOAL: 4},
}

POSITION_CONSTRAINTS = {
    Names.GOALKEEPER: {Names.TOTAL: 2, 'min_start': 1, 'max_start': 1},
    Names.DEFENDER: {Names.TOTAL: 5, 'min_start': 3, 'max_start': 5},
    Names.MIDFIELDER: {Names.TOTAL: 5, 'min_start': 2, 'max_start': 5},
    Names.FORWARD: {Names.TOTAL: 3, 'min_start': 1, 'max_start': 3},
}

TEAM_CONSTRAINTS = {
    Names.STARTERS: 11, 'max_players_per_team': 3, Names.BUDGET: 1000,
    Names.TOTAL: sum(pos[Names.TOTAL] for pos in POSITION_CONSTRAINTS.values())
}


class FPLSettings(BaseModel):
    """Class for FPL-specific settings that can be tuned."""

    n_form: int = Field(default=3, gt=0)

    n_fixtures: int = Field(default=3, ge=1)

    metric: Literal[
        Columns.EXPECTED_POINTS_PREDICTED_SUM
        ] = Field(default=Columns.EXPECTED_POINTS_PREDICTED_SUM)

    prediction_model: Literal[Names.FORMULA, Names.XGBOOST] = Field(default=Names.XGBOOST)

    ml_target: Literal[Names.POINTS] = Field(default=Names.POINTS)

    features_scale: Literal[Names.FORM, Names.FORM_PER_90] = Field(default=Names.FORM)
