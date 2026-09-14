
from pydantic import BaseModel, Field

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# 'corners_and_indirect_freekicks_order', 'direct_freekicks_order', 'penalties_order'
PLAYERS_COLUMNS_KEEP = [
    'web_name', 'now_cost', 'chance_of_playing_next_round', 'code', 'id',
    'can_select', 'status', 'team', 'element_type', 'event_points',
    # 'ep_next'
    ]
PLAYERS_COLUMNS_RENAME = {
    'web_name': 'player_name', 'id': 'player_id',
    'chance_of_playing_next_round': 'availability', 'team': 'team_id',
    'event_points': 'PTS',
    # 'ep_next': 'xP_FPL'
    }

TEAMS_COLUMNS_KEEP = [
    'id', 'short_name',
    # 'strength_attack_home', 'strength_attack_away',
    # 'strength_defence_home', 'strength_defence_away'
    ]
TEAMS_COLUMNS_RENAME = {
    "id": "team_id", "short_name": "team_name",
    # 'strength_attack_home': 'OFF_home', 'strength_attack_away': 'OFF_away',
    # 'strength_defence_home': 'DEF_home', 'strength_defence_away': 'DEF_away'
    }

FIXTURES_COLUMNS_KEEP = ['finished', 'event', 'team_a', 'team_h']
FIXTURES_COLUMNS_RENAME = {'event': 'GW'}

PLAYERS_GW_COLUMNS_KEEP = [
    'element', 'total_points', 'round', 'minutes',
    'expected_goals', 'expected_assists', 'expected_goals_conceded',
    'clean_sheets', 'defensive_contribution', 'saves', 'yellow_cards',
    'red_cards', 'penalties_missed', 'penalties_saved', 'own_goals', 'bonus',
    ]
PLAYERS_GW_COLUMNS_RENAME = {
    'element': 'player_id', 'total_points': 'PTS', 'round': 'GW', 'minutes': 'MIN',
    'expected_goals': 'xG', 'expected_assists': 'xA',
    'expected_goals_conceded': 'xGC', 'clean_sheets': 'CS',
    'defensive_contribution': 'DC', 'saves': 'SA', 'yellow_cards': 'YC',
    'red_cards': 'RC', 'penalties_missed': 'PM', 'penalties_saved': 'PS',
    'own_goals': 'OG', 'bonus': 'BP'
    }

# expected values from 2022-23, DC from 2025-26
PAST_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']

# missing 'code', 'team_id'
SEASONS_COLUMNS_KEEP = [
    'name', 'position', 'team', 'bonus', 'clean_sheets', 'expected_assists',
    'expected_goals', 'expected_goals_conceded', 'minutes', 'own_goals', 'penalties_missed',
    'penalties_saved', 'red_cards', 'yellow_cards', 'saves', 'total_points',
    'GW', 'opponent_team', 'was_home', 'element',
    # 'defensive_contribution'
    ]
SEASONS_COLUMNS_RENAME = {
    'name': 'player_name', 'position': 'position_name', 'element': 'player_id', 'minutes': 'MIN',
    'expected_goals': 'xG', 'expected_assists': 'xA', 'expected_goals_conceded': 'xGC',
    'clean_sheets': 'CS', 'saves': 'SA', 'yellow_cards': 'YC', 'red_cards': 'RC',
    'penalties_missed': 'PM', 'penalties_saved': 'PS', 'own_goals': 'OG', 'bonus': 'BP',
    'total_points': 'PTS', 'opponent_team': 'opponent_team_id', 'was_home': 'is_home',
    # 'defensive_contribution': 'DC'
    }

SEASONS_TEAMS_COLUMNS_KEEP = [
    'id', 'name', 'short_name', 'strength_attack_home', 'strength_attack_away',
    'strength_defence_home', 'strength_defence_away'
    ]
SEASONS_TEAMS_COLUMNS_RENAME = {
    'id': 'team_id', 'name': 'team', 'short_name': 'team_name',
    'strength_attack_home': 'OFF_home', 'strength_attack_away': 'OFF_away',
    'strength_defence_home': 'DEF_home', 'strength_defence_away': 'DEF_away'
    }

EXCLUDED_FROM_NEGATIVE_CHECK = ['change', 'balance', 'xP', 'bps', 'points', 'form', 'value', 'PTS']

FEATURES_FORM = [
    'MIN', 'CS', 'BP', 'SA', 'YC', 'RC', 'PTS',
    'PM', 'PS', 'OG', 'xG', 'xA', 'xGC',
    # 'DC'
    ]

PREDICTION_MODEL = 'xgboost'

ML_FEATURES_DICT = {
    'GKP': ['MIN_form', 'xGC_form', 'CS_form', 'BP_form', 'SA_form', 'YC_form', 'RC_form',
            'PS_form', 'OG_form', 'FDR_def',
        #     'PTS_form'
            ],
    'DEF': ['MIN_form', 'xG_form', 'xA_form', 'xGC_form', 'CS_form', 'BP_form',
            'YC_form', 'RC_form', 'PM_form', 'OG_form', 'FDR_def', 'FDR_off',
        #     'PTS_form'
            ],
    'MID': ['MIN_form', 'xG_form', 'xA_form', 'CS_form', 'BP_form', 'YC_form', 'RC_form',
            'PM_form', 'OG_form', 'FDR_def', 'FDR_off',
        #     'PTS_form'
            ],
    'FWD': ['MIN_form', 'xG_form', 'xA_form', 'BP_form', 'YC_form', 'RC_form', 'PM_form',
            'OG_form', 'FDR_off',
        #     'PTS_form'
            ]
}


class FPLSettings(BaseModel):

    live_url: str = Field(default="https://fantasy.premierleague.com/api/bootstrap-static/")
    fixtures_url: str = Field(default="https://fantasy.premierleague.com/api/fixtures/")
    gw_url: str = Field(default="https://fantasy.premierleague.com/api/element-summary/")

    past_url: str = Field(default="https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/")
    past_players_suffix: str = Field(default="/gws/merged_gw.csv")
    past_teams_suffix: str = Field(default="/teams.csv")

    n_form: int = Field(default=3, gt=0)

    n_fixtures: int = Field(default=3, ge=1)

    metric: str = Field(default="xP_Predicted_sum")

    total_budget: float = Field(default=1000, gt=0)

    total_team_size: int = Field(default=15, gt=0)
    starting_team_size: int = Field(default=11, gt=0)

    max_players_per_team: int = Field(default=3, gt=0)

    n_gkp: int = Field(default=2, ge=0)
    n_def: int = Field(default=5, ge=0)
    n_mid: int = Field(default=5, ge=0)
    n_fwd: int = Field(default=3, ge=0)

    n_gkp_start_min: int = Field(default=1, ge=0)
    n_def_start_min: int = Field(default=3, ge=0)
    n_mid_start_min: int = Field(default=2, ge=0)
    n_fwd_start_min: int = Field(default=1, ge=0)

    n_gkp_start_max: int = Field(default=1, ge=0)
    n_def_start_max: int = Field(default=5, ge=0)
    n_mid_start_max: int = Field(default=5, ge=0)
    n_fwd_start_max: int = Field(default=3, ge=0)

    xA_coefficient: int = Field(default=3, ge=0)
    xPS_coefficient: int = Field(default=5, ge=0)
    xYC_coefficient: int = Field(default=1, ge=0)
    xRC_coefficient: int = Field(default=3, ge=0)
    xOG_coefficient: int = Field(default=2, ge=0)
    xPM_coefficient: int = Field(default=2, ge=0)

    xMIN_low_coefficient: int = Field(default=1, ge=0)
    xMIN_high_coefficient: int = Field(default=2, ge=0)
    xMIN_barrier: int = Field(default=60, ge=0)
    xMIN_threshold: float = Field(default=0.1, ge=0)
    xMIN_penalty: int = Field(default=1, ge=0)

    xSA_coefficient: float = Field(default=1.0/3.0, ge=0)
    xGC_coefficient: float = Field(default=0.5, ge=0)

    xCS_coefficient: int = Field(default=4, ge=0)
    xCS_coefficient_mid: int = Field(default=1, ge=0)

    xDC_coefficient: int = Field(default=2, ge=0)
    xDC_barrier_def: int = Field(default=10, ge=0)
    xDC_barrier_mid: int = Field(default=12, ge=0)

    xG_coefficient_gkp: int = Field(default=10, ge=0)
    xG_coefficient_def: int = Field(default=6, ge=0)
    xG_coefficient_mid: int = Field(default=5, ge=0)
    xG_coefficient_fwd: int = Field(default=4, ge=0)
