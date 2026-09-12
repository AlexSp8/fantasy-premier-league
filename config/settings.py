
from pydantic import BaseModel, Field

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# 'influence', 'creativity', 'threat', 'ict_index', 'starts'
# 'recoveries', 'tackles', 'clearances_blocks_interceptions'
# 'bps', 'total_points', 'points_per_game', 'form', 'event_points'
# 'value_form', 'value_season'
# 'corners_and_indirect_freekicks_order', 'direct_freekicks_order', 'penalties_order'
COLUMNS_KEEP = [
    'web_name', 'now_cost', 'minutes', 'expected_goals_per_90',
    'expected_assists_per_90', 'expected_goals_conceded_per_90', 'clean_sheets',
    'defensive_contribution', 'saves', 'yellow_cards', 'red_cards',
    'penalties_missed', 'penalties_saved', 'own_goals', 'bonus',
    'chance_of_playing_next_round', 'code', 'event_points',
    'can_select', 'status', 'team', 'element_type',
    # 'ep_next'
    ]

COLUMNS_RENAME = {
    'web_name': 'player_name', 'minutes': 'MIN',
    'expected_goals_per_90': 'xG_per_90', 'expected_assists_per_90': 'xA_per_90',
    'expected_goals_conceded_per_90': 'xGC_per_90', 'clean_sheets': 'CS',
    'defensive_contribution': 'DC', 'saves': 'SA', 'yellow_cards': 'YC',
    'red_cards': 'RC', 'penalties_missed': 'PM', 'penalties_saved': 'PS',
    'own_goals': 'OG', 'bonus': 'BP',
    'chance_of_playing_next_round': 'availability',
    'event_points': 'PTS_GW',
    # 'ep_next': 'xP_FPL'
    }

EXCLUDED_FROM_NEGATIVE = ['change', 'balance', 'xP', 'bps', 'points', 'form', 'value', 'PTS']

# No 'xPTS_GW'
ML_FEATURES_DICT: dict[str, list[str]] = {
    'GKP': ['xMIN', 'xGC', 'xCS', 'xBP', 'xSA', 'xYC', 'xRC', 'xPS', 'xOG', 'FDR_def',
        #     'xPTS_GW'
            ],
    'DEF': ['xMIN', 'xG', 'xA', 'xGC', 'xCS', 'xBP', 'xYC', 'xRC', 'xPM', 'xOG', 'FDR_def', 'FDR_off',
        #     'xPTS_GW'
            ],
    'MID': ['xMIN', 'xG', 'xA', 'xCS', 'xBP', 'xYC', 'xRC', 'xPM', 'xOG', 'FDR_def', 'FDR_off',
        #     'xPTS_GW'
            ],
    'FWD': ['xMIN', 'xG', 'xA', 'xBP', 'xYC', 'xRC', 'xPM', 'xOG', 'FDR_off',
        #     'xPTS_GW'
            ]
}


class FPLSettings(BaseModel):

    n_form: int = Field(default=3, gt=0)

    n_fixtures: int = Field(default=3, ge=1)

    metric: str = Field(default='xP_FPL_sum')

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
