
from enum import StrEnum


class NamesRaw(StrEnum):

    PLAYERS = 'elements'
    PLAYER_ID = 'id'
    PLAYER_HISTORY = 'history'
    TEAMS = 'teams'

class Names(StrEnum):

    SEPARATOR = '_'

    NAME = 'name'
    ID = 'id'
    CODE = 'code'
    COST = 'cost'

    PLAYERS = 'players'
    TEAMS = 'teams'

    PLAYER = 'player'
    POSITION = 'position'
    TEAM = 'team'
    OPPONENT = 'opponent'
    FIXTURES = 'fixtures'
    HOME = 'home'
    AWAY = 'away'
    FINISHED = 'finished'

    GAMEWEEK = 'gw'
    GW_HISTORY = GAMEWEEK + SEPARATOR + 'history'

    SEASON = 'season'
    PAST = 'past'

    PAST_TEAMS = PAST + SEPARATOR + TEAMS
    PAST_PLAYERS = PAST + SEPARATOR + PLAYERS

    LIVE = 'live'

    RAW = 'raw'
    CLEAN = 'clean'

    POINTS = 'PTS'
    MINUTES = 'MIN'
    GOAL = 'G'
    ASSIST = 'A'
    GOAL_CONCEDED = 'GC'
    OWN_GOALS = 'OG'
    CLEAN_SHEET = 'CS'
    YELLOW_CARD = 'YC'
    RED_CARD = 'RC'
    BONUS_POINTS = 'BP'
    SAVES = 'SAV'
    MISSES = 'MIS'
    DEFENSIVE_CONTRIBUTIONS = 'DC'
    PENALTY = 'PEN'
    FIXTURE_DIFFICULTY = 'fdr'
    OFFENSE = 'off'
    DEFENSE = 'def'

    EXPECTED = 'x'

    GOALKEEPER = 'GKP'
    DEFENDER = 'DEF'
    MIDFIELDER = 'MID'
    FORWARD = 'FWD'

    FORM = 'form'
    TOTAL = 'total'
    STARTERS = 'starters'
    BUDGET = 'budget'
    XGBOOST = 'xgb'
    FORMULA = 'formula'

    ID_SUFFIX = SEPARATOR + ID
    OPPONENT_SUFFIX = SEPARATOR + OPPONENT
    FORM_SUFFIX = SEPARATOR + FORM
    PREDICTED_SUFFIX = SEPARATOR + 'predicted'
    SUM_SUFFIX = SEPARATOR + 'sum'
    HOME_SUFFIX = SEPARATOR + HOME
    AWAY_SUFFIX = SEPARATOR + AWAY
    PER_90_SUFFIX = SEPARATOR + 'per' + SEPARATOR + '90'

    FORM_PER_90 = FORM + PER_90_SUFFIX
    FORM_PER_90_SUFFIX = SEPARATOR + FORM_PER_90
