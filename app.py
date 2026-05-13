from pathlib import Path
import traceback

import pandas as pd
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from nba_api.live.nba.endpoints import scoreboard, playbyplay

from src.predict import WinProbabilityPredictor


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

predictor = None
current_replay_id = 0
current_live_id = 0

replay_state = {
    "paused": False,
    "speed": 1.0,
    "selected_game_id": None,
}

PBP_PATH = Path("data/play_by_play.csv")
GAMES_PATH = Path("data/raw_games.csv")


TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}

TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751,
    "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}

TEAM_LOGOS = {
    abbr: f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"
    for abbr, team_id in TEAM_IDS.items()
}


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def clock_to_seconds_remaining(period: int, clock: str) -> int:
    if pd.isna(clock):
        return 0

    try:
        clock = str(clock)

        if clock.startswith("PT"):
            minutes_part = clock.split("M")[0].replace("PT", "")
            seconds_part = clock.split("M")[1].replace("S", "")
            minutes = int(minutes_part)
            seconds = int(float(seconds_part))
        elif ":" in clock:
            minutes, seconds = clock.split(":")
            minutes = int(minutes)
            seconds = int(float(seconds))
        else:
            return 0

        period_remaining = minutes * 60 + seconds
    except Exception:
        period_remaining = 0

    if period <= 4:
        return (4 - period) * 12 * 60 + period_remaining

    return period_remaining


def parse_home_away(row):
    team = row["TEAM_ABBREVIATION"]
    matchup = row["MATCHUP"]

    if " vs. " in matchup:
        home = team
        away = matchup.split(" vs. ")[1].strip()
    elif " @ " in matchup:
        away = team
        home = matchup.split(" @ ")[1].strip()
    else:
        home = "HOME"
        away = "AWAY"

    return home, away


def get_game_teams(game_id: str):
    games = pd.read_csv(GAMES_PATH, dtype={"GAME_ID": str})
    row = games[games["GAME_ID"] == game_id].iloc[0]

    home, away = parse_home_away(row)

    return {
        "home_abbr": home,
        "away_abbr": away,
        "home_name": TEAM_NAMES.get(home, home),
        "away_name": TEAM_NAMES.get(away, away),
        "home_logo": TEAM_LOGOS.get(home, ""),
        "away_logo": TEAM_LOGOS.get(away, ""),
    }


def build_pregame_strength_table():
    games = pd.read_csv(GAMES_PATH, dtype={"GAME_ID": str})
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    records = {}
    strength_by_game = {}

    for _, row in games.iterrows():
        game_id = str(row["GAME_ID"])
        home, away = parse_home_away(row)

        for team in [home, away]:
            if team not in records:
                records[team] = {"wins": 0, "losses": 0}

        home_wins = records[home]["wins"]
        home_losses = records[home]["losses"]
        away_wins = records[away]["wins"]
        away_losses = records[away]["losses"]

        home_games = home_wins + home_losses
        away_games = away_wins + away_losses

        home_pre_win_pct = home_wins / home_games if home_games > 0 else 0.5
        away_pre_win_pct = away_wins / away_games if away_games > 0 else 0.5

        strength_by_game[game_id] = {
            "home_pre_wins": home_wins,
            "home_pre_losses": home_losses,
            "away_pre_wins": away_wins,
            "away_pre_losses": away_losses,
            "home_pre_win_pct": home_pre_win_pct,
            "away_pre_win_pct": away_pre_win_pct,
            "pre_game_win_pct_diff": home_pre_win_pct - away_pre_win_pct,

            "home_elo": 1500.0,
            "away_elo": 1500.0,
            "elo_diff": 0.0,

            "home_last10_win_pct": 0.5,
            "away_last10_win_pct": 0.5,
            "last10_win_pct_diff": 0.0,

            "home_avg_points_for": 110.0,
            "away_avg_points_for": 110.0,
            "home_avg_points_allowed": 110.0,
            "away_avg_points_allowed": 110.0,

            "offensive_strength_diff": 0.0,
            "defensive_strength_diff": 0.0,
}
        

        listed_team = row["TEAM_ABBREVIATION"]
        listed_team_won = row["WL"] == "W"

        if listed_team == home:
            home_win = listed_team_won
        else:
            home_win = not listed_team_won

        if home_win:
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        else:
            records[away]["wins"] += 1
            records[home]["losses"] += 1

    return strength_by_game


PREGAME_STRENGTH = {}


def get_default_strength():
    return {
        "home_pre_wins": 0,
        "home_pre_losses": 0,
        "away_pre_wins": 0,
        "away_pre_losses": 0,
        "home_pre_win_pct": 0.5,
        "away_pre_win_pct": 0.5,
        "pre_game_win_pct_diff": 0.0,

        "home_elo": 1500.0,
        "away_elo": 1500.0,
        "elo_diff": 0.0,

        "home_last10_win_pct": 0.5,
        "away_last10_win_pct": 0.5,
        "last10_win_pct_diff": 0.0,

        "home_avg_points_for": 110.0,
        "away_avg_points_for": 110.0,
        "home_avg_points_allowed": 110.0,
        "away_avg_points_allowed": 110.0,

        "offensive_strength_diff": 0.0,
        "defensive_strength_diff": 0.0,
    }


def get_strength_for_game(game_id: str):
    return PREGAME_STRENGTH.get(str(game_id), get_default_strength())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/games")
def games():
    df = pd.read_csv(GAMES_PATH, dtype={"GAME_ID": str})
    df = df.sort_values("GAME_DATE", ascending=False)

    game_list = []

    for _, row in df.iterrows():
        game_id = row["GAME_ID"]
        teams = get_game_teams(game_id)
        label = f'{row["GAME_DATE"]} — {teams["away_abbr"]} @ {teams["home_abbr"]}'

        game_list.append({
            "game_id": game_id,
            "date": row["GAME_DATE"],
            "label": label,
            "home_team": teams["home_abbr"],
            "away_team": teams["away_abbr"],
        })

    return jsonify(game_list)


@app.route("/live-games")
def live_games():
    try:
        board = scoreboard.ScoreBoard()
        data = board.get_dict()

        games_data = data.get("scoreboard", {}).get("games", [])
        live_list = []

        for game in games_data:
            game_id = game.get("gameId")
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})

            home_abbr = home.get("teamTricode", "HOME")
            away_abbr = away.get("teamTricode", "AWAY")

            status = game.get("gameStatusText", "Unknown status")
            label = f"{away_abbr} @ {home_abbr} — {status}"

            live_list.append({
                "game_id": game_id,
                "label": label,
                "status": status,
                "home_team": home_abbr,
                "away_team": away_abbr,
            })

        return jsonify(live_list)

    except Exception:
        traceback.print_exc()
        return jsonify([])


def emit_prediction(game_id, teams, strength, home_score, away_score, period,
                    seconds_remaining, possessions, home_fouls, away_fouls,
                    description, mode):
    features = {
        "period": period,
        "seconds_remaining": seconds_remaining,
        "home_score": home_score,
        "away_score": away_score,
        "score_diff_home": home_score - away_score,
        "abs_score_diff": abs(home_score - away_score),
        "possessions_proxy": possessions,
        "home_fouls_proxy": home_fouls,
        "away_fouls_proxy": away_fouls,
        "is_clutch_time": int(seconds_remaining <= 300 and abs(home_score - away_score) <= 5),

        "home_pre_wins": strength["home_pre_wins"],
        "home_pre_losses": strength["home_pre_losses"],
        "away_pre_wins": strength["away_pre_wins"],
        "away_pre_losses": strength["away_pre_losses"],
        "home_pre_win_pct": strength["home_pre_win_pct"],
        "away_pre_win_pct": strength["away_pre_win_pct"],
        "pre_game_win_pct_diff": strength["pre_game_win_pct_diff"],
        "home_elo": strength["home_elo"],
        "away_elo": strength["away_elo"],
        "elo_diff": strength["elo_diff"],

        "home_last10_win_pct": strength["home_last10_win_pct"],
        "away_last10_win_pct": strength["away_last10_win_pct"],
        "last10_win_pct_diff": strength["last10_win_pct_diff"],

        "home_avg_points_for": strength["home_avg_points_for"],
        "away_avg_points_for": strength["away_avg_points_for"],

        "home_avg_points_allowed": strength["home_avg_points_allowed"],
        "away_avg_points_allowed": strength["away_avg_points_allowed"],

        "offensive_strength_diff": strength["offensive_strength_diff"],
        "defensive_strength_diff": strength["defensive_strength_diff"],
    }

    home_prob = predictor.predict_home_win_probability(features)

    socketio.emit(
        "win_probability_update",
        {
            "mode": mode,
            "game_id": game_id,
            "home_team": teams["home_abbr"],
            "away_team": teams["away_abbr"],
            "home_name": teams["home_name"],
            "away_name": teams["away_name"],
            "home_logo": teams["home_logo"],
            "away_logo": teams["away_logo"],
            "home_score": home_score,
            "away_score": away_score,
            "period": period,
            "seconds_remaining": seconds_remaining,
            "home_win_probability": round(home_prob * 100, 2),
            "away_win_probability": round((1 - home_prob) * 100, 2),
            "last_play": description,
            "home_record": f'{strength["home_pre_wins"]}-{strength["home_pre_losses"]}',
            "away_record": f'{strength["away_pre_wins"]}-{strength["away_pre_losses"]}',
            "home_elo": round(strength["home_elo"], 1),
            "away_elo": round(strength["away_elo"], 1),
            "home_last10": round(strength["home_last10_win_pct"] * 100, 1),
            "away_last10": round(strength["away_last10_win_pct"] * 100, 1),
            "speed": replay_state["speed"],
            "paused": replay_state["paused"],
        },
    )


def replay_game_loop(game_id: str, replay_id: int):
    global current_replay_id

    try:
        pbp = pd.read_csv(PBP_PATH, dtype={"GAME_ID": str})
        teams = get_game_teams(game_id)
        strength = get_strength_for_game(game_id)

        game = pbp[pbp["GAME_ID"] == game_id].copy()
        game = game.sort_values("actionNumber").reset_index(drop=True)

        replay_state["selected_game_id"] = game_id
        replay_state["paused"] = False

        home_score = 0
        away_score = 0
        possessions = 0
        home_fouls = 0
        away_fouls = 0

        i = 0
        while i < len(game):
            if replay_id != current_replay_id:
                break

            while replay_state["paused"]:
                if replay_id != current_replay_id:
                    break
                socketio.sleep(0.1)

            if replay_id != current_replay_id:
                break

            r = game.iloc[i]

            period = safe_int(r.get("period"), 1)
            home_score = safe_int(r.get("scoreHome"), home_score)
            away_score = safe_int(r.get("scoreAway"), away_score)

            seconds_remaining = clock_to_seconds_remaining(
                period,
                r.get("clock", "PT00M00.00S")
            )

            action_type = str(r.get("actionType", "")).lower()
            sub_type = str(r.get("subType", "")).lower()
            description = str(r.get("description", ""))

            if action_type in ["made shot", "turnover", "rebound"]:
                possessions += 1

            if "foul" in action_type or "foul" in sub_type or "foul" in description.lower():
                home_fouls += 0.5
                away_fouls += 0.5

            emit_prediction(
                game_id,
                teams,
                strength,
                home_score,
                away_score,
                period,
                seconds_remaining,
                possessions,
                home_fouls,
                away_fouls,
                description,
                "Historical Replay",
            )

            i += 1

            delay = 0.35 / replay_state["speed"]
            socketio.sleep(delay)

    except Exception:
        traceback.print_exc()


def live_game_loop(game_id: str, live_id: int):
    global current_live_id

    last_action_number = None
    possessions_proxy = 0
    home_fouls = 0
    away_fouls = 0

    try:
        while live_id == current_live_id:
            pbp_endpoint = playbyplay.PlayByPlay(game_id=str(game_id))
            data = pbp_endpoint.get_dict()

            game = data.get("game", {})
            actions = game.get("actions", [])

            if not actions:
                socketio.emit("status_message", {
                    "message": "No live play-by-play actions found yet for this game."
                })
                socketio.sleep(8)
                continue

            latest = actions[-1]

            action_number = latest.get("actionNumber")

            if action_number == last_action_number:
                socketio.sleep(5)
                continue

            last_action_number = action_number

            home_team = game.get("homeTeam", {})
            away_team = game.get("awayTeam", {})

            home_abbr = home_team.get("teamTricode", "HOME")
            away_abbr = away_team.get("teamTricode", "AWAY")

            teams = {
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_name": TEAM_NAMES.get(home_abbr, home_abbr),
                "away_name": TEAM_NAMES.get(away_abbr, away_abbr),
                "home_logo": TEAM_LOGOS.get(home_abbr, ""),
                "away_logo": TEAM_LOGOS.get(away_abbr, ""),
            }

            home_score = safe_int(latest.get("scoreHome"), 0)
            away_score = safe_int(latest.get("scoreAway"), 0)

            period = safe_int(latest.get("period"), 1)
            clock = latest.get("clock", "PT00M00.00S")
            seconds_remaining = clock_to_seconds_remaining(period, clock)

            action_type = str(latest.get("actionType", "")).lower()
            sub_type = str(latest.get("subType", "")).lower()
            description = str(latest.get("description", "Live play update"))

            if action_type in ["made shot", "turnover", "rebound"]:
                possessions_proxy += 1

            if "foul" in action_type or "foul" in sub_type or "foul" in description.lower():
                team_tricode = latest.get("teamTricode")

                if team_tricode == home_abbr:
                    home_fouls += 1
                elif team_tricode == away_abbr:
                    away_fouls += 1
                else:
                    home_fouls += 0.5
                    away_fouls += 0.5

            strength = get_default_strength()

            emit_prediction(
                game_id,
                teams,
                strength,
                home_score,
                away_score,
                period,
                seconds_remaining,
                possessions_proxy,
                home_fouls,
                away_fouls,
                description,
                "Live Play-by-Play",
            )

            socketio.sleep(5)

    except Exception:
        traceback.print_exc()
        socketio.emit("status_message", {
            "message": "Live play-by-play failed. Check terminal for error."
        })
        
@socketio.on("start_replay")
def start_replay(data):
    global current_replay_id, current_live_id

    game_id = data.get("game_id")
    if not game_id:
        return

    current_live_id += 1
    current_replay_id += 1
    replay_id = current_replay_id

    replay_state["selected_game_id"] = game_id
    replay_state["paused"] = False

    socketio.start_background_task(replay_game_loop, game_id, replay_id)


@socketio.on("pause_replay")
def pause_replay():
    replay_state["paused"] = True
    socketio.emit("status_message", {"message": "Replay paused."})


@socketio.on("resume_replay")
def resume_replay():
    replay_state["paused"] = False
    socketio.emit("status_message", {"message": "Replay resumed."})


@socketio.on("restart_replay")
def restart_replay():
    global current_replay_id, current_live_id

    game_id = replay_state.get("selected_game_id")
    if not game_id:
        return

    current_live_id += 1
    current_replay_id += 1
    replay_id = current_replay_id

    replay_state["paused"] = False

    socketio.start_background_task(replay_game_loop, game_id, replay_id)


@socketio.on("set_speed")
def set_speed(data):
    speed = float(data.get("speed", 1.0))
    replay_state["speed"] = max(0.25, min(speed, 4.0))
    socketio.emit("status_message", {
        "message": f'Replay speed set to {replay_state["speed"]}x.'
    })


@socketio.on("start_live")
def start_live(data):
    global current_live_id, current_replay_id

    game_id = data.get("game_id")
    if not game_id:
        return

    current_replay_id += 1
    current_live_id += 1
    live_id = current_live_id

    socketio.start_background_task(live_game_loop, game_id, live_id)


@socketio.on("connect")
def handle_connect():
    print("Client connected")


if __name__ == "__main__":
    if not Path("models/winprob_model.pt").exists():
        raise FileNotFoundError("Train the model first: python src/train_model.py")

    predictor = WinProbabilityPredictor()
    PREGAME_STRENGTH = build_pregame_strength_table()

    socketio.run(
        app,
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )