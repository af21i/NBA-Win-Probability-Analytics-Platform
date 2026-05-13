from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


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


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


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


def get_final_scores_by_game(pbp: pd.DataFrame) -> dict:
    final_scores = {}

    for game_id, game in pbp.groupby("GAME_ID"):
        game = game.sort_values("actionNumber")

        home_score = 0
        away_score = 0

        for _, row in game.iterrows():
            home_score = safe_int(row.get("scoreHome"), home_score)
            away_score = safe_int(row.get("scoreAway"), away_score)

        final_scores[str(game_id)] = {
            "home_score": home_score,
            "away_score": away_score,
        }

    return final_scores


def expected_score(team_elo: float, opponent_elo: float) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def build_pregame_strength_table(games: pd.DataFrame, pbp: pd.DataFrame) -> dict:
    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    final_scores = get_final_scores_by_game(pbp)

    team_state = {}
    strength_by_game = {}

    def init_team(team):
        if team not in team_state:
            team_state[team] = {
                "wins": 0,
                "losses": 0,
                "elo": 1500.0,
                "last10": [],
                "points_for": [],
                "points_allowed": [],
            }

    for _, row in games.iterrows():
        game_id = str(row["GAME_ID"])
        home, away = parse_home_away(row)

        init_team(home)
        init_team(away)

        home_state = team_state[home]
        away_state = team_state[away]

        home_games = home_state["wins"] + home_state["losses"]
        away_games = away_state["wins"] + away_state["losses"]

        home_win_pct = home_state["wins"] / home_games if home_games > 0 else 0.5
        away_win_pct = away_state["wins"] / away_games if away_games > 0 else 0.5

        home_last10 = (
            sum(home_state["last10"][-10:]) / len(home_state["last10"][-10:])
            if len(home_state["last10"][-10:]) > 0 else 0.5
        )

        away_last10 = (
            sum(away_state["last10"][-10:]) / len(away_state["last10"][-10:])
            if len(away_state["last10"][-10:]) > 0 else 0.5
        )

        home_avg_pf = (
            sum(home_state["points_for"][-10:]) / len(home_state["points_for"][-10:])
            if len(home_state["points_for"][-10:]) > 0 else 110.0
        )

        away_avg_pf = (
            sum(away_state["points_for"][-10:]) / len(away_state["points_for"][-10:])
            if len(away_state["points_for"][-10:]) > 0 else 110.0
        )

        home_avg_pa = (
            sum(home_state["points_allowed"][-10:]) / len(home_state["points_allowed"][-10:])
            if len(home_state["points_allowed"][-10:]) > 0 else 110.0
        )

        away_avg_pa = (
            sum(away_state["points_allowed"][-10:]) / len(away_state["points_allowed"][-10:])
            if len(away_state["points_allowed"][-10:]) > 0 else 110.0
        )

        strength_by_game[game_id] = {
            "home_pre_wins": home_state["wins"],
            "home_pre_losses": home_state["losses"],
            "away_pre_wins": away_state["wins"],
            "away_pre_losses": away_state["losses"],
            "home_pre_win_pct": home_win_pct,
            "away_pre_win_pct": away_win_pct,
            "pre_game_win_pct_diff": home_win_pct - away_win_pct,

            "home_elo": home_state["elo"],
            "away_elo": away_state["elo"],
            "elo_diff": home_state["elo"] - away_state["elo"],

            "home_last10_win_pct": home_last10,
            "away_last10_win_pct": away_last10,
            "last10_win_pct_diff": home_last10 - away_last10,

            "home_avg_points_for": home_avg_pf,
            "away_avg_points_for": away_avg_pf,
            "home_avg_points_allowed": home_avg_pa,
            "away_avg_points_allowed": away_avg_pa,

            "offensive_strength_diff": home_avg_pf - away_avg_pf,
            "defensive_strength_diff": away_avg_pa - home_avg_pa,
        }

        if game_id not in final_scores:
            continue

        home_score = final_scores[game_id]["home_score"]
        away_score = final_scores[game_id]["away_score"]

        home_win = home_score > away_score

        home_expected = expected_score(home_state["elo"], away_state["elo"])
        away_expected = expected_score(away_state["elo"], home_state["elo"])

        k = 20

        home_state["elo"] += k * ((1 if home_win else 0) - home_expected)
        away_state["elo"] += k * ((0 if home_win else 1) - away_expected)

        if home_win:
            home_state["wins"] += 1
            away_state["losses"] += 1
            home_state["last10"].append(1)
            away_state["last10"].append(0)
        else:
            away_state["wins"] += 1
            home_state["losses"] += 1
            away_state["last10"].append(1)
            home_state["last10"].append(0)

        home_state["points_for"].append(home_score)
        home_state["points_allowed"].append(away_score)

        away_state["points_for"].append(away_score)
        away_state["points_allowed"].append(home_score)

    return strength_by_game

def build_rows_for_game(game: pd.DataFrame, strength: dict) -> list[dict]:
    game = game.sort_values("actionNumber").copy()

    home_score = 0
    away_score = 0
    possessions_proxy = 0
    home_fouls_proxy = 0
    away_fouls_proxy = 0

    rows = []

    game_id = str(game["GAME_ID"].iloc[0])

    default_strength = {
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

    game_strength = strength.get(game_id, default_strength)

    for _, r in game.iterrows():
        period = safe_int(r.get("period"), 1)

        home_score = safe_int(r.get("scoreHome"), home_score)
        away_score = safe_int(r.get("scoreAway"), away_score)

        seconds_remaining = clock_to_seconds_remaining(period, r.get("clock", "PT00M00.00S"))

        action_type = str(r.get("actionType", "")).lower()
        sub_type = str(r.get("subType", "")).lower()
        description = str(r.get("description", "")).lower()

        if action_type in ["made shot", "turnover", "rebound"]:
            possessions_proxy += 1

        if "foul" in action_type or "foul" in sub_type or "foul" in description:
            home_fouls_proxy += 0.5
            away_fouls_proxy += 0.5

        score_diff_home = home_score - away_score

        rows.append({
            "GAME_ID": game_id,
            "period": period,
            "seconds_remaining": seconds_remaining,
            "home_score": home_score,
            "away_score": away_score,
            "score_diff_home": score_diff_home,
            "abs_score_diff": abs(score_diff_home),
            "possessions_proxy": possessions_proxy,
            "home_fouls_proxy": home_fouls_proxy,
            "away_fouls_proxy": away_fouls_proxy,
            "is_clutch_time": int(seconds_remaining <= 5 * 60 and abs(score_diff_home) <= 5),

            "home_pre_wins": game_strength["home_pre_wins"],
            "home_pre_losses": game_strength["home_pre_losses"],
            "away_pre_wins": game_strength["away_pre_wins"],
            "away_pre_losses": game_strength["away_pre_losses"],
            "home_pre_win_pct": game_strength["home_pre_win_pct"],
            "away_pre_win_pct": game_strength["away_pre_win_pct"],
            "pre_game_win_pct_diff": game_strength["pre_game_win_pct_diff"],
            "home_elo": game_strength["home_elo"],
            "away_elo": game_strength["away_elo"],
            "elo_diff": game_strength["elo_diff"],
            "home_last10_win_pct": game_strength["home_last10_win_pct"],
            "away_last10_win_pct": game_strength["away_last10_win_pct"],
            "last10_win_pct_diff": game_strength["last10_win_pct_diff"],
            "home_avg_points_for": game_strength["home_avg_points_for"],
            "away_avg_points_for": game_strength["away_avg_points_for"],
            "home_avg_points_allowed": game_strength["home_avg_points_allowed"],
            "away_avg_points_allowed": game_strength["away_avg_points_allowed"],
            "offensive_strength_diff": game_strength["offensive_strength_diff"],
            "defensive_strength_diff": game_strength["defensive_strength_diff"],
        })

    if not rows:
        return []

    final_home_score = rows[-1]["home_score"]
    final_away_score = rows[-1]["away_score"]
    home_win = int(final_home_score > final_away_score)

    for row in rows:
        row["home_win"] = home_win

    return rows


def main():
    pbp_path = DATA_DIR / "play_by_play.csv"
    games_path = DATA_DIR / "raw_games.csv"

    if not pbp_path.exists():
        raise FileNotFoundError("Missing data/play_by_play.csv. Run src/download_data.py first.")

    if not games_path.exists():
        raise FileNotFoundError("Missing data/raw_games.csv. Run src/download_data.py first.")

    pbp = pd.read_csv(pbp_path, dtype={"GAME_ID": str})
    games = pd.read_csv(games_path, dtype={"GAME_ID": str})

    strength_by_game = build_pregame_strength_table(games,pbp)

    all_rows = []

    for game_id, game in pbp.groupby("GAME_ID"):
        rows = build_rows_for_game(game, strength_by_game)
        all_rows.extend(rows)

    dataset = pd.DataFrame(all_rows)
    dataset = dataset.dropna()

    dataset.to_csv(DATA_DIR / "model_dataset.csv", index=False)

    print(f"Saved {len(dataset):,} training rows to data/model_dataset.csv")


if __name__ == "__main__":
    main()