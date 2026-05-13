import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get_games_for_season(
    season: str,
    season_type: str,
    limit: int | None = None
) -> pd.DataFrame:
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",
    )

    games = finder.get_data_frames()[0]

    games = games.sort_values("GAME_DATE").drop_duplicates("GAME_ID")

    games = games[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "MATCHUP",
            "WL",
        ]
    ]

    games["SEASON_TYPE"] = season_type

    if limit:
        games = games.head(limit)

    return games


def download_play_by_play(games: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for i, game_id in enumerate(games["GAME_ID"], start=1):
        print(f"[{i}/{len(games)}] Downloading play-by-play for {game_id}")

        try:
            pbp = playbyplayv3.PlayByPlayV3(
                game_id=str(game_id)
            ).get_data_frames()[0]

            pbp["GAME_ID"] = str(game_id)
            frames.append(pbp)

            time.sleep(0.7)

        except Exception as exc:
            print(f"Could not download {game_id}: {exc}")
            time.sleep(2)

    if not frames:
        raise RuntimeError("No play-by-play data downloaded.")

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help="Example: 2025-26")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        choices=["Regular Season", "Playoffs"],
        help="Regular Season or Playoffs",
    )
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    games = get_games_for_season(
        season=args.season,
        season_type=args.season_type,
        limit=args.limit,
    )

    raw_games_path = DATA_DIR / "raw_games.csv"
    pbp_path = DATA_DIR / "play_by_play.csv"

    if raw_games_path.exists():
        old_games = pd.read_csv(raw_games_path, dtype={"GAME_ID": str})
        games = pd.concat([old_games, games], ignore_index=True)
        games = games.drop_duplicates("GAME_ID")

    new_pbp = download_play_by_play(games)

    if pbp_path.exists():
        old_pbp = pd.read_csv(pbp_path, dtype={"GAME_ID": str})
        pbp = pd.concat([old_pbp, new_pbp], ignore_index=True)
        pbp = pbp.drop_duplicates(["GAME_ID", "actionNumber"])
    else:
        pbp = new_pbp

    games.to_csv(raw_games_path, index=False)
    pbp.to_csv(pbp_path, index=False)

    print("Saved data/raw_games.csv and data/play_by_play.csv")


if __name__ == "__main__":
    main()