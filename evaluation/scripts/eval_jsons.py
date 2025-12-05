import argparse
import glob
import json
import os

import numpy as np


def is_valid_number(value):
    if isinstance(value, (int, float)):
        return np.isfinite(value)
    return False


def aggregate_statistics(folder_path, total_chunks):
    # Initialize lists to store data
    distances_to_goal = []
    successes = []
    spls = []
    ndtws = []
    path_lengths = []
    oracle_successes = []
    steps_taken = []
    total_episodes = 0
    invalid_spls = 0
    invalid_distances = 0

    # Iterate over all JSON files in the folder
    for id in range(total_chunks):

        pattern = os.path.join(folder_path, f"*_{total_chunks}-{id}.json")
        file_list = glob.glob(pattern)
        if len(file_list) != 1:
            print(f"Missing {total_chunks}-{id}.json")
            continue
        file_path = file_list[0]

        if os.path.exists(file_path):
            with open(file_path) as file:
                data = json.load(file)
                # Aggregate data
                for episode_id, episode_data in data.items():
                    if is_valid_number(episode_data["distance_to_goal"]):
                        distances_to_goal.append(episode_data["distance_to_goal"])
                    else:
                        invalid_distances += 1

                    successes.append(episode_data["success"])

                    if is_valid_number(episode_data["spl"]):
                        spls.append(episode_data["spl"])
                    else:
                        invalid_spls += 1

                    ndtws.append(episode_data["ndtw"])
                    path_lengths.append(episode_data["path_length"])
                    oracle_successes.append(episode_data["oracle_success"])
                    steps_taken.append(episode_data["steps_taken"])
                    total_episodes += 1

    # Calculate statistics
    stats = {
        "mean_distance_to_goal": np.mean(distances_to_goal) if distances_to_goal else "N/A",
        "mean_success": np.mean(successes),
        "mean_spl": np.mean(spls) if spls else "N/A",
        "mean_ndtw": np.mean(ndtws),
        "mean_path_length": np.mean(path_lengths),
        "mean_oracle_success": np.mean(oracle_successes),
        "mean_steps_taken": np.mean(steps_taken),
        "total_episodes": total_episodes,
        "invalid_spls": invalid_spls,
        "invalid_distances": invalid_distances,
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate statistics from JSON files.")
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing JSON files.",
    )
    parser.add_argument("total_chunks", type=int, help="Total chunks of JSON files to process.")
    args = parser.parse_args()

    statistics = aggregate_statistics(args.folder_path, args.total_chunks)
    print(json.dumps(statistics, indent=4))
