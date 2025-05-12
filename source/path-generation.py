#
#
# Path generation for test purposes
#
#
import numpy as np
import pandas as pd
import os
from typing import Callable

import numpy as np
import pandas as pd
import os, sys
from typing import Callable, Tuple

def load_bar(current: int, total: int, bar_length: int = 50) -> None:
    """Displays a load bar in the terminal."""
    progress = current / total
    block = int(round(bar_length * progress))
    bar = "#" * block + "-" * (bar_length - block)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({progress * 100:.2f}%)")
    sys.stdout.flush()


def generate_tool_path_data(equation: Callable[[float], Tuple[float, float, float]], noise_factor: float, trials: int, rounds: int, output_path: str) -> None:
    """Generates a series of tool points (x, y, z) based on a Gaussian Process deviation around a second-order optimal path and saves it to a CSV file.

    Args:
        equation: A callable that defines the tool path as a function of time, returning (x, y, z).
        noise_factor: The standard deviation of the Gaussian noise to add to the tool path.
        trials: The number of tool points to generate per round.
        rounds: The number of rounds to generate data for.
        output_path: The file path to save the CSV data.
    """
    data = []

    for _ in range(rounds):
        round_data = []
        previous_point = np.array(equation(0))
        for i in range(trials):
            time = i / trials  # Normalize time to [0, 1]
            target_point = np.array(equation(time))

            # Gaussian deviation around the optimal path
            noise = np.random.normal(0, noise_factor, size=3)
            deviation = noise * time * (1 - time)

            # Compute the next point
            next_point = previous_point + (target_point - previous_point) / 2 + deviation
            round_data.extend(next_point)
            previous_point = next_point

        data.append(round_data)
        load_bar(_, rounds)
    load_bar(rounds, rounds)
    print(
        f"\nGenerated {trials * rounds} tool points for {rounds} rounds."
    )
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, header=False)


def main():
    # Example usage with a quadratic path
    def quadratic_path(t) -> Tuple[float, float, float]:
        x = 5 * t ** 2
        y = -3 * t ** 2 + 4 * t
        z = 2 * t ** 2 - t + 3
        return x, y, z

    generate_tool_path_data(quadratic_path, noise_factor=0.3, trials=1000, rounds=100,
                            output_path="../data/tool_path_data.csv")


if __name__ == "__main__":
    print("IceCube Path generation for ENE701")
    main()


