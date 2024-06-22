import re, csv
import pandas as pd


def parse_poker_log(file_path, ending_chip_count):
    with open(file_path, 'r') as file:
        log = file.read()

    # Split the log into sections based on the "----" delimiter
    games = log.split('-----------------------------------------------------------------')

    # Define patterns to extract player order and ending player chips
    player_order_pattern = re.compile(r'\[(.*?)\]')
    ending_chips_pattern = re.compile(r'Ending Player Chips: \[(.*?)\]')

    result = []

    for game_index, game in enumerate(games, start=1):
        if "RIVER" in game:
            stage = "river"
        elif "TURN" in game:
            stage = "turn"
        elif "FLOP" in game.split("PREFLOP")[-1]:
            stage = "flop"
        else:
            stage = "preflop"

        if not game.strip():
            continue

        # Extract player order
        player_order_match = player_order_pattern.search(game)
        if not player_order_match:
            continue

        player_order_str = player_order_match.group(1)
        player_order = eval(player_order_str)

        # Extract ending player chips
        ending_chips_match = ending_chips_pattern.search(game)
        if not ending_chips_match:
            continue

        ending_chips_str = ending_chips_match.group(1)
        ending_chips = list(map(int, ending_chips_str.split(',')))
        if sum(ending_chips) != ending_chip_count:
            continue

        # Construct the list of tuples with game number, player model, and chips
        for player, chip in zip(player_order, ending_chips):
            result.append({'game_number': game_index, 'game_stage': stage, 'model_name': player[0], 'ending_chip': chip})

    return result

def report_winning(result_path: str, starting_chip: int|float):
    result = pd.read_csv(result_path)
    result['winning'] = result['ending_chip'] - starting_chip
    # print(sum(result['winning']))
    winnings = {}
    print((result.groupby('model_name').sum()['winning']) / max(result['game_number']))

if __name__ == "__main__":
    result_txt_path = 'history/8b_1600_vs_8b_800_headsup_game_result.txt'
    output_csv_path = 'history/8b_1600_vs_8b_800_headsup_game_result.csv'
    starting_chip = 200
    num_players = 2
    ending_chip_count = 400

    parsed_results = parse_poker_log(result_txt_path, ending_chip_count)
    result_csv = pd.DataFrame(parsed_results)
    result_csv.to_csv(output_csv_path)

    print(f"Results have been saved to {output_csv_path}")
    print("Total")
    report_winning(output_csv_path, 200)
    print(f"Total Number of Games: {sum(result_csv.groupby('game_stage').count()['game_number'].sort_values(ascending=False)/num_players)}")
    print(result_csv.groupby('game_stage').count()['game_number'].sort_values(ascending=False)/num_players)
    for stage in ['preflop', 'flop', 'turn', 'river']:
        print(stage)
        subset = result_csv[result_csv['game_stage']==stage].copy()
        subset['winning'] = subset['ending_chip'] - 200
        # print(sum(result['winning']))
        winnings = {}
        print((subset.groupby('model_name').sum()['winning']) / max(subset['game_number']))



