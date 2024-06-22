import pandas as pd
import ast,json,random

random.seed(42)
def preflop_csv_to_json(preflop_dataset: pd.DataFrame):
    def parse_action(line):
        if line == "":
            return "there has been no action yet"
        parts = line.split('/')
        description = []
        
        # Helper function to get the verbal action
        def get_action(index, bet):
            if 'bb' in bet:
                return f"{parts[index]} raise {bet.split('bb')[0]}"
            elif 'allin' in bet:
                return f"{parts[index]} all in"
            elif 'fold' in bet:
                return f"{parts[index]} fold"
            elif 'call' in bet:
                return f"{parts[index]} call"
        
        # Iterate over parts and construct the description list
        i = 0
        while i < len(parts):
            if i+1 < len(parts):
                description.append(get_action(i, parts[i+1]))
            i += 2
        
        # Properly join all elements with commas, and 'and' for the last element
        if len(description) > 1:
            # Join all elements with a comma except the last two
            result = ', '.join(description[:-1])
            # Add 'and' before the last element
            result += f", and {description[-1]}"
        else:
            result = description[0] if description else ''
        
        return result
    def parse_holding(hand):
        suits = ["Spade", 'Heart', "Club", "Diamond"]
        rank_dict = {"2": "Two", "3": "Three", "4": "Four", "5":"Five", "6": "Six", "7": "Seven",
                    "8": "Eight", "9": "Nine", "T": "Ten", "J":"Jack", "Q": "Queen", "K": "King", "A": "Ace"}
        suit_sample = random.sample(suits, 2)
        if len(hand) == 2:
            return f"{rank_dict[hand[0]]} of {suit_sample[0]} and {rank_dict[hand[1]]} of {suit_sample[1]}"
        else:
            if 's' in hand:
                return f"{rank_dict[hand[0]]} of {suit_sample[0]} and {rank_dict[hand[1]]} of {suit_sample[0]}"
            else:
                return f"{rank_dict[hand[0]]} of {suit_sample[0]} and {rank_dict[hand[1]]} of {suit_sample[1]}"
    def parse_moves(moves):
        moves_ls = ast.literal_eval(moves)
        return [f"raise {move.split('bb')[0]}" if 'bb' in move else move.upper() for move in moves_ls]
    def construct_prompt(row: pd.Series):
        # print(row)
        preflop_action_summary = parse_action(row['prev_line'])
        hero_position = row['hero_pos']
        hero_holding = parse_holding(row['hero_holding'])
        current_pot_size = row['pot_size']
        available_moves = parse_moves(row['available_moves'])

        prompt = f"""You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision. 

Here is a game summary: 

The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. 
The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.
In this hand, your position is {hero_position}, and your holding is [{hero_holding}].
Before the flop, {preflop_action_summary}. Assume that all other players that is not mentioned folded.

Now it is your turn to make a move. 
To remind you, the current pot size is {current_pot_size} chips, and your holding is [{hero_holding}].

Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.
Your optimal action is:"""

        return prompt, available_moves, f"raise {row['correct_decision'].split('bb')[0]}" if 'bb' in row['correct_decision'] else row['correct_decision'].upper()
    preflop_dataset_json = [{"instruction": construct_prompt(preflop_dataset.iloc[i])[0],
                     "output": construct_prompt(preflop_dataset.iloc[i])[2]} 
                     for i in range(preflop_dataset.shape[0])]
    
    return preflop_dataset_json

def postflop_csv_to_json(postflop_dataset: pd.DataFrame):
    def parse_preflop_action(preflop_action):
        position_map = {"UTG": "UTG", "HJ": "HJ", "CO": "CO", 
                        "BTN": "BTN", "SB": "SB", "BB": "BB"}
        action_list = preflop_action.split('/')
        # print(action_list)
        if len(action_list) == 4:
            position_1 = position_map[action_list[0]]
            position_1_raise_size = action_list[1]
            position_2 = position_map[action_list[2]]
            position_2_action = action_list[3]
            return f"{position_1} RAISE {position_1_raise_size}, and {position_2} CALL"
        elif len(action_list) == 6:
            position_1 = position_map[action_list[0]]
            position_1_raise_size = action_list[1]
            position_2 = position_map[action_list[2]]
            position_2_reraise_size = action_list[3]
            position_1_action = action_list[5]
            return f"{position_1} RAISE {position_1_raise_size}, {position_2} RAISE {position_2_reraise_size}, and {position_1} CALL"

        else:
            raise ValueError("Unseen Preflop Action")
    def parse_board(board):
        rank_map = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", 
                    "T": "Ten", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
        suit_map = {'h': 'Heart', 'c': 'Club', 'd': 'Diamond', 's': 'Spade'}
        board_list = board.split(",")
        processed_board_list = [f"{rank_map[card[0]]} Of {suit_map[card[1]]}" for card in board_list]

        return ', '.join(processed_board_list[:-1]) + ', and ' + processed_board_list[-1] \
            if len(processed_board_list) > 1 else processed_board_list[0]
    def parse_relative_position(preflop_action):
        position_map = {"UTG": "UTG", "HJ": "HJ", "CO": "CO", 
                        "BTN": "BTN", "SB": "SB", "BB": "BB"}
        relative_position_map = {"UTG": 3, "HJ": 4, "CO": 5, "BTN": 6, "SB": 1, "BB": 2}
        action_list = preflop_action.split('/')
        position_1 = action_list[0]
        position_2 = action_list[2]
        return {'OOP': position_map[position_2], 'IP': position_map[
            position_1]} if relative_position_map[position_1] > relative_position_map[position_2] else {
                'OOP': position_map[position_1], 'IP': position_map[position_2]}
    def parse_postflop_action(preflop_action, postflop_action):
        relative_position_map = parse_relative_position(preflop_action)
        if pd.isna(postflop_action) or postflop_action=="":
            return {"flop": "there has been no action yet"}
        action_list = postflop_action.split('/')
        # print(action_list)
        def process_action_list(action_list):
            processed_action_list = []
            for i in range(len(action_list)):
                action = action_list[i]
                # print(action)
                if "CHECK" in action:
                    action = action.replace("_", " ")
                if "BET" in action or "RAISE" in action:
                    action = action.replace("_", " ") + " chips"
                elif 'CALL' in action:
                    pos = "IP" if action_list[i-1].split("_")[0] == "OOP" else "OOP"
                    action = pos + " " + action
                # print(action_list)
                processed_action_list.append(action)
            if len(processed_action_list) == 0:
                return "there has been no action yet"
            return ', '.join(processed_action_list[:-1]) + ', and ' + processed_action_list[-1] \
            if len(processed_action_list) > 1 else processed_action_list[0]

        if 'dealcards' in action_list:
            sep_index = action_list.index('dealcards')
            flop_action_list = action_list[:sep_index]
            turn_action_list = action_list[sep_index+2:]
            processed_flop_action_list = process_action_list(flop_action_list).replace("OOP", relative_position_map['OOP']).\
                replace("IP", relative_position_map['IP'])
            processed_turn_action_list = process_action_list(turn_action_list).replace("OOP", relative_position_map['OOP']).\
                replace("IP", relative_position_map['IP'])
            return {"flop": processed_flop_action_list, "turn": processed_turn_action_list}
        else:
            processed_flop_action_list = process_action_list(action_list).replace("OOP", relative_position_map['OOP']).\
                replace("IP", relative_position_map['IP'])
            return {"flop": processed_flop_action_list}
    def parse_holding(holding):
        rank_map = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", 
                    "T": "Ten", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
        suit_map = {'h': 'Heart', 'c': 'Club', 'd': 'Diamond', 's': 'Spade'}

        return f"[{rank_map[holding[0]]} Of {suit_map[holding[1]]} and {rank_map[holding[2]]} Of {suit_map[holding[3]]}]"
    def parse_available_moves(available_moves):
        # print(available_moves)
        def parse_bet_raise(move):
            action_name, amount = move.rsplit(' ', 1)
            return f"{action_name} {int(round(float(amount)))}"
        return ", ".join([parse_bet_raise(move) if ("BET" in move or "RAISE" in move) else move for move in available_moves])

    def construct_prompt(row: pd.Series):
        # print(row)
        relative_position_map = parse_relative_position(row['preflop_action'])
        hero_position = relative_position_map[row['hero_position']]
        hero_holding = parse_holding(row['holding'])
        preflop_action_summary = parse_preflop_action(row['preflop_action']).replace("bb", " chips")
        flop_summary = f"The flop comes {parse_board(row['board_flop'])}, then {parse_postflop_action(row['preflop_action'], row['postflop_action'])['flop']}."
        eval_at_turn = row['evaluation_at'] == "Turn"
        if eval_at_turn:
            turn_summary = f"The turn comes {parse_board(row['board_turn'])}, then {parse_postflop_action(row['preflop_action'], row['postflop_action'])['turn']}."
        else:
            turn_summary = ""
        current_pot_size = row['pot_size']
        # print(row['Available_Moves'])
        available_moves = parse_available_moves(ast.literal_eval(row['available_moves']))
        # print(available_moves)

        prompt = f"""You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision. 

Here is a game summary: 

The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. 
The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.
In this hand, your position is {hero_position}, and your holding is {hero_holding}.
Before the flop, {preflop_action_summary}. Assume that all other players that is not mentioned folded.
{flop_summary}
{turn_summary}

Now it is your turn to make a move. 
To remind you, the current pot size is {current_pot_size} chips, and your holding is {hero_holding}.

Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer. 
Your optimal action is:"""
        
        return prompt, row['correct_decision']

    # row_index = random.randint(0, dataset.shape[0])
    # print(row_index)
    # prompt, correct_decision = construct_prompt(postflop_dataset.iloc[row_index])

    # print(prompt)
    # print(correct_decision)

    postflop_dataset_json = [{"instruction": construct_prompt(postflop_dataset.iloc[i])[0],
                     "output": construct_prompt(postflop_dataset.iloc[i])[1]} 
                     for i in range(postflop_dataset.shape[0])]
    
    return postflop_dataset_json

def poker_csv_to_json(dataset: pd.DataFrame, preflop=True):
    def replace_keywords(data):
        if isinstance(data, dict):
            return {k: replace_keywords(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [replace_keywords(item) for item in data]
        elif isinstance(data, str):
            data = data.replace("ALLIN", "all in")
            data = data.replace("CALL", "call")
            data = data.replace("RAISE", "raise")
            data = data.replace("CHECK", "check")
            data = data.replace("BET", "bet")
            data = data.replace("FOLD", "fold")
            data = data.replace("UNDER_THE_GUN", "UTG")
            data = data.replace("HIJACK", "HJ")
            data = data.replace("CUTOFF", "CO")
            data = data.replace("BUTTON", "BTN")
            data = data.replace("SMALL_BLIND", "SB")
            data = data.replace("BIG_BLIND", "BB")
            data = data.replace("BIG_BLIND", "BB")
            return data
        else:
            return data
        
    if preflop:
        dataset_json = preflop_csv_to_json(dataset)
    else:
        dataset_json = postflop_csv_to_json(dataset)
    
    dataset_json = replace_keywords(dataset_json)

    return dataset_json

CSV_FILENAME = "postflop_eval_dataset_filtered.csv"
IS_PREFLOP = False
JSON_FILENAME = "sft_postflop_full.json"

dataset = pd.read_csv(CSV_FILENAME).fillna("")
dataset_json = poker_csv_to_json(dataset, preflop=IS_PREFLOP)
with open(JSON_FILENAME, 'w') as json_file:
    random.shuffle(dataset_json)
    json.dump(dataset_json, json_file, indent=2)

