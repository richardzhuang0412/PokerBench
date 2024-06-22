import re
import requests
from openai import OpenAI
from together import Together
from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem import ActionType
from texasholdem.game.player_state import PlayerState
from texasholdem.card.card import Card
from texasholdem.card.deck import Deck
import requests, json, random, os, signal, subprocess, time
from tqdm import tqdm

def query_model_api(prompt:str, model_name:str, 
                    openai_api_key:str, 
                    together_api_key:str):
    openai_client = OpenAI(api_key=openai_api_key)
    together_client = Together(api_key=together_api_key)

    if "gpt" in model_name:
        # Uncomment this if using OpenAI Client
        completion = openai_client.chat.completions.create(
            temperature=0.1,
            model= model_name,
            max_tokens=50,
            messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
            ],
        )
        response = completion.choices[0].message.content

    elif "llama" in model_name:
        # Uncomment this if using Together Client Chat Model
        completion = together_client.chat.completions.create(
            temperature=0.05,
            model= model_name,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content
    
    return response

def query_vllm_host(prompt:str, port:int, model_name:str):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.1,
    max_tokens=20
    )

    return completion.choices[0].message.content

def query_litgpt_host(prompt:str, port:int):
    response = requests.post(
        f"http://127.0.0.1:{port}/predict", 
        json={"prompt": prompt}
    )

    return response.json()['output'].split("### Response:")[-1]

def check_host_working(port:str, litgpt:bool=True):
    success = False
    i = 1
    TIMEOUT_THRESHOLD = 2
    while not success:
        if i >= TIMEOUT_THRESHOLD:
            raise TimeoutError("Host Server Not Responding")
        try:
            if litgpt:
                query_litgpt_host("test test test ", port)
            else:
                query_vllm_host("test test test", port)
            success = True
        except:
            print(f"Inference request failed {i} time for port {port}")
            i += 1
        time.sleep(5)

    print(f"Port {port} is working!")
    return

def host_litgpt_model(gpu_id:int, checkpoint_path:str, port:str):
    hosting_command = f"export MKL_SERVICE_FORCE_INTEL=1 && export CUDA_VISIBLE_DEVICES={gpu_id} && litgpt serve --checkpoint_dir {checkpoint_path} --temperature 0.1 --max_new_tokens 50 --port {port}"
    host_process = subprocess.Popen(hosting_command, shell=True)    
    time.sleep(30)
    check_host_working(port)
    return

def host_vllm_model(gpu_id:int, checkpoint_path:str, port:str):
    pass

def kill_host(gpu_ids:list[int], ports:list[int]):
    def get_user_of_pid(pid):
        try:
            result = subprocess.check_output(['ps', '-o', 'user=', '-p', str(pid)]).decode().strip()
            return result
        except subprocess.CalledProcessError:
            return None

    def kill_processes_on_gpu(gpu_id, username='richard'):
        # Get the list of processes running on the specified GPU
        try:
            result = subprocess.check_output(['nvidia-smi', f'--id={gpu_id}', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
        except subprocess.CalledProcessError as e:
            print(f"Failed to query GPU {gpu_id}: {e}")
            return

        # Parse the PIDs
        gpu_processes = result.split('\n')
        gpu_processes = [int(pid) for pid in gpu_processes if pid]

        if not gpu_processes:
            print(f"No processes found on GPU {gpu_id}")
            return

        # Kill each process that belongs to the specified user
        for pid in gpu_processes:
            process_user = get_user_of_pid(pid)
            if process_user == username:
                try:
                    subprocess.Popen(f"kill -9 {pid}", shell=True)
                    print(f"Terminated process with PID: {pid} owned by {username}")
                except ProcessLookupError:
                    print(f"Process with PID {pid} not found")
                except Exception as e:
                    print(f"Failed to kill process with PID {pid}: {e}")
            else:
                print(f"Skipped terminating process with PID: {pid} (owned by {process_user})")

    for gpu_id in gpu_ids:
        kill_processes_on_gpu(gpu_id, username='richard')
    for port in ports:
        result = subprocess.run(f"lsof -i :{port}", shell=True, capture_output=True, text=True)
        if result.stdout:
            lines = result.stdout.splitlines()
            for k in range(1, len(lines)):
                pid = lines[k].split()[1]
                subprocess.run(f"kill -9 {pid}", shell=True)
        result = subprocess.run(f"lsof -i :{port}", shell=True, capture_output=True, text=True)
        while result.stdout:
            time.sleep(3)
            result = subprocess.run(f"lsof -i :{port}", shell=True, capture_output=True, text=True)
        print(f"All process on port {port} killed")
        
def extract_player_hands(text):
    """
    Takes in processed hand history and return a dictionary of player and their hands
    """
    lines = text.split('\n')
    player_order = []
    player_cards = []
    player_hands = {}

    # Extract player order
    for line in lines:
        if line.startswith("Player Order"):
            player_order_str = line.split(":")[1].strip()  # Get the part after the colon
            player_order = [int(x.strip()) for x in player_order_str.split(',')]  # Convert to list of integers
            break

    # Extract player cards
    for line in lines:
        if line.startswith("Player Cards"):
            player_cards_str = line.split(":")[1].strip()  # Get the part after the colon
            player_cards = player_cards_str.split('],')  # Split the cards string into a list
            player_cards = [cards.replace('[', '').replace(']', '').strip() for cards in player_cards]  # Clean up
            break

    # Map cards to player IDs based on order
    for player_id, cards in zip(player_order, player_cards):
        player_hands[player_id] = cards

    return player_hands

def extract_player_stacks(text):
    # Initialize an empty dictionary to hold player IDs and their stack sizes
    player_stacks = {}

    # Split the text into lines
    lines = text.split('\n')

    # Variables to hold extracted player order and chip counts
    player_order = []
    chip_counts = []

    # Extract player order and chip counts from the text
    for line in lines:
        if line.startswith("Player Order"):
            # Extract and convert player order to a list of integers
            player_order_str = line.split(":")[1].strip()
            player_order = [int(x.strip()) for x in player_order_str.split(',')]
        
        if line.startswith("Player Chips"):
            # Extract and convert chip counts to a list of integers
            chip_counts_str = line.split(":")[1].strip()
            chip_counts = [int(x.strip()) for x in chip_counts_str.split(',')]

    # Map player IDs to their corresponding stack sizes
    for player_id, chips in zip(player_order, chip_counts):
        player_stacks[player_id] = chips

    return player_stacks

def process_hand_history(game: TexasHoldEm, gui: TextGUI, text):
    """
    take in game.hand_history.to_string(),
    and return an appropriate hand history for prompting
    """
    button_id = (game.bb_loc - 2 + game.max_players) % game.max_players
    lines = text.split('\n')
    processed_lines = []
    prompt_lines = []

    for line in lines:

        if "Player Cards" in line:
            player_cards = line
            processed_lines.append(line)
            continue

        if "Player Chips" in line:
            player_order = "Player Order (Start with Button, then Small Blind, the Big Blind): " + \
                           ",".join(str((i + button_id) % game.max_players) 
                                    for i in range(game.max_players)
                                    if game.players[((i + button_id) % game.max_players)].state != PlayerState.SKIP)
            processed_lines.append(player_order)
            prompt_lines.append(player_order)
            processed_lines.append(line)
            prompt_lines.append(line)
            continue

        if "New Cards" in line:
            processed_lines.append(line)
            processed_lines.append("Action: ")
            prompt_lines.append(line)
            prompt_lines.append("Action: ")
            continue

        # Adjust player IDs in action lines
        def adjust_player_id(match):
            player_id = int(match.group(1))
            print(player_id)
            adjusted_id = (player_id + button_id) % game.max_players
            return f"({adjusted_id},{match.group(2)}"

        line = re.sub(r'\((\d+),([A-Z_]+)', adjust_player_id, line)

        if "SETTLE" in line:
            break
        
        processed_lines.append(line)
        prompt_lines.append(line)

    processed_history = '\n'.join(processed_lines)
    player_hands = extract_player_hands(processed_history)
    player_chips = extract_player_stacks(processed_history)
    prompt = '\n'.join(prompt_lines)

    prompt = f"You are playing a {game.max_players}-handed No Limit Texas Holdem game, the game information is as follows: \n" + prompt
    prompt += f"You are Player {game.current_player}, it is your turn to perform an action.\n"
    prompt += f"You have {game.players[game.current_player].chips} chips. Your hand is [{player_hands[game.current_player]}] \n"
    prompt += f"(Card suit follows the card number, s = Spade, c = Club, h = Heart, and d = Diamond)\n"
    available_actions = ', '.join(re.split(r'\s{2,}', gui._available_actions_block()[0]))
    prompt += f"Your available moves are: {available_actions} \n"
    prompt += f"Please Announce your action: "

    return processed_history, prompt, text

def generate_prompt(game: TexasHoldEm, gui: TextGUI):
    """
    Generate an appropriate prompt that provides hand history 
    and ask LLM for an action of appropriate format

    Returns: A string of prompt
    """
    processed_history, prompt, raw_history = process_hand_history(game=game, gui=gui, text=game.hand_history.to_string())

    with open("game_simulation/hand_history/prompt.txt", "w") as file:
        file.write(prompt)
    with open("game_simulation/hand_history/hand_history.txt", "w") as file:
        file.write(processed_history)
    with open("game_simulation/hand_history/raw_history.txt", "w") as file:
        file.write(raw_history)

    return prompt

def accept_model_input(game:TexasHoldEm, model:str, prompt:str, random:bool = False):
    """
    Feed the prompt to model and return its response
    Parse the response into readble ActionType format for the simulator

    Returns: action, total
    """
    if random:
        moves = game.get_available_moves()
        return moves.sample()
    else:
        parsed_prompt = parse_prompt(prompt)
        response = get_model_response(parsed_prompt, model=model)
        action, total = parse_response(response)

        return action, total

def parse_prompt(prompt, game:TexasHoldEm=None):
    def parse_card(card):
        value = card[0]
        suit = card[1]
        
        value_map = {
            'A': 'Ace',
            'K': 'King',
            'Q': 'Queen',
            'J': 'Jack',
            'T': '10'
        }
        suit_map = {
            'c': 'Clubs',
            'd': 'Diamonds',
            'h': 'Hearts',
            's': 'Spades'
        }
    
        return f"{value_map.get(value, value)} of {suit_map[suit]}"
    
    lines = prompt.strip().split('\n')
    
    # Parse prehand information
    big_blind = int(next(line for line in lines if line.startswith('Big Blind:')).split(':')[1].strip())
    small_blind = int(next(line for line in lines if line.startswith('Small Blind:')).split(':')[1].strip())
    
    # Parse action history for each stage
    stages = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
    action_history = {}
    board_cards = []
    current_stage = None
    for line in lines:
        if line.strip() in stages:
            current_stage = line.strip()
            action_history[current_stage] = []
        elif line.startswith('New Cards:'):
            cards = line.split(':')[1].strip().strip('[]').split(',')
            board_cards.extend([parse_card(card.strip()) for card in cards if card.strip()])
        elif re.match(r'^\d+\.', line):
            actions = line.split('.')[-1].strip().split(';')
            for action in actions:
                player, action_type, *args = action.strip('()').split(',')
                if action_type == 'RAISE':
                    amount = int(args[0])
                    action_history[current_stage].append(f"Player {player} has raised to {amount} chips")
                else:
                    action_history[current_stage].append(f"Player {player} has {action_type.lower()}")
    
    # Parse player information
    header = next(line for line in lines if line.startswith('You are playing a'))
    player_line = next(line for line in lines if line.startswith('You are Player'))
    stack_size_line = next(line for line in lines if line.startswith('You have'))
    hand_line = next(line for line in lines if 'Your hand is' in line)
    player_id = int(player_line.split('Player')[1].strip().split(',')[0])
    stack_size = int(stack_size_line.split()[2])
    hand = [parse_card(card) for card in hand_line.split('Your hand is')[1].strip().strip('[').strip(']').split()]
    
    # Parse available moves
    available_moves_line = next(line for line in lines if line.startswith('Your available moves are:'))
    available_moves = available_moves_line.split(':')[1].strip().split(', ')
    def transform_raise_text(action_string):
        # Use regular expression to find two numbers separated by a hyphen
        match = re.search(r'RAISE to (\d+) - (\d+)', action_string)
        if match:
            # Extract the two numbers
            lower_amount, upper_amount = match.group(1), match.group(2)
            # Format and return the new sentence
            return f"RAISE to [amount], where [amount] should be strictly between {lower_amount} and {upper_amount}"
        return None  # Return None if the pattern does not match
    # for i in range(len(available_moves)):
    #     if "RAISE" in available_moves[i]:
    #         available_moves[i] = transform_raise_text(available_moves[i])

    # Generate natural language prompt for each stage
    prompt = header + f"\n\n"
    prompt += f"The small blind is {small_blind} chips and the big blind is {big_blind} chips.\n\n"
    
    for stage in stages:
        if stage in action_history:
            prompt += f"{stage}:\n"
            if action_history[stage]:
                prompt += "The action in this stage has been:\n"
                for i, action_line in enumerate(action_history[stage], start=1):
                    prompt += f"{i}. {action_line}\n"
                prompt += "\n"
            else:
                prompt += "There has been no action in this stage.\n\n"
    
    if board_cards:
        prompt += "The board cards are:\n"
        prompt += ', '.join(board_cards) + "\n\n"
    
    prompt += f"You are Player {player_id}, it is your turn to make a move.\n"
    prompt += f"You have a stack of {stack_size} chips and your hand is [{' and '.join(hand)}].\n\n"
    
    prompt += "Your available actions are:\n"
    prompt += '\n'.join(f"- {move}" for move in available_moves) + '\n\n'
    prompt += "Decide on an action based on the strength of your hand, your position, and actions before you. \n\n"
    # prompt += "Fold if you think your hand is weak, call if you think your hand is medium strength, and raise if you think your hand is strong and has high probability of winning\n\n"
    prompt += f"Do not explain your answer. Only give your decision in the format above.\n\n"

    prompt += "Your optimal action is: "
    with open("game_simulation/hand_history/parsed_prompt.txt", "w") as file:
        file.write(prompt)
    return prompt

def get_model_response(parsed_prompt, model):
    def llama_template(prompt):
        return f"""[INST] {prompt} [/INST]"""
    if model == "Llama-13b-Chat":
        data = {
        'model_prompt': llama_template(parsed_prompt),
        'temperature': 0.2,
        'max_tokens': 20,
        'logprobs':10
        }

        response = requests.post('http://crtx.eecs.berkeley.edu:8002/query_model', json=data)
        return response.json()['response']['choices'][0]['text']
    elif model == "Llama-70b-Chat":
        data = {
        'model_prompt': llama_template(parsed_prompt),
        'temperature': 0.2,
        'max_tokens': 20,
        'logprobs':10
        }

        response = requests.post('http://lexi.eecs.berkeley.edu:8002/query_model', json=data)
        return response.json()['response']['choices'][0]['text']
    elif model == "ChatGPT-3.5-Turbo":
        client = OpenAI(api_key="")
        response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=parsed_prompt,
        max_tokens=20,
        temperature=0
        )

        return response.choices[0].text

def parse_response(response_text):
    if "CALL" in response_text:
        return ActionType.CALL, None
    elif "CHECK" in response_text:
        return ActionType.CHECK, None
    elif "FOLD" in response_text:
        return ActionType.FOLD, None
    elif "ALL" in response_text:
        return ActionType.ALL_IN, None
    elif "RAISE" in response_text:
        def parse_raise_amount(action_string):
            match = re.search(r'RAISE(?: to)? (\d+)', action_string)
            if match:
                return int(match.group(1))
            return None
        amount = parse_raise_amount(response_text)
        return ActionType.RAISE, amount
    else:
        raise ValueError("Response not parsable")
    
def get_current_player_hand(game:TexasHoldEm):
    hand = game.hands[game.current_player]
    return f"{parse_new_cards(str(hand[0]))} and {parse_new_cards(str(hand[1]))}"

def get_current_player_stack(game:TexasHoldEm):
    return game.players[game.current_player].chips

def get_current_player_position(game: TexasHoldEm):
    def find_keys_by_value(d, target_value):
        return [key for key in d if d[key] == target_value]
    position_map = get_player_positions(game)
    player_pos_ls = find_keys_by_value(position_map, game.current_player)
    assert len(player_pos_ls) == 1
    return player_pos_ls[0]

def get_current_pot(game:TexasHoldEm):
    return game.pots[game._last_pot_id()].amount + sum(
        [entry[1] for entry in game.pots[game._last_pot_id()].player_amounts.items()])

def get_player_positions(game:TexasHoldEm):
    if game.max_players == 6:
        player_position_map = {"SB":game.sb_loc, "BB":game.bb_loc, "UTG":(game.bb_loc + 1) % game.max_players, 
                           "HJ":(game.bb_loc + 2) % game.max_players, "CO":(game.bb_loc + 3) % game.max_players, "BTN":game.btn_loc}
    elif game.max_players == 2:
        player_position_map = {"BB":game.bb_loc, "SB":game.btn_loc}
    return player_position_map

def get_summary(game:TexasHoldEm, text:str):
    pattern = r'\((\d+),(\w+)(?:,(\d+\.?\d*))?\)'
    matches = re.findall(pattern, text)

    result = []
    for match in matches:
        first = int(match[0])
        second = match[1]
        third = float(match[2]) if match[2] else None
        result.append((first, second, third) if third is not None else (first, second))

    position_map = get_player_positions(game)
    summary = ""
    def find_keys_by_value(d, target_value):
        return [key for key in d if d[key] == target_value]
    for item in result:
        assert len(item) <= 3
        adjusted_id = (item[0] + game.btn_loc) % game.max_players
        player_pos_ls = find_keys_by_value(position_map, adjusted_id)
        assert len(player_pos_ls) == 1
        if len(item) == 2:
            summary += f"{player_pos_ls[0]} {item[1].lower()}, "
        elif len(item) == 3:
            summary += f"{player_pos_ls[0]} {item[1].lower()} {item[2]}, "
    summary = summary[::-1].replace(',', '.', 1)[::-1]
    return summary

def parse_new_cards(card:str):
    assert len(card) == 2, "Invalid Card Length"
    rank_map = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight",
                "9": "Nine", "T": "Ten", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
    suit_map = {"c": "Club", "s": "Spade", "d": "Diamond", "h": "Heart"}
    return f"{rank_map[card[0]]} of {suit_map[card[1]]}"

def get_preflop_summary(game:TexasHoldEm):
    preflop_text = game.hand_history.to_string().split("PREFLOP")[-1]
    if "FLOP" in preflop_text:
        text = preflop_text.split("FLOP")[0]
    elif "SETTLE" in preflop_text: # Preflop All In
        text = preflop_text.split("SETTLE")[0]
    elif preflop_text:
        text = preflop_text
    else:
        raise ValueError
    
    if get_summary(game, text).strip() == '':
        return "Before the flop, nothing happens yet."
    return "Before the flop, " + get_summary(game, text)
    
def get_flop_summary(game:TexasHoldEm):
    flop_text = game.hand_history.to_string().split("PREFLOP")[-1].split("FLOP")[-1]
    flop_cards_str = flop_text.split("New Cards: ")[1].split("\n")[0]
    flop_new_card = ", ".join([parse_new_cards(flop_cards_str[1:3]), 
                               parse_new_cards(flop_cards_str[4:6]), 
                               parse_new_cards(flop_cards_str[7:9])])
    if "TURN" in flop_text:
        text = flop_text.split("TURN")[0]
    elif "SETTLE" in flop_text:
        text = flop_text.split("SETTLE")[0]
    elif flop_text:
        text = flop_text
    else:
        raise ValueError

    if get_summary(game, text).strip() == '':
        return f"The flop comes {flop_new_card}. Nothing happens yet.\n"
    return f"The flop comes {flop_new_card}. {get_summary(game, text)}\n"

def get_turn_summary(game:TexasHoldEm):
    turn_text = game.hand_history.to_string().split("TURN")[-1]
    turn_cards_str = turn_text.split("New Cards: ")[1].split("\n")[0]
    turn_new_card = parse_new_cards(turn_cards_str[1:3])
    if "RIVER" in turn_text:
        text = turn_text.split("RIVER")[0]
    elif "SETTLE" in turn_text: 
        text = turn_text.split("SETTLE")[0]
    elif turn_text:
        text = turn_text
    else:
        raise ValueError
    if get_summary(game, text).strip() == '':
        return f"The turn comes {turn_new_card}. Nothing happens yet.\n"
    return f"The turn comes {turn_new_card}. {get_summary(game, text)}\n"

def get_river_summary(game:TexasHoldEm):
    river_text = game.hand_history.to_string().split("RIVER")[-1]
    river_cards_str = river_text.split("New Cards: ")[1].split("\n")[0]
    river_new_card = parse_new_cards(river_cards_str[1:3])
    if "SETTLE" in river_text:
        text = river_text.split("SETTLE")[0]
    elif river_text:
        text = river_text
    else:
        raise ValueError
    if get_summary(game, text).strip() == '':
        return f"The river comes {river_new_card}. Nothing happens yet.\n"
    return f"The river comes {river_new_card}. {get_summary(game, text)}\n"

def construct_prompt(game: TexasHoldEm):
    # Separate cases for preflop spot and postflop spots
    # Aggregate flop/turn/river if applicable
    preflop_summary = get_preflop_summary(game)
    flop_summary = get_flop_summary(game) if "FLOP" in game.hand_history.to_string().split("PREFLOP")[-1] else ""
    turn_summary = get_turn_summary(game) if "TURN" in game.hand_history.to_string().split("PREFLOP")[-1] else ""
    river_summary = get_river_summary(game) if "RIVER" in game.hand_history.to_string().split("PREFLOP")[-1] else ""
    pot_summary = get_current_pot(game) / 2 # Map to 0.5/1 Game
    stack_summary = get_current_player_stack(game) / 2 # Map to 0.5/1 Game
    hand_summary = get_current_player_hand(game)
    position_summary = get_current_player_position(game)
    def replace_numbers_with_half(text):
        pattern = r'\d+\.\d+|\d+'
        def replace_with_half(match):
            number = float(match.group(0))
            half_number = number / 2
            # Return the half value as a string, remove trailing .0 for integers
            return str(half_number)
        modified_text = re.sub(pattern, replace_with_half, text)
    
        return modified_text
    preflop_summary = replace_numbers_with_half(preflop_summary)
    flop_summary = replace_numbers_with_half(flop_summary)
    turn_summary = replace_numbers_with_half(turn_summary)
    river_summary = replace_numbers_with_half(river_summary)

    prompt = f"""You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision. 

Here is a game summary: 

The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. 
The player positions involved in this game are {', '.join(list(get_player_positions(game).keys()))}.
In this hand, your position is {position_summary}, and your holding is [{hand_summary}].
{preflop_summary}
{flop_summary + turn_summary + river_summary}

Now it is your turn to make a move.
To remind you, the current pot size is {pot_summary} chips, your remaining stack size is {stack_summary}, and your holding is [{hand_summary}].

Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.
Your optimal action is:"""
    return prompt

def parse_model_output(game:TexasHoldEm, response:str):
    response = response.lower()
    if "fold" in response:
        return (ActionType.FOLD, None)
    if "call" in response:
        return (ActionType.CALL, None)
    if "all in" in response:
        return (ActionType.ALL_IN, None)
    if "check" in response:
        return (ActionType.CHECK, None)
    if "bet" in response or "raise" in response:
        pattern = r'\d+\.\d+|\d+'
        match = re.search(pattern, response)
        if match:
            number_str = match.group(0)
            number = float(number_str)
            size = int(round(number, 0))
        else:
            size = None
        return (ActionType.RAISE, size * 2) # Map back to 1/2 Game
    available_moves = game.get_available_moves().action_types
    if ActionType.CHECK in available_moves:
        return (ActionType.CHECK, None)
    else:
        return (ActionType.FOLD, None)
    
def adjust_action(game:TexasHoldEm, gui:TextGUI, action:ActionType, size:float|int|None):
    available_moves = game.get_available_moves().action_types
    raise_range = game.get_available_moves().raise_range
    if game.validate_move(action=action, total=size):
        # print("Valid Move!")
        if ActionType.CHECK in available_moves and action == ActionType.FOLD: # Avoid folding when can check
            if gui:
                gui.display_error("Check Available But Choose to Fold, Adjusted to Checking")
            return (ActionType.CHECK, None)
        return (action, size)
    # Can always FOLD or ALL_IN
    if action == ActionType.CALL and ActionType.CHECK in available_moves:
        return (ActionType.CHECK, None)
    if action == ActionType.RAISE:
        if len(raise_range) == 0:
            return (ActionType.CALL, None)
        if size < min(raise_range):
            return (ActionType.RAISE, min(raise_range))
        if size > max(raise_range):
            return (ActionType.RAISE, max(raise_range))
    if ActionType.CHECK in available_moves:
        return (ActionType.CHECK, None)
    else:
        return (ActionType.FOLD, None)
        
def check_if_early_stop(game:TexasHoldEm):  
    if "FLOP" not in game.hand_history.to_string().split("PREFLOP")[-1]:
        return False
    return sum([player.state in [PlayerState.ALL_IN, PlayerState.TO_CALL, PlayerState.IN] for player in game.players]) > 2

def store_game_result(game:TexasHoldEm, model_list:list[int], output_path:str, game_num:int, recorded_deck:Deck):
    game_num_text = f"GAME {game_num + 1}"
    model_order_text = str(model_list)
    adjusted_id_map = str({id: (id + game.btn_loc) % game.max_players for id in range(game.max_players)})
    game_summary = game.hand_history.to_string()
    ending_chips_summary = f"Ending Player Chips: {[player.chips for player in game.players]}"
    recorded_deck_summary = str(recorded_deck.cards)
    player_cards_summary = str(game.hands)
    linebreak = "-----------------------------------------------------------------"
    return "\n".join([game_num_text, model_order_text, adjusted_id_map, game_summary, 
                      ending_chips_summary, recorded_deck_summary, player_cards_summary,
                      linebreak])

# Create a few-shot-example set to sample from
def create_example_bank(full_data:list, preflop:bool, sample_per_option:int):
    # random.seed(42)
    if preflop:
        all_possible_options = ['raise', 'fold', 'call', 'all in', 'check']
    else:
        all_possible_options = ['raise', 'fold', 'call', 'check', 'bet']
    bank = {}
    for option in all_possible_options:
        option_subset = [entry for entry in full_data if option in entry['output']]
        bank[option] = random.sample(option_subset, min(sample_per_option, len(option_subset)))
    return bank

# Function to sample k entries and construct prompt
def construct_k_shot_prompt(test_prompt:str, example_bank:list, k:int=5):
    # random.seed(42)
    examples = []
    for option in example_bank.keys():
        examples.extend(random.sample(example_bank[option], 1))
    random.shuffle(examples)
    examples = [sample['instruction'] + " " + sample['output'] + "\n" for sample in examples]
    examples_str = "\n".join(examples)
    # print(examples_str + "\n" + test_prompt)
    return examples_str + "\n" + test_prompt