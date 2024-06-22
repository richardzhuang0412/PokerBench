import time
#import pyautogui
import re
import random
from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem import ActionType
from texasholdem.agents.basic import random_agent
from texasholdem.game.player_state import PlayerState
from utils import *

BUY_IN = 500
SMALL_BLIND = 2
BIG_BLIND = 5
MAX_PLAYERS = 6
game = TexasHoldEm(buyin=BUY_IN, small_blind=SMALL_BLIND,big_blind=BIG_BLIND, max_players=MAX_PLAYERS)
gui = TextGUI(game=game)

def play_game(game:TexasHoldEm, gui:TextGUI, model_list: list[str],
              output_path:str='game_simulation/hand_history'):
    """
    Main loop for game simulation
    """
    while game.is_game_running():
        game.start_hand()
        
        while game.is_hand_running():
            if not gui.game.is_hand_running():
                break
            
            gui.display_state()
            i = 0
            valid = False
            while not valid:
                if i > 0:
                    gui.display_error(f"Invalid Move for {i} time")
                    time.sleep(2)
                if i == 10:
                    raise ValueError("Invalid Move Loop")
                # gui.prompt_input()
                prompt = generate_prompt(game, gui)
                current_model = model_list[game.current_player]
                gui.display_error(f"Player ID: {game.current_player}, Current Model: {current_model}")
                action, total = accept_model_input(game=game, model=current_model, prompt=prompt, random=False)
                valid = game.validate_move(action=action, total=total)
                i += 1
            i = 0
            # Take the action in the game
            gui.game.take_action(action, total=total)
            time.sleep(2)

            # Display the winners if the hand ended
            if not gui.game.is_hand_running():
                # A line to record game result
                gui.display_state()
                time.sleep(5)
                # gui.display_win() # Need to "Press Enter to Continue"

        path = game.export_history(output_path)     # save history
        # gui.replay_history(path)                 # replay history

model_list = ["ChatGPT-3.5-Turbo" for i in range(6)]
random.shuffle(model_list)
play_game(game, gui, model_list=model_list, output_path='game_simulation/hand_history/all_chatgpt')