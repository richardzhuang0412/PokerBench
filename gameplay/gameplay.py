from texasholdem.game.game import TexasHoldEm
from texasholdem.gui.text_gui import TextGUI
from texasholdem import ActionType
from texasholdem.agents.basic import random_agent
from texasholdem.game.player_state import PlayerState
from texasholdem.game import GameState
from texasholdem.card.card import Card
from texasholdem.card.deck import Deck
import time, random, re
import numpy as np
from utils import *
import requests, json, random, os, signal, subprocess, time
from tqdm import tqdm

def play_game(game:TexasHoldEm, gui:TextGUI|None, model_list: list[int], output_path:str, game_num:int,
              test_mode:bool, sleep_time:float|int, preflop_bank:dict|None=None, postflop_bank:dict|None=None, 
              custom_deck:Deck|None=None, custom_hands:dict|None=None):
    """
    Main loop for game simulation
    """
    normal_end = True
    while game.is_game_running():
        game.start_hand()
        if custom_deck and custom_hands:
            game._deck = custom_deck
            game.hands = custom_hands
        recorded_deck = game._deck
        if test_mode:
            # print(game._deck.cards)
            # print(Deck._get_full_deck())
            # random.seed(42)
            # game._deck = 
            pass
        while game.is_hand_running():
            if check_if_early_stop(game):
                if gui:
                    gui.display_error("Not Heads Up Postflop")
                time.sleep(2)
                normal_end = False
                break
            if not game.is_hand_running():
                break
            if gui: 
                gui.display_state()
            i = 0
            valid = False
            while not valid:
                if i > 0:
                    if gui: 
                        gui.display_error(f"Invalid Move for {i} time")
                    # print(f"Invalid Move for {i} time")
                    time.sleep(2)
                if i == 10:
                    raise ValueError("Invalid Move Loop")
                # gui.prompt_input()
                current_model = model_list[game.current_player]
                if gui:
                    gui.display_error(f"Game #{game_num + 1} Player ID: {game.current_player}, Current Model: {current_model[0]}")
                if "BOT" in current_model[0]:
                    action, total = (ActionType.FOLD, None)
                elif not test_mode:
                    prompt = construct_prompt(game)
                    # print(prompt)
                    if "meta-llama" in current_model[0]:
                        if "FLOP" not in game.hand_history.to_string().split("PREFLOP")[-1]:
                            bank = preflop_bank
                        else:
                            bank = postflop_bank
                        prompt = construct_k_shot_prompt(prompt, bank)
                        model_response = query_vllm_host(prompt=prompt, port=current_model[1], model_name=current_model[0])
                        # print(prompt)
                    elif "gpt" in current_model[0]:
                        if "FLOP" not in game.hand_history.to_string().split("PREFLOP")[-1]:
                            bank = preflop_bank
                        else:
                            bank = postflop_bank
                        prompt = construct_k_shot_prompt(prompt, bank)
                        model_response = query_model_api(prompt=prompt, model_name=current_model[0])
                    else:
                        model_response = query_litgpt_host(prompt=prompt, port=current_model[1])
                    # print(f"Model {current_model[0]} Response: {model_response}")
                    parsed_response = parse_model_output(game, model_response)
                    # print(f"Model {current_model[0]} Parsed Response: {model_response}")
                    # print(f"Parsed Response: {parsed_response}")
                    # print(f"Available Actions: {game.get_available_moves().action_types}")
                    # print(f"Available Raise Range: {game.get_available_moves().raise_range}")
                    adjusted_action = adjust_action(game, gui, parsed_response[0], parsed_response[1])
                    # print(f"Adjusted Action Valid: {game.validate_move(action=adjusted_action[0], total=adjusted_action[1])}")
                    # print(f"Player ID: {game.current_player}, Current Model: {current_model}")
                    # if gui:
                    #     gui.display_error(f"Game #{game_num + 1} Player ID: {game.current_player}, Current Model: {current_model[0]}")
                    # time.sleep(3)
                    # action, total = accept_model_input(game=game, model=current_model, prompt=prompt, random=True)
                    # print(f"Random Agent Decision: {(action, total)}")
                    # time.sleep(3)
                    action, total = adjusted_action
                else:
                    prompt = construct_prompt(game)
                    print(prompt)
                    action, total = game.get_available_moves().sample()
                valid = game.validate_move(action=action, total=total)
                # if valid:
                #     gui.display_error("Valid Move Taken!")
                i += 1
            i = 0
            # Take the action in the game
            game.take_action(action, total=total)
            # print(game.hand_history.to_string())
            time.sleep(sleep_time)

            # Display the winners if the hand ended
            if not game.is_hand_running():
                # A line to record game result
                if gui: 
                    gui.display_state()                
                    # print(game.hand_history.to_string())
                time.sleep(sleep_time)
                # gui.display_win() # Need to "Press Enter to Continue"

        # path = game.export_history(output_path)     # save history
        # TODO: Store Result
        game.game_state = GameState.STOPPED
        return (normal_end, store_game_result(game, model_list, output_path, game_num, recorded_deck))
        # gui.replay_history(path)                 # replay history

if __name__ == "__main__":
    SMALL_BLIND = 1
    BIG_BLIND = 2
    BUY_IN = 200
    MAX_PLAYERS = 2
    # GAME = TexasHoldEm(buyin=BUY_IN, small_blind=SMALL_BLIND,big_blind=BIG_BLIND, max_players=MAX_PLAYERS)
    # GUI = TextGUI(game=GAME)
    # GUI = None
    CHECKPOINT_1 = "../out/llama-3-8b-110k/step-005000"
    CHECKPOINT_2 = "gpt-4-turbo"
    # GPU_ID = 10
    # PORT_1 = 0
    # PORT_2 = 1
    GPU_ID = 7
    PORT_1 = 9000
    PORT_2 = 9000
    NUM_GAME = 1000
    GAME_LOG_OUTPUT_PATH = './history'
    SIMULATION_RESULT_OUTPUT_PATH = './history/gpt_4_turbo_vs_8b_5000_headsup_game_result.txt'
    # MODEL_LIST = [(CHECKPOINT_1, PORT_1), (CHECKPOINT_1, PORT_1), (CHECKPOINT_1, PORT_1),
    #               (CHECKPOINT_2, PORT_2), (CHECKPOINT_2, PORT_2), (CHECKPOINT_2, PORT_2)]
    # MODEL_LIST = [(CHECKPOINT_1, PORT_1), ("PLACEHOLDER_BOT", None), ("PLACEHOLDER_BOT", None),
    #             (CHECKPOINT_2, PORT_2), ("PLACEHOLDER_BOT", None), ("PLACEHOLDER_BOT", None)]
    MODEL_LIST = [(CHECKPOINT_1, PORT_1), (CHECKPOINT_2, PORT_2)]
    SLEEP_TIME = 0.01
    TEST_MODE = False

    if not TEST_MODE:
        # Initialize Model Servers First
        kill_host(gpu_ids=[GPU_ID], ports=[PORT_1, PORT_2])
        # kill_host(gpu_ids=[GPU_ID], ports=[PORT_1])
        host_litgpt_model(gpu_id=GPU_ID, checkpoint_path=CHECKPOINT_1, port=PORT_1)
        # host_litgpt_model(gpu_id=GPU_ID, checkpoint_path=CHECKPOINT_2, port=PORT_2)
        with open(SIMULATION_RESULT_OUTPUT_PATH, 'w') as file:
            pass

    game_num = 0
    PREFLOP_FULL_PATH = "../SFT/sft_preflop_full.json"
    POSTFLOP_FULL_PATH = "../SFT/sft_postflop_full.json"
    with open(PREFLOP_FULL_PATH, 'r') as file:
        preflop_full = json.load(file)
    with open(POSTFLOP_FULL_PATH, 'r') as file:
        postflop_full = json.load(file)
    preflop_bank = create_example_bank(preflop_full, preflop=True, sample_per_option=50)
    postflop_bank = create_example_bank(postflop_full, preflop=False, sample_per_option=50)

    while game_num < NUM_GAME:
        random.shuffle(MODEL_LIST)
        GAME = TexasHoldEm(buyin=BUY_IN, small_blind=SMALL_BLIND,big_blind=BIG_BLIND, max_players=MAX_PLAYERS)
        GUI = TextGUI(game=GAME)
        # GUI = None
        normal_end, result = play_game(GAME, GUI, MODEL_LIST, GAME_LOG_OUTPUT_PATH, 
                                       game_num, test_mode = TEST_MODE, sleep_time=SLEEP_TIME,
                                       preflop_bank=preflop_bank, postflop_bank=postflop_bank)
        if normal_end:
            game_num += 1
        if not TEST_MODE:
            with open(SIMULATION_RESULT_OUTPUT_PATH, 'a') as file:
                file.write(result + '\n')
    if not TEST_MODE:
        kill_host(gpu_ids=[GPU_ID], ports=[PORT_1, PORT_2])
    
