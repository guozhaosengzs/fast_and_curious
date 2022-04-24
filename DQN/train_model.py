import argparse
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

import pandas as pd

RENDER                        = True
STARTING_EPISODE              = 1
ENDING_EPISODE                = 1000
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 25
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    # parser.add_argument('-t', '--trial', help='Specify the number of trial for model and csv to save.')
    args = parser.parse_args()

    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=args.epsilon)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end
    # if args.trial:
    #     TRIAL = args.trial
    TRIAL = "CNN_allP_600_FINAL"

    # metric
    steering_records = []
    gas_records = []
    break_records = []
    reward_records = []

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        steering_count = 0
        gas_count = 0
        break_count = 0

        # loop
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            # reward tracking
            reward = 0

            for _ in range(SKIP_FRAMES+1):
                next_state, r, done, info = env.step(action)
                reward += r
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Extra bonus for the model if it does not using gas and break at the same time
            if (action[1] == 1 and action[2] == 0): 
                reward *= 1.1
            if (action[1] == 0 and action[2] == 0.2):
                reward *= 1.1

            # Extra bonus for no sudden steering
            if len(agent.memory) >= 2:
                T_minus_1_action = agent.action_space[agent.memory[-1][1]]

                T_minus_1_steering = T_minus_1_action[0]
                T_minus_0_steering = action[0]

                if T_minus_1_steering * T_minus_0_steering == -1:
                    reward -= 0.1 # penalty
                
                # Metric tracker
                if T_minus_1_steering != T_minus_0_steering: 
                    steering_count += 1 # steering change

            # Extra bonus for no continuos break or gas 
            if len(agent.memory) >= 10:

                action_list = [action] # from current state to T-9 state
                for i in range(-1, -9, -1):   
                    action_list.append(agent.action_space[agent.memory[i][1]])

                recent_gas = [a[1] for a in action_list]  
                recent_break = [a[2] for a in action_list]    

                if all(g == 1 for g in recent_gas):
                    reward -= 0.05 # penalty

                if all(b == 0.2 for b in recent_break):
                    reward -= 0.05 # penalty

                if all(g == 0 for g in recent_gas) and all(b == 0 for b in recent_break):
                    reward -= 5 # penalty
            
            # Metric tracker
            if action[1] == 1: # Gas
                gas_count += 1

            if action[2] == 0.2: # Break
                break_count += 1

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.5}, Epsilon: {:.2}'.format(e, ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.epsilon)))
                # save metrics
                steering_records.append(steering_count / time_frame_counter)
                gas_records.append(gas_count / time_frame_counter)
                break_records.append(break_count / time_frame_counter)
                reward_records.append(total_reward)
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            model_save_path = './save/' + TRIAL + '.h5'
            agent.save(model_save_path.format(e))

        # export saved data
        metric_df = pd.DataFrame({'steering': steering_records, 'gas': gas_records, 'break': break_records, 'rewards': reward_records})
        metric_df.to_csv(TRIAL + '.csv')

    env.close()
