import torch
import plotly.express as px

def train_model(optimizer, env, num_episodes = 20, target_update = 10):
    import os
    if not os.path.exists("results"):
        os.mkdir("results")

    snapshots = dict()
    statistics = dict()
    statistics['scores'] = []
    statistics['num_boxes_solved'] = []
    statistics['episode'] = []
    for i_episode in range(num_episodes):
        statistics['episode'].append(i_episode)
        env.reset()
        states = []
        actions =[]
        rewards = []
        done = False
        bad_streak = 0
        while(not done and bad_streak < 35):
            current_screen = env.get_screen()
            states.append(current_screen)

            # Select and perform an action
            action = env.select_action(current_screen, optimizer.policy_net)
            actions.append(action)
            _, reward, done, _ = env.step(action)
            bad_streak += 1
            if (reward >= 0):
                reward += 3.5
                bad_streak = 0
            rewards.append(reward)

        # Perform one step of the optimization (on the target network)
        print(f'\nfinished episode {i_episode}:')
        statistics['scores'].append(optimizer.optimize_model(states, actions, rewards))
        # Update the target network, copying all weights and biases in DQN
        statistics['num_boxes_solved'].append(env.get_no_of_solved_boxes())
        if i_episode % target_update == 0:
            snapshots[f"model_snapshot_@{i_episode}"] = optimizer.policy_net.state_dict()
            print(f'saved snapshot @ episode {i_episode}')
            optimizer.target_net.load_state_dict(optimizer.policy_net.state_dict())
            fig1 = px.line(statistics, 'episode', 'scores')
            fig1.write_image(f'results/scores.pdf')

            fig2 = px.line(statistics, 'episode', 'num_boxes_solved')
            fig2.write_image(f'results/boxes.pdf')
    snapshots[f"model_snapshot_@{num_episodes}"] = optimizer.policy_net.state_dict()

    print('Complete')
    torch.save(snapshots, './results/snapshots')
    return statistics

