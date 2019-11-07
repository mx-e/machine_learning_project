import torch

def train_model(optimizer, env, num_episodes = 20, target_update = 10):
    import os
    if not os.path.exists("results"):
        os.mkdir("results")

    snapshots = dict()
    for i_episode in range(num_episodes):

        env.reset()
        states = []
        actions =[]
        rewards = []

        done = False
        while(not done):
            current_screen = env.get_screen()
            states.append(current_screen)

            # Select and perform an action
            action = env.select_action(current_screen, optimizer.policy_net)
            actions.append(action)
            _, reward, done, _ = env.step(action)
            rewards.append(reward)

        # Perform one step of the optimization (on the target network)
        optimizer.optimize_model(states, actions, rewards)
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            snapshots[f"model_snapshot_@{i_episode}"] = optimizer.policy_net.state_dict()
            print(f'saved snapshot @ episode {i_episode}')
            optimizer.target_net.load_state_dict(optimizer.policy_net.state_dict())

    snapshots[f"model_snapshot_@{num_episodes}"] = optimizer.policy_net.state_dict()
    print(f'saved snapshot @ episode {num_episodes}')

    print('Complete')
    torch.save(snapshots, './results/snapshots')

