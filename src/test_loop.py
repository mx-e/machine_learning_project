from itertools import count

def test_model(episodes, env, policy_net):
    iteration_count = 0
    reward_sum = 0
    episode_len_sum = 0
    for i_episode in range(episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = env.get_screen()
        current_screen = env.get_screen()
        for t in count():
            iteration_count += 1
            current_screen = env.get_screen()
            # Select and perform an action
            action = policy_net(current_screen).max(1)[1].view(1, 1).item()
            _, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                episode_len_sum += t
                break
    env.close()
    average_reward = reward_sum / iteration_count
    average_ep_len = episode_len_sum / episodes
    return average_reward, average_ep_len