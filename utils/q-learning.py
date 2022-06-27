# Dependencies
import gym
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
env = gym.make('MountainCar-v0')
env.reset()

# Constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 24001
SAVE_EVERY = 100
SHOW_EVERY = 2000
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Setup Q-Table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Metrics
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
    # Function to return the correct discrete bucket our current state is in
    ds = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(ds.astype(int))


for episode in range(EPISODES):
    episode_reward = 0
    show = False
    done = False
    discrete_state = get_discrete_state(env.reset())
    if episode % SHOW_EVERY == 0:
        show = True

    # Core loop for agent interaction with the environment
    while not done:
        if np.random.random() > epsilon:
            # Do best action from Q-Table
            action = np.argmax(q_table[discrete_state])
        else:
            # Do a random action
            action = np.random.randint(0, env.action_space.n)
        # Step forward and recalculate
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        episode_reward += reward

        if show:
            # Render occasionally to observe progress
            env.render()
        if not done:
            # Calculate values and propagate reward
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            # Calculate our new Q-value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f'Successful completion at: {episode}')
            q_table[discrete_state + (action, )] = 0
        # Move state forward to new state
        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Append metrics
    ep_rewards.append(episode_reward)
    if not episode % SAVE_EVERY:
        # Save Q-Table
        np.save(f'q-tables/{episode}-qtable.npy', q_table)
    if not episode % SHOW_EVERY:
        # Append aggregate metrics
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f'Episode: {episode}, Average: {average_reward}, Min: {min(ep_rewards[-SHOW_EVERY:])} ' +
              f'Max: {max(ep_rewards[-SHOW_EVERY:])}')
# Close the environment after training
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
