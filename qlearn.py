import numpy as np
import gym

# Create the environment
env = gym.make('Taxi-v3')

# Set the hyperparameters
num_episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Initialize the Q-table to zeros
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Run the Q-learning algorithm
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    done = False
    print(state)
    while not done:
        # Choose an action using an epsilon-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(Q[state[0]])  # exploit
        
        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _, prs= env.step(action)
        print(env.step(action))
        
        # Update the Q-table
        Q[state[0], action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state[0], action])
        
        # Update the state for the next iteration
        state = (next_state, prs)
        print(state, reward, done, 'check')
    
    # Print the total reward for this episode
    print(f"Episode {episode + 1}: Total reward = {reward}")
    
# Close the environment
env.close()
