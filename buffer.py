import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_size, n_actions, sad_robot=False, 
                 augment_data=False, augment_rewards=False, expert_data_ratio=0.1,
                 augment_noise_ratio=0.1):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, input_size))
        self.new_state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.sad_robot = sad_robot
        self.augment_data = augment_data
        self.augment_rewards = augment_rewards
        self.augment_noise_ratio = augment_noise_ratio # Only relevant if augment rewards is set. 
        self.expert_data_ratio = expert_data_ratio
        self.expert_data_cutoff = 0


    def __len__(self):
        return self.mem_ctr

    def can_sample(self, batch_size):
        if self.mem_ctr > (batch_size * 1000):
            return True
        else:
            return False

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        
        if self.expert_data_ratio > 0:
            expert_batch_quantity = int(batch_size * self.expert_data_ratio)
            random_batch = np.random.choice(max_mem, batch_size - expert_batch_quantity)
            expert_batch = np.random.choice(self.expert_data_cutoff, expert_batch_quantity)
            batch = np.concatenate((random_batch, expert_batch))
        else:
            batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        if self.augment_data:
            # Compute dynamic noise levels based on the average absolute values
            state_noise_std = self.augment_noise_ratio * np.mean(np.abs(states))
            action_noise_std = self.augment_noise_ratio * np.mean(np.abs(actions))
            reward_noise_std = self.augment_noise_ratio * np.mean(np.abs(rewards))

            # Adding dynamic noise to states, actions, and rewards
            states = states + np.random.normal(0, state_noise_std, states.shape)
            actions = actions + np.random.normal(0, action_noise_std, actions.shape)
            # rewards = rewards + np.random.normal(0, reward_noise_std, rewards.shape)

        if self.augment_rewards:
            rewards = rewards * 100

            if self.sad_robot:
                rewards = rewards - 1

        return states, actions, rewards, states_, dones

    def save_to_csv(self, filename='checkpoints/memory.npz'):
        np.savez(filename,
                 state=self.state_memory[:self.mem_ctr],
                 action=self.action_memory[:self.mem_ctr],
                 reward=self.reward_memory[:self.mem_ctr],
                 next_state=self.new_state_memory[:self.mem_ctr],
                 done=self.terminal_memory[:self.mem_ctr])
        print(f"Saved {filename}")

    def load_from_csv(self, filename='checkpoints/memory.npz', expert_data=True):
        try:
            data = np.load(filename)
            self.mem_ctr = len(data['state'])
            self.state_memory[:self.mem_ctr] = data['state']
            self.action_memory[:self.mem_ctr] = data['action']
            self.reward_memory[:self.mem_ctr] = data['reward']
            self.new_state_memory[:self.mem_ctr] = data['next_state']
            self.terminal_memory[:self.mem_ctr] = data['done']
            print(f"Successfully loaded {filename} into memory")
            print(f"{self.mem_ctr} memories loaded")
            
            if(expert_data):
                self.expert_data = expert_data
                self.expert_data_cutoff = self.mem_ctr

        except:
            print(f"Unable to load memory from {filename}")


def classify_records_in_replay_buffer(replay_buffer, goal_start_map):
    # Initialize a dictionary to count the occurrences of each goal
    goal_counts = {goal: 0 for goal in goal_start_map.keys()}
    
    # Iterate over all records in the replay buffer
    for i in range(len(replay_buffer)):
        # Extract the state
        state = replay_buffer.state_memory[i]
        
        # The goal-related part of the state starts after the first 59 entries
        goal_part = state[59:]
        
        # Dictionary to store the "score" for each goal based on non-zero values
        goal_scores = {goal: 0 for goal in goal_start_map.keys()}
        
        # Evaluate each goal segment separately
        for goal, start_idx in goal_start_map.items():
            # Determine the length of the segment by looking for where the next goal starts
            if goal != 'kettle':  # If not the last goal, look for the next goal's start
                end_idx = min([idx for g, idx in goal_start_map.items() if idx > start_idx])
            else:  # If 'kettle', it's the last goal, so take the rest of the array
                end_idx = len(goal_part)
                
            # Extract the relevant segment for this goal
            goal_segment = goal_part[start_idx:end_idx]
            
            # Calculate the score as the sum of absolute values (magnitude of non-zero values)
            goal_scores[goal] = np.sum(np.abs(goal_segment))
        
        # Determine the goal with the highest score
        most_likely_goal = max(goal_scores, key=goal_scores.get)
        
        # Increment the count for the most likely goal
        goal_counts[most_likely_goal] += 1
    
    return goal_counts