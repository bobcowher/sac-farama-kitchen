import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_utils import *
from model import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from buffer import ReplayBuffer



class Agent(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy, target_update_interval,
                 automatic_entropy_tuning, hidden_size, learning_rate, exploration_scaling_factor):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Initialize the predictive model
        self.predictive_model = PredictiveModel(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.predictive_model_optim = Adam(self.predictive_model.parameters(), lr=learning_rate)

        self.exploration_scaling_factor = exploration_scaling_factor

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        # else:
        #     self.alpha = 0
        #     self.automatic_entropy_tuning = False
        #     self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        #     self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def pretrain_actor(self, memory : ReplayBuffer, epochs=100, batch_size=64, summary_writer_name="", noise_ratio=0.1):
        self.policy.train()
        
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(len(memory) // batch_size):
                state_batch, action_batch, _, _, _ = memory.sample_buffer(batch_size=batch_size, augment_data=True, noise_ratio=noise_ratio)
                
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                
                predicted_actions, _, _ = self.policy.sample(state_batch)
                loss = F.mse_loss(predicted_actions, action_batch)
                
                self.policy_optim.zero_grad()
                loss.backward()
                self.policy_optim.step()
                
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / (len(memory) // batch_size)
            writer.add_scalar('pretrain_actor/loss', avg_epoch_loss, epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss}")

        writer.close()


    def pretrain_actor_and_critic(self, memory : ReplayBuffer, epochs=100, batch_size=64, summary_writer_name="", noise_ratio=0.1):
        self.policy.train()
        
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)
        
        for epoch in range(epochs):
            epoch_loss_actor = 0
            epoch_loss_critic = 0

            for _ in range(len(memory) // batch_size):
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)
                
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
                
                if epoch % 2 == 0:
                    predicted_actions, _, _ = self.policy.sample(state_batch)
                    loss = F.mse_loss(predicted_actions, action_batch)
                    
                    self.policy_optim.zero_grad()
                    loss.backward()
                    self.policy_optim.step()
                    
                    epoch_loss_actor += loss.item()
                else: 
                    with torch.no_grad():
                        next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                        next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

                    qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
                    qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                    qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                    qf_loss = qf1_loss + qf2_loss

                    # Update the critic network
                    self.critic_optim.zero_grad()
                    qf_loss.backward()
                    self.critic_optim.step()
                    
                    epoch_loss_critic += qf_loss.item()

            if epoch % 2 == 0:
                avg_epoch_loss_actor = epoch_loss_actor / (len(memory) // batch_size)
                writer.add_scalar('pretrain_actor/loss', avg_epoch_loss_actor, epoch)
                print(f"Epoch {epoch+1}/{epochs}, Actor Loss: {avg_epoch_loss_actor}")
            else:
                avg_epoch_loss_critic = epoch_loss_critic / (len(memory) // batch_size)
                writer.add_scalar('pretrain_critic/loss', avg_epoch_loss_critic, epoch)
                print(f"Epoch {epoch+1}/{epochs}, Critic Loss: {avg_epoch_loss_critic}")

            
            soft_update(self.critic_target, self.critic, self.tau)


        writer.close()



    def pretrain_critic_with_human_data(self, memory : ReplayBuffer, epochs=100, batch_size=64, summary_writer_name="", noise_ratio=0.1):
        self.critic.train()
        
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(len(memory) // batch_size):
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample_buffer(batch_size=batch_size, augment_data=True, noise_ratio=noise_ratio)
                
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
                
                with torch.no_grad():
                    qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, action_batch)  # Human's next action
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    target_q_value = reward_batch + self.gamma * (1 - done_batch) * min_qf_next_target
                
                qf1, qf2 = self.critic(state_batch, action_batch)
                qf1_loss = F.mse_loss(qf1, target_q_value)
                qf2_loss = F.mse_loss(qf2, target_q_value)
                qf_loss = qf1_loss + qf2_loss
                
                self.critic_optim.zero_grad()
                qf_loss.backward()
                self.critic_optim.step()
                
                epoch_loss += qf_loss.item()

            avg_epoch_loss = epoch_loss / (len(memory) // batch_size)
            writer.add_scalar('pretrain_critic/loss', avg_epoch_loss, epoch)
            print(f"Epoch {epoch+1}/{epochs}, Critic Loss: {avg_epoch_loss}")

        writer.close()

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Predict the next state using the predictive model
        # predicted_next_state = self.predictive_model(state_batch, action_batch)

        # # Calculate prediction loss as an intrinsic reward
        # # print("Predicted next state:", predicted_next_state.shape)
        # # print("Actual next state:", next_state_batch.shape)
        # prediction_error = F.mse_loss(predicted_next_state, next_state_batch)
        # prediction_error_no_reduction = F.mse_loss(predicted_next_state, next_state_batch, reduce=False)

        # scaled_intrinsic_reward = prediction_error_no_reduction.mean(dim=1)
        # scaled_intrinsic_reward = self.exploration_scaling_factor * torch.reshape(scaled_intrinsic_reward, (batch_size, 1))

        # Calculate penalty for stagnation
        # stagnation_penalty = -0.1  # Adjust the value as needed
        # if (state_batch == next_state_batch).all(dim=1).sum() > 0:
        #     reward_batch += stagnation_penalty

        # print(f"Scaled Intrinsic Reward(mean): {scaled_intrinsic_reward.mean()}")

        # reward_batch = reward_batch + scaled_intrinsic_reward # TODO: Uncomment for intrinsic reward

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        # Update the critic network
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # # Update the predictive model
        # self.predictive_model_optim.zero_grad()
        # prediction_error.backward()
        # self.predictive_model_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), 0, alpha_tlogs.item()


    def train(self, env, env_name, memory, episodes=1000, batch_size=64, updates_per_step=1, summary_writer_name="", max_episode_steps=100):

        warmup = 20

        # Tesnorboard
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        # Training Loop
        total_numsteps = 0
        updates = 0

        for i_episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, _ = env.reset()

            while not done and episode_steps < max_episode_steps:
                if warmup > i_episode:
                    action = env.action_space.sample()  # Sample random action
                else:
                    action = self.select_action(state)  # Sample action from policy

                if memory.can_sample(batch_size=batch_size):
                    # Number of updates per step in environment
                    for i in range(updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, prediction_error_loss, alpha = self.update_parameters(memory,
                                                                                                            batch_size,
                                                                                                            updates)

                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('loss/prediction_error', prediction_error_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        updates += 1

                next_state, reward, done, _, _ = env.step(action)  # Step
                    
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                mask = 1 if episode_steps == max_episode_steps else float(not done)
                
                memory.store_transition(state, action, reward, next_state, mask)  # Append transition to memory

                state = next_state

            writer.add_scalar('reward/train', episode_reward, i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                        episode_steps,
                                                                                        round(episode_reward, 2)))
            if i_episode % 10 == 0:
                self.save_checkpoint(env_name=env_name)


    def test(self, env, episodes=10, max_episode_steps=500):

        for i_episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, _ = env.reset()

            while not done and episode_steps < max_episode_steps:

                action = self.select_action(state)  # Sample action from policy
                # print(f"Action: {action}")

                next_state, reward, done, _, _ = env.step(action)  # Step
                # print(next_state.shape)
                episode_steps += 1

                if reward == 1:
                    done = True
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == max_episode_steps else float(not done)

                state = next_state

            print("Episode: {}, episode steps: {}, reward: {}".format(i_episode,
                                                                                        episode_steps,
                                                                                        round(episode_reward, 2)))

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        self.predictive_model.save_checkpoint()
    # Save model parameters
    def save_checkpoint(self, env_name, suffix=""):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print('Saving models')
        self.policy.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()


    # Load model parameters
    def load_checkpoint(self, evaluate=False):

        try:
            print('Loading models...')
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            print('Successfully loaded models')
        except:
            if evaluate:
                raise Exception("Unable to load models. Can't evaluate model")
            else:
                print("Unable to load models. Starting from scratch")

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()



