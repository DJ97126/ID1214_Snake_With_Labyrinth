import agents,modules,replay_buffers,explorers
import torch
import os
import time
import gameEnvironment
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


class TrainManager():

    def __init__(self,
                 env,
                 episodes=1000,
                 batch_size=64,
                 num_steps=4,
                 memory_size = 2000,
                 replay_start_size = 200,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 update_target_steps=200,
                 gamma=0.9,
                 e_greed=0.3,
                 decay_rate=0.001,
                 num_decay=10,
                 factor=0.8,
                 patience=500,
                 threshold=1e-6,
                 min_lr=1e-6
                 ):
        self.env = env
        self.episodes = episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_act = env.n_act
        n_obs = env.n_obs
        
        # Create NEW models from scratch (no pretrained weights)
        actor = modules.Actor(obs_size=n_obs,n_act=n_act)
        critic = modules.Critic(obs_size=n_obs,n_act=n_act)
        
        print(f"\n{'='*60}")
        print(f"Training from SCRATCH with labyrinth (n_obs={n_obs})")
        print(f"{'='*60}\n")
        
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=actor_lr)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=critic_lr)
        rb = replay_buffers.ReplayBuffer(memory_size,num_steps)
        explorer = explorers.EpsilonGreedy(n_act,e_greed,decay_rate,num_decay)
        actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='min', factor=factor, patience=patience,threshold=threshold, min_lr=min_lr)
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='min', factor=factor, patience=patience,threshold=threshold, min_lr=min_lr)
        self.agent = agents.DQNAgent(
            actor=actor,
            actor_optimizer=actor_optimizer,
            actor_scheduler=actor_scheduler,
            critic=critic,
            critic_optimizer=critic_optimizer,
            critic_scheduler=critic_scheduler,
            replay_buffer = rb,
            batch_size=batch_size,
            replay_start_size = replay_start_size,
            n_act=n_act,
            gamma=gamma,
            update_target_steps=update_target_steps,
            explorer=explorer,
        )

    def train_episode(self,episodes):
        total_reward = 0
        obs = self.env.reset()
        num = 0
        while True:
            action = self.agent.act(obs,episodes)
            next_obs, reward, done,length = self.env.step(action)
            total_reward += reward
            next_obs = torch.tensor(next_obs).to(self.device)
            reward = torch.tensor(reward).to(self.device)
            done = torch.tensor(done).to(self.device)
            self.agent.learn(obs, action, reward, next_obs, done,num)
            obs = next_obs
            num += 1
            if done: break
        return total_reward,length,num

    def train(self):
        # Use non-blocking backend to prevent window from stealing focus
        matplotlib.use('TkAgg')  # Non-intrusive backend
        plt.ioff()  # Turn off interactive mode to prevent focus stealing
        
        # Larger figure for better readability
        fig, ax = plt.subplots(figsize=(14, 8))
        plt.show(block=False)  # Show once without blocking
        
        rewards = []
        line, = ax.plot(rewards, linewidth=2)
        ax.set_title('Training from Scratch - Labyrinth', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episodes (window of last 1000)', fontsize=14)
        ax.set_ylabel('Episode Reward', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        split_num = int(self.episodes / 1000)
        e = 0
        t = time.time()
        sum_avr_length = 0
        
        # Curriculum learning: 4 stages × 25,000 episodes = 100,000 total
        level_thresholds = [25000, 50000, 75000]  # Transition episodes
        current_level = 0
        
        for i in range(split_num):
            with tqdm(total=int(self.episodes / split_num), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.episodes / split_num)):
                    ep_reward, length, num = self.train_episode(episodes=e)
                    sum_avr_length += length
                    
                    # Curriculum learning: increase difficulty at milestones
                    if current_level < len(level_thresholds) and e == level_thresholds[current_level]:
                        current_level += 1
                        self.env.set_difficulty_level(current_level)
                    
                    # Update plot less frequently to avoid interruptions (every 200 episodes)
                    if e % 200 == 0:
                        rewards.append(ep_reward)
                        if len(rewards) > 1000:
                            rewards.pop(0)
                        line.set_ydata(rewards)
                        line.set_xdata(list(range(len(rewards))))
                        ax.relim()
                        ax.autoscale_view(True, True, True)
                        fig.canvas.draw()
                        fig.canvas.flush_events()  # Non-blocking update
                    if e % 1000 == 0:
                        checkpoint = {
                            'actor': self.agent.actor_pred_func.state_dict(),
                            'critic': self.agent.critic_pred_func.state_dict()
                        }
                        file_path = "./model_labyrinth/trained_model_" + str(e) + ".pth"
                        os.makedirs("./model_labyrinth", exist_ok=True)
                        torch.save(checkpoint, file_path)
                        print(f"\nCheckpoint saved: {file_path}")
                    if e % 50 == 0:
                        avr_length = sum_avr_length / 50
                        sum_avr_length = 0
                        actor_lr = self.agent.actor_optimizer.param_groups[0]['lr']
                        critic_lr = self.agent.critic_optimizer.param_groups[0]['lr']
                        pbar.set_postfix({'episode': '%d' % e, 'reward': '%.3f' % ep_reward,'length': '%.6f' % avr_length, 'step_num': '%.3f' % num,'actor_lr': '%.8f' % actor_lr,'critic_lr':'%.8f' %critic_lr,'e_greed':'%.3f' %self.agent.explorer.epsilon,'time': '%.3f' % (time.time() - t)})
                        t = time.time()
                    e += 1
                    pbar.update(1)

        try:
            plt.ioff()
            os.makedirs('results', exist_ok=True)
            fname = os.path.join('results', 'scratch_training_{}.png'.format(int(time.time())))
            fig.savefig(fname)
            print('Saved training plot to', fname)
        except Exception as ex:
            print('Failed to save training plot:', ex)


if __name__ == '__main__':
    env1 = gameEnvironment.game()
    
    # Train from scratch with higher exploration for labyrinth
    tm = TrainManager(
        env1,
        update_target_steps=200,
        episodes=100000,     # More episodes - learning from zero
        actor_lr=6e-3,      # Higher learning rates
        critic_lr=3e-2,
        batch_size=512,
        memory_size=5000,
        replay_start_size=1024,
        decay_rate=0.00002,
        e_greed=0.40,       # Higher exploration
        num_decay=20,
        gamma=0.9,
        num_steps=20,
        min_lr=1e-8,
        factor=0.95,
        patience=200,
        threshold=1e-7
    )
    tm.train()
