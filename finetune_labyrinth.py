import agents,modules,replay_buffers,explorers
import torch
import os
import time
import gameEnvironment
from tqdm import tqdm
import matplotlib.pyplot as plt

class TrainManager():

    def __init__(self,
                 env,  # environment
                 episodes=1000,  # number of episodes
                 batch_size=64,  # batch size
                 num_steps=4,  # frequency of learning
                 memory_size = 2000,  # replay buffer capacity
                 replay_start_size = 200,  # number of steps before replay starts
                 actor_lr=0.001,  # value function learning rate
                 critic_lr=0.001,  # policy function learning rate
                 update_target_steps=200,
                 gamma=0.9,  # discount rate for rewards
                 e_greed=0.3 , # exploration probability in exploration vs exploitation
                 decay_rate=0.001, # exploration probability decay rate
                 num_decay=10, # exploration probability decay frequency
                 factor=0.8, # factor by which learning rate is reduced
                 patience=500, # number of epochs to wait before reducing learning rate after metric stops improving
                 threshold=1e-6, # threshold for determining if metric has stopped improving
                 min_lr=1e-6, # minimum learning rate
                 pretrained_checkpoint=None  # NEW: Path to expanded checkpoint for transfer learning
                 ):
        self.env = env
        self.episodes = episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_act = env.n_act
        n_obs = env.n_obs
        
        # Create models
        actor = modules.Actor(obs_size=n_obs,n_act=n_act)
        critic = modules.Critic(obs_size=n_obs,n_act=n_act)
        
        # Load pretrained weights if provided
        if pretrained_checkpoint is not None:
            print(f"Loading pretrained weights from: {pretrained_checkpoint}")
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            actor.load_state_dict(checkpoint['actor'])
            critic.load_state_dict(checkpoint['critic'])
            print("Pretrained weights loaded successfully!")
            print("Starting fine-tuning with labyrinth awareness...")
        
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

    # train one game episode
    def train_episode(self,episodes):
        total_reward = 0
        obs = self.env.reset()
        num = 0
        while True:
            action = self.agent.act(obs,episodes)
            next_obs, reward, done,length = self.env.step(action,episodes)
            total_reward += reward
            next_obs = torch.tensor(next_obs).to(self.device)
            reward = torch.tensor(reward).to(self.device)
            done = torch.tensor(done).to(self.device)
            self.agent.learn(obs, action, reward, next_obs, done,num)
            obs = next_obs
            num += 1
            if done: break
        return total_reward,length,num

    # test one game episode
    def test_episode(self,episodes):
        total_reward = 0
        num = 0
        obs = self.env.reset()
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, length = self.env.step(action,episodes)
            total_reward += reward
            obs = torch.tensor(next_obs).to(self.device)
            obs = next_obs
            self.env.render(True,episodes)
            num += 1
            if done: break
        return total_reward,length,num

    def train(self):
        # 1. enable matplotlib interactive mode
        plt.ion()
        # 2. initialize figure
        fig, ax = plt.subplots()
        rewards = []
        line, = ax.plot(rewards)
        ax.set_title('Fine-tuning Reward Over Time (Labyrinth)')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        split_num = int(self.episodes / 1000)
        e = 0
        t = time.time()
        sum_avr_length = 0
        for i in range(split_num):
            with tqdm(total=int(self.episodes / split_num), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.episodes / split_num)):
                    ep_reward, length, num = self.train_episode(episodes=e)
                    sum_avr_length += length
                    if e % 50 == 0:
                        rewards.append(ep_reward)
                        if len(rewards) > 1000:
                            rewards.pop(0)
                        line.set_ydata(rewards)
                        line.set_xdata(list(range(len(rewards))))
                        ax.relim()
                        ax.autoscale_view(True, True, True)
                        plt.draw()
                        plt.pause(0.001)
                    if e % 1000 == 0:
                        # Save both actor and critic in a single checkpoint
                        checkpoint = {
                            'actor': self.agent.actor_pred_func.state_dict(),
                            'critic': self.agent.critic_pred_func.state_dict()
                        }
                        file_path = "./model4/finetuned_model_" + str(e) + ".pth"
                        os.makedirs("./model4", exist_ok=True)
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

        # training finished: save reward plot
        try:
            plt.ioff()
            os.makedirs('results', exist_ok=True)
            fname = os.path.join('results', 'finetuning_reward_{}.png'.format(int(time.time())))
            fig.savefig(fname)
            print('Saved training plot to', fname)
        except Exception as ex:
            print('Failed to save training plot:', ex)


if __name__ == '__main__':
    env1 = gameEnvironment.game()
    
    # Expand checkpoint
    # from expand_model import save_expanded_checkpoint
    # save_expanded_checkpoint("./model/actor/trained_model_25000.pth", "./model2/expanded_model.pth")
    
    # Step 2: Fine-tune with expanded checkpoint
    tm = TrainManager(
        env1,
        update_target_steps=200,
        episodes=10000,  # Fewer episodes for fine-tuning
        actor_lr=3e-3,    # Increased learning rate to overcome old habits
        critic_lr=1e-2,   # Increased learning rate
        batch_size=512,
        memory_size=5000,
        replay_start_size=1024,
        decay_rate=0.00001,  # Slower decay to maintain exploration longer
        e_greed=0.50,     # MUCH higher exploration - force it to try new paths
        num_decay=20,
        gamma=0.9,
        num_steps=20,
        min_lr=1e-8,
        factor=0.95,
        patience=200,
        threshold=1e-7,
        pretrained_checkpoint="./model3/finetuned_model_9000.pth"  # Load expanded checkpoint
    )
    tm.train()
