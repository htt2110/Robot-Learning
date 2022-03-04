import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import time
import random
import os
from cartpole_env import make_cart_pole_env, configure_pybullet
from replay_buffer import ReplayBuffer
from q_network import QNetwork
import torch.nn.functional as F

random.seed(5)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--save_dir', type=str, default='models',
                        help="the root folder for saving the checkpoints")
    parser.add_argument('--gui', action='store_true', default=False,
                        help="whether to turn on GUI or not")
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed:
        args.seed = int(time.time())
    return args

  

def train_dqn(env, args, device):
    # set up seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # ---
    # Write your code here to train DQN
    
    #print(env.action_space.n)
    #print(env.observation_space.shape[0])
    batch_size = 150
    gamma = 0.999
    stp = 0.005
    epsilon = 1
    target_update = 5
    
    # Intializing Q_A & Q_T
    policy_net = QNetwork(env).to(device)
    target_net = QNetwork(env).to(device)
    
    # Setting Q_T = Q_A
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    #Initializing Replay Memory 
    memory = ReplayBuffer(1000)
    
    num_episodes = 500
    transition = [[]]
    
    for episode in range(num_episodes):
        
        episode_reward = 0
        state = env.reset() 

        for t in range(500):
            # Epsilon goes from 1 to 0.01 and remains 0.01 after 
            if epsilon > 0.01:
                epsilon -= stp
            else:
                epsilon = 0.01    
                
            #Sampling values between 0 & 1 i.e. probablity
            sample = random.random()  
            #print(sample)
            
            #Condition for getting actions 
            if sample > epsilon:
                with torch.no_grad():
                    logits = policy_net(torch.Tensor(state.reshape((1,) + state.shape)), device)
                    action = torch.argmax(logits, dim=1).tolist()[0]
                    #print(action)
            else:
                action = np.random.choice(range(env.action_dims))
                
            #Executing action a_t
            next_state, reward, done, _ = env.step(action)

            #Updating reward
            episode_reward += reward
            
            transition = [[state], [action], [reward], [next_state],[done]]

            #Storing transition values in the buffer
            memory.put(transition)
            
            #Training the DQN
            if len(memory.buffer) > batch_size:
                # Sampling the Minibatch from the Memory Buffer
                transitions = memory.sample(batch_size)
                state_batch = transitions[0].reshape(batch_size,4)
                action_batch = transitions[1].reshape(batch_size)
                reward_batch = transitions[2].reshape(batch_size)
                non_final_next_states = transitions[3].reshape(batch_size,4)
                done_batch = transitions[4].reshape(batch_size)
                
                neg_done = np.logical_not(done_batch)*1
                
                pred_action_values=policy_net(torch.Tensor(state_batch).float(),device)
                state_action_values=torch.tensor(action_batch).to(device)
                action_values = pred_action_values.gather(1,state_action_values.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    next_state_values = target_net(torch.Tensor(non_final_next_states).float(),device).detach().numpy()
                expected_state_action_values=(reward_batch+neg_done*gamma*np.amax(next_state_values,axis=1))
                loss=F.mse_loss(action_values,torch.Tensor(expected_state_action_values).float())
                
         
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                
            #Setting s_t = s_t+1
            state = next_state
            
            #Ending Epsidoe if S_t+1 terminal
            if done == True:
                break
        
        model_folder_name = f'episode_{episode:06d}_reward_{round(episode_reward):03d}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(policy_net.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')

        # Q_T = Q_A after every k episode
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())    

                                     
    # ---


if __name__ == "__main__":
    args = get_args()
    configure_pybullet(rendering=args.gui, debug=True)
    env = make_cart_pole_env()
    device = torch.device('cpu')

    train_dqn(env, args, device)
