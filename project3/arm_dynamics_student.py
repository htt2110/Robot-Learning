from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from train_dynamics import Net


class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, device):
        # ---
        # Your code hoes here
        # Initialize the model loading the saved model from provided model_path
        self.model = Net(9)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        # ---
        self.model_loaded = True

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Your code goes here
            # Use the loaded model to predict new state given the current state and action
            
            x = np.append(state.reshape(1,6), action.reshape(1,3))
            x = torch.Tensor(x)
            new_state = self.model(x).detach().numpy().reshape(6,1)
            return new_state
            # ---
        else:
            return state
