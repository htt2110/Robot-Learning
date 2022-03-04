import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os
import time
np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.
    
    t = 0
    

    X = np.zeros((9, 1))
    Y = np.zeros((6, 1)) 
    current_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  # position + velocity
    current_state[0] = -np.pi / 2.0
    #print(current_state)
    torque = np.linspace(-1.75, 1.75, 1000)
    
    for i in range(0,1000):
        j = np.random.randint(1000, size = 1000)
        action = [torque[j[i]], 0, 0]
        arm_teacher.set_t(0)
        arm_teacher.set_state(current_state)
        arm_teacher.set_action(action)
        X_new = np.zeros((9,1))
        Y_new = np.zeros((6,1))
        while arm_teacher.get_t() < 5:
            X_new = np.append(X_new, np.append(arm_teacher.get_state(), action).reshape(9,1), axis=1)
            arm_teacher.advance()
            Y_new = np.append(Y_new, arm_teacher.get_state(), axis=1)
        X_new = np.delete(X_new, 0, 1)
        Y_new = np.delete(Y_new, 0, 1)       
        X = np.hstack((X, X_new))
        Y = np.hstack((Y, Y_new))
        X = np.delete(X, 0, 1)
        Y = np.delete(Y, 0, 1) 
        print(X.shape, Y.shape)
        #print(current_state)
    # ---

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
