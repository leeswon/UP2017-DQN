import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as pat

# Sample code how to use this library (dynamics with Runge-Kutta Numerical Estimation and drawer)
'''
import numpy as np
from simulator_pendulum import pendulum_simulator

pend_sim = pendulum_simulator(n_state=2, n_action=11, n_validation_state=16, env_para=[0.02, 3.0, 1.5, 1.5, 9.82])
print pend_sim.validation_state

state = np.array([0.2*np.pi, 0.0], dtype=np.float32)
for T in range(1, 1000):
    state = pend_sim.dynamics(state)
    print "reward", pend_sim.reward(state)
    pend_sim.simulator_draw(state)
'''

def angle_limit_transform(angle, limit_list):
    ### limit_list = [min_angle, max_angle]
    if angle < limit_list[0]:
        angle = angle_limit_transform(angle+2.0*np.pi, limit_list)
    elif angle >= limit_list[1]:
        angle = angle_limit_transform(angle-2.0*np.pi, limit_list)
    return angle

# env_para : [time_step, pend_length, pend_mass, pend_damper, g]
# n_validation_state : must be multiples of 4
# n_action : must be odd integer
class pendulum_simulator():
    def __init__(self, n_state, n_action, n_validation_state, env_para):
        self.num_states = n_state	# dimension of state vector
        self.num_actions = n_action
        self.para = env_para

        # initial states for validation
        self.validation_state = []
        self.num_validation_states = n_validation_state
        angle_tmp = np.concatenate((np.linspace(0.1*np.pi, 0.6*np.pi, num=self.num_validation_states//4), np.linspace(1.4*np.pi, 1.9*np.pi, num=self.num_validation_states//4)))
        np.random.shuffle(angle_tmp)
        for cnt in range(len(angle_tmp)):
            self.validation_state.append([angle_tmp[cnt], (np.random.rand(1)[0]-0.5)])
        np.random.shuffle(angle_tmp)
        for cnt in range(len(angle_tmp)):
            self.validation_state.append([angle_tmp[cnt], (np.random.rand(1)[0]-0.5)])
        del angle_tmp

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def train_reward(self, state):
        if len(state.shape) < 2:   # vector case
            if np.cos(state[0]) > 0.975:
                return (1.0, True)
            elif np.cos(state[0]) < -0.975:
                return (-1.0, True)
            else:
                return (-0.001, False)
        elif len(state.shape) < 3:   # array case
            reward_vec = np.cos(state[:, 0])
            return (reward_vec, np.logical_or( (reward_vec>0.975), (reward_vec<-0.975) ))

    def reward(self, state):
        if len(state.shape) < 2:   # vector case
            return (np.cos(state[0]), np.cos(state[0])<-0.95)
        elif len(state.shape) < 3:   # array case
            return (np.cos(state[:, 0]), np.cos(state[:, 0])<-0.95)

    def action_to_control(self, action):
        max_control = self.para[1]*self.para[2]*self.para[4]
        delta_control = 5.0*max_control/(self.num_actions-1)
        return max_control-action*delta_control

    def dynamics(self, state, action=-1):
        if action < 0:
            control_torque = 0.0
        else:
            control_torque = self.action_to_control(action)

        k1 = np.array([state[1], 1.5*self.para[4]*np.sin(state[0])/self.para[1]+3.0*(control_torque-self.para[3]*state[1])/(self.para[2]*(self.para[1]**2))], dtype=np.float32)
        state1 = state + 0.5*self.para[0]*k1
        k2 = np.array([state1[1], 1.5*self.para[4]*np.sin(state1[0])/self.para[1]+3.0*(control_torque-self.para[3]*state1[1])/(self.para[2]*(self.para[1]**2))], dtype=np.float32)
        state2 = state + 0.5*self.para[0]*k2
        k3 = np.array([state2[1], 1.5*self.para[4]*np.sin(state2[0])/self.para[1]+3.0*(control_torque-self.para[3]*state2[1])/(self.para[2]*(self.para[1]**2))], dtype=np.float32)
        state3 = state + self.para[0]*k3
        k4 = np.array([state3[1], 1.5*self.para[4]*np.sin(state3[0])/self.para[1]+3.0*(control_torque-self.para[3]*state3[1])/(self.para[2]*(self.para[1]**2))], dtype=np.float32)
        state = state + (self.para[0]/6.0)*(k1 + k4 + 2.0*(k2+k3))
        state[0] = angle_limit_transform(state[0], [-np.pi, np.pi])
        return state

    def simulator_draw(self, state):
        r1 = pat.Rectangle((-0.1, 0), 0.2, self.para[1], color="red")
        t1 = mpl.transforms.Affine2D().rotate_deg(state[0]*180.0/np.pi)+self.ax.transData
        r1.set_transform(t1)

        self.ax.clear()
        self.ax.add_patch(r1)
        plt.xlim(-1.5*self.para[1], 1.5*self.para[1])
        plt.ylim(-1.5*self.para[1], 1.5*self.para[1])
        plt.grid(True)

        plt.pause(self.para[0])
