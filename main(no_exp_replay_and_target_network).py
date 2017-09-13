import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from simulator_pendulum import pendulum_simulator
from agent_nn_model import FFNN3_Q_agent


# control log of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

dim_state, n_action, n_valid_state, valid_episode_max_step = 2, 17, 16, 200

env = pendulum_simulator(n_state=dim_state, n_action=n_action, n_validation_state=n_valid_state, env_para=[0.02, 3.0, 1.5, 1.5, 9.82])

save_period = 10

lr, gamma, epsilon_0 = 0.0002, 0.95, 0.15	#learning_rate, reward_discount, epsilon_greedy_prob
num_episodes = 30000
#num_episodes = 100
num_hidden = 32

agent = FFNN3_Q_agent(n_state=dim_state, n_action=n_action, n_hidden=num_hidden, reward_discount=gamma, lr=lr)

weights = tf.trainable_variables()
print "# of trainable variables : ", len(weights)

init = tf.initialize_all_variables()

validation_step_cnt_hist, validation_avg_reward_hist = [], []
with tf.Session(config=config) as sess:
    sess.run(init)
    episode_cnt, epsilon = 0, epsilon_0
    while episode_cnt < num_episodes:
        state = np.array([np.pi*(np.random.rand(1)[0]-0.5), np.random.rand(1)[0]*2.2-1.1])
        episode_end_condition = False
        while not episode_end_condition:
            # action selection
            if np.random.rand(1) < epsilon:
                action = np.random.randint(agent.num_actions)
            else:
                action = sess.run(agent.chosen_action, feed_dict={agent.state_in:state})

            # new state and reward
            new_state = env.dynamics(state, action)
            reward, _ = env.reward(state)

            # target Q
            currQ = sess.run(agent.Qout, feed_dict={agent.state_in:state})
            nextQ_tmp = sess.run(agent.Qout, feed_dict={agent.state_in:new_state})
            nextQ = reward + gamma*np.max(nextQ_tmp)
            targetQ = np.array(currQ)
            targetQ[action] = nextQ
            #print "\n\naction :", action, "reward :", reward, "nextQ : ", nextQ
            #print "current :", currQ
            #print "nextQ tmp :", nextQ_tmp
            #print "target :", targetQ

            # train the network
            #sess.run(agent.update, feed_dict={agent.state_in:state, agent.nextQ_holder:targetQ})
            sess.run(agent.update, feed_dict={agent.state_in:state, agent.nextQ_holder:targetQ, agent.epoch:episode_cnt})

            #_, ww = sess.run([agent.update, weights], feed_dict={agent.state_in:state, agent.nextQ_holder:targetQ})
            #print "\n\nParameters of model"
            #for i in range(len(ww)):
            #    print ww[i]

            # update environment (state)
            _, episode_end_condition = env.reward(new_state)
            state = new_state

        ### End of training of one episode
        episode_cnt = episode_cnt+1
        epsilon = epsilon_0/(1.0 + (episode_cnt/50.0) )

        ### Validation Test
        if episode_cnt % save_period == 0:
        #if episode_cnt % (num_episodes //20) == 0:
        #if episode_cnt % (num_episodes //100) == 0:
            valid_step_cnt_list, valid_total_return_list = [], []
            for valid_epi_cnt in range(n_valid_state):
                state = np.array(env.validation_state[valid_epi_cnt])
                valid_epi_end_cond, valid_epi_step_cnt, valid_epi_return = False, 0, 0.0
                while (not valid_epi_end_cond) and (valid_epi_step_cnt <= valid_episode_max_step):
                    action = sess.run(agent.chosen_action, feed_dict={agent.state_in:state})
                    new_state = env.dynamics(state, action)
                    reward, _ = env.reward(state)
                    _, valid_epi_end_cond = env.reward(new_state)

                    valid_epi_step_cnt, valid_epi_return = valid_epi_step_cnt + 1, valid_epi_return + reward
                    state = new_state
                valid_step_cnt_list.append(valid_epi_step_cnt)
                valid_total_return_list.append(valid_epi_return)
            valid_epi_mean_step_cnt, valid_epi_mean_return = np.mean(valid_step_cnt_list), np.sum(valid_total_return_list)/np.sum(valid_step_cnt_list)
            print "Validation result - epoch %d - Avg. step : %f, Avg. return : %f" %(episode_cnt, valid_epi_mean_step_cnt, valid_epi_mean_return)
            #print valid_step_cnt_list
            #print valid_total_return_list
            print "\n"

            validation_step_cnt_hist.append(valid_epi_mean_step_cnt)
            validation_avg_reward_hist.append(valid_epi_mean_return)


### Plot History of Validation Test's Result
plot_x = np.arange(1, len(validation_step_cnt_hist)+1) * save_period
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(plot_x, np.array(validation_step_cnt_hist))
axarr[0].set_title('History of Average Number of Steps')
axarr[1].scatter(plot_x, np.array(validation_avg_reward_hist))
axarr[1].set_title('History of Average Return')

plt.show()
