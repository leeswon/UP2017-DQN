import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from simulator_pendulum import pendulum_simulator
import agent_nn_model


### control log of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

### set-up environment of RL problem
dim_state, n_action, n_valid_state, valid_episode_max_step = 2, 17, 16, 300

env = pendulum_simulator(n_state=dim_state, n_action=n_action, n_validation_state=n_valid_state, env_para=[0.02, 3.0, 1.5, 1.5, 9.82])

### hyper-parameters of agent/training procedure
lr, gamma, epsilon_0 = 0.01, 0.95, 0.3	#learning_rate, reward_discount, epsilon_greedy
num_hidden = 32
batch_size = 16

max_epochs = 30000
test_period, target_update_period, min_number_of_exp = 10, 20, 30
train_cnt, train_conti_cond, epsilon = 0, True, epsilon_0
experience_storage = []
validation_step_cnt_hist, validation_avg_reward_hist = [], []

if batch_size > min_number_of_exp:
    print "Set size of mini-batch smaller than or equal to minimum size of experience storage!"
    raise

### choose agent
dim_layers = [dim_state, num_hidden, n_action]
#agent = agent_nn_model.FFNN3_Q_agent(n_state=dim_state, n_action=n_action, n_hidden=num_hidden, reward_discount=gamma, lr=lr)
agent = agent_nn_model.FFNN_Q_batch(dim_layers=dim_layers, reward_discount=gamma, learning_rate=lr)

init = tf.initialize_all_variables()
with tf.Session(config=config) as sess:
    sess.run(init)

    #### construct target Q network
    curr_param = sess.run(agent.param)
    target_agent = agent_nn_model.FFNN_Q_batch(dim_layers=dim_layers, reward_discount=gamma, learning_rate=lr, param_list=curr_param)

    while train_cnt < max_epochs and train_conti_cond:
        state = np.array([np.pi*(np.random.rand(1)[0]-0.5), np.random.rand(1)[0]*2.2-1.1])
        episode_end_condition = False
        while not episode_end_condition:
            #### action selection
            if np.random.rand(1) < epsilon:
                action = np.random.randint(agent.num_actions)
            else:
                action = sess.run(agent.chosen_action, feed_dict={agent.state_in:np.array([state])})[0]

            #### new state and reward
            new_state = env.dynamics(state, action)
            reward, _ = env.train_reward(state)

            #### update environment (state)
            _, episode_end_condition = env.train_reward(new_state)
            episode_end_condition = episode_end_condition or (train_cnt >= max_epochs)
            state = new_state

            #### save experience
            case_summary = [state, action, reward, new_state]
            experience_storage.append(case_summary)


            #### training agent in this case
            if len(experience_storage) >= min_number_of_exp:
                #### sampling training cases
                states_for_train, new_states_for_train = np.zeros(shape=(batch_size, dim_state), dtype=np.float32), np.zeros(shape=(batch_size, dim_state), dtype=np.float32)
                action_for_train, reward_for_train = np.zeros(shape=(batch_size), dtype=np.int32), np.zeros(shape=(batch_size), dtype=np.float32)

                indices = np.arange(len(experience_storage))
                np.random.shuffle(indices)

                for case_cnt in range(batch_size):
                    states_for_train[case_cnt,:] = experience_storage[indices[case_cnt]][0]
                    action_for_train[case_cnt] = experience_storage[indices[case_cnt]][1]
                    reward_for_train[case_cnt] = experience_storage[indices[case_cnt]][2]
                    new_states_for_train[case_cnt,:] = experience_storage[indices[case_cnt]][3]

                targetQ = sess.run(agent.Qout, feed_dict={agent.state_in:states_for_train})
                nextQ_tmp = sess.run(target_agent.Qout, feed_dict={target_agent.state_in:new_states_for_train})
                nextQ = reward_for_train + gamma*np.max(nextQ_tmp, axis=1)

                for case_cnt in range(batch_size):
                    targetQ[case_cnt, action_for_train[case_cnt]] = nextQ[case_cnt]

                #### update model
                sess.run(agent.update, feed_dict={agent.state_in:states_for_train, agent.nextQ_holder:targetQ, agent.epoch:train_cnt})

                #### increase train_cnt / decrease epsilon
                train_cnt = train_cnt+1
                epsilon = epsilon_0/(1.0 + (train_cnt/200.0) )

            #### update target Q network
            if train_cnt % target_update_period == 0 and train_cnt > 0:
                curr_param = sess.run(agent.param)
                target_agent = agent_nn_model.FFNN_Q_batch(dim_layers=dim_layers, reward_discount=gamma, learning_rate=lr, param_list=curr_param)


            #### Validation Test
            if train_cnt % test_period == 0 and train_cnt > 0:
                valid_step_cnt_list, valid_total_return_list = [], []
                for valid_epi_cnt in range(n_valid_state):
                    valid_state = np.array(env.validation_state[valid_epi_cnt])
                    valid_epi_end_cond, valid_epi_step_cnt, valid_epi_return = False, 0, 0.0

                    while (not valid_epi_end_cond) and (valid_epi_step_cnt <= valid_episode_max_step):
                        action = sess.run(agent.chosen_action, feed_dict={agent.state_in:np.array([valid_state])})[0]
                        new_valid_state = env.dynamics(valid_state, action)
                        reward, _ = env.reward(valid_state)
                        _, valid_epi_end_cond = env.reward(new_valid_state)

                        valid_epi_step_cnt, valid_epi_return = valid_epi_step_cnt + 1, valid_epi_return + reward
                        valid_state = new_valid_state
                    valid_step_cnt_list.append(valid_epi_step_cnt)
                    valid_total_return_list.append(valid_epi_return)
                valid_epi_mean_step_cnt, valid_epi_mean_return = np.mean(valid_step_cnt_list), np.sum(valid_total_return_list)/np.sum(valid_step_cnt_list)
                print "Validation result - epoch %d - Avg. step : %f, Avg. return : %f" %(train_cnt, valid_epi_mean_step_cnt, valid_epi_mean_return)
                #print valid_step_cnt_list
                #print valid_total_return_list
                #print "\n"

                validation_step_cnt_hist.append(valid_epi_mean_step_cnt)
                validation_avg_reward_hist.append(valid_epi_mean_return)


### Plot History of Validation Test's Result
plot_x = np.arange(1, len(validation_step_cnt_hist)+1) * test_period
x_lim = [0, (len(validation_step_cnt_hist)+1)*test_period]
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(plot_x, np.array(validation_step_cnt_hist))
axarr[0].set_title('History of Average Number of Steps')
axarr[0].set_xlim(x_lim)
axarr[1].plot(plot_x, np.array(validation_avg_reward_hist))
axarr[1].set_title('History of Average Return')
axarr[1].set_xlim(x_lim)

plt.show()
