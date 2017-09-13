import numpy as np
import tensorflow as tf

#### function to generate weight parameter
def new_placeholder(shape):
    return tf.placeholder(shape=shape, dtype=tf.float32)

#### function to generate bias parameter
def new_weight_or_bias(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

#### function to add fully-connected layer
def new_fc_layer(layer_input, input_dim, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None):
    if weight is None:
        weight = new_weight_or_bias(shape=[input_dim, output_dim])
    elif type(weight) is np.ndarray:
        weight = tf.constant(weight, dtype=tf.float32)

    if bias is None:
        bias = new_weight_or_bias(shape=[output_dim])
    elif type(bias) is np.ndarray:
        bias = tf.constant(bias, dtype=tf.float32)

    if activation_fn is None:
        layer = tf.matmul(layer_input, weight) + bias
    else:
        layer = activation_fn( tf.matmul(layer_input, weight) + bias )
    return layer, [weight, bias]

#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)



#### Q value-based agent
class FFNN3_Q_agent():
    def __init__(self, n_state, n_action, n_hidden, reward_discount, lr):
        self.num_states = n_state
        self.num_actions = n_action
        self.num_hidden_units = n_hidden
        self.reward_discount = reward_discount
        self.learn_rate = lr

        # placeholders for the agent
        self.state_in= tf.placeholder(shape=[self.num_states], dtype=tf.float32)
        self.nextQ_holder = tf.placeholder(shape=[self.num_actions],dtype=tf.float32)
        self.epoch = tf.placeholder(dtype=tf.float32)
        #self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        #self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

        # feed-forward network
        hidden_output = tf.contrib.layers.fully_connected([self.state_in], n_hidden, biases_initializer=tf.random_uniform_initializer(), activation_fn=tf.nn.relu, weights_initializer=tf.random_uniform_initializer())
        output = tf.contrib.layers.fully_connected(hidden_output, self.num_actions, biases_initializer=tf.random_uniform_initializer(), activation_fn=None, weights_initializer=tf.random_uniform_initializer())
        self.Qout = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.Qout, 0)

        # training proceedure
        self.loss = tf.reduce_sum(tf.square(self.nextQ_holder - self.Qout))
        #self.update = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/50.0)).minimize(self.loss)


#### Q value-based Feedforward Neural Net agent
class FFNN_Q_batch():
    def __init__(self, dim_layers, reward_discount, learning_rate, param_list=None):
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.num_states = dim_layers[0]
        self.num_actions = dim_layers[-1]
        self.learn_rate = learning_rate

        #### placeholder of model
        self.state_in = new_placeholder([None, self.layers_size[0]])
        self.nextQ_holder = new_placeholder([None, self.layers_size[-1]])
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model
        self.layers, self.param = [], []
        for cnt in range(self.num_layers):
            if cnt == 0 and param_list is None:
                layer_tmp, para_tmp = new_fc_layer(self.state_in, self.layers_size[0], self.layers_size[1])
            elif cnt == 0:
                layer_tmp, para_tmp = new_fc_layer(self.state_in, self.layers_size[0], self.layers_size[1], weight=param_list[2*cnt], bias=param_list[2*cnt+1])
            elif cnt == self.num_layers-1 and param_list is None:
                layer_tmp, para_tmp = new_fc_layer(self.layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], activation_fn=None)
            elif cnt == self.num_layers-1:
                layer_tmp, para_tmp = new_fc_layer(self.layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], activation_fn=None, weight=param_list[2*cnt], bias=param_list[2*cnt+1])
            elif param_list is None:
                layer_tmp, para_tmp = new_fc_layer(self.layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1])
            else:
                layer_tmp, para_tmp = new_fc_layer(self.layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], weight=param_list[2*cnt], bias=param_list[2*cnt+1])
            self.layers.append(layer_tmp)
            #if param_list is None:
            self.param = self.param + para_tmp

        #### functions of model
        self.Qout = self.layers[-1]
        self.chosen_action = tf.argmax(self.Qout, axis=1)

        if param_list is None:
            self.loss = tf.reduce_sum(tf.square(self.nextQ_holder - self.Qout))

            #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
            self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/100.0)).minimize(self.loss)
