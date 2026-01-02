import os

import numpy as np
import tensorflow as tf

from utils import ReplayMemory

INPUT_HEIGHT = 84     
INPUT_WIDTH = 84     
CHANNELS = 4       
N_OUTPUTS = 4  # Number of possible actions that the agent can make (the four directions)


# To be more robust to outliers, we use a quadratic loss for small errors, and a linear loss for large ones.
def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class CNNModel(tf.keras.Model):
    def __init__(self, name):
        super(CNNModel, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu', padding='same', name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu', padding='same', name='conv2')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(), name='fc1')
        self.logits = tf.keras.layers.Dense(N_OUTPUTS, kernel_initializer=tf.keras.initializers.VarianceScaling(), name='logits')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.logits(x)


class ActorCritic:
    def __init__(self, training_steps=5000000, learning_rate=0.0001, momentum=0.95,
                 memory_size=100000, discount_rate=0.95, eps_min=0.05):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.memory_size = memory_size
        self.memory = ReplayMemory(self.memory_size)
        self.discount_rate = discount_rate
        self.eps_min = eps_min
        self.eps_decay_steps = int(training_steps / 2)

        self.online_model = CNNModel('online')
        self.target_model = CNNModel('target')
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=True)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')

        # Build models
        dummy_input = tf.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, CHANNELS))
        self.online_model(dummy_input)
        self.target_model(dummy_input)

    def start(self, checkpoint_path):
        """
        Initialize the model or restore the model if it already exists.
        
        :return: Iteration that we want the model to start training
        """
        if os.path.isfile(checkpoint_path):
            # Load weights
            self.online_model.load_weights(checkpoint_path)
            self.target_model.load_weights(checkpoint_path)
            training_start = 1  
            print('Restoring model...')
        else:
            # Make the model warm up before training
            training_start = 10000  
            self.copy_online_to_target()
            print('New model...')
        return training_start

    @tf.function
    def train_step(self, states, actions, targets):
        with tf.GradientTape() as tape:
            q_values = self.online_model(states)
            q_value = tf.reduce_sum(q_values * tf.one_hot(actions, N_OUTPUTS), axis=1, keepdims=True)
            error = tf.abs(targets - q_value)
            loss = tf.reduce_mean(clipped_error(error))
        grads = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_model.trainable_variables))
        self.global_step.assign_add(1)
        return loss

    def train(self, checkpoint_path, file_writer, mean_score):
        """
        Trains the agent and writes regularly a training summary.

        :param checkpoint_path: The path where the model will be saved
        :param file_writer: The file where the training summary will be written for Tensorboard visualization
        :param mean_score: The mean game score
        """
        copy_steps = 5000  
        save_steps = 2000   
        summary_steps = 500 

        cur_states, actions, rewards, next_states, dones = self.sample_memories()

        next_q_values = self.target_model(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1, keepdims=True)
        y_vals = rewards + (1 - dones) * self.discount_rate * max_next_q_values

        loss_val = self.train_step(cur_states, actions, y_vals)

        step = self.global_step.numpy()

        # Regularly copy the online DQN to the target DQN
        if step % copy_steps == 0:
            self.copy_online_to_target()

        # Save the model regularly
        if step % save_steps == 0:
            self.online_model.save_weights(checkpoint_path)

        # Write the training summary regularly
        if step % summary_steps == 0:
            with file_writer.as_default():
                tf.summary.scalar('loss', loss_val, step=step)
                tf.summary.scalar('mean score', mean_score, step=step)

    def predict(self, cur_state):
        """
        Makes the actor predict q-values based on the current state of the game.
        
        :param cur_state: Current state of the game
        :return The Q-values predicted by the actor
        """
        q_values = self.online_model(tf.expand_dims(cur_state, 0))
        return q_values.numpy()

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def act(self, cur_state, step):
        """
        :param cur_state: Current state of the game
        :param step: Training step
        :return: Action selected by the agent
        """
        q_values = self.predict(cur_state)
        eps_max = 1.0
        epsilon = max(self.eps_min, eps_max - (eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(N_OUTPUTS)  # Random action
        else:
            return np.argmax(q_values)  # Optimal action

    def copy_online_to_target(self):
        """
        Copies the weights from online to target model.
        """
        self.target_model.set_weights(self.online_model.get_weights())

    def sample_memories(self, batch_size=32):
        """
        Extracts memories from the agent's memory.
        
        :param batch_size: Size of the batch that we extract from the memory
        :return: State, action, reward, next_state, and done values as tensors
        """
        cols = [[], [], [], [], []]  # state, action, reward, next_state, done
        for memory in self.memory.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        
        states = np.array(cols[0], dtype=np.float32)
        actions = np.array(cols[1], dtype=np.int32)
        rewards = np.array(cols[2], dtype=np.float32).reshape(-1, 1)
        next_states = np.array(cols[3], dtype=np.float32)
        dones = np.array(cols[4], dtype=np.float32).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones