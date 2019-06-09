import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.core.framework import summary_pb2
import os

from config import Config
from model_base import model

class DQN(model):
	"""
	Implement Neural Network with Tensorflow
	"""
	def add_placeholders_op(self):
		state_length = self._config.state_length
		state_history = self._config.state_history

		# self.state: batch of states, type = float32
		# self.a: batch of actions, type = int32
		# self.r: batch of rewards, type = float32
		# self.state_p: batch of next states, type = float32
		# self.lr: learning rate, type = float32

		self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_length, state_history])
		self.a = tf.placeholder(dtype=tf.int32, shape=[None])
		self.r = tf.placeholder(dtype=tf.float32, shape=[None])
		self.state_p = tf.placeholder(dtype=tf.float32, shape=[None, state_length, state_history])
		self.lr = tf.placeholder(dtype=tf.float32, shape=[])

	def get_q_values_op(self, state, scope, reuse=False):
		num_actions = self._config.numActions
		with tf.variable_scope(scope, reuse=reuse):
			state_flattened = layers.flatten(state)
			l1 = layers.fully_connected(state_flattened, self._config.hidden_size[0], activation_fn=None)
			l1 = layers.batch_norm(l1)
			l1 = tf.nn.relu(l1)
			l1 = tf.nn.dropout(l1, self._config.dropout)
			
			l2 = layers.fully_connected(l1, self._config.hidden_size[1], activation_fn=None)
			l2 = layers.batch_norm(l2)
			l2 = tf.nn.relu(l2)
			l2 = tf.nn.dropout(l2, self._config.dropout)

			l3 = layers.fully_connected(l2, self._config.hidden_size[2], activation_fn=None)
			l3 = layers.batch_norm(l3)
			l3 = tf.nn.relu(l3)
			l3 = tf.nn.dropout(l3, self._config.dropout)

			out = layers.fully_connected(l3, num_actions, activation_fn=None)
		return out

	def add_update_target_op(self, q_scope, target_q_scope):
		q_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
		target_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
		self.update_target_op = tf.group(*[tf.assign(var1, var2) for var1, var2 in zip(target_var, q_var)])

	def add_loss_op(self, q, target_q):
		num_actions = self._config.numActions
		Q_samp = self.r + self._config.gamma * tf.reduce_max(target_q, axis=1)
		Q_s_a = tf.reduce_sum(tf.one_hot(self.a, num_actions) * q, axis=1)
		self.loss = tf.reduce_mean(tf.square(Q_samp - Q_s_a))

	def add_optimizer_op(self, scope):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		grads = optimizer.compute_gradients(self.loss, var_list=var_list)
		if self._config.grad_clip:
			grads = [(tf.clip_by_norm(grad, self._config.clip_val), var) for grad, var in grads]
		self.train_op = optimizer.apply_gradients(grads)
		self.grad_norm = tf.global_norm([grad for grad, _ in grads])

	def build(self):
		self.add_placeholders_op()
		self.q = self.get_q_values_op(self.state, scope='q', reuse=False)
		self.target_q = self.get_q_values_op(self.state_p, scope='target_q', reuse=False)

		self.add_update_target_op('q','target_q')
		self.add_loss_op(self.q, self.target_q)
		self.add_optimizer_op('q')

	def initialize(self):
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		if self._config.mode == 'train':
			self.sess.run(tf.global_variables_initializer())
			print('running training mode')
		elif self._config.mode == 'test':
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self._config.model_output))
			print('running test mode')
		self.sess.run(self.update_target_op)

	def train(self):
		print("Start to sample buffer")
		self.sampling_buffer()
		print("Finished sample buffer")
		t = 0
		total_loss = 0

		sw = tf.summary.FileWriter(self._config.model_output, self.sess.graph)

		while t < self._config.nsteps_train:
			t += 1
			self._lr_schedule.update(t)
			self._eps_schedule.update(t)
			loss_t = self.train_step(t, self._config.batch_size, self._lr_schedule.get_epsilon())
			total_loss += loss_t
			if t % self._config.print_freq == 0:
				eps = self._eps_schedule.get_epsilon()
				sys.stdout.write('Iter {} \t Loss {} \t Eps {} \n'.format(t, total_loss / t, eps))
				sys.stdout.flush()
			value_loss_train = summary_pb2.Summary.Value(tag='Train_epoch_loss', simple_value=total_loss / t)
			summary = summary_pb2.Summary(value=[value_loss_train])
			sw.add_summary(summary, global_step = t)
			sw.flush()

	def train_step(self, t, batch_size, lr):
		states, states_p, actions, rewards = self._bf.sample(batch_size)
		feed_dict = {self.state: states, self.state_p: states_p, 
			self.a: actions, self.r: rewards, self.lr:lr}
		loss_eval, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

		if t % self._config.target_update_freq == 0:
			self.sess.run(self.update_target_op)
		if t % self._config.saving_freq == 0:
			print(self._config.model_output)
			if not os.path.exists(self._config.model_output):
				os.makedirs(self._config.model_output)
			self.saver.save(self.sess, save_path=os.path.join(self._config.model_output, 'model'))
		if t % self._config.simulation_freq == 0:
			self.sampling_buffer()
		return loss_eval

	def get_random_action(self, state):
		action = np.random.randint(self._config.numActions)
		q = self.get_q_values(state)[0]
		q_value = q_value = q[action]
		return (action, q_value)

	def get_best_action(self, state):
		q = self.get_q_values(state)[0]
		action = np.argmax(q)
		# if self._config.mode == "test":
			# print("Q value and best action:")
			# print(q)
			# print(state)
			# self._sim.print_grid()
			# print(action)

		q_value = q[action]
		return (action, q_value)

	def get_q_values(self, state):
		q, = self.sess.run([self.q], feed_dict={self.state:state})
		return q

	def get_best_action_fn(self):
		def action_fn(state):
			action = self.get_best_action(state)[0]
			return action
		return action_fn

if __name__ == '__main__':
	config = Config()
	model = DQN(config)
	print("Model Build!")
	model.initialize()
	print("Initialized!")
	model.train()