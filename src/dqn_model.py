import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os

from config import Config
from model_base import model




class DQN(model):
	"""
	Implement Neural Network with Tensorflow
	"""
	def add_placeholders_op(self):






	def get_q_values_op(self, state, scope, reuse=False):




	def add_update_target_op(self, q_scope, target_q_scope):




	def add_loss_op(self, q, target_q):





	def add_optimizer_op(self, scope):





	def build(self):




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





	def train_step(self, t, batch_size, lr):




	def get_random_action(self, state):





	def get_best_action(self, state):




	def get_q_values(self, state):



	def get_best_action_fn(self):




if __name__ == '__main__':
	config = Config()
	model = DQN(config)
	model.initialize()
	model.train()