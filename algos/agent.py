class ACAgent(object):
	def __init__(self, algo, storage):
		self.algo = algo
		self.storage = storage

	def update(self):
		info = self.algo.update(self.storage)
		self.storage.after_update()

		return info

	def to(self, device):
		self.algo.actor_critic.to(device)
		self.storage.to(device)

		return self

	def train(self):
		self.algo.actor_critic.train()

	def eval(self):
		self.algo.actor_critic.eval()

	def random(self):
		self.algo.actor_critic.random = True

	def process_action(self, action):
		if hasattr(self.algo.actor_critic, 'process_action'):
			return self.algo.actor_critic.process_action(action)
		else:
			return action

	def act(self, *args, **kwargs):
		return self.algo.actor_critic.act(*args, **kwargs)

	def get_value(self, *args, **kwargs):
		return self.algo.actor_critic.get_value(*args, **kwargs)

	def insert(self, *args, **kwargs):
		return self.storage.insert(*args, **kwargs)