import torch
import torch.nn as nn

class Env():
	def __init__(self, node_embeddings,fist_node_id,num_victim_nodes,device):
		super().__init__()
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
			--> self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			demand: (batch, n_nodes-1)
			node_embeddings: (batch, n_nodes, embed_dim)

			is_next_depot: (batch, 1), e.g., [[True], [True], ...]
			Nodes that have been visited will be marked with True.
		"""
		self.device = device
		self.fist_node_id = fist_node_id

		self.num_victim_nodes = num_victim_nodes
		self.node_embeddings = node_embeddings
		self.batch, self.n_nodes, self.embed_dim = node_embeddings.size()

		self.is_next_depot = torch.ones([self.batch, 1], dtype = torch.bool).to(self.device)
		self.visited_customer = torch.zeros((self.batch, self.n_nodes, 1), dtype = torch.bool).to(self.device)

		#compute cost
		"""
		self.x_natural = x_natural
		self.y = y
		self.rand_start_mode = rand_start_mode
		self.traffic_mdeol = model
		self.epsilon = epsilon
		self.A_wave = A_wave
		self.edge_weights = edge_weights
		"""


	def get_mask(self, visited_mask):
		""" next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = torch.int32), [0] denotes going to depot
			visited_mask **includes depot**: (batch, n_nodes, 1)
			visited_mask[:,1:,:] **excludes depot**: (batch, n_nodes-1, 1)
			customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...], [0] denotes 0th customer, not depot
			self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			D: (batch, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_nodes-1)
			visited_customer **excludes depot**: (batch, n_nodes-1, 1)
		 	is_next_depot: (batch, 1), e.g. [[True], [True], ...]
		 	return mask: (batch, n_nodes, 1)		
		"""

		self.visited_customer = self.visited_customer | visited_mask

		return self.visited_customer
	
	def _get_step(self, next_node):
		""" next_node **includes depot** : (batch, 1) int, range[0, n_nodes-1]
			--> one_hot: (batch, 1, n_nodes)
			node_embeddings: (batch, n_nodes, embed_dim)
			demand: (batch, n_nodes-1)
			--> if the customer node is visited, demand goes to 0 
			prev_node_embedding: (batch, 1, embed_dim)
			context: (batch, 1, embed_dim+1)
		"""

		one_hot = torch.eye(self.n_nodes)[next_node]
		visited_mask = one_hot.type(torch.bool).permute(0,2,1).to(self.device)
		mask = self.get_mask(visited_mask)

		prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].repeat(1,1,self.embed_dim))

		step_context = prev_node_embedding
		return mask, step_context

	def _create_t1(self):
		mask_t1 = self.create_mask_t1()
		#print(mask_t1)
		step_context_t1 = self.create_context_t1()
		#print("step context t1", step_context_t1.shape)
		return mask_t1, step_context_t1

	def create_mask_t1(self):
		mask_customer = self.visited_customer.to(self.device)
		if self.fist_node_id is not None:
			mask_depot = self.fist_node_id * torch.ones([self.batch, 1, 1]).to(self.device)
		else:
			mask_depot =  torch.zeros([self.batch, 1, 1]).to(self.device)


		mask_customer.scatter(dim=1, index=mask_depot.long(), value=True)
		return mask_customer

	def create_context_t1(self):

		if self.fist_node_id is not None:
			depot_idx = self.fist_node_id * torch.ones([self.batch, 1] , dtype = torch.long).to(self.device)# long == int64
		else:
			depot_idx =  torch.zeros([self.batch, 1] , dtype = torch.long).to(self.device)# long == int64



		depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].repeat(1,1,self.embed_dim))

		return depot_embedding

	def get_log_likelihood(self, _log_p, pi):
		""" _log_p: (batch, decode_step, n_nodes)
			pi: (batch, decode_step), predicted tour
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])
		return torch.sum(log_p.squeeze(-1), 1)

	def get_costs(self):
		""" self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			pi: (batch, decode_step), predicted tour
			d: (batch, decode_step, 2)
			Note: first element of pi is not depot, the first selected node in the path
		"""
		x = torch.randn(32,256,12)
		y = torch.randn(32,256,12)
		loss  =torch.nn.MSELoss(reduction="none")
		cost = loss(x,y).mean(dim=-1).mean(dim=-1)


		return cost

class Sampler(nn.Module):
	""" args; logits: (batch, n_nodes)
		return; next_node: (batch, 1)
		TopKSampler <=> greedy; sample one with biggest probability
		CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
	"""
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		
class TopKSampler(Sampler):
	def forward(self, logits):
		return torch.topk(logits, self.n_samples, dim = 1)[1]# == torch.argmax(log_p, dim = 1).unsqueeze(-1)

class CategoricalSampler(Sampler):
	def forward(self, logits):
		return torch.multinomial(logits.exp(), self.n_samples)
