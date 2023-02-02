import torch
import torch.nn as nn

from envs.layers import MultiHeadAttention, DotProductAttention
from envs.decoder_utils import TopKSampler, CategoricalSampler, Env
#from data import generate_data
class DecoderCell(nn.Module):
	def __init__(self,victim_nodes,device, embed_dim = 128, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		self.victim_nodes = victim_nodes
		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_step = nn.Linear(embed_dim, embed_dim, bias = False)
		
		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim)
		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
		self.env = Env
		self.device = device
	def compute_static(self, node_embeddings, graph_embedding):
		self.Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])
		self.K1 = self.Wk1(node_embeddings)
		self.V = self.Wv(node_embeddings)
		self.K2 = self.Wk2(node_embeddings)
		
	def compute_dynamic(self, mask, step_context):
		Q_step = self.Wq_step(step_context)
		Q1 = self.Q_fixed + Q_step
		Q2 = self.MHA([Q1, self.K1, self.V], mask = mask)
		Q2 = self.Wout(Q2)
		logits = self.SHA([Q2, self.K2, None], mask = mask)

		return logits.squeeze(dim = 1)

	def forward(self, first_node_index, encoder_output, return_pi = False, decode_type = 'sampling'):

		node_embeddings, graph_embedding = encoder_output
		batch_size = node_embeddings.size(0)
		self.compute_static(node_embeddings, graph_embedding)
		env = Env(node_embeddings, first_node_index, self.victim_nodes, self.device)

		mask, step_context = env._create_t1()

		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
		log_ps, tours = [], []
		# if the staring node is not appointment, the policy net would learn, else, the fist node is appointment
		if first_node_index is not None:
			for i in range(env.num_victim_nodes-1):
				logits = self.compute_dynamic(mask, step_context)
				log_p = torch.log_softmax(logits, dim = -1)
				next_node = selecter(log_p)
				mask, step_context = env._get_step(next_node)
				tours.append(next_node.squeeze(1))
				log_ps.append(log_p)

			first_selection = first_node_index * torch.ones(batch_size,1).to(self.device)
			pi = torch.stack(tours, 1)
			ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)
			pi = torch.cat([first_selection.long(),pi], dim =1)
		else:
			for i in range(env.num_victim_nodes):
				logits = self.compute_dynamic(mask, step_context)
				log_p = torch.log_softmax(logits, dim=-1)
				next_node = selecter(log_p)
				# print("next_node shape", next_node.shape)
				mask, step_context = env._get_step(next_node)
				tours.append(next_node.squeeze(1))
				log_ps.append(log_p)
			pi = torch.stack(tours, 1)
			ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)

		#cost = env.get_costs()
		if return_pi:
			return  ll, pi
		return  ll

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	#print(device)
	batch, n_nodes, embed_dim = 32, 325, 64
	#data = generate_data(device= device,n_samples = batch, n_customer = n_nodes-1)
	decoder = DecoderCell(device,embed_dim, n_heads = 8, clip = 10.).to(device)
	#print("data", data)
	node_embeddings = torch.rand((batch, n_nodes, embed_dim), dtype = torch.float).to(device)
	graph_embedding = torch.rand((batch, embed_dim), dtype = torch.float).to(device)
	encoder_output = (node_embeddings, graph_embedding)
	# a = graph_embedding[:,None,:].expand(batch, 7, embed_dim)
	# a = graph_embedding[:,None,:].repeat(1, 7, 1)
	# print(a.size())

	decoder.train()

	data = torch.rand((batch, n_nodes, embed_dim), dtype = torch.float).to(device)


	cost, ll, pi = decoder(1,encoder_output, return_pi = True, decode_type = 'sampling')
	print('\ncost: ', cost.size(), cost)
	print('\nll: ', ll.size(), ll)
	print('\npi: ', pi.size(), pi)

	# cnt = 0
	# for i, k in decoder.state_dict().items():
	# 	print(i, k.size(), torch.numel(k))
	# 	cnt += torch.numel(k)
	# print(cnt)

	# ll.mean().backward()
	# print(decoder.Wk1.weight.grad)
	# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634	