from distutils.command.build_scripts import first_line_re
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

from einops import rearrange
# from rotary_embedding_torch import RotaryEmbedding

import pdb


def padded_stack(tensors, padding=0):
	dim_count = None
	for t in tensors:
		if not t.shape[0]==0:
			dim_count = len(t.shape)
			break
	try:
		if not dim_count:
			dim_count = len(tensors[0].shape)
		max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
	except:
		pdb.set_trace()
	# max_shape = [max([get_shape_for_padding(t, d) for t in tensors]) for d in range(dim_count)]

	padded_tensors = []

	for t in tensors:
		try:
			e = extend_tensor(t, max_shape, fill=padding)
		except:
			pdb.set_trace()

		padded_tensors.append(e)

	stacked = torch.stack(padded_tensors)
	return stacked

def batched_index(inputs, indices):
	"""
	inputs: b x n x d
	indices: b x m
	output: b x m x d
	"""
	input_size = inputs.shape[-1]
	# b x m x d
	indices = indices.unsqueeze(-1).expand(list(indices.shape) + [input_size])
	indexed_tensor = torch.gather(inputs, dim=1, index=indices)
	return indexed_tensor


def batch_index(tensor, index, pad=False):
	if tensor.shape[0] != index.shape[0]:
		raise Exception()

	if not pad:
		# tensor[i][index[i]] shape: nr x 2 x d (2: entity idx)
		return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])    # i: batch, index[i]: relation x entity_idx, tensor[i]: entity x emb
	else:
		return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def get_ent_mask1d(n_ents, max_num=None):
	"""
	n_ents: shape (b,), gold ent number.
	"""
	if max_num is None:
		max_num = max(n_ents)
	bs = len(n_ents)
	mask = torch.arange(max_num, device=n_ents.device).unsqueeze(0).repeat(bs,1)
	mask = mask < n_ents.reshape(bs,1)
	return mask

def get_ent_mask2d(n_ents):
	"""
	n_ents: shape (bs,), ent number.
	return b x max_n_ent x max_n_ent
	"""
	max_num = max(n_ents)

	if isinstance(n_ents, list):
		n_ents = torch.tensor(n_ents)
	n_ents = n_ents.reshape(-1,1)
	bs = n_ents.shape[0]
	mask0 = torch.arange(max_num, device=n_ents.device).unsqueeze(0).repeat(bs,1)	# bs x max_num
	mask1 = mask0.unsqueeze(-1).repeat(1,1,max_num)		# bs x max_num x max_num
	mask2 = mask0.unsqueeze(-2).repeat(1,max_num, 1)	
	mask3 = n_ents.unsqueeze(-1).repeat(1,max_num,max_num)	# bs x max_num x max_num
	mask = (mask1<mask3)*(mask2<mask3)
	return mask

def get_ent_mask3d(n_ents):
	"""
	mask2d: bs x ne x ne
	return bs x ne x ne x ne
	"""
	mask2d = get_ent_mask2d(n_ents)
	bs, ne, _ = mask2d.shape
	m1 = mask2d.unsqueeze(-1).repeat(1,1,1,ne)		# bs x n1 x n2 x n3
	m2 = mask2d.unsqueeze(-2).repeat(1,1,ne,1)		# bs x n1 x n3 x n2
	mask = m1*m2
	return mask


def get_span_mask(start_ids, end_ids, max_len):
	b, n = start_ids.shape
	# b x ns x n    (max_len: ns), end_ids: not for slicing. subtoken of end_ids is in span
	tmp = torch.arange(max_len, device=start_ids.device).unsqueeze(0).unsqueeze(-1).expand(b, max_len, n)
	batch_start_ids = start_ids.unsqueeze(1).expand_as(tmp)
	batch_end_ids = end_ids.unsqueeze(1).expand_as(tmp)
	mask = ((tmp >= batch_start_ids).float() * (tmp <= batch_end_ids).float())
	return mask



def cat_encode(repr1, repr2):
	repr = torch.cat((repr1, repr2), dim=-1)
	return repr


def max_pool(*reprs):

	dims = [int(x.shape[-1]) for x in reprs]
	assert len(set(dims))==1					# reprs have one hidden dimension
	
	reprs = [x.unsqueeze(-1) for x in reprs]
	reprs = torch.cat(reprs, dim=-1)
	reprs = torch.max(reprs, dim=-1)[0]
	
	return reprs


# def init_weights(m):




class CatEncoder(nn.Module):
	def __init__(self, input_dims, output_dim=None, proj=True):
		super().__init__()
		inputdims = [dim for dim in input_dims]
		self.input_dims = inputdims
		self.proj = proj
		self.output_dim = output_dim if self.proj else sum(self.inputdims)
		if proj:
			self.projection = nn.Linear(sum(inputdims), output_dim)


	def forward(self, *reprs):
		repr = torch.cat(reprs, dim=-1)
		if self.proj:
			repr = self.projection(repr)	
		return repr



class CatEncoderCross(nn.Module):
	def __init__(self, input_dims, output_dim, proj=True):
		super().__init__()
		inputdims = [dim for dim in input_dims]
		self.input_dims = inputdims
		self.output_dim = output_dim
		self.proj = proj
		if self.proj:
			self.proj = nn.Linear(sum(inputdims), output_dim)


	def forward(self, input1, input2):
		assert input1.shape[0]==input2.shape[0]
		bs, n_ent1, _ = input1.shape
		bs, n_ent2, _ = input2.shape
		# n_ent1 x n_ent2 x 2
		# rel_coos2 = [(i, j) for i in range(n_ent1) for j in range(n_ent2)]
		# inputs = [torch.cat((input1[i,j], input2[i,k]),dim=-1) for i in range(bs) for j, k in rel_coos2 ]
		# repr1 = torch.stack(inputs).to(input1.device).reshape(bs, n_ent1, n_ent2, -1)
		input1 = input1.unsqueeze(2).expand(-1, -1, n_ent2,-1)
		input2 = input2.unsqueeze(1).expand(-1, n_ent1, -1, -1)
		repr = torch.cat([input1,input2], dim=-1)
		if self.proj:
			repr = self.proj(repr)
			
		return repr

	def get_output_dim(self):
		return self.output_dim 





class BiafEncoder(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim, rank=768, factorize=False, bias_1=True, bias_2=True ):
		super().__init__()
		self.factorize = factorize
		if self.factorize:
			self.proj1 = nn.Linear(input_dim1, rank)
			self.proj2 = nn.Linear(input_dim2, rank)
			self.encoder = nn.Linear(rank, output_dim)
		else:
			self.bias_1 = bias_1
			self.bias_2 = bias_2
			self.weight = nn.Parameter(torch.Tensor(input_dim1+bias_1, input_dim2+bias_2, output_dim))
			self.bias = nn.Parameter(torch.Tensor(output_dim))
			self.reset_parameters()
	
	def reset_parameters(self):
		for w in [self.weight]:
			torch.nn.init.xavier_normal_(w)
		self.bias.data.fill_(0)
		return

	def forward(self, input1, input2):
		if self.factorize:
			input1 = self.proj1(input1)
			input2 = self.proj2(input2)
			repr = self.encoder(input1*input2)
		else:
			if self.bias_1:
				input1 = torch.cat((input1, torch.ones_like(input1[..., :1])), -1)
			if self.bias_2:
				input2 = torch.cat((input2, torch.ones_like(input2[..., :1])), -1)
			if len(input1.shape)==3:
				layer = torch.einsum('bnd,bne,deo->bno', input1, input2, self.weight.to(input1.dtype))
			elif len(input1.shape)==4:
				layer = torch.einsum('bnmd,bnme,deo->bnmo', input1, input2, self.weight.to(input1.dtype))
			repr = layer + self.bias.to(layer.dtype)
		return repr

# class Triaffine(nn.Module):

class BiafCrossEncoder(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim, rank=768, factorize=False, bias_1=True, bias_2=True ):
		super().__init__()
		self.input_dim1 = input_dim1
		self.input_dim2 = input_dim2
		self.output_dim = output_dim
		self.rank = rank
		self.factorize = factorize
		if self.factorize:
			self.proj1 = nn.Linear(input_dim1, rank)
			self.proj2 = nn.Linear(input_dim2, rank)
			self.encoder = nn.Linear(rank, output_dim)
		else:
			self.bias_1 = bias_1
			self.bias_2 = bias_2
			self.weight = nn.Parameter(torch.Tensor(input_dim1+bias_1, input_dim2+bias_2, output_dim))
			self.bias = nn.Parameter(torch.Tensor(output_dim))
			self.reset_parameters()
	
	def reset_parameters(self):
		for w in [self.weight]:
			torch.nn.init.xavier_normal_(w)
		self.bias.data.fill_(0)
		return

	def forward(self, input1, input2):
		if self.factorize:
			bs, n1, _ = input1.shape
			_, n2, _ = input2.shape
			input1 = input1.unsqueeze(2).expand(-1, -1, n2,-1)		# bs x n1 x n2 x d
			input2 = input2.unsqueeze(1).expand(-1, n1, -1, -1)		# bs x n1 x n2 x d
			input1 = self.proj1(input1)								# bs x n1 x n2 x d_rank
			input2 = self.proj2(input2)
			repr = self.encoder(input1*input2)						# bs x n1 x n2 x d_output
		else:
			if self.bias_1:
				input1 = torch.cat((input1, torch.ones_like(input1[..., :1])), -1)
			if self.bias_2:
				input2 = torch.cat((input2, torch.ones_like(input2[..., :1])), -1)
			layer = torch.einsum('bnd,bme,deo->bnmo', input1, input2, self.weight.to(input1.dtype))
			repr = layer + self.bias.to(layer.dtype)
		return repr		


class BiaffineSpanRepr(nn.Module):
	def __init__(self, input_size, hidden_dim, span_dim, rank, factorize=False, mode=0):
		"""
		mode 0: repr1 = biaf(e1, e2), repr2 = biaf(m1, m2), repr = biaf(repr1, repr2)
		mode 1: repr1 = biaf(e1, e2), repr2 = biaf(m1, m2), repr = cat(repr1, repr2)
		mode 2: repr1 = cat(e1, e2), repr2 = cat(m1, m2), repr = biaf(repr1, repr2)
		mode 3: repr1 = cat(e1, m1),  repr2 = cat(e2, m2), repr = biaf(repr1, repr2)
		mode 4: repr1 = biaf(e1, m1), repr2 = biaf(e2, m2), repr = biaf(repr1, repr2)
		mode 5: repr1 = biaf(e1, m1), repr2 = biaf(e2, m2), repr = cat(repr1, repr2)
		mode 6: repr = tetrafine(e1, m1, e2, m2)

		factorize: for repr1 + repr2 --> repr
		"""
		super().__init__()

		self.span_dim = span_dim
		self.factorize = factorize
		self.mode = mode
		
		self.e1_proj = nn.Linear(input_size, rank)
		self.e2_proj = nn.Linear(input_size, rank)
		self.m1_proj = nn.Linear(input_size, rank)
		self.m2_proj = nn.Linear(input_size, rank)  
		# step 1, e/m --> repr1, repr2
		if mode in {0, 1, 4, 5, 6}:
			hid_size = rank
		elif mode in {2,3}:
			hid_size = 2*rank
		else:
			pdb.set_trace()

		self.proj1 = nn.Linear(hid_size, hidden_dim)	# for repr_e, repr_h
		self.proj2 = nn.Linear(hid_size, hidden_dim)	# for repr_m, repr_t
		
		# step2, repr1/repr2 --> span_repr
			
		if mode in {1,5}:
			self.encode_proj = nn.Linear(2*hidden_dim, span_dim)
		elif mode==6:
			self.encode_proj = nn.Linear(hid_size, span_dim)
		else:
			if factorize:
				self.encode_proj = nn.Linear(hidden_dim, span_dim) 
			else:
				self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim, span_dim))
				self.bias = nn.Parameter(torch.Tensor(span_dim))
				self.reset_parameters()	

	def reset_parameters(self):
		for w in [self.weight]:
			torch.nn.init.xavier_normal_(w)
		self.bias.data.fill_(0)
		return


	def forward(self, e1, e2, m1, m2):
		"""
		e1, e2, m1, m2: bs * n_ent * input_size (bert output dim)
		e1, e2: entity start/end (subtokens)
		m1, m2: entity start/end (markers with the same position encoding as subtokens)
		"""
		
		# bs * n_ent * input_size --> bs * n_ent * rank
		e1 = self.e1_proj(e1)
		e2 = self.e2_proj(e2)
		m1 = self.m1_proj(m1)
		m2 = self.m2_proj(m2)

		# step1
		if self.mode in {0, 1, 2, 6}:
			input11, input12, input21, input22 = e1, e2, m1, m2
		elif self.mode in {3, 4, 5}:
			input11, input12, input21, input22 = e1, m1, e2, m2
		
		if self.mode in {0,1,4,5}:
			repr1 = self.proj1(input11*input12)
			repr2 = self.proj2(input21*input22)
		elif self.mode in {2,3}:
			repr1 = self.proj1(cat_encode(input11, input12))
			repr2 = self.proj1(cat_encode(input21, input22))
		

		# step2
		if self.mode in {1,5}:
			span_repr = self.encode_proj(torch.cat((repr1, repr2), dim=-1))
		elif self.mode == 6:
			span_repr = self.encode_proj(input11*input12*input21*input22)
		else:
			if self.factorize:
				span_repr = self.encode_proj(repr1*repr2)
			else:
				# if self.fp16 and self.training:
				layer = torch.einsum('bnd,bne,deo->bno', repr1, repr2, self.weight.to(repr1.dtype))
				span_repr = layer + self.bias.to(layer.dtype)
				# else:
				# 	layer = torch.einsum('bnd,bne,deo->bno', repr1, repr2, self.weight)
				# 	span_repr = layer + self.bias
		return span_repr


	
	@property
	def output_dim(self):
		return self.span_dim
		# return 2*self.rank
		# return 2*self.input_size
		
class BiSpanRepr(nn.Module):
	def __init__(self, input_size, span_dim, hidden_size):
		"""
		repr1 = cat(e1, m1),  repr2 = cat(e2, m2), repr = biaf(repr1, repr2)
		"""
		super().__init__()
		self.span_dim = span_dim
		self.proj11 = nn.Linear(input_size, hidden_size)
		self.proj12 = nn.Linear(input_size, hidden_size)
		self.proj21 = nn.Linear(input_size, hidden_size)
		self.proj22 = nn.Linear(input_size, hidden_size)
		self.proj1 = nn.Linear(2*hidden_size, hidden_size)
		self.proj2 = nn.Linear(2*hidden_size, hidden_size)
		# self.encode_proj = nn.Linear(rank, span_dim) 
		self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size, span_dim))
		self.bias = nn.Parameter(torch.Tensor(span_dim))
		self.reset_parameters()	


	def reset_parameters(self):
		for w in [self.weight]:
			torch.nn.init.xavier_normal_(w)
		self.bias.data.fill_(0)
		return
		
	def forward(self, e1, e2, m1, m2):
		"""
		e1, e2, m1, m2: bs * n_ent * input_size (bert output dim)
		e1, e2: entity start/end (subtokens)
		m1, m2: entity start/end (markers with the same position encoding as subtokens)
		"""
		e1 = self.proj11(e1)
		e2 = self.proj12(e2)
		m1 = self.proj21(m1)
		m2 = self.proj22(m2)

		repr1 = self.proj1(torch.cat((e1,m1), dim=-1))
		repr2 = self.proj2(torch.cat((e2,m2), dim=-1))
		# span_repr = self.encode_proj(repr1*repr2)
		layer = torch.einsum('bnd,bne,deo->bno', repr1, repr2, self.weight.to(repr1.dtype))
		span_repr = layer + self.bias.to(layer.dtype)
		return span_repr


	@property
	def output_dim(self):
		return self.span_dim



# class SpanRepr(nn.Module):
# 	"""Abstract class describing span representation."""

# 	def __init__(self, input_dim, use_proj=False, proj_dim=256, dropout=0.3):
# 		super(SpanRepr, self).__init__()
# 		self.input_dim = input_dim
# 		self.proj_dim = proj_dim
# 		self.use_proj = use_proj
# 		self.dropout =dropout
# 		if use_proj:
# 			# self.proj = nn.Linear(input_dim, proj_dim)
# 			model_list = []
# 			model_list.append(nn.Linear(input_dim, proj_dim))
# 			model_list.append(nn.GELU())
# 			model_list.append(nn.Dropout(p=dropout))
# 			self.proj = nn.Sequential(*model_list)

			
# 	@abstractmethod
# 	def forward(self, encoded_input, start_ids, end_ids):
# 		raise NotImplementedError

# 	def get_input_dim(self):
# 		return self.input_dim

# 	@abstractmethod
# 	def get_output_dim(self):
# 		raise NotImplementedError


class AttnSpanRepr(nn.Module):
	"""Class implementing the attention-based span representation."""

	def __init__(self, input_dim, proj_dim, output_dim, dropout=0.1):
		"""If use_endpoints is true then concatenate the end points to attention-pooled span repr.
		Otherwise just return the attention pooled term.
		"""
		super(AttnSpanRepr, self).__init__()
		self.proj = nn.Sequential(
					nn.Linear(input_dim, proj_dim),
					nn.GELU(),
					nn.Dropout(p=dropout)
					)
		self.output_dim = output_dim
		self.attention_params = nn.Linear(proj_dim, 1)
		self.output_proj = nn.Linear(proj_dim, output_dim)

		# Initialize weight to zero weight
		# self.attention_params.weight.data.fill_(0)
		# self.attention_params.bias.data.fill_(0)

	def forward(self, encoded_input, mention_pos):
		"""
		encoded_input: bs x seq_len x dh
		"""
		start_ids = mention_pos[:, :, 0]		# bs x max_n_ent	
		end_ids = mention_pos[:, :, 1]			# bs x max_n_ent
		encoded_input = self.proj(encoded_input)
		# b x ns x ne    (ns: subtokens, ne: entities)
		span_mask = get_span_mask(start_ids, end_ids, encoded_input.shape[1])
		attn_mask = (1 - span_mask) * (-1e4)
		# b x ns x 1 + b x ns x ne --> b x ns x ne
		attn_logits = self.attention_params(encoded_input) + attn_mask
		# b x ns x ne --> b x ne x ns
		attention_wts = nn.functional.softmax(attn_logits, dim=1).permute(0,2,1)
		attention_term = torch.einsum('bes, bsd->bed', attention_wts, encoded_input)
		span_repr = self.output_proj(attention_term)

		return span_repr




class BiaffineRelationCls(BiaffineSpanRepr):
	def __init__(self, input_size, hidden_dim, num_labels, rank, factorize=False, mode=0):
		super().__init__(input_size, hidden_dim, num_labels, rank, factorize, mode)
		"""
		input_size: subject/object entity head/tail token repr size
		input four reprs, subj_head, subj_tail, obj_head, obj_tail(s1, s2, o1, o2);
		"""


	def forward(self, subj_head, subj_tail, obj_head, obj_tail):
		rel_scores = super().forward(subj_head, subj_tail, obj_head, obj_tail)

		return rel_scores



class Tetrafine(BiaffineSpanRepr):
	def __init__(self, input_dim, hidden_dim, output_dim, rank, factorize=False, mode=6):
		super().__init__(input_dim, hidden_dim, output_dim, rank, factorize, mode)
		"""
		input_size: subject/object entity head/tail token repr size
		input four reprs, subj_head, subj_tail, obj_head, obj_tail(s1, s2, o1, o2);
		"""


	def forward(self, input1, input2, input3, input4):
		rel_scores = super().forward(input1, input2, input3, input4)

		return rel_scores


class CPDTrilinear(nn.Module):
	def __init__(self, input_dim1, input_dim2, input_dim3, rank, output_dim):
		super().__init__()
		"""
		input three tensor with the same shape.
		"""
		self.proj1 = nn.Linear(input_dim1, rank)
		self.proj2 = nn.Linear(input_dim2, rank)
		self.proj3 = nn.Linear(input_dim3, rank)

		self.encode_proj = nn.Linear(rank, output_dim)

	def forward(self, input1, input2, input3):
		layer1 = self.proj1(input1)
		layer2 = self.proj2(input2)
		layer3 = self.proj3(input3)

		return self.encode_proj(layer1*layer2*layer3)




class LinearMessegePasser(nn.Module):
	def __init__(self, sender_dim, receiver_dim):
		super().__init__()
		self.s_dim = sender_dim
		self.r_dim = receiver_dim
		self.net = nn.Linear(self.s_dim+self.r_dim, self.r_dim)

	def forward(self, x_s, x_r):
		return self.net(torch.cat([x_s, x_r], dim=-1)) 

class BiaffineMessagePasser(nn.Module):
	def __init__(self, sender_dim, receiver_dim):
		super().__init__()
		self.net = BiafEncoder(input_dim1=sender_dim, input_dim2=receiver_dim, output_dim=receiver_dim)

	def forward(self, x_s, x_r):
		return self.net(x_s, x_r)













#-------------------------------------
class HyperGNNTernaryGraph(nn.Module):
	def __init__(self, ent_dim, rel_dim, dropout, args):
		super(HyperGNNTernaryGraph, self).__init__()
		self.args = args
		self.iter = args.iter
		
		self.hyperedgelayer = HyperGNNTernaryComposeLayer(ent_dim, rel_dim, dropout=dropout, args=args)
		aggregator = HyperGNNTernaryAggregateLayer
		self.aggregate = aggregator(ent_dim, rel_dim, dropout, args)

		

	def forward(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
		"""
		sub_reprs: bs x ns x no x d
		"""
		mask1d = get_ent_mask1d(ent_numbers)
		mask2d = get_ent_mask2d(ent_numbers)
		
		for i in range(self.iter):
			factor = self.hyperedgelayer(sub_reprs, obj_reprs, rel_reprs)
			sub_reprs, obj_reprs, rel_reprs = self.aggregate(sub_reprs, obj_reprs, rel_reprs, factor, ent_numbers)
			sub_reprs *= mask1d.unsqueeze(-1)
			obj_reprs *= mask1d.unsqueeze(-1)
			rel_reprs *= mask2d.unsqueeze(-1)
		return sub_reprs, obj_reprs, rel_reprs

class HyperGNNBinaryGraph(nn.Module):
	def __init__(self, rel_dim, dropout, args):
		super(HyperGNNBinaryGraph, self).__init__()
		self.args = args
		self.iter = args.iter
		aggregator = HyperGNNBinaryAggregateLayer

		self.hyperedgelayer = HyperGNNBinaryComposeLayer(rel_dim, dropout=dropout, args=args)
		self.aggregate = aggregator(rel_dim, dropout, args)

	def forward(self, rel_reprs, ent_numbers):
		"""
		xx_reprs: node reprs
		"""
		mask2d = get_ent_mask2d(ent_numbers)
		for i in range(self.iter):
			factor = self.hyperedgelayer(rel_reprs)
			rel_reprs = self.aggregate(rel_reprs, factor, ent_numbers)
			rel_reprs *= mask2d.unsqueeze(-1)
		return rel_reprs


class HyperGNNHybridGraph(nn.Module):
	def __init__(self, ent_dim, rel_dim, dropout, args):
		super(HyperGNNHybridGraph, self).__init__()
		self.args = args
		self.iter = args.iter

		aggregator = HyperGNNHybridAggregateLayer

		self.hyperedgelayer1 = HyperGNNTernaryComposeLayer(ent_dim, rel_dim, dropout=dropout, args=args)
		self.hyperedgelayer2 = HyperGNNBinaryComposeLayer(rel_dim, dropout=dropout, args=args)
		self.aggregate = aggregator(ent_dim, rel_dim, dropout, args)

	def forward(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
		"""
		xx_reprs: node reprs
		"""
		mask1d = get_ent_mask1d(ent_numbers)
		mask2d = get_ent_mask2d(ent_numbers)

		for i in range(self.iter):
			factor_ter = self.hyperedgelayer1(sub_reprs, obj_reprs, rel_reprs)
			if self.args.factor_type in {'tersib', 'tercop', 'tergp'}:
				factor_b1 = self.hyperedgelayer2(rel_reprs)
				factors = (factor_ter, factor_b1)
			elif self.args.factor_type in {'tersibcop', 'tersibgp', 'tercopgp'}:
				factor_b1, factor_b2 = self.hyperedgelayer2(rel_reprs)
				factors = (factor_ter, factor_b1, factor_b2)
			elif self.args.factor_type == 'tersibcopgp':
				factor_b1, factor_b2, factor_b3 = self.hyperedgelayer2(rel_reprs)
				factors = (factor_ter, factor_b1, factor_b2, factor_b3)
			sub_reprs, obj_reprs, rel_reprs = self.aggregate(sub_reprs, obj_reprs, rel_reprs, factors, ent_numbers)
			sub_reprs *= mask1d.unsqueeze(-1)
			obj_reprs *= mask1d.unsqueeze(-1)
			rel_reprs *= mask2d.unsqueeze(-1)
			
		return sub_reprs, obj_reprs, rel_reprs


class HyperGNNTernaryComposeLayer(nn.Module):
	"""
	update hyperedge features
	"""
	def __init__(self, ent_dim, rel_dim, dropout, args):
		super(HyperGNNTernaryComposeLayer, self).__init__()
		mem_dim = args.mem_dim
		self.dropout = nn.Dropout(dropout)
		layernorm = args.layernorm
		dims = (ent_dim, ent_dim, rel_dim)
		if args.factor_encoder=='biaf':
			self.factor_compose = CPDTrilinear(ent_dim, ent_dim, rel_dim, mem_dim, mem_dim)
		elif args.factor_encoder=='cat':
			self.factor_compose = CatEncoder(input_dims=dims, output_dim=mem_dim, proj=True)

		self.layernorm = nn.LayerNorm(mem_dim, eps=1e-6) if layernorm else nn.Identity()	# for cell state of node1
		# self.layernorm_2 = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
		# self.layernorm_3 = nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()

		# self.aggregate = HyperGNNTernaryAggregateLayer(ent_dim, rel_dim, dropout, args)

	def forward(self, sub_reprs, obj_reprs, rel_reprs):
		"""
		input old vertex features, update hyperedge features, then update vertex features
		"""
		bs, ne, _ = sub_reprs.shape

		# initial the node reprs in hyperedge
		sub_h = sub_reprs.unsqueeze(-2).expand(-1, -1, ne, -1)		# bs x ns x no x de
		obj_h = obj_reprs.unsqueeze(-3).expand(-1, ne, -1, -1)		# bs x ns x no x de
		rel_h = rel_reprs

		# update hyperedge feature
		factor = self.layernorm(self.dropout(self.factor_compose(sub_h, obj_h, rel_h)))

		# sub_reprs, obj_reprs, rel_reprs = self.aggregate(sub_reprs, obj_reprs, rel_reprs, factor, batch_mask)
		return factor

class HyperGNNBinaryComposeLayer(nn.Module):
	"""
	update hyperedge features
	"""
	def __init__(self, rel_dim, dropout, args):
		super(HyperGNNBinaryComposeLayer, self).__init__()
		self.args = args
		self.factor_type = args.factor_type
		mem_dim = args.mem_dim
		self.dropout = nn.Dropout(dropout)
		layernorm = args.layernorm
		dims = (rel_dim, rel_dim)
		if args.factor_type in {'sib', 'cop', 'gp', 'tersib', 'tercop', 'tergp'}:
			if args.factor_encoder=='biaf':
				self.factor_compose1 = BiafEncoder(input_dim1=rel_dim, input_dim2=rel_dim, output_dim=mem_dim, rank=mem_dim, factorize=True)
			elif args.factor_encoder=='cat':
				self.factor_compose1 = CatEncoder(input_dims=dims, output_dim=mem_dim, proj=True)
			self.layernorm1 = nn.LayerNorm(mem_dim, eps=1e-6) if layernorm else nn.Identity()
		elif args.factor_type in {'sibcop', 'sibgp', 'copgp', 'tersibcop','tersibgp', 'tercopgp'}:
			if args.factor_encoder=='biaf':
				self.factor_compose1 = BiafEncoder(input_dim1=rel_dim, input_dim2=rel_dim, output_dim=mem_dim, rank=mem_dim, factorize=True)
				self.factor_compose2 = BiafEncoder(input_dim1=rel_dim, input_dim2=rel_dim, output_dim=mem_dim, rank=mem_dim, factorize=True)
			elif args.factor_encoder=='cat':
				self.factor_compose1 = CatEncoder(input_dims=dims, output_dim=mem_dim, proj=True)
				self.factor_compose2 = CatEncoder(input_dims=dims, output_dim=mem_dim, proj=True)
			self.layernorm1 = nn.LayerNorm(mem_dim, eps=1e-6) if layernorm else nn.Identity()
			self.layernorm2 = nn.LayerNorm(mem_dim, eps=1e-6) if layernorm else nn.Identity()
		elif args.factor_type in {'sibcopgp', 'tersibcopgp'}:
			if args.factor_encoder=='biaf':
				self.factor_compose1 = BiafEncoder(input_dim1=rel_dim, input_dim2=rel_dim, output_dim=mem_dim, rank=mem_dim, factorize=True)
				self.factor_compose2 = BiafEncoder(input_dim1=rel_dim, input_dim2=rel_dim, output_dim=mem_dim, rank=mem_dim, factorize=True)
				self.factor_compose3 = BiafEncoder(input_dim1=rel_dim, input_dim2=rel_dim, output_dim=mem_dim, rank=mem_dim, factorize=True)
			elif args.factor_encoder=='cat':
				self.factor_compose1 = CatEncoder(input_dims=dims, output_dim=mem_dim, proj=True)
				self.factor_compose2 = CatEncoder(input_dims=dims, output_dim=mem_dim, proj=True)
				self.factor_compose3 = CatEncoder(input_dims=dims, output_dim=mem_dim, proj=True)
			self.layernorm1 = nn.LayerNorm(mem_dim, eps=1e-6) if layernorm else nn.Identity()
			self.layernorm2 = nn.LayerNorm(mem_dim, eps=1e-6) if layernorm else nn.Identity()
			self.layernorm3 = nn.LayerNorm(mem_dim, eps=1e-6) if layernorm else nn.Identity()


	def forward(self, rel_reprs):
		"""
		input old vertex features, update hyperedge features, then update vertex features
		"""
		# if self.args.unirel:
		# 	rel_ha0 = rel_hb0 = reprs
		# else:
		# 	rel_ha0, rel_hb0 = reprs
		bs, ns, no, _ = rel_reprs.shape
		# initial the node reprs in hyperedge
		if self.factor_type in {'sib','tersib'}:							# score for rij and rik
			b1_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b1_hb = rel_reprs.unsqueeze(-3).expand(-1, -1, no, -1, -1)		# bs x ns x no1 x no x dm
		elif self.factor_type in {'cop', 'tercop'}:							# score for rik and rjk
			b1_ha = rel_reprs.unsqueeze(-3).expand(-1, -1, ns, -1, -1)		# bs x ns x ns1 x no x dm
			b1_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm
		elif self.factor_type in {'gp', 'tergp'}:							# score for rij and rjk
			b1_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b1_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm
		elif self.factor_type in {'sibcop', 'tersibcop'}:					
			b1_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b1_hb = rel_reprs.unsqueeze(-3).expand(-1, -1, no, -1, -1)		# bs x ns x no1 x no x dm
			b2_ha = rel_reprs.unsqueeze(-3).expand(-1, -1, ns, -1, -1)		# bs x ns x ns1 x no x dm
			b2_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm
		elif self.factor_type in {'sibgp', 'tersibgp'}:
			b1_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b1_hb = rel_reprs.unsqueeze(-3).expand(-1, -1, no, -1, -1)		# bs x ns x no1 x no x dm
			b2_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b2_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm
		elif self.factor_type in {'copgp', 'tercopgp'}:
			b1_ha = rel_reprs.unsqueeze(-3).expand(-1, -1, ns, -1, -1)		# bs x ns x ns1 x no x dm
			b1_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm
			b2_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b2_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm
		elif self.factor_type in {'sibcopgp', 'tersibcopgp'}:
			b1_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b1_hb = rel_reprs.unsqueeze(-3).expand(-1, -1, no, -1, -1)		# bs x ns x no1 x no x dm
			b2_ha = rel_reprs.unsqueeze(-3).expand(-1, -1, ns, -1, -1)		# bs x ns x ns1 x no x dm
			b2_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm
			b3_ha = rel_reprs.unsqueeze(-2).expand(-1, -1, -1, no, -1)		# bs x ns x no x no1 x dm
			b3_hb = rel_reprs.unsqueeze(-4).expand(-1, ns, -1, -1, -1)		# bs x ns1 x ns x no x dm


		# update hyperedge feature
		if self.args.factor_type in {'sib', 'cop', 'gp', 'tersib', 'tercop','tergp'}:
			factor = self.layernorm1(self.dropout(self.factor_compose1(b1_ha, b1_hb)))
			return factor
		elif self.args.factor_type in {'sibcop', 'sibgp', 'copgp', 'tersibcop','tersibgp', 'tercopgp'}:
			factor1 = self.layernorm1(self.dropout(self.factor_compose1(b1_ha, b1_hb)))
			factor2 = self.layernorm2(self.dropout(self.factor_compose2(b2_ha, b2_hb)))
			return (factor1, factor2)
		elif self.factor_type in {'sibcopgp', 'tersibcopgp'}:
			factor1 = self.layernorm1(self.dropout(self.factor_compose1(b1_ha, b1_hb)))
			factor2 = self.layernorm2(self.dropout(self.factor_compose2(b2_ha, b2_hb)))
			factor3 = self.layernorm3(self.dropout(self.factor_compose3(b3_ha, b3_hb)))
			return (factor1, factor2, factor3)


class HyperGNNTernaryAggregateLayer(nn.Module):
	def __init__(self, ent_dim, rel_dim, dropout, args):
		super(HyperGNNTernaryAggregateLayer, self).__init__()
		self.args = args
		mem_dim = args.mem_dim
		layernorm = args.layernorm
		self.dropout = nn.Dropout(dropout)

		# for ablation study
		self.proj_s = nn.Linear(ent_dim, mem_dim)
		self.proj_o = nn.Linear(ent_dim, mem_dim)
		self.attn_combine_s = nn.Linear(mem_dim + ent_dim, mem_dim)
		self.attn_combine_o = nn.Linear(mem_dim + ent_dim, mem_dim)
		
		# if args.attn_encoder=='nonlinear':
		self.attn_combine_s = nn.Sequential(self.attn_combine_s, nn.GELU())
		self.attn_combine_o = nn.Sequential(self.attn_combine_o, nn.GELU())
		# self.attn_combine_r = nn.Sequential(self.attn_combine_r, nn.GELU())
			
		self.sv = nn.Linear(mem_dim, 1, bias=False)
		self.ov = nn.Linear(mem_dim, 1, bias=False)

		self.fc_s = nn.Linear(mem_dim, ent_dim)
		self.fc_o = nn.Linear(mem_dim, ent_dim)

		if self.args.attn_self:
			self.proj_r = nn.Linear(rel_dim, mem_dim)
			self.attn_combine_r = nn.Linear(mem_dim + rel_dim, mem_dim)
			self.attn_combine_r = nn.Sequential(self.attn_combine_r, nn.GELU())
			self.rv = nn.Linear(mem_dim, 1, bias=False)
			self.fc_r = nn.Linear(mem_dim, rel_dim)
		else:
			self.encode_r = LinearMessegePasser(mem_dim, rel_dim)

		self.layernorm_s = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
		self.layernorm_o = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
		self.layernorm_r = nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()

	def update_rel(self, rel_reprs, factor):
		"""
		rel_reprs: bs x ns x no x dr
		factor: bs x ns x no x dm
		return updated rel_reprs
		"""
		# ht = self._apply_mask(ht, mask)
		# ht_new = self._apply_mask(ht_new, mask)

		res = rel_reprs
		if self.args.attn_self:
			ht = self.proj_r(rel_reprs).unsqueeze(-2)			# bs x ns x no x 1 x dm
			total_h = torch.cat([ht, factor.unsqueeze(-2)], dim=-2)		# bs x ns x no x 2 x dm
			ht = rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
			comb = torch.cat([ht, total_h], dim=-1)						# bs x ns x no x 2 x (dr + dm)
			energy = self.attn_combine_r(comb)    						# bs x ns x no x 2 x 1
			energy = self.rv(energy).squeeze(-1) 						# bs x ns x no x 2
			attention = energy.softmax(dim=-1)							# bs x ns x no x 2
			output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))
			output = self.dropout(self.fc_r(output)) + res
			output = self.layernorm_r(output)
		else:
			output = self.dropout(self.encode_r(factor, rel_reprs)) + res
			output = self.layernorm_r(output) 
		# output = torch.max(total_h, dim=-2)[0]

		return output

	def update_sub(self, sub_reprs, factor, ent_numbers):
		"""
		factor: bs x ns x no x dm
		sub_reprs: bs x ns x ds
		batch_mask: bs x ns x no
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
		res = sub_reprs
		bs, ne, _, dm = factor.shape
		ht = self.proj_s(res)
		# total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
		total_h = torch.cat([ht.unsqueeze(-2), factor], dim=-2) if self.args.attn_self else factor	# bs x ne x (1+ne) x dm or bs x ne x ne x dm

		ht = sub_reprs.unsqueeze(-2).repeat(1, 1, total_h.shape[-2], 1).contiguous()		# bs x ne x ne x dm
		comb = torch.cat([ht, total_h], dim=-1)							# bs x ne x (1+ne) x dm+dr
		# pdb.set_trace()
		energy = self.attn_combine_s(comb)    							# bs x ne x (1+ne) x dm
		energy = self.sv(energy).squeeze(-1)     						# bs x ne x (1+ne) 
		attn_mask = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)						# bs x ne x ne
		attn_mask = (attn_mask + ~batch_mask).bool()					# bs x ne x ne
		if self.args.attn_self:
			attn_self = torch.zeros((bs, ne, 1), device=self.args.device).bool()		#  bs x ne x 1
			attn_mask = torch.cat((attn_self, attn_mask), axis=-1)			# bs x ne x (1+ne)

		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-1)								# ns x no
		output = torch.einsum('bij,bijd->bid', attention, total_h.to(attention.dtype))		# ns x dm
		output = self.dropout(self.fc_s(output)) + res
		output = self.layernorm_s(output)

		return output

	def update_obj(self, obj_reprs, factor, ent_numbers):
		"""
		factor: bs x ns x no x dm
		obj_reprs: bs x no x de
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
		res = obj_reprs
		bs, ne, _, dm = factor.shape
		ht = self.proj_o(res)											# (b, no, dh)
		# total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
		total_h = torch.cat([ht.unsqueeze(-3), factor], dim=-3) if self.args.attn_self else factor	# bs*(1+ne)*ne*dm
		
		ht = obj_reprs.unsqueeze(-3).repeat(1,total_h.shape[-3], 1, 1).contiguous()		# bs x (1+ne) x ne x de
		comb = torch.cat([ht, total_h], dim=-1)							# bs x (1+ne) x ne x (de+dm)

		energy = self.attn_combine_o(comb)    							# bs x (1+ne) x ne x dm
		energy = self.ov(energy).squeeze(-1)     						# bs x (1+ne) x ne
		attn_mask = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)					# bs x ne x ne
		attn_mask = (attn_mask + ~batch_mask).bool()					# bs x ne x ne
		if self.args.attn_self:
			attn_self = torch.zeros((bs, 1, ne), device=self.args.device).bool()	 		# bs x 1 x ne
			attn_mask = torch.cat((attn_self, attn_mask), axis=-2)			# bs x (1+ne) x ne

		# attn_mask = torch.sum(ht, dim=-1) == 0							
		energy = energy.masked_fill(attn_mask, -1e4)

		attention = energy.softmax(dim=-2)								# bs x (1+ne) x ne
		output = torch.einsum('bij,bijd->bjd', attention, total_h.to(attention.dtype))		# bs x no x dm
		output = self.dropout(self.fc_o(output)) + res
		output = self.layernorm_o(output)

		return output 


	def forward(self, sub_reprs, obj_reprs, rel_reprs, factor, ent_numbers):
		"""
		rel_reprs: bs x ns x no x dr
		sub_reprs: bs x ns x de
		obj_reprs: no x de	
		rel_h, sub_h, obj_h: ns x no x dx
		"""
		sub_reprs = self.update_sub(sub_reprs, factor, ent_numbers)
		obj_reprs = self.update_obj(obj_reprs, factor, ent_numbers)
		rel_reprs = self.update_rel(rel_reprs, factor)
		return sub_reprs, obj_reprs, rel_reprs


class HyperGNNBinaryAggregateLayer(nn.Module):
	def __init__(self, rel_dim, dropout, args):
		super(HyperGNNBinaryAggregateLayer, self).__init__()

		self.factor_type = args.factor_type
		mem_dim = args.mem_dim
		self.args = args
		layernorm = args.layernorm
		self.dropout = nn.Dropout(dropout)

		self.attn_combine = nn.Linear(mem_dim + rel_dim, mem_dim)
		self.attn_combine = nn.Sequential(self.attn_combine, nn.GELU())
		self.v = nn.Linear(mem_dim, 1, bias=False)
		self.fc = nn.Linear(mem_dim, rel_dim)
		self.layernorm = nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()


	def update_single(self, ht, factor, ent_numbers):
		"""
		ht is rel repr: bs x ns x no x dr, factor: bs x ni x nj x nk x dm
		if edgetype=='sib'
			factor: score for rij and rik; j,k are obj axis
			ha: ht in i,j axis; hb: ht in ik axis of factor 
			ha: bs x ni x nj x nk x dm; hb:  bs x ni x nk x nj x dm 
		if edgetype=='cop'
			factor: score of rik and rjk.
			ha: ht in ik axis of factor; hb: ht in jk axis of factor
			ha: bs x ni x nk x nj x d; hb: bs x nj x nk x ni x d	
		if edgetype=='gp'
			factor: score for rij and rjk.
			bin_ha: bs x ni x nj x nk x d; bin_hb: bs x nj x nk x ni x d
		return updated ht
		"""
		bs, ne, _, dr = ht.shape

		if self.factor_type =='sib':
			ha = factor
			hb = factor.permute(0,1,3,2,4)
		elif self.factor_type=='cop':
			ha = factor.permute(0,1,3,2,4)
			hb = factor.permute(0,2,3,1,4)
		elif self.factor_type=='gp':
			ha = factor
			hb = factor.permute(0,2,3,1,4)
		ht_new = torch.cat((ha, hb), dim=-2)

		res = ht
		
		total_h = ht_new
		ht = ht.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()		# bs x ns x no x nc x dm
		comb = torch.cat([ht, total_h], dim=-1)											# bs x ns x no x nc x 2*dm

		energy = self.attn_combine(comb)    			# bs x ns x no x nc x dm
		energy = self.v(energy).squeeze(-1)     		# bs x ns x no x nc 
		# attn_mask = torch.sum(ht, dim=-1) == 0		# bs x ns x no x nc
		batch_mask3d = get_ent_mask3d(ent_numbers)		# bs*ne*ne*ne
		if self.factor_type=='sib':
			m1 = torch.eye(ne,dtype=torch.int,device=self.args.device).unsqueeze(0).repeat(bs,1,1)	# bs x no x no1	, r(i,j) not att to r(i,j)
			m1 = m1.unsqueeze(-3).repeat(1,ne,1,1).bool()						# bs*ne*ne*ne, bs x ns x no x no1
			m1 = (m1 + ~batch_mask3d).bool()
			attn_mask = torch.stack((m1,m1), dim=-2).reshape(bs, ne, ne, -1)	# bs x ns x no x nc
		elif self.factor_type=='cop':
			m1 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
			m1 = m1.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
			m1 = (m1 + ~batch_mask3d).bool()
			attn_mask = torch.stack((m1,m1), dim=-2).reshape(bs, ne, ne, -1)	# bs x ns x no x 2ns1
			# unsqueeze(-2).repeat(1,no,1).bool()	# ns x no x 2*ns1
		elif self.factor_type=='gp':
			m1 = ~batch_mask3d
			attn_mask = torch.stack((m1,m1), dim=-2).reshape(bs, ne, ne, -1)	# bs x ns x no x 2no
		else:
			raise ValueError('factor_type is not correct')
		energy = energy.masked_fill(attn_mask, -1e4)
		
		attention = energy.softmax(dim=-1) 			# bs x ns x no x nc
		output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))	# ns x no x dm

		output = self.dropout(self.fc(output)) + res
		output = self.layernorm(output)

		return output

	def update_double(self, ht, factors, ent_numbers):
		"""
		factor_type: sibcop, sibgp, copgp
		ht: bs x ns x no x d
		if edgetype=='sib'
			factor: j,k are obj axis; ij, ik are two relations
			ha: ht in i,j axis; hb: ht in ik axis of factor 
			ha: bs x ni x nj x nk x dm; hb:  bs x ni x nk x nj x dm 
		if edgetype=='cop'
			factor: ik,jk are two relations.
			ha: ht in ik axis of factor; hb: ht in jk axis of factor
			ha: bs x ni x nk x nj x d; hb: bs x nj x nk x ni x d	
		if edgetype=='gp'
			factor: ij, jk are two relations
			bin_ha: bs x ni x nj x nk x d; bin_hb: bs x nj x nk x ni x d
		"""
		(factor1, factor2) = factors
		res = ht
		bs, ne, _, dm = ht.shape
		if self.factor_type in {'sibcop', 'sibgp'}:
			f1_ha1 = factor1									
			f1_hb1 = factor1.permute(0,1,3,2,4)	
		elif self.factor_type == 'copgp':
			f1_ha1 = factor1.permute(0,1,3,2,4)					
			f1_hb1 = factor1.permute(0,2,3,1,4)	
		f1_h = torch.cat([f1_ha1, f1_hb1], dim=-2)

		if self.factor_type in {'copgp', 'sibgp'}:
			f2_ha1 = factor2									# bs x ns x no x no1 x d	gp
			f2_hb1 = factor2.permute(0,2,3,1,4)	
		elif self.factor_type =='sibcop':
			f2_ha1 = factor2.permute(0,1,3,2,4)					
			f2_hb1 = factor2.permute(0,2,3,1,4)	
		f2_h = torch.cat([f2_ha1, f2_hb1], dim=-2)

		total_h = torch.cat([f1_h, f2_h], dim=-2)					# bs x ns x no x (3no1+ns1) x d   
		ht = ht.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()		# bs x ns x no x nc x dm,   (nc=(3no1+ns1)
		comb = torch.cat([ht, total_h], dim=-1)											# ns x no x nc x 2*dm

		energy = self.attn_combine(comb)    		# bs x ns x no x nc x dm
		energy = self.v(energy).squeeze(-1)     	# bs x ns x no x nc 
		batch_mask3d = get_ent_mask3d(ent_numbers)		# bs x ns x no x no1
		
		if self.factor_type=='sibgp':
			m1 = torch.eye(ne,dtype=torch.int,device=self.args.device).unsqueeze(0).repeat(bs,1,1)	# bs x no x no1	, r(i,j) not att to r(i,j)
			m1 = m1.unsqueeze(-3).repeat(1,ne,1,1).bool()					# bs x ns x no x no1
			m1 = (m1 + ~batch_mask3d).bool()
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)	# for sib,  bs x ns x no x 2no1
			m2 = ~batch_mask3d
			m2 = torch.cat((m2,m2), dim=-1).reshape(bs, ne, ne, -1)	# for gp, bs x ns x no x 2ns1
		elif self.factor_type=='sibcop':
			m1 = torch.eye(ne,dtype=torch.int,device=self.args.device).unsqueeze(0).repeat(bs,1,1)	# bs x no x no1	, r(i,j) not att to r(i,j)
			m1 = m1.unsqueeze(-3).repeat(1,ne,1,1).bool()					# bs x ns x no x no1
			m1 = (m1 + ~batch_mask3d).bool()
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)	# for sib,  bs x ns x no x 2no1
			m2 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
			m2 = m2.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
			m2 = (m2 + ~batch_mask3d).bool()
			m2 = torch.cat((m2,m2), dim=-1).reshape(bs, ne, ne, -1)			# bs x ns x no x 2ns1
		elif self.factor_type=='copgp':
			#cop
			m1 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
			m1 = m1.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
			m1 = (m1 + ~batch_mask3d).bool()
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2ns1
			# gp
			m2 = ~batch_mask3d
			m2 = torch.cat((m2,m2), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2no
		else:
			raise ValueError('factor_type is not correct')
		attn_mask = torch.cat((m1,m2), dim=-1).reshape(bs, ne, ne, -1)

		energy = energy.masked_fill(attn_mask, -1e4)

		attention = energy.softmax(dim=-1) 			# bs x ns x no x nc
		output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))	# ns x no x dm
		# if self.args.attn_res:
		# 	output = self.dropout(self.fc(output)) + res
		# else:
		# 	output = self.dropout(self.fc(output))
		output = self.dropout(self.fc(output)) + res
		output = self.layernorm(output)

		return output

	def update_triple(self, ht, factors, ent_numbers):
		"""
		self.factor_type == sibcopgp
		ht: bs x ns x no x d
		sib:
			ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
		cop:
			ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
		gp:
			ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
		"""
		(factor1, factor2, factor3) = factors
		res = ht
		bs, ne, _, dm = ht.shape
		# sib
		f1_ha1 = factor1									# bs x ns x no x no1 x d	sib
		f1_hb1 = factor1.permute(0,1,3,2,4)					# bs x ns x no x no1 x d
		f1_h = torch.cat([f1_ha1, f1_hb1], dim=-2)
		# cop
		f2_ha1 = factor2.permute(0,1,3,2,4)					# bs x ns x no x ns1 x d	cop
		f2_hb1 = factor2.permute(0,2,3,1,4)					# bs x ns x no x ns1 x d
		f2_h = torch.cat([f2_ha1, f2_hb1], dim=-2)
		# gp
		f3_ha1 = factor3									# bs x ns x no x no1 x d	gp
		f3_hb1 = factor3.permute(0,2,3,1,4)					# bs x ns x no x ns1 x d
		f3_h = torch.cat([f3_ha1, f3_hb1], dim=-2)

		total_h = torch.cat([f1_h, f2_h, f3_h], dim=-2)					# bs x ns x no x (3no1+3ns1) x d   
		ht = ht.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()		# bs x ns x no x nc x dm,   (nc=(3no1+ns1)
		comb = torch.cat([ht, total_h], dim=-1)											# bs x ns x no x nc x 2*dm

		energy = self.attn_combine(comb)    		# bs x ns x no x nc x dm
		energy = self.v(energy).squeeze(-1)     	# bs x reprsreprsns x no x nc 
		batch_mask3d = get_ent_mask3d(ent_numbers)		# bs x ns x no x no1
		
		# sib
		m1 = torch.eye(ne,dtype=torch.int,device=self.args.device).unsqueeze(0).repeat(bs,1,1)	# bs x no x no1	, r(i,j) not att to r(i,j)
		m1 = m1.unsqueeze(-3).repeat(1,ne,1,1).bool()					# bs x ns x no x no1
		m1 = (m1 + ~batch_mask3d).bool()
		m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)	# for sib,  bs x ns x no x 2no1
		# cop
		m2 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
		m2 = m2.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
		m2 = (m2 + ~batch_mask3d).bool()
		m2 = torch.cat((m2,m2), dim=-1).reshape(bs, ne, ne, -1)			# bs x ns x no x 2ns1
		# gp
		m3 = ~batch_mask3d
		m3 = torch.cat((m3,m3), dim=-2).reshape(bs, ne, ne, -1)			# bs x ns x no x (no1+ns1)
		
		attn_mask = torch.cat((m1,m2,m3), dim=-1).reshape(bs, ne, ne, -1)

		energy = energy.masked_fill(attn_mask, -1e4)

		attention = energy.softmax(dim=-1) 			# bs x ns x no x nc
		output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))	# ns x no x dm

		output = self.dropout(self.fc(output)) + res
		output = self.layernorm(output)

		return output


	def forward(self, *inputs):
		if self.factor_type in {'sib', 'cop','gp'}:
			return self.update_single(*inputs)
		elif self.factor_type in {'sibcop', 'sibgp', 'copgp'}:
			return self.update_double(*inputs)
		elif self.factor_type == 'sibcopgp':
			return self.update_triple(*inputs)
		else:
			raise ValueError('factor_type is not correct')

class HyperGNNHybridAggregateLayer(nn.Module):
	def __init__(self, ent_dim, rel_dim, dropout, args):
		super(HyperGNNHybridAggregateLayer, self).__init__()

		self.factor_type = args.factor_type
		mem_dim = args.mem_dim
		self.args = args
		layernorm = args.layernorm
		self.dropout = nn.Dropout(dropout)
		
		self.proj_s = nn.Linear(ent_dim, mem_dim)
		self.proj_o = nn.Linear(ent_dim, mem_dim)

		if self.args.attn_self:
			self.proj_r = nn.Linear(rel_dim, mem_dim)


		self.attn_combine_s = nn.Linear(mem_dim + ent_dim, mem_dim)
		self.attn_combine_o = nn.Linear(mem_dim + ent_dim, mem_dim)
		self.attn_combine_r = nn.Linear(mem_dim + rel_dim, mem_dim)
		# if args.attn_encoder=='nonlinear':
		self.attn_combine_s = nn.Sequential(self.attn_combine_s, nn.GELU())
		self.attn_combine_o = nn.Sequential(self.attn_combine_o, nn.GELU())
		self.attn_combine_r = nn.Sequential(self.attn_combine_r, nn.GELU())
			
		self.sv = nn.Linear(mem_dim, 1, bias=False)
		self.ov = nn.Linear(mem_dim, 1, bias=False)
		self.rv = nn.Linear(mem_dim, 1, bias=False)

		self.fc_s = nn.Linear(mem_dim, ent_dim)
		self.fc_o = nn.Linear(mem_dim, ent_dim)
		self.fc_r = nn.Linear(mem_dim, rel_dim)

		self.layernorm_s = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
		self.layernorm_o = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
		self.layernorm_r = nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()

	def update_rel_single(self, rel_reprs, factors, ent_numbers):
		"""
		rel_reprs: bs x ns x no x dr
		factor_ter: bs x ns x no x d
		sib:
			ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
		cop:
			ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
		gp:
			ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
		return updated ht
		"""	
		
		factor_ter, factor_b1 = factors
		bs, ne, _, dm = rel_reprs.shape
		res = rel_reprs

		ter_ht_new = factor_ter.unsqueeze(-2)			# bs x ns x no x 1 x dm

		# ablation
		if self.factor_type =='tersib':
			ha = factor_b1
			hb = factor_b1.permute(0,1,3,2,4)
		elif self.factor_type=='tercop':
			ha = factor_b1.permute(0,1,3,2,4)
			hb = factor_b1.permute(0,2,3,1,4)
		elif self.factor_type=='tergp':
			ha = factor_b1
			hb = factor_b1.permute(0,2,3,1,4)
		b1_ht_new = torch.cat((ha, hb), dim=-2)
		
		total_h = torch.cat([ter_ht_new, b1_ht_new], dim=-2)  			# bs x ns x no x nc x dm, for sib, nc=1+2no1, for cop, nc = 1+2ns1

		if self.args.attn_self:
			ht = self.proj_r(rel_reprs).unsqueeze(-2)				# bs x ns x no x 1 x dm
			total_h = torch.cat([ht, total_h], dim=-2)				# bs x ns x no x (1+nc) x dm
			
			
		ht = rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()			# ns x no x nc x dm
		comb = torch.cat([ht, total_h], dim=-1)											# ns x no x nc x 2*dm

		energy = self.attn_combine_r(comb)    		# ns x no x nc x dm
		energy = self.rv(energy).squeeze(-1)     	# ns x no x nc 
		# attn_mask = torch.sum(ht, dim=-1) == 0		# ns x no x nc
		batch_mask2d = get_ent_mask2d(ent_numbers).to(self.args.device)
		batch_mask3d = get_ent_mask3d(ent_numbers)		# bs x ns x no x no1
		mask_ter = ~batch_mask2d.unsqueeze(-1)							# bs x ns x no x 1
		if self.factor_type=='tersib':
			m1 = torch.eye(ne,dtype=torch.int,device=self.args.device).unsqueeze(0).repeat(bs,1,1)	# bs x no x no1	, r(i,j) not att to r(i,j)
			m1 = m1.unsqueeze(-3).repeat(1,ne,1,1).bool()					# bs x ns x no x no1
			m1 = (m1 + ~batch_mask3d).bool()
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)		# bs x ns x no x 2no1
		elif self.factor_type=='tercop':
			m1 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
			m1 = m1.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
			m1 = (m1 + ~batch_mask3d).bool()
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2ns1
		elif self.factor_type=='tergp':
			m1 = ~batch_mask3d
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2no
		attn_mask = torch.cat((mask_ter,m1), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x (1+2ns1)

		if self.args.attn_self:
			attn_self = torch.zeros((bs, ne, ne, 1), device=self.args.device).bool()	
			attn_mask = torch.cat((attn_self, attn_mask), axis=-1)				# bs x ns x no x (2+2ns1)

		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-1) 												# bs x ns x no x nc
		output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))	# bs x ns x no x dm

		output = self.dropout(self.fc_r(output)) + res
		output = self.layernorm_r(output)

		return output


	def update_rel_double(self, rel_reprs, factors, ent_numbers):
		"""
		rel_reprs: bs x ns x no x dr
		factor_ter: bs x ns x no x d
		factors for 
		sib:
			ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
		cop:
			ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
		gp:
			ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
		factor_ter, factor_b1, factor_b2 = factors
		res = rel_reprs
		bs, ne, _, _ = rel_reprs.shape
		
		ter_ht_new = factor_ter.unsqueeze(-2)			# bs x ns x no x 1 x dm

		if self.factor_type in {'tersibcop', 'tersibgp'}:
			f1_ha1 = factor_b1									
			f1_hb1 = factor_b1.permute(0,1,3,2,4)	
		elif self.factor_type == 'tercopgp':
			f1_ha1 = factor_b1.permute(0,1,3,2,4)					
			f1_hb1 = factor_b1.permute(0,2,3,1,4)	
		f1_h = torch.cat([f1_ha1, f1_hb1], dim=-2)

		if self.factor_type in {'tercopgp', 'tersibgp'}:
			f2_ha1 = factor_b2									# bs x ns x no x no1 x d	gp
			f2_hb1 = factor_b2.permute(0,2,3,1,4)	
		elif self.factor_type =='tersibcop':
			f2_ha1 = factor_b2.permute(0,1,3,2,4)					
			f2_hb1 = factor_b2.permute(0,2,3,1,4)	
		f2_h = torch.cat([f2_ha1, f2_hb1], dim=-2)


		# b1_h = torch.cat((b1_ha1, b1_hb1), dim=-2)					# bs x ns x no x nc1	nc1=2no1 sib; 2ns1 cop; no1+ns1 gp;
		# b2_h = torch.cat((b2_ha1, b2_hb1), dim=-2)					# bs x ns x no x nc2	nc2=2ns1 cop; no1+ns1 gp;
		ht_new = torch.cat([f1_h, f2_h], dim=-2)					# bs x ns x no x nc, nc= 2no1+2ns1 sibcop; 3no1+ns1 sibgp; 3ns1+no1 copgp

		total_h = torch.cat([ter_ht_new, ht_new], dim=-2)			# bs x ns x no x 1+nc x d

		if self.args.attn_self:
			ht = self.proj_r(rel_reprs).unsqueeze(-2)								# bs x ns x no x 1 x dm
			total_h = torch.cat([ht, total_h], dim=-2)				# bs x ns x no x (1+nc) x dm

		ht = rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()			# bs x ns x no x nc x dm,   
		comb = torch.cat([ht, total_h], dim=-1)											# bs x ns x no x nc x 2*dm

		energy = self.attn_combine_r(comb)    		# ns x no x nc x dm
		energy = self.rv(energy).squeeze(-1)     	# ns x no x nc
		
		# attn mask
		batch_mask3d = get_ent_mask3d(ent_numbers)		# bs x ns x no x no1
		mask_ter = ~batch_mask.unsqueeze(-1)							# bs x ns x no x 1

		if self.args.factor_type in {'tersibcop', 'tersibgp'}:
			m1 = torch.eye(ne,dtype=torch.int,device=self.args.device).unsqueeze(0).repeat(bs,1,1)	# bs x no x no1	, r(i,j) not att to r(i,j)
			m1 = m1.unsqueeze(-3).repeat(1,ne,1,1).bool()					# bs x ns x no x no1
			m1 = (m1 + ~batch_mask3d).bool()
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)		# bs x ns x no x 2no1
		elif self.args.factor_type == 'tercopgp':
			m1 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
			m1 = m1.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
			m1 = (m1 + ~batch_mask3d).bool()
			m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2ns1

		if self.args.factor_type in {'tercopgp', 'tersibgp'}:
			m2 = ~batch_mask3d
			m2 = torch.cat((m2,m2), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2no
		elif self.args.factor_type == 'tersibcop':
			m2 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
			m2 = m2.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
			m2 = (m2 + ~batch_mask3d).bool()
			m2 = torch.cat((m2,m2), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2ns1

		attn_mask = torch.cat((mask_ter, m1, m2), dim=-1).reshape(bs, ne, ne, -1)		# bs x ns x no x (1+nc)

		if self.args.attn_self:
			attn_self = torch.zeros((bs, ne, ne, 1), device=self.args.device).bool()	
			attn_mask = torch.cat((attn_self, attn_mask), axis=-1)				# bs x ns x no x (2+nc)


		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-1) 		# bs x ns x no x nc

		output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))	# bs x ns x no x dm

		output = self.dropout(self.fc_r(output)) + res
		output = self.layernorm_r(output)

		return output

	def update_rel_triple(self, rel_reprs, factors, ent_numbers):
		"""
		rel_reprs: bs x ns x no x d
		factor_ter: bs x ns x no x d
		factors for 
		sib:
			ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
		cop:
			ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
		gp:
			ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)

		factor_ter, factor1, factor2, factor3 = factors
		res = rel_reprs
		bs, ne, _, _ = rel_reprs.shape
		
		ter_ht_new = factor_ter.unsqueeze(-2)			# bs x ns x no x 1 x dm
		# sib
		f1_ha1 = factor1									# bs x ns x no x no1 x d	sib
		f1_hb1 = factor1.permute(0,1,3,2,4)		
		# cop
		f2_ha1 = factor2.permute(0,1,3,2,4)					# bs x ns x no x ns1 x d	cop
		f2_hb1 = factor2.permute(0,2,3,1,4)	
		# gp
		f3_ha1 = factor3									# bs x ns x no x no1 x d	gp
		f3_hb1 = factor3.permute(0,2,3,1,4)	
		# b1_ha1 = factor_b1							# bs x ns x no x no1 x d
		# b1_hb1 = factor_b1.permute(0,1,3,2,4)		# bs x ns x no x no1 x d
		# b2_ha1 = factor_b2.permute(0,1,3,2,4)		# bs x ns x no x ns1 x d
		# b2_hb1 = factor_b2.permute(0,2,3,1,4)		# bs x ns x no x ns1 x d
		# b3_ha1 = factor_b3							# bs x ns x no x no1 x d
		# b3_hb1 = factor_b3.permute(0,2,3,1,4)		# bs x ns x no x ns1 x d


		b1_h = torch.cat((f1_ha1, f1_hb1), dim=-2)					# bs x ns x no x nc1	nc1=2no1 sib; 2ns1 cop; no1+ns1 gp;
		b2_h = torch.cat((f2_ha1, f2_hb1), dim=-2)					# bs x ns x no x nc2	nc2=2ns1 cop; no1+ns1 gp;
		b3_h = torch.cat((f3_ha1, f3_hb1), dim=-2)
		ht_new = torch.cat([b1_h, b2_h, b3_h], dim=-2)					# bs x ns x no x nc, nc= 2no1+2ns1 sibcop; 3no1+ns1 sibgp; 3ns1+no1 copgp

		total_h = torch.cat([ter_ht_new, ht_new], dim=-2)			# bs x ns x no x 1+nc x d

		if self.args.attn_self:
			ht = self.proj_r(rel_reprs).unsqueeze(-2)								# bs x ns x no x 1 x dm
			total_h = torch.cat([ht, total_h], dim=-2)				# bs x ns x no x (1+nc) x dm

		ht = rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()			# bs x ns x no x nc x dm,   
		comb = torch.cat([ht, total_h], dim=-1)											# bs x ns x no x nc x 2*dm

		energy = self.attn_combine_r(comb)    		# ns x no x nc x dm
		energy = self.rv(energy).squeeze(-1)     	# ns x no x nc
		
		# attn mask
		batch_mask3d = get_ent_mask3d(ent_numbers)		# bs x ns x no x no1
		mask_ter = ~batch_mask.unsqueeze(-1)							# bs x ns x no x 1

		#sib
		m1 = torch.eye(ne,dtype=torch.int,device=self.args.device).unsqueeze(0).repeat(bs,1,1)	# bs x no x no1	, r(i,j) not att to r(i,j)
		m1 = m1.unsqueeze(-3).repeat(1,ne,1,1).bool()					# bs x ns x no x no1
		m1 = (m1 + ~batch_mask3d).bool()
		m1 = torch.cat((m1,m1), dim=-1).reshape(bs, ne, ne, -1)		# bs x ns x no x 2no1
		#cop
		m2 = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)			# bs x ns x ns1
		m2 = m2.unsqueeze(-2).repeat(1,1,ne,1).bool()						# bs x ns x no x ns1
		m2 = (m2 + ~batch_mask3d).bool()
		m2 = torch.cat((m2,m2), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2ns1
		# gp
		m3 = ~batch_mask3d
		m3 = torch.cat((m3,m3), dim=-1).reshape(bs, ne, ne, -1)	# bs x ns x no x 2no

		attn_mask = torch.cat((mask_ter, m1, m2, m3), dim=-1).reshape(bs, ne, ne, -1)		# bs x ns x no x (1+nc)

		if self.args.attn_self:
			attn_self = torch.zeros((bs, ne, ne, 1), device=self.args.device).bool()	
			attn_mask = torch.cat((attn_self, attn_mask), axis=-1)				# bs x ns x no x (2+nc)

		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-1) 		# bs x ns x no x nc
		output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))	# bs x ns x no x dm

		output = self.dropout(self.fc_r(output)) + res
		output = self.layernorm_r(output)

		return output


	
	def update_sub(self, sub_reprs, factor_ter, ent_numbers):
		"""
		sub_reprs: bs x ns x dm
		factor_ter: bs x ns x no x dm
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
		res = sub_reprs
		bs, ne, _, dm = factor_ter.shape
		ht = self.proj_s(res)
		total_h = torch.cat([ht.unsqueeze(-2), factor_ter], dim=-2) if self.args.attn_self else factor_ter	# ns x (1+no) x dm or ns x no x dm

		# total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
		# total_h = factor_ter
		ht = sub_reprs.unsqueeze(-2).repeat(1, 1, total_h.shape[-2], 1).contiguous()		# bs x ns x no x dm
		comb = torch.cat([ht, total_h], dim=-1)							# bs x ns x no x dm+dr

		energy = self.attn_combine_s(comb)    							# bs x ns x no x dm
		energy = self.sv(energy).squeeze(-1)     						# bs x ns x no 
		attn_mask = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)						# bs x ns x ns
		attn_mask = (attn_mask + ~batch_mask).bool()
		if self.args.attn_self:
			attn_self = torch.zeros((bs, ne, 1), device=self.args.device).bool()		#  bs x ne x 1
			attn_mask = torch.cat((attn_self, attn_mask), axis=-1)	
		# attn_mask = torch.sum(ht, dim=-1) == 0							# ns x (no+1)
		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-1)								# ns x no
		output = torch.einsum('bij,bijd->bid', attention, total_h.to(attention.dtype))		# ns x dm
		output = self.dropout(self.fc_s(output)) + res
		output = self.layernorm_s(output)

		return output

	def update_obj(self, obj_reprs, factor_ter, ent_numbers):
		"""
		factor_ter: bs x ns x no x dm
		obj_reprs: bs x no x dm
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
		res = obj_reprs
		bs, ne, _, dm = factor_ter.shape
		# total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
		ht = self.proj_o(res)
		total_h = torch.cat([ht.unsqueeze(-3), factor_ter], dim=-3) if self.args.attn_self else factor_ter	# bs x (1+ne) x ne x dm or bs x ne x ne x dm

		ht = obj_reprs.unsqueeze(-3).repeat(1, total_h.shape[-3], 1, 1).contiguous()		
		comb = torch.cat([ht, total_h], dim=-1)							# bs x nc x ne x dm+do

		energy = self.attn_combine_o(comb)    							# bs x nc x ne x dm
		energy = self.ov(energy).squeeze(-1)     						# bs x nc x ne 
		attn_mask = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)					# bs x ne x ne
		attn_mask = (attn_mask + ~batch_mask).bool()
		if self.args.attn_self:
			attn_self = torch.zeros((bs, 1, ne), device=self.args.device).bool()	 		# bs x 1 x ne
			attn_mask = torch.cat((attn_self, attn_mask), axis=-2)			# bs x (1+ne) x ne
		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-2)								# bs x (1+ne) x ne
		output = torch.einsum('bij,bijd->bjd', attention, total_h.to(attention.dtype))		# bs x no x dm
		output = self.dropout(self.fc_o(output)) + res
		output = self.layernorm_o(output)						# no x dm 

		return output

	def forward(self, *inputs):
		sub_reprs, obj_reprs, rel_reprs, factors, ent_numbers = inputs
		if self.factor_type in {'tersib', 'tercop','tergp'}:
			factor_ter = factors[0]
			sub_reprs = self.update_sub(sub_reprs, factor_ter, ent_numbers)
			obj_reprs = self.update_obj(obj_reprs, factor_ter, ent_numbers)  
			rel_reprs = self.update_rel_single(rel_reprs, factors, ent_numbers)
		elif self.factor_type in {'tersibcop', 'tersibgp', 'tercopgp'}:
			factor_ter = factors[0]
			sub_reprs = self.update_sub(sub_reprs, factor_ter, ent_numbers)
			obj_reprs = self.update_obj(obj_reprs, factor_ter, ent_numbers) 
			rel_reprs = self.update_rel_double(rel_reprs, factors, ent_numbers)
		elif self.factor_type == 'tersibcopgp':
			factor_ter = factors[0]
			sub_reprs = self.update_sub(sub_reprs, factor_ter, ent_numbers)
			obj_reprs = self.update_obj(obj_reprs, factor_ter, ent_numbers) 
			rel_reprs = self.update_rel_triple(rel_reprs, factors, ent_numbers)
		else:
			raise ValueError('factor_type is not correct')
			

		return sub_reprs, obj_reprs, rel_reprs









#----------------------------------------------
# MFVI
class MFVI(nn.Module):
	def __init__(self, ent_dim, rel_dim, mem_dim, n_ent_labels, n_rel_labels, args):
		super().__init__()
		self.args = args
		self.ent_dim = ent_dim
		self.iter = args.iter
		self.re = n_ent_labels
		self.rr = n_rel_labels

		self.ter_scorer =CPDTrilinear(input_dim1=ent_dim, input_dim2=ent_dim, input_dim3=rel_dim,
										rank=mem_dim, output_dim=n_ent_labels**2*n_rel_labels)

		self.bin_scorer = BiafEncoder(input_dim1=rel_dim, input_dim2=rel_dim, 
											output_dim=n_rel_labels**2, 
											rank=mem_dim, factorize=True)
											
	def _ter_potential(self, qs, qo, qr, jointscores):
		"""
		qs, qo, qr: maskd distributions
		qs: bs x ni x nei
		qo: bs x nj x nej
		qr: bs x ni x nj x nrij
		jointscores: bs x ni x nj x nei x nej x nrij
		return:
		Fs : bs x ns x nse
		Fo : bs x no x noe
		Fr : bs x ns x no x nr
		"""
		# bs x ni x nj x nrij --> bs x ni x nj x 1 x 1 x nrij
		qr = qr.unsqueeze(-2).unsqueeze(-2)
		# bs x ni x nei --> bs x ni x nj x nei x nej
		qs = qs.unsqueeze(-2).unsqueeze(-1)
		# bs x 1 x no x 1 x nso
		qo = qo.unsqueeze(-3).unsqueeze(-2)
		# bs x ns x no x nse x nso
		Fso = (qr*jointscores).sum(-1)		
		# bs x ns x nse
		Fs = (qo*Fso).sum(axis=(-3,-1))
		# bs x no x noe
		Fo = (qs*Fso).sum(axis=(-4,-2))
		# bs x ns x no x nr
		Fr = (qs.unsqueeze(-1)*qo.unsqueeze(-1)*jointscores).sum(axis=(-2,-3))
		
		
		return Fs, Fo, Fr

	def _sib_potential(self, qr, jointscores):
		"""
		qr: bs x ns x no x nr
		jointscores: bs x ni x nj x nk x nrij x nrik; j,k are objects
		"""
		bs, ne, _, _, nr, _ = jointscores.shape
		qrij = qr.unsqueeze(-2).unsqueeze(-1).repeat(1,1,1,ne,1,nr)		# qr: bs x ni x nj x nr --> bs x ni x nj x nk x nr x nrjk
		qrik = qr.unsqueeze(-3).unsqueeze(-2).repeat(1,1,ne,1,nr,1)		# qr: bs x ni x nk x nr

		Frij = (qrik*jointscores).sum(axis=(-3,-1))						# bs x ns x no x nr
		Frik = (qrij*jointscores).sum(axis=(-4,-2))
		
		return Frij + Frik
	
	def _cop_potential(self, qr, jointscores):
		"""
		qr: bs x ns x no x nr
		jointscores: bs x ni x nj x nk x nrik x nrjk, k is the obj dim.
		"""
		bs, ne, _, _, nr, _ = jointscores.shape
		qrik = qr.unsqueeze(-3).unsqueeze(-1).repeat(1,1,ne,1,1,nr)		# qr: bs x ni x nk x nrik --> bs x ni x nj x nk x nrik x nrjk
		qrjk = qr.unsqueeze(-4).unsqueeze(-2).repeat(1,ne,1,1,nr,1)		# qr: bs x nj x nk x nrjk --> bs x ni x nj x nk x nrik x nrjk

		Frik = (qrik*jointscores).sum(axis=(-4,-1))						# bs x ni x nk x nrik
		Frjk = (qrjk*jointscores).sum(axis=(-5,-2))
		
		return Frik + Frjk

	def _gp_potential(self, qr, jointscores):
		"""
		qr: bs x ns x no x nr
		jointscores: bs x ni x nj x nk x nrij x nrjk; j is obj for rij, jis sub for rjk, k is obj dim
		
		"""
		bs, ne, _, _, nr, _ = jointscores.shape
		qrij = qr.unsqueeze(-2).unsqueeze(-1).repeat(1,1,1,ne,1,nr)			# qr: bs x ni x nj x nrij --> bs x ni x nj x nk x nrij x nrjk
		qrjk = qr.unsqueeze(-4).unsqueeze(-2).repeat(1,ne,1,1,nr,1)			# qr: bs x nj x nk x nrjk

		Frij = (qrij*jointscores).sum(axis=(-3,-1))
		Frjk = (qrjk*jointscores).sum(axis=(-5,-2))

		return Frij + Frjk


	def mfvi_ternary(self, sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers):
		"""
		subscores: bs x ns x nse
		objscores: bs x no x nso
		relscores: bs x ns x no x nr
		jointscores: bs x ns x no x nse x noe x nr
		"""
		bs, ne, _ = sub_reprs.shape
		batch_mask1d = get_ent_mask1d(ent_numbers)	# bs x ns(no)
		batch_mask2d = get_ent_mask2d(ent_numbers)	# bs x ns x no
		jointscores = self.ter_scorer(sub_reprs.unsqueeze(-2).expand(rel_reprs.shape), obj_reprs.unsqueeze(-2).expand(rel_reprs.shape), rel_reprs).reshape(bs, ne, ne, self.re, self.re, self.rr)			# ns x no x nse x noe x nr
		jointscores *= batch_mask2d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		qsv = subscores.clone()
		qov = objscores.clone()
		qrv = relscores.clone()

		for i in range(self.iter):
			qsv = qsv.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
			qov = qov.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
			qrv = qrv.masked_fill(~batch_mask2d.unsqueeze(-1), -1e4)
			qs = qsv.softmax(dim=-1)		# ns x nse
			qo = qov.softmax(dim=-1)		# no x noe
			qr = qrv.softmax(dim=-1)		# ns x no x nr
			Fs, Fo, Fr = self._ter_potential(qs, qo, qr, jointscores)
			#update scores
			qsv = qsv + Fs
			qov = qov + Fo
			qrv = qrv + Fr

		return qsv, qov, qrv



	def mfvi_hybrid(self, sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers):
		"""
		subscores: bs x ns x nse
		objscores: bs x no x nso
		relscores: bs x ns x no x nr

		"""
		bs, ne, _ = sub_reprs.shape
		batch_mask1d = get_ent_mask1d(ent_numbers)	# bs x ns(no)
		batch_mask2d = get_ent_mask2d(ent_numbers)	# bs x ns x no
		batch_mask3d = get_ent_mask3d(ent_numbers)
		qsv = subscores.clone()
		qov = objscores.clone()
		qrv = relscores.clone()

		ter_scores = self.ter_scorer(sub_reprs.unsqueeze(-2).expand(rel_reprs.shape), obj_reprs.unsqueeze(-2).expand(rel_reprs.shape), rel_reprs).reshape(bs, ne, ne, self.re, self.re, self.rr)			# ns x no x nse x noe x nr
		ter_scores *= batch_mask2d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		if 'sib' in self.args.factor_type:
			sib_scores = self.bin_scorer(rel_reprs.unsqueeze(-2).repeat(1,1,1,ne,1), rel_reprs.unsqueeze(-3).repeat(1,1,ne,1,1)	).reshape(bs, ne, ne, ne, self.rr, self.rr)
			sib_scores *= batch_mask3d.unsqueeze(-1).unsqueeze(-1)
		if 'cop' in self.args.factor_type:
			cop_scores = self.bin_scorer(rel_reprs.unsqueeze(-3).repeat(1,1,ne,1,1), rel_reprs.unsqueeze(-4).repeat(1,ne,1,1,1)).reshape(bs, ne, ne, ne, self.rr, self.rr)
			cop_scores *= batch_mask3d.unsqueeze(-1).unsqueeze(-1)
		if 'gp' in self.args.factor_type:
			gp_scores = self.bin_scorer(rel_reprs.unsqueeze(-2).repeat(1,1,1,ne,1), rel_reprs.unsqueeze(-4).repeat(1,ne,1,1,1)).reshape(bs, ne, ne, ne, self.rr, self.rr)
			gp_scores *= batch_mask3d.unsqueeze(-1).unsqueeze(-1)

		for i in range(self.iter):
			qsv = qsv.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
			qov = qov.masked_fill(~batch_mask1d.unsqueeze(-1), -1e4)
			qrv = qrv.masked_fill(~batch_mask2d.unsqueeze(-1), -1e4)
			qs = qsv.softmax(dim=-1)		# ns x nse
			qo = qov.softmax(dim=-1)		# no x noe
			qr = qrv.softmax(dim=-1)		# ns x no x nr
			ter_fs, ter_fo, ter_fr = self._ter_potential(qs, qo, qr, ter_scores)
			frs = []
			if 'sib' in self.args.factor_type:
				frs.append(self._sib_potential(qr, sib_scores))
			if 'cop' in self.args.factor_type:
				frs.append(self._cop_potential(qr, cop_scores))
			if 'gp' in self.args.factor_type:
				frs.append(self._gp_potential(qr, gp_scores))
			bin_fr = sum(frs)

			
			#update scores
			qsv = qsv + ter_fs
			qov = qov + ter_fo
			qrv = qrv + ter_fr + bin_fr

		return qsv, qov, qrv



	def forward(self, sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers):
		"""
		sub_reprs: bs x ns x d
		obj_reprs: bs x no x d
		rel_reprs: bs x ns x no x d
		subscores: bs x ns x nse
		objscores: bs x no x nso
		relscores: bs x ns x no x nr
		"""
		if self.args.factor_type=='ternary':
			subscores, objscores, relscores = self.mfvi_ternary(sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers)
		elif self.args.factor_type in {'tersib', 'tercop', 'tergp', 'tersibcop', 'tersibgp', 'tercopgp', 'tersibcopgp'}:
			subscores, objscores, relscores = self.mfvi_hybrid(sub_reprs, obj_reprs, rel_reprs, subscores, objscores, relscores, ent_numbers)
		else:
			raise ValueError('We do not experiment on binary config')
		
		return subscores, objscores, relscores







#----------------------------------------------
# GNN with no hyper-edge



class GNN(nn.Module):

	def __init__(self, ent_dim, rel_dim, dropout, args):
		super().__init__()
		self.args = args
		self.iter = args.iter
		mem_dim = args.mem_dim
		layernorm = args.layernorm
		self.dropout = nn.Dropout(dropout)

		
		self.proj_kv_s = nn.Linear(ent_dim, mem_dim)
		self.proj_kv_o = nn.Linear(ent_dim, mem_dim)
		self.proj_kv_rs = nn.Linear(rel_dim, mem_dim)
		self.proj_kv_ro = nn.Linear(rel_dim, mem_dim)
		self.attn_combine_s = nn.Linear(mem_dim + ent_dim, mem_dim)
		self.attn_combine_o = nn.Linear(mem_dim + ent_dim, mem_dim)
		
		self.attn_combine_s = nn.Sequential(self.attn_combine_s, nn.GELU())
		self.attn_combine_o = nn.Sequential(self.attn_combine_o, nn.GELU())
			
		self.sv = nn.Linear(mem_dim, 1, bias=False)
		self.ov = nn.Linear(mem_dim, 1, bias=False)

		self.fc_s = nn.Linear(mem_dim, ent_dim)
		self.fc_o = nn.Linear(mem_dim, ent_dim)

		# if self.args.attn_self:
		self.proj_kv_r = nn.Linear(rel_dim, mem_dim)
		self.proj_kv_sr = nn.Linear(ent_dim, mem_dim)
		self.proj_kv_or = nn.Linear(ent_dim, mem_dim)
		self.attn_combine_r = nn.Linear(mem_dim + rel_dim, mem_dim)
		self.attn_combine_r = nn.Sequential(self.attn_combine_r, nn.GELU())
		self.rv = nn.Linear(mem_dim, 1, bias=False)
		self.fc_r = nn.Linear(mem_dim, rel_dim)
		# else:
		# 	self.encode_r = LinearMessegePasser(ent_dim, rel_dim)

		self.layernorm_s = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
		self.layernorm_o = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
		self.layernorm_r = nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()

	def update_rel(self, sub_reprs, obj_reprs, rel_reprs):
		"""
		sub_reprs: bs x ns x de
		obj_reprs: bs x no x de
		rel_reprs: bs x ns x no x dr
		return updated rel_reprs
		"""
		# ht = self._apply_mask(ht, mask)
		# ht_new = self._apply_mask(ht_new, mask)
		res = rel_reprs
		bs, ne, _, _ = rel_reprs.shape
		hs = self.proj_kv_sr(sub_reprs).unsqueeze(-2).repeat(1,1,ne,1).unsqueeze(-2)			# bs x ns x no x 1 x dm
		ho = self.proj_kv_or(obj_reprs).unsqueeze(-3).repeat(1,ne,1,1).unsqueeze(-2)			# bs x ns x no x 1 x dm
		if self.args.attn_self:
			hr = self.proj_kv_r(rel_reprs).unsqueeze(-2)											# bs x ns x no x 1 x dm
			total_h = torch.cat([hr, hs, ho], dim=-2)		# bs x ns x no x 3 x dm
		else:
			total_h = torch.cat([hs,ho], dim=-2)
		ht = rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()		# bs x ns x no x 3 x dr
		comb = torch.cat([ht, total_h], dim=-1)												# bs x ns x no x 3 x (dr + dm)
		energy = self.attn_combine_r(comb)    												# bs x ns x no x 3 x 1
		energy = self.rv(energy).squeeze(-1) 												# bs x ns x no x 3
		attention = energy.softmax(dim=-1)													# bs x ns x no x 3
		output = torch.einsum('bijk,bijkd->bijd', attention, total_h.to(attention.dtype))
		output = self.dropout(self.fc_r(output)) + res
		output = self.layernorm_r(output)
		# else:
		# 	output = self.dropout(self.encode_r(sub_reprs, obj_reprs)) + res
		# 	output = self.layernorm_r(output) 
		# output = torch.max(total_h, dim=-2)[0]

		return output

	def update_sub(self, sub_reprs, rel_reprs, ent_numbers):
		"""
		factor: bs x ns x no x dr
		sub_reprs: bs x ns x ds
		batch_mask: bs x ns x no
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)

		res = sub_reprs
		bs, ne, _, _ = rel_reprs.shape
		hs = self.proj_kv_s(res)				# bs x ne x dm
		hr = self.proj_kv_rs(rel_reprs)			# bs x ne x ne x dm
		total_h = torch.cat([hs.unsqueeze(-2), hr], dim=-2) if self.args.attn_self else hr	# bs x ne x (1+ne) x dm or bs x ne x ne x dm

		ht = sub_reprs.unsqueeze(-2).repeat(1, 1, total_h.shape[-2], 1).contiguous()		# bs x ne x ne x dm
		comb = torch.cat([ht, total_h], dim=-1)							# bs x ne x (1+ne) x dm+dr
		energy = self.attn_combine_s(comb)    							# bs x ne x (1+ne) x dm
		energy = self.sv(energy).squeeze(-1)     						# bs x ne x (1+ne) 
		attn_mask = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)						# bs x ne x ne
		attn_mask = (attn_mask + ~batch_mask).bool()					# bs x ne x ne
		if self.args.attn_self:
			attn_self = torch.zeros((bs, ne, 1), device=self.args.device).bool()		#  bs x ne x 1
			attn_mask = torch.cat((attn_self, attn_mask), axis=-1)			# bs x ne x (1+ne)

		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-1)								# ns x no
		output = torch.einsum('bij,bijd->bid', attention, total_h.to(attention.dtype))		# ns x dm
		output = self.dropout(self.fc_s(output)) + res
		output = self.layernorm_s(output)

		return output

	def update_obj(self, obj_reprs, rel_reprs, ent_numbers):
		"""
		factor: bs x ns x no x dm
		obj_reprs: bs x no x de
		"""
		batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
		res = obj_reprs
		bs, ne, _, _ = rel_reprs.shape
		ho = self.proj_kv_o(res)										# bs x ne x dm
		hr = self.proj_kv_ro(rel_reprs)									# bs x ne x ne x dm
		total_h = torch.cat([ho.unsqueeze(-3), hr], dim=-3) if self.args.attn_self else hr	# bs*(1+ne)*ne*dm
		
		ht = obj_reprs.unsqueeze(-3).repeat(1,total_h.shape[-3], 1, 1).contiguous()		# bs x (1+ne) x ne x de
		comb = torch.cat([ht, total_h], dim=-1)							# bs x (1+ne) x ne x (de+dm)

		energy = self.attn_combine_o(comb)    							# bs x (1+ne) x ne x dm
		energy = self.ov(energy).squeeze(-1)     						# bs x (1+ne) x ne
		attn_mask = torch.eye(ne,device=self.args.device).unsqueeze(0).repeat(bs,1,1)					# bs x ne x ne
		attn_mask = (attn_mask + ~batch_mask).bool()					# bs x ne x ne
		if self.args.attn_self:
			attn_self = torch.zeros((bs, 1, ne), device=self.args.device).bool()	 		# bs x 1 x ne
			attn_mask = torch.cat((attn_self, attn_mask), axis=-2)			# bs x (1+ne) x ne
				
		energy = energy.masked_fill(attn_mask, -1e4)
		attention = energy.softmax(dim=-2)								# bs x (1+ne) x ne
		output = torch.einsum('bij,bijd->bjd', attention, total_h.to(attention.dtype))		# bs x no x dm
		output = self.dropout(self.fc_o(output)) + res
		output = self.layernorm_o(output)

		return output 


	def aggregate(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
		"""
		rel_reprs: bs x ns x no x dr
		sub_reprs: bs x ns x de
		obj_reprs: no x de	
		rel_h, sub_h, obj_h: ns x no x dx
		"""
		sub_reprs_new = self.update_sub(sub_reprs, rel_reprs, ent_numbers)
		obj_reprs_new = self.update_obj(obj_reprs, rel_reprs, ent_numbers)
		rel_reprs_new = self.update_rel(sub_reprs, obj_reprs, rel_reprs)
		return sub_reprs_new, obj_reprs_new, rel_reprs_new




	def forward(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
		mask1d = get_ent_mask1d(ent_numbers)
		mask2d = get_ent_mask2d(ent_numbers)
		for i in range(self.iter):
			sub_reprs, obj_reprs, rel_reprs = self.aggregate(sub_reprs, obj_reprs, rel_reprs, ent_numbers)
			sub_reprs *= mask1d.unsqueeze(-1)
			obj_reprs *= mask1d.unsqueeze(-1)
			rel_reprs *= mask2d.unsqueeze(-1)
		return sub_reprs, obj_reprs, rel_reprs

