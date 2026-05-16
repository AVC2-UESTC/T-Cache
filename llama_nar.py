# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.cache_utils import Cache
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast


# sinusoidal positional encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)

## lazy import
from types import MethodType

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` containing the query and key tensors rotated using Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Custom Llama attention used by T-Cache to switch between full attention
# and partial prompt-cache attention during inference.
def custom_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = True,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    cache_dic=None, 
    current=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    assert current is not None, "current should not be None"
    
    # check the condition for partial attention
    if (current.get('prompt_prefix_len',None) is not None) and (current['step']>current['thres_step']) and current.get('should_cache',False) and current['prompt_cache']: 
        attention_type="partial"
    else: attention_type="full"

    ####################################FULL ATTENTION COMPUTATION ##############################
    if attention_type=="full":
        bsz, q_len, _ = hidden_states.size()
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        cache_dic['attn_map'][-1][current['layer']] = attn_weights  # save the attention weight in the cache
        
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
    
        # upcast attention to fp32            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)                
        attn_output = self.o_proj(attn_output)  
       
        if not output_attentions:
            attn_weights = None      

        return attn_output, attn_weights, past_key_value 
    
    ########################### PARTIAL ATTENTION COMPUTATION  ################################

    elif attention_type=="partial":
        bsz, q_len, _ = hidden_states.size()
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        prompt_prefix_len = current['prompt_prefix_len']
        q_prompt, q_input = query_states[:,:,:prompt_prefix_len,:], query_states[:,:,prompt_prefix_len:,:]
        k_prompt, k_input = key_states[:,:,:prompt_prefix_len,:], key_states[:,:,prompt_prefix_len:,:]        
        
        A_pi= cache_dic['attn_map'][-1][current['layer']][:,:,:prompt_prefix_len,prompt_prefix_len:]
        A_pp = cache_dic['attn_map'][-1][current['layer']][:,:,:prompt_prefix_len,:prompt_prefix_len]
        A_ip = torch.matmul(q_input, k_prompt.transpose(2, 3)) / math.sqrt(self.head_dim)
        A_ii = torch.matmul(q_input, k_input.transpose(2, 3)) / math.sqrt(self.head_dim)
           
        top = torch.cat([A_pp, A_pi], dim=3)
        bottom = torch.cat([A_ip, A_ii], dim=3)
               
        A_full = torch.cat([top, bottom], dim=2)

        assert A_full.shape == (bsz, self.num_heads, q_len, q_len),f"the size should match, but the shape is {A_full.shape}"
        
        attn_weights = A_full
        cache_dic['attn_map'][-1][current['layer']] = attn_weights

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :,:, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:            
            attn_output = self.o_proj(attn_output)  
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value
    
    else: raise f"Attention type is not defined. It should be either full or partial, but got {attention_type}"

class LlamaNARDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        """Override to adaptive layer norm"""
        super().__init__(config, layer_idx)  # init attention, mlp, etc.
        self.layer_idx = layer_idx
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        
        # Patch self_attn to behave as described in the T-Cache paper.
        self.self_attn.forward = MethodType(
            custom_attention_forward, self.self_attn
        )

    # add `cond` in forward function
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = True,
        use_cache: Optional[bool] = False,
        current = None,
        cache_dic = None,
        
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # Check whether the forward function being called is the patched one.
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        if current.get('use_baseline',False):
            layer = current.get('layer',None)
            assert layer is not None, "layer should not be None"
        
            residual = hidden_states
            hidden_states = self.input_layernorm(
                hidden_states, cond_embedding=cond_embedding
            )
            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_dic=cache_dic, 
                current=current,
            )

            hidden_states = residual + hidden_states  

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(
                hidden_states, cond_embedding=cond_embedding
            )

            hidden_states = self.mlp(hidden_states)                 
            hidden_states = residual + hidden_states   
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)
            return outputs
           
        elif current.get('use_tcache',False):
            
            step = current.get('step',None)
            layer = current.get('layer',None)
            cache_steps = current.get('cache_steps',None)            
            prompt_prefix_len = current.get('prompt_prefix_len',None)   # prompt_prefix_len can be None
            thres_step = current.get('thres_step',None)  # threshold step to start to activate prompt-reuse method
            
            
            assert cache_steps is not None,f"cache_steps should not be None, but got {cache_steps}"
            assert step is not None and layer is not None, "step and layer should not be None"
            assert thres_step is not None , "thres_step should not be None"
                       
            if step in cache_steps:                
                should_cache = True
                current['should_cache'] = True  # Save it for cfg computation.
            else:
                should_cache = False
                current['should_cache'] = False  # Save it for cfg computation.
            
            if should_cache :
                current['module'] = 'attn'
                residual = hidden_states
                
                hidden_states = self.input_layernorm(
                    hidden_states, cond_embedding=cond_embedding
                )

                # self_attn.forward has been patched to custom_attention_forward.
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_dic=cache_dic, 
                    current=current,
                )

                cache_dic['cache'][-1][layer]['attn'] = hidden_states                 
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states                
                
                hidden_states = self.post_attention_layernorm(
                    hidden_states, cond_embedding=cond_embedding
                )

                current['module'] = 'mlp'
                
                if  step > thres_step and current['should_cache'] and prompt_prefix_len is not None and current['prompt_cache']:  
                    # This applies prompt reuse at the MLP layer.
                    a = cache_dic['cache'][-1][layer]['mlp'][:,:prompt_prefix_len,:]
                    b = self.mlp(hidden_states[:,prompt_prefix_len:,:])
                    hidden_states=torch.cat([a, b], dim=1)                            
                else :
                    hidden_states = self.mlp(hidden_states) 

                cache_dic['cache'][-1][layer]['mlp'] = hidden_states  
                hidden_states = residual + hidden_states
                outputs = (hidden_states,)

                if output_attentions:
                    outputs += (self_attn_weights,)

                if use_cache:
                    outputs += (present_key_value,)

                return outputs
            
            else:            
                residual = hidden_states
                current['module'] = 'attn'
                
                hidden_states = cache_dic['cache'][-1][layer]['attn']
                self_attn_weights = cache_dic['attn_map'][-1][layer]                
                present_key_value=None
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states                 
                hidden_states = self.post_attention_layernorm(
                    hidden_states, cond_embedding=cond_embedding
                )                  
                mlp_output = cache_dic['cache'][-1][layer]['mlp']                
                hidden_states = residual + mlp_output
                outputs = (hidden_states,)
                if output_attentions:
                    outputs += (self_attn_weights,)
                if use_cache:
                    outputs += (present_key_value,)
                return outputs
        
        else : raise "Caching mode is not defined."

class DiffLlama(LlamaModel):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        config=LlamaConfig(0, 256, 1024, 1, 1),
    ):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    LlamaConfig(
                        hidden_size=hidden_size,
                        num_attention_heads=num_heads,
                        max_position_embeddings=4096,
                        intermediate_size=hidden_size * 4,
                    ),
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # self.position_embedding = PositionalEncoding(hidden_size, dropout=0.0)

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )

        self.post_init()

        # self.reset_parameters()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create noncausal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        x,
        diffusion_step,
        cond,
        x_mask,
        input_ids: torch.LongTensor = None,  # [num_quant, B, T]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        current=None,
        cache_dic=None,
        
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # retrieve some shape info
        batch_size, seq_length, _ = x.shape

        # condition mlp
        cond_embedding = self.cond_mlp(cond)  # (B, T, C)

        # diffusion step embedding
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step = self.diff_step_mlp(diffusion_step)  # (B, C)
        x = x + cond_embedding

        inputs_embeds = x
        attention_mask = x_mask

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):            
            current['layer'] = idx                   
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cond_embedding=diffusion_step,
                    current= current,
                    cache_dic=cache_dic,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states


class DiffLlamaPrefix(LlamaModel):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        use_phone_cond=True,
        config=LlamaConfig(0, 256, 1024, 1, 1),
    ):
        super().__init__(config)

        self.use_phone_cond = use_phone_cond
        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    LlamaConfig(
                        hidden_size=hidden_size,
                        num_attention_heads=num_heads,
                        max_position_embeddings=4096,
                        intermediate_size=hidden_size * 4,
                    ),
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )
        
        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        if self.use_phone_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )

        self.embed_tokens = None

        self.post_init()
        

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create noncausal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        x,
        diffusion_step,
        x_mask,
        phone_embedding: Optional[torch.LongTensor] = None,
        phone_mask: Optional[torch.FloatTensor] = None,
        input_ids: torch.LongTensor = None,  # [num_quant, B, T]
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        current=None,
        cache_dic=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # retrieve some shape info
        if self.use_phone_cond and phone_embedding is not None:
            phone_embedding = self.cond_mlp(phone_embedding)  # (B, T, C)
            phone_length = phone_embedding.shape[1]
            inputs_embeds = torch.cat([phone_embedding, x], dim=1)
            attention_mask = torch.cat([phone_mask, x_mask], dim=1)
        else:
            inputs_embeds = x
            attention_mask = x_mask
            phone_length = 0

        # diffusion step embedding
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step = self.diff_step_mlp(diffusion_step)  # (B, C)

        batch_size, seq_length, _ = inputs_embeds.shape

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None        
        
        for idx, decoder_layer in enumerate(self.layers):

            current['layer'] = idx # assign the layer idx 


            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cond_embedding=diffusion_step,
                    current= current,
                    cache_dic=cache_dic,

                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
   


        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return hidden_states[
            :,
            phone_length:,
        ]
