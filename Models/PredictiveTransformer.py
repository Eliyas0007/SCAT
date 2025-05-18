import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from .activations import Swish
from .convolution import ConformerConvModule

class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio) -> None:
        super().__init__()

        self.MLP = nn.Sequential(*[
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            Swish(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim * mlp_ratio),
            Swish(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        ])

    def forward(self, x):
        return self.MLP(x)
    
class Blocks(nn.Module):
    def __init__(self, embed_dim, num_head, mlp_ratio, num_cls, dropout, conf=False, cross_attention=True) -> None:
        super(Blocks, self).__init__()

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        # spacial self attention and cross attention
        self.attention_spatial_sa = nn.MultiheadAttention(embed_dim, num_head, batch_first=True)
        self.proj_spatial_sa = nn.Linear(embed_dim, embed_dim)
        if cross_attention:
            self.attention_spatial_ca = nn.MultiheadAttention(embed_dim, num_head, batch_first=True)
            self.mix_spatial = nn.Sequential(
                nn.Linear(embed_dim * (num_cls - 1), embed_dim),
                Swish()
            )
            self.proj_spatial_ca = nn.Linear(embed_dim, embed_dim)
            self.norm_ca_y_s = nn.LayerNorm(embed_dim)
            
        else:
            self.linear_spatial = FeedForward(embed_dim, mlp_ratio)

        self.feed_forward_s = FeedForward(embed_dim, mlp_ratio)
        self.norm1_1 = nn.LayerNorm(embed_dim)
        self.norm1_2 = nn.LayerNorm(embed_dim)
        self.norm1_3 = nn.LayerNorm(embed_dim)

        # temporal self attention and cross attention
        self.attention_temporal_sa = nn.MultiheadAttention(embed_dim, num_head, batch_first=True)
        if conf:
            self.proj_temporal_sa = ConformerConvModule(in_channels=embed_dim, kernel_size=3, dropout_p=dropout, causal=True)
        else:
            self.proj_temporal_sa = nn.Linear(embed_dim, embed_dim)
        if cross_attention:  
            self.attention_temporal_ca = nn.MultiheadAttention(embed_dim, num_head, batch_first=True)
            self.mix_temporal = nn.Sequential(
                nn.Linear(embed_dim * (num_cls - 1), embed_dim),
                Swish()
            )
            if conf:
                self.proj_temporal_ca = ConformerConvModule(in_channels=embed_dim, kernel_size=3, dropout_p=dropout, causal=True)
            else:
                self.proj_temporal_ca = nn.Linear(embed_dim, embed_dim)
            self.norm_ca_y_t = nn.LayerNorm(embed_dim)
            
        else:
            self.linear_temporal = FeedForward(embed_dim, mlp_ratio)

        self.feed_forward_t = FeedForward(embed_dim, mlp_ratio)
        self.norm2_1 = nn.LayerNorm(embed_dim)
        self.norm2_2 = nn.LayerNorm(embed_dim)
        self.norm2_3 = nn.LayerNorm(embed_dim)

    
    def create_causal_mask(self, size, device, reverse=False):
        mask = torch.triu(torch.ones((size, size), dtype=torch.bool), diagonal=1).to(device)
        if reverse:
            mask = torch.flip(mask, dims=(0, 1))
        return mask

    '''
    spatial attention
    '''
    def self_attention_spatial(self, x):
        x = self.norm1_1(x)
        x = self.attention_spatial_sa(x, x, x)[0]
        x = self.dropout(x)
        return x
    
    def cross_attention_spatial(self, x, y):
        x = self.norm1_2(x)
        y = self.norm_ca_y_s(y)
        x = self.attention_spatial_ca(x, y, y)[0]
        x = self.dropout(x)
        return x

    def mix_sp(self, x):
        x = self.mix_spatial(x)
        x = self.dropout(x)
        return x

    def feed_forward_spatial(self, x):
        x = self.feed_forward_s(self.norm1_3(x))
        x = self.dropout(x)
        return x
    

    '''
    temporal attention
    '''
    def self_attention_temporal(self, x, reverse=False):
        x = self.norm2_1(x)
        mask = self.create_causal_mask(x.shape[1], x.device, reverse)
        x = self.attention_temporal_sa(x, x, x, attn_mask=mask)[0]
        x = self.dropout(x)
        return x
    
    def cross_attention_temporal(self, x, y, reverse=False):
        x = self.norm2_2(x)
        y = self.norm_ca_y_t(y)
        mask = self.create_causal_mask(x.shape[1], x.device, reverse)
        x = self.attention_temporal_ca(x, y, y, attn_mask=mask)[0]
        x = self.dropout(x)
        return x
    
    def mix_tmp(self, x):
        x = self.mix_temporal(x)
        x = self.dropout(x)
        return x
    
    def feed_forward_temporal(self, x):
        x = self.feed_forward_t(self.norm2_3(x))
        x = self.dropout(x)
        return x

    def forward(self, x1, x2):
        
        x1 = self.self_attention_spatial(x1)
        x1 = self.cross_attention_spatial(x1, x2)
        x1 = self.feed_forward_spatial(x1)

        x1 = self.self_attention_temporal(x1)
        x1 = self.cross_attention_temporal(x1, x2)
        x1 = self.feed_forward_temporal(x1)

        return x1

class CAT(nn.Module):
    def __init__(self, vqvae_dim, grid_size, seq_len, num_cls, num_instance,
                embed_dim, num_head, mlp_ratio, depth, is_stochastic, droput, device, conf, cross_attention) -> None:
        super().__init__()

        self.depth = depth
        self.device = device
        self.num_cls = num_cls
        self.seq_len = seq_len
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.vqvae_dim = vqvae_dim
        self.num_instance = num_instance
        self.is_stochastic = is_stochastic
        self.instances = numpy.sum(num_instance)
        self.intervals = self.generate_intervals()
        self.cross_attention = cross_attention

        # changed to embedding per class
        self.spatial_embeds = []
        self.temporal_embeds = []
        self.spatio_temporal_embeds = []
        for _ in range(self.instances):
            self.spatial_embeds.append(nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim)))
            self.temporal_embeds.append(nn.Parameter(torch.randn(1, seq_len, embed_dim)))
            self.spatio_temporal_embeds.append(nn.Parameter(torch.randn(1, seq_len, grid_size * grid_size * embed_dim)))
        self.spatial_embeds = nn.ParameterList(self.spatial_embeds)
        self.temporal_embeds = nn.ParameterList(self.temporal_embeds)
        self.spatio_temporal_embeds = nn.ParameterList(self.spatio_temporal_embeds)

        self.embed = nn.Linear(vqvae_dim, embed_dim)

        self.class_blocks = []
        for _ in range(num_cls):
            blocks = []
            for _ in range(depth):
                blocks.append(Blocks(embed_dim, num_head, mlp_ratio, self.instances, droput, conf, cross_attention))
            blocks = nn.ModuleList(blocks)
            self.class_blocks.append(blocks)
        self.class_blocks = nn.ModuleList(self.class_blocks)
        
        if not is_stochastic:
            self.cl_out = nn.Linear(embed_dim, vqvae_dim)
        else:
            self.cl_out = nn.Linear(embed_dim, self.codebook_size)
            
        self.apply(weights_init)
    
    def generate_intervals(self):
        intervals = []
        start = 0
        for num in self.num_instance:
            end = start + num
            intervals.append(list(range(start, end)))
            start = end
        return intervals
    
    def get_index_from_input_with_intervals(self, input:int):
        for i, interval in enumerate(self.intervals):
            if input in interval:
                return i
        raise ValueError(f"Input {input} not in intervals")
    

    # gradient checkpointing
    def forward(self, input, reverse=False):
        return checkpoint.checkpoint(self._forward, input, reverse, use_reentrant=True)

    def _forward(self, x, reverse=False):

        B, T, _, _ = x.shape

        x = rearrange(x, 'b t cl (e h w) -> b t cl h w e', e=self.vqvae_dim, h=self.grid_size, w=self.grid_size)
        x = self.embed(x)
        
        new_x = []
        for d in range(self.depth):

            x = rearrange(x, 'b t cl h w e -> cl b t (e h w)')
            for i in range(sum(self.num_instance)):
                x[i] = x[i] + self.spatio_temporal_embeds[i]
            x = rearrange(x, 'cl b t (e h w) -> cl (b t) (h w) e', e=self.embed_dim, h=self.grid_size, w=self.grid_size)
            saved_x = torch.zeros_like(x)
            
            for cl_1 in range(self.instances):
                idx_1 = self.get_index_from_input_with_intervals(cl_1)
                # spatial self attention for main class
                if self.cross_attention:
                    if cl_1 == 0:
                        x1 = x[cl_1] + self.spatial_embeds[cl_1]
                        x1 = x1 + self.class_blocks[idx_1][d].self_attention_spatial(x1)
                        x1 = x1 + self.class_blocks[idx_1][d].proj_spatial_sa(x1)
                        saved_x[cl_1] = x1
                    else:
                        x1 = saved_x[cl_1]
                else:
                    x1 = x[cl_1] + self.spatial_embeds[cl_1]
                    x1 = x1 + self.class_blocks[idx_1][d].self_attention_spatial(x1)
                    x1 = x1 + self.class_blocks[idx_1][d].proj_spatial_sa(x1)

                # spatial cross attention for other classes
                if self.cross_attention:
                    count = 0
                    for cl_2 in range(self.instances):
                        if cl_1 != cl_2:
                            idx_2 = self.get_index_from_input_with_intervals(cl_2)
                            if cl_1 == 0:
                                x2 = x[cl_2] + self.spatial_embeds[cl_2]
                                x2 = x2 + self.class_blocks[idx_2][d].self_attention_spatial(x2)
                                x2 = x2 + self.class_blocks[idx_2][d].proj_spatial_sa(x2)
                                saved_x[cl_2] = x2
                            else:
                                x2 = saved_x[cl_2]

                            ca = self.class_blocks[idx_1][d].cross_attention_spatial(x1, x2)
                            if count == 0:
                                x1_ca = ca
                            else:
                                x1_ca = torch.cat([x1_ca, ca], dim=-1)
                            count = count + 1
                    x1 = x1 + self.class_blocks[idx_1][d].mix_sp(x1_ca) 
                    x1 = x1 + self.class_blocks[idx_1][d].proj_spatial_ca(x1)
                else:
                    x1_norm = self.class_blocks[idx_1][d].norm1_2(x1)
                    x1 = x1 + self.class_blocks[idx_1][d].linear_spatial(x1_norm)
                # feed forward for spatial attention
                x1 = x1 + self.class_blocks[idx_1][d].feed_forward_spatial(x1)
                new_x.append(x1) # append to save for temporal attention
                        
            # stack the processed data for temporal attention
            x = torch.stack(new_x)
            new_x = [] # reset for next iteration

            # changing the dimentions for temporal attention
            x = rearrange(x, 'cl (b t) (h w) e -> cl (b h w) t e', t=T, h=self.grid_size, w=self.grid_size)
            saved_x = torch.zeros_like(x)
            for cl_1 in range(self.instances):
                idx_1 = self.get_index_from_input_with_intervals(cl_1)
                if self.cross_attention:
                    if cl_1 == 0:              
                        x1 = x[cl_1] + self.temporal_embeds[cl_1]
                        # temporal self attention for main class
                        x1 = x1 + self.class_blocks[idx_1][d].self_attention_temporal(x1, reverse)
                        x1 = x1 + self.class_blocks[idx_1][d].proj_temporal_sa(x1)
                        saved_x[cl_1] = x1
                    else:
                        x1 = saved_x[cl_1]
                else:
                    x1 = x[cl_1] + self.temporal_embeds[cl_1]
                    x1 = x1 + self.class_blocks[idx_1][d].self_attention_temporal(x1, reverse)
                    x1 = x1 + self.class_blocks[idx_1][d].proj_temporal_sa(x1)
                
                if self.cross_attention:
                    # temporal cross attention for other classes
                    count = 0
                    for cl_2 in range(self.instances):
                        if cl_1 != cl_2:
                            idx_2 = self.get_index_from_input_with_intervals(cl_2)
                            if cl_1 == 0:
                                x2 = x[cl_2] + self.temporal_embeds[cl_2]
                                x2 = x2 + self.class_blocks[idx_2][d].self_attention_temporal(x2, reverse)
                                x2 = x2 + self.class_blocks[idx_2][d].proj_temporal_sa(x2)
                                saved_x[cl_2] = x2
                            else:
                                x2 = saved_x[cl_2]
                            ca = self.class_blocks[idx_1][d].cross_attention_temporal(x1, x2, reverse)
                            if count == 0:
                                x1_ca = ca
                            else:
                                x1_ca = torch.cat([x1_ca, ca], dim=-1)
                            count = count + 1

                    x1 = x1 + self.class_blocks[idx_1][d].mix_tmp(x1_ca)
                    x1 = x1 + self.class_blocks[idx_1][d].proj_temporal_ca(x1)
                else:
                    x1_norm = self.class_blocks[idx_1][d].norm2_2(x1)
                    x1 = x1 + self.class_blocks[idx_1][d].linear_temporal(x1_norm)
                # feed forward for temporal attention
                x1 = x1 + self.class_blocks[idx_1][d].feed_forward_temporal(x1)
                new_x.append(x1) # append to save for spatio-temporal attention

            # stack the processed data for spatio-temporal attention
            x = torch.stack(new_x)
            new_x = [] # reset for next iteration

            if self.depth - 1 != d:
                x = rearrange(x, 'cl (b h w) t e -> b t cl h w e', t=T, h=self.grid_size, w=self.grid_size)
                            
        x = rearrange(x, 'cl (b h w) t e -> cl t b h w e', b=B, h=self.grid_size, w=self.grid_size)
        if self.is_stochastic:
            x = rearrange(x, 'cl t b h w e -> b t cl (h w) e')
            return x
        final_out = self.cl_out(x)
            
        final_out = rearrange(final_out, 'cl t b h w e -> b t (cl e h w)')
        return final_out
    

def get_one_hot_encoding_with_indices(encoding_inds, codebook):
        K, D = codebook.shape
        encoding_inds = encoding_inds.unsqueeze(1)  # [BHW, 1]
        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_inds.size(0), K, device=encoding_inds.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, codebook)  # [BHW, D]

        return encoding_one_hot, quantized_latents

def get_one_hot_by_indices(indecies, codebooks):

    B, T, F, CL = indecies.shape
    reshaped = rearrange(indecies, 'b t f cl -> b cl (t f)')
    for b in range(B):
        reshaped_b = reshaped[b]
        for cl in range(CL):
            if cl < 4:
                index = 0
            else:
                index = 1
            in_ae = reshaped_b[cl]
            x, q_x = get_one_hot_encoding_with_indices(in_ae, codebooks[index])
            q_x = rearrange(q_x, '(t f) (c cl) -> t cl (f c)', t=T, f=F, cl=1)
            x = rearrange(x, '(t f) one_hot -> t f one_hot', t=T, f=F).unsqueeze(0)
            if cl == 0:
                b_quantized_x = x
                real_q_x = q_x
            else:
                b_quantized_x = torch.concat([b_quantized_x, x], dim=0)
                real_q_x = torch.concat([real_q_x, q_x], dim=1)
        b_quantized_x = b_quantized_x.unsqueeze(0)
        real_q_x = real_q_x.unsqueeze(0)
        if b == 0:
            quantized_x = b_quantized_x
            real_quantized_x = real_q_x
        else:
            quantized_x = torch.concat([quantized_x, b_quantized_x], dim=0)
            real_quantized_x = torch.concat([real_quantized_x, real_q_x], dim=0)

    quantized_x = rearrange(quantized_x, 'b cl t f one_hot -> b t cl f one_hot')
    return quantized_x

def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        # print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)

    
