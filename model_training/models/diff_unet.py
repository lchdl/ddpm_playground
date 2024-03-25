import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def nonlinearity():
    # Also known as Swish function. 
    # See: https://en.wikipedia.org/wiki/Swish_function.
    return nn.SiLU() 

def linear(in_feat, out_feat):
    return nn.Linear(in_feat, out_feat, bias=True)

def normalize(in_channels, num_groups):
    # apply group/instance normalization to input.
    if num_groups: 
        return nn.GroupNorm(num_groups=num_groups,
                        num_channels=in_channels)
    else:
        return nn.InstanceNorm2d(num_features=in_channels)

def nin(in_channels, out_channels):
    # Network in Network (NiN): 
    # See: https://arxiv.org/abs/1312.4400.
    # which is equivalent to 1x1 convolution.
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size=1, stride=1, padding=0)

def conv2d(in_channels, out_channels):
    # Ordinary 2D convolution using 3x3 kernels.
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, stride=1, padding=1)

def downsample(in_channels, out_channels):
    # Down sampling layer (0.5x spatial size).
    # Using kernel_size=3 to avoid bed-of-nail effect.
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, stride=2, padding=1)

def up_sample(in_channels, out_channels):
    # Up sampling layer (2x spatial size).
    # Using kernel_size=3 to avoid bed-of-nail effect.
    return nn.ConvTranspose2d(in_channels, out_channels, 
                     kernel_size=3, stride=2, padding=1, 
                     output_padding=1)

def scaled_dot_product(q, k, v, mask=None, return_attn=False):
    '''
    q, k, v: [..., seq_len, d_k]
    mask: [seq_len, seq_len]
    '''
    assert q.shape == k.shape == v.shape
    assert len(q.shape) >= 2
    
    seq_len, d_k = q.shape[-2], q.shape[-1]
    attn_weights = torch.matmul(q, torch.transpose(k, -2, -1)) / np.sqrt(d_k)
    if mask is not None:
        assert len(mask.shape) == 2
        assert mask.shape[-2] == mask.shape[-1] == seq_len
        # reshape mask so that it can be broadcasted
        pad_dim = len(attn_weights.shape) - len(mask.shape)
        adapted_mask_shape = [1] * pad_dim + [seq_len, seq_len]
        mask = mask.reshape(*adapted_mask_shape)
        # apply mask to attention
        attn_weights = attn_weights.masked_fill(mask == 0, -torch.inf)
    attn_weights = F.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn_weights, v)
    if return_attn:
        return out, attn_weights
    else:
        return out

class MultiheadAttention(nn.Module):
    '''
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    '''
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim(%d) should be divisible by num_heads(%d).' % (embed_dim, num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, q, k, v, mask=None, return_attn=False):
        '''
        q, k, v: [batch_size, seq_len, input_dim], 
            => concatenated qkv tensor for self/cross attention.
        mask: [seq_len, seq_len]
        '''
        assert q.shape == k.shape == v.shape
        batch_size, seq_len, embed_dim = v.shape[0], v.shape[1], v.shape[2]
        assert embed_dim == self.embed_dim
        q = self.q_proj(q).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k_proj(k).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v_proj(v).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        # q, k, v: [BatchSize, Heads, NumTokens, HeadDim]
        sdp_out = scaled_dot_product(q,k,v,mask=mask, return_attn=return_attn)
        out, attn_weights = None, None
        if return_attn:
            out, attn_weights = sdp_out
        else:
            out = sdp_out
        out = out.permute(0,2,1,3) # [batch_size, seq_len, num_heads, head_dim]
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        out = self.o_proj(out)
        if return_attn:
            return out, attn_weights
        else:
            return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = MultiheadAttention(embed_dim, num_heads)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)
    def forward(self, q, k, v, mask=None):
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        x = self.mha(q, k, v, mask=mask)
        assert x.shape == v.shape
        return x + v

class PixelSelfAttention(nn.Module):
    '''
    2D pixel-wise self-attention block:
    input:  [b, in_ch, h, w]
    output: [b, in_ch, h, w]
    '''
    def __init__(self, in_ch, groups=None, emb_dim=None):
        super().__init__()
        self.in_ch = in_ch
        self.emb_dim = emb_dim if emb_dim is not None else in_ch
        self.groups = groups
        self.norm_layer = nn.GroupNorm(
            # default to instance norm if groups is not set.
            num_groups=groups if groups is not None else in_ch, 
            num_channels=in_ch) 
        self.q_proj = linear(in_ch, self.emb_dim)
        self.k_proj = linear(in_ch, self.emb_dim)
        self.v_proj = linear(in_ch, self.emb_dim)
        self.o_proj = linear(self.emb_dim, in_ch)
    def forward(self, x):
        '''
        x: [b, in_ch, h, w]
        '''
        assert len(x.shape) == 4, 'Expected a 4D tensor with shape [b, c, h, w].'
        batch, in_channels, height, width = x.shape
        x0 = self.norm_layer(x)
        x0 = torch.permute(x0, [0, 2, 3, 1]) # channel dim as embedding dim
        x0 = torch.reshape(x0, [batch, height * width, in_channels])
        Q = self.q_proj(x0) # [b, num_tok, emb_dim]
        K = self.k_proj(x0) # [b, num_tok, emb_dim]
        W = torch.softmax(torch.bmm(Q, torch.permute(K, [0, 2, 1])) / np.sqrt(self.emb_dim), dim=2) # [b, num_tok, num_tok]
        V = self.v_proj(x0) # [b, num_tok, emb_dim]
        y = torch.bmm(W, V) # [b, num_tok, emb_dim]
        y = self.o_proj(y) # [b, num_tok, in_channels]
        y = torch.reshape(y, [batch, height, width, self.in_ch])
        y = torch.permute(y, [0, 3, 1, 2]) # [b, c_in, h, w]
        return x + y # residual connection

class Residual(nn.Module):
    def __init__(self, in_ch, mid_ch, cond_dim, dropout_p=None, num_groups=None):
        super().__init__()
        self.block0 = nn.Sequential(
            normalize(in_ch, num_groups), 
            nonlinearity(), 
            conv2d(in_ch, mid_ch))
        self.block_temb = nn.Sequential(
            nonlinearity(), 
            linear(cond_dim, mid_ch))
        self.block1 = [
            normalize(mid_ch, num_groups), 
            nonlinearity()]
        if dropout_p: 
            self.block1.append(nn.Dropout(p=dropout_p))
        self.block1.append(conv2d(mid_ch, mid_ch))
        self.block1.append(nin(mid_ch, in_ch))
        self.block1 = nn.Sequential(*self.block1)
    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        '''
        x: [b, in_ch, h, w]
        '''
        x0 = self.block0(x)
        t0 = self.block_temb(temb)[:, :, None, None] # [b, c] -> [b, c, 1, 1]
        x0 += t0
        x0 = self.block1(x0)
        return x0 + x

class TimeEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block0 = nn.Sequential(
            linear(in_dim, out_dim),
            nonlinearity(),
            linear(out_dim, out_dim))
    def forward(self, cond):
        ''' cond: [b, cond_dim] '''
        return self.block0(cond)

class TagEmbedding(nn.Module):
    def __init__(self, max_tokens, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb0 = nn.Embedding(max_tokens, emb_dim, padding_idx=0)
        # a tiny transformer for processing tag tokens
        self.xfm0 = TransformerBlock(emb_dim, num_heads)

    def forward(self, inds):
        ''' inds: [b, ind_dim] '''
        assert len(inds.shape) == 2
        batch_size, ind_len = inds.shape[0], inds.shape[1]
        x = self.emb0(inds)
        assert x.shape[0] == batch_size and x.shape[1] == ind_len and x.shape[2] == self.emb_dim
        x = self.xfm0(x, x, x).sum(dim=1)
        return x

class Encoding(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, arch="rr", 
                 dropout_p=None, num_groups=None, apply_ds=True):
        super().__init__()
        self.arch = arch
        self.start = nin(in_ch, in_ch)
        self.blocks = nn.ModuleList()
        for arch_ch in arch:
            if arch_ch == 'r':
                self.blocks.append(Residual(in_ch, in_ch, cond_dim, dropout_p, num_groups))
            elif arch_ch == 'a':
                self.blocks.append(PixelSelfAttention(in_ch, num_groups))
            else:
                raise RuntimeError('Unknown architecture definition "%s".' % arch_ch)
        if apply_ds:
            self.end = downsample(in_ch, out_ch)
        else:
            self.end = conv2d(in_ch, out_ch)
    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        '''
        x: [b, in_ch, h, w]
        '''
        x = self.start(x)
        for i, arch_ch in enumerate(self.arch):
            if arch_ch == 'r':
                x = self.blocks[i](x, temb)
            elif arch_ch == 'a':
                x = self.blocks[i](x)
        x = self.end(x)
        return x

class Decoding(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, arch="rr", 
                 dropout_p=None, num_groups=None, apply_us=True):
        super().__init__()
        self.arch = arch
        if apply_us:
            self.start = up_sample(in_ch, out_ch)
        else:
            self.start = conv2d(in_ch, out_ch)
        self.blocks = nn.ModuleList()
        for arch_ch in arch:
            if arch_ch == 'r':
                self.blocks.append(Residual(out_ch, out_ch, cond_dim, dropout_p, num_groups))
            elif arch_ch == 'a':
                self.blocks.append(PixelSelfAttention(out_ch, num_groups))
            else:
                raise RuntimeError('Unknown architecture definition "%s".' % arch_ch)
        self.end = nin(out_ch, out_ch)
    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        '''
        x: [b, in_ch, h, w]
        '''
        x = self.start(x)
        for i, arch_ch in enumerate(self.arch):
            if arch_ch == 'r':
                x = self.blocks[i](x, temb)
            elif arch_ch == 'a':
                x = self.blocks[i](x)
        x = self.end(x)
        return x

class Concatenate(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=None):
        super().__init__()
        self.blocks = nn.Sequential(
            normalize(in_ch, num_groups),
            nonlinearity(),
            conv2d(in_ch, out_ch))
    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_ch, cond_dim, arch="rr", 
                 dropout_p=None, num_groups=None):
        super().__init__()
        self.arch = arch
        self.blocks = nn.ModuleList()
        for arch_ch in arch:
            if arch_ch == 'r':
                self.blocks.append(Residual(in_ch, in_ch, cond_dim, dropout_p, num_groups))
            elif arch_ch == 'a':
                self.blocks.append(PixelSelfAttention(in_ch, num_groups))
            else:
                raise RuntimeError('Unknown architecture definition "%s".' % arch_ch)
    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        for i, arch_ch in enumerate(self.arch):
            if arch_ch == 'r':
                x = self.blocks[i](x, temb)
            elif arch_ch == 'a':
                x = self.blocks[i](x)
        return x

class Diffusion_UNet(nn.Module):
    '''
    Description
    ------------
    U-Net gaussian noise estimator for diffusion model.

    Model Input
    ------------
    x:    [b, c_in, h, w], batched input image
    temb: [b, cond_dim],   batched time embedding info
    cemb: [b, cond_dim],   batched conditional embedding info

    Model Output
    ------------
    noise_pred: [b, c_in, h, w], estimated standard Gaussian noise.

    Target
    ------------
    noise: [b, c_in, h, w], where each pixel $\sim \mathcal{N}(0,I)$.
    '''
    def __init__(self, 
                 in_ch=3, out_ch=3, base_ch=16,
                 ch_mult=[1, 2, 4, 8], 
                 down_arch=["rr", "rr", "rr", "rr"],
                 bottle_arch="rra",
                 up_arch=["rr", "rr", "rr", "rr"],
                 temb_dims=[1024, 512],
                 tag_dim=512,
                 num_groups=8,
                 dropout_p=None,
                 n_tags=None,
    ):
        '''
        Parameters
        ------------
        @param `in_ch`      : Number of input channels.
        @param `out_ch`     : Number of output channels.
        @param `base_ch`    : Output channels for the first convolutional layer. 
        @param `ch_mult`    : Channel number multiplier for every resolution.
        @param `down_arch`  : Architecture of the down-sampling block in each resolution.
        @param `bottle_arch`: Architecture of the bottleneck block in each resolution.
        @param `up_arch`    : Architecture of the up-sampling block in each resolution.
        @param `temb_dims`  : Time embedding vector lengths in each MLP layer.
        @param `tag_dim`    : Tag embedding vector length.
        @param `num_groups` : Number of groups used in GroupNorm.
        @param `num_heads`  : Number of heads in attention module.
        @param `dropout_p`  : Dropout rate (0.0~1.0). "None" to disable dropout. 
        @param `n_tags`     : Maximum number of tags supported.
        '''
        assert len(ch_mult) == len(down_arch) == len(up_arch), \
            'Invalid model architecture definition. The number of elements in `ch_mult`, '\
            '`down_arch`, and `up_arch` should be equal. Got %d, %d, and %d instead.' % \
            (len(ch_mult), len(down_arch), len(up_arch))
        assert len(temb_dims) == 2, \
            'Invalid time embedding layer definition. Expected a list with two integers but got '\
            '%s.' % str(temb_dims)
        assert n_tags is not None, '`n_tags` should not be None, expected an integer > 1.'

        super().__init__()
        ch_cfg = []
        for mul in ch_mult:
            ch_cfg.append(base_ch*mul)
        nres = len(ch_mult)
        self.time_embed = TimeEmbedding(temb_dims[0], temb_dims[1])
        self.tag_embed = TagEmbedding(n_tags, tag_dim, 4)
        # start block (for initialization)
        self.initial = conv2d(in_ch, ch_cfg[0])
        # downsample block
        self.encoder = nn.ModuleList()
        for res_i in range(nres):
            last_res = (res_i == nres-1)
            self.encoder.append(Encoding(
                in_ch=ch_cfg[res_i], 
                out_ch=ch_cfg[res_i+1] if not last_res else ch_cfg[res_i], 
                cond_dim=temb_dims[-1]+tag_dim, arch=down_arch[res_i], 
                dropout_p=dropout_p, num_groups=num_groups, apply_ds=not last_res))
        # bottle neck
        self.bottleneck = Bottleneck(
            in_ch=ch_cfg[-1], cond_dim=temb_dims[-1]+tag_dim, arch=bottle_arch, 
            dropout_p=dropout_p, num_groups=num_groups)
        # concat + upsample block
        self.decoder = nn.ModuleList()
        self.concat = nn.ModuleList()
        for res_i in reversed(range(nres)):
            last_res = (res_i == nres-1)
            if not last_res:
                self.concat.append(Concatenate(
                    in_ch=ch_cfg[res_i+1]*2, out_ch=ch_cfg[res_i+1], 
                    num_groups=num_groups))
                self.decoder.append(Decoding( 
                    in_ch=ch_cfg[res_i+1], out_ch=ch_cfg[res_i],
                    cond_dim=temb_dims[-1]+tag_dim, arch=up_arch[res_i], 
                    dropout_p=dropout_p, num_groups=num_groups, apply_us=(res_i!=nres-1)))
            else:
                self.concat.append(nn.Identity()) # just a placeholder and not used.
                self.decoder.append(Decoding(
                    in_ch=ch_cfg[res_i], out_ch=ch_cfg[res_i],
                    cond_dim=temb_dims[-1]+tag_dim, arch=up_arch[res_i], 
                    dropout_p=dropout_p, num_groups=num_groups, apply_us=(res_i!=nres-1)))
        # end block
        self.final = nn.Sequential(
            normalize(ch_cfg[0] * 2, num_groups=num_groups),
            nonlinearity(),
            conv2d(ch_cfg[0] * 2, out_ch))
     
    def nparams(self, as_string=False):
        '''
        Print the number of trainable parameters in the model.
        '''
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if as_string:
            if   n > 1e9: return '%.1fB' % (n/1e9)
            elif n > 1e6: return '%.1fM' % (n/1e6)
            elif n > 1e3: return '%.1fK' % (n/1e3)
            else: return str(n)
        else:
            return n

    def forward(self, x, temb, tag_ids=None):
        '''
        x: [b, c_in, h, w]
        temb: [b, temb_dim], time embedding vectors
        tag_ids: [b, num_tags], conditional embedding vectors
        '''
        assert len(x.shape) == 4, \
            'Only accept 4D image input with shape [b, c_in, h, w].'
        assert len(temb.shape) == 2, \
            'Only accept 2D time embedding input with shape [b, temb_dim]. ' \
            'Note: you need to pass scalar time value to an embedding layer.'
        assert len(self.encoder) == len(self.decoder)

        h = []
        nres = len(self.encoder)
        t0 = self.time_embed(temb)
        if tag_ids is not None:
            c0 = self.tag_embed(tag_ids)
        else:
            batch_size, cemb_dim = x.shape[0], self.tag_embed.emb_dim
            c0 = torch.zeros(batch_size, cemb_dim).to(t0.device)
        t0 = torch.cat((t0,c0), dim=-1)
        x0 = self.initial(x)
        h.append(x0)
        for i in range(nres):
            last_res = (i==nres-1)
            x0 = self.encoder[i](x0, t0)
            if not last_res:
                h.append(x0)
        x0 = self.bottleneck(x0, t0)
        for i in range(nres):
            last_res = (i==0)
            if not last_res:
                x0 = torch.cat((h.pop(), x0), dim=1)
                x0 = self.concat[i](x0)
            x0 = self.decoder[i](x0, t0)
        x0 = torch.cat((h.pop(), x0), dim=1)
        x0 = self.final(x0)
        return x0

if __name__ == '__main__':
    x = torch.tensor([[1,2,3], [2,3,4]]).long()
    layer = TagEmbedding(8, 6, 2)
    y = layer(x)
    print(y)
    print(y.shape)


