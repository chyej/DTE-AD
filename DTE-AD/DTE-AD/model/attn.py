import torch
import torch.nn as nn
import torch.nn.functional as F


def create_missing_value_mask(data):
    '''
    :param data: [batch_size, seq_len, feature_dim]
    :return: [batch_size, 1, seq_len]
    '''
    # NaN-True
    mask = torch.isnan(data)
    mask = mask.any(dim=-1, keepdim=True)
    return (~mask).float()

def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).expand(seq.size(0), seq.size(1), seq.size(2))
    subsequent_mask = subsequent_mask.triu(diagonal=1)
    # upper triangular part of a matrix(2-D)
    return subsequent_mask


class FFT2DLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_state):
        real = torch.fft.fft(torch.fft.fft(hidden_state, dim=-1), dim=-2).real
        imag = torch.fft.fft(torch.fft.fft(hidden_state, dim=-1), dim=-2).imag
        # 2D FFT
        #fft = torch.fft.fft(torch.fft.fft(hidden_state, dim=-1), dim=-2)
        return real, imag #fft


class IFFT2DLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_state):
        ifft = torch.fft.ifft(torch.fft.ifft(hidden_state, dim=-2), dim=-1).real
        # 2D IFFT
        return ifft


class FFT1DLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_state):
        fft_real = torch.fft.fft(hidden_state, dim=-1).real
        fft_imag = torch.fft.fft(hidden_state, dim=-1).imag
        return fft_real, fft_imag


class IFFT1DLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_state):
        ifft = torch.fft.ifft(hidden_state, dim=-1).real
        return ifft


class p_scaledotproductattention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn_dropout = nn.Dropout(0.)
        self.res_attention = False
        self.head_dim = d_model // n_heads  ## d_model = n_heads*d_k
        self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5), requires_grad=False)

    def forward(self, q, k, v, mask=None):
        '''
            q : [batch, n_head, patch_num*feat_dim, d_k]
            k : [batch, n_heads, d_k, patch_num*feat_dim]
            v : [batch, n_heads, patch_num*feat_dim, d_v]
        '''

        # scaled matmul(q, k)
        attn_scores = torch.matmul(q, k) * self.scale
        #print(f'attn_score shape : {attn_scores.shape}')

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1, -1)  # [batch_size, n_head, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        #print(f'scale_output = {output.shape}\nscale_attn_weights = {attn_weights.shape}')

        return output, attn_weights


class multiheadattention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=True)

        # scale dot product attention
        self.sdp_attn = p_scaledotproductattention(d_model, n_heads)

        # project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(0.))

    def forward(self, q, k, v, mask=None):
        bs = int(q.size(0))

        if k is None: k = q
        if v is None: v = q

        # Linear ( + split in multiple heads )
        q_s = self.W_Q(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(k).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(v).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        # scale dot product attention : multi-heads
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, mask)
        #print(f'attn output : {output.shape}')
        #print(f'attn_weights output : {attn_weights.shape}')

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        #print(f'multi-head before output : {output.shape}')
        output = self.to_out(output)
        
        #print(f'multi-head output : {output.shape}')
        #print(f'attn_weights = {attn_weights.shape}')

        return output, attn_weights
