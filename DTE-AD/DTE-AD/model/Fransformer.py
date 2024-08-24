from .attn import *
from .embed import DataEmbedding
from .decomposition import *


class t_EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(t_EncoderLayer, self).__init__()
        # attn대신 fourier 하기 때문에 d_model=window_size
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, mask = None, src_mask=None, src_key_padding_mask=None):
        output, attn_weights = self.attention(x, x, x, mask)
        res_x = x + self.dropout(output)
        res_x = self.norm1(res_x)
        inter_x = self.conv1(res_x.transpose(-1, 1))
        inter_x = self.dropout(self.activation(inter_x))
        inter_x = self.conv2(inter_x).transpose(-1, 1)
        inter_x = self.dropout(inter_x)
        out = self.norm2(res_x + inter_x)
        return out, attn_weights


class f_EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.mixing_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.feed_forward = nn.Linear(d_model, d_ff)
        self.output_dense = nn.Linear(d_ff, d_model)
        self.output_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, res_x, src_mask=None, src_key_padding_mask=None):
        fft_output = self.mixing_layer_norm(res_x)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        return output



# Transformer Encoder
class t_TransformerEncoder(nn.Module):
    def __init__(self, attention, d_model, nhead, d_ff, num_layers):
        super(t_TransformerEncoder, self).__init__()
        self.encoder_layer = t_EncoderLayer(attention, d_model, d_ff=None, dropout=0.1, activation="relu")
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
    def forward(self, x, mask=None):
        time_attention = []
        for _ in range(self.encoder.num_layers):
            x, time_attn = self.encoder_layer(x, mask)
            time_attention.append(time_attn)
        return x, time_attention



class f_TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers):
        super(f_TransformerEncoder, self).__init__()
        self.frequency = FFT2DLayer()
        self.inv_frequency = IFFT2DLayer()         
        #self.frequency = FFT1DLayer()
        #self.inv_frequency = IFFT1DLayer()  
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        fft_real, fft_imag = self.frequency(x)
        enc = self.encoder(fft_real)
        enc = torch.complex(enc, fft_imag)
        ifft = self.inv_frequency(enc)
        return ifft

class co_feed(nn.Module):
    def __init__(self, d_hid, d_ffn):
        super(co_feed, self).__init__()
        self.w1 = nn.Linear(d_hid, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_hid)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_hid)

    def forward(self, x):
        res = x
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        output = x
        return output


class DTE_AD(nn.Module):
    def __init__(self, feats, n_window, n_heads=8, n_layers=3, d_ff=256, d_hid=256, d_model=128):  # feats : [batch_size, length, feature]
        super(DTE_AD, self).__init__()

        self.revin = RevIN(feats, affine=True, subtract_last=False)
        self.decomp = series_decomp(kernel_size=3)
        self.embedding = DataEmbedding(c_in=feats, d_model=d_model, dropout=0.0)
        # 하나의 임베딩을 사용함으로써 Trend와 residual 사이의 내재된 관계를 더 잘 학습할 수 있음
        self.t_attn = multiheadattention(d_model, n_heads)
        self.time_encoder = t_TransformerEncoder(self.t_attn, d_model, n_heads, d_ff, n_layers)
        self.freq_encoder = f_TransformerEncoder(d_model, n_heads, d_ff, n_layers)
        self.co_attn_t = multiheadattention(d_model, n_heads)
        self.co_attn_f = multiheadattention(d_model, n_heads)
        self.co_feed1 = co_feed(d_hid=n_window, d_ffn=d_ff)
        self.co_feed2 = co_feed(d_hid=n_window, d_ffn=d_ff)
        self.decoder = nn.Linear(d_hid, feats)

    def forward(self, x):
        # 누락된 값 채움
        mask_en = create_missing_value_mask(x)
        # RevIN normalization (input과 output 사이 scale 또는 shift 방지)
        x = self.revin(x, 'norm')
        # decomposition (시계열 분해 -> trend, residual)
        res, trend = self.decomp(x)

        # 각각 임베딩
        em_t = self.embedding(trend) # time domain input
        em_r = self.embedding(res)  # freq domain input
        # time domain encoder, frequency domain encoder
        z_time, time_attn = self.time_encoder(em_t, mask_en) # [batch, win_len, d_model]
        z_freq = self.freq_encoder(em_r) # [batch, win_len, d_model]

        # co-attention (z_time, z_freq 관계)
        co_attn1, t_attn = self.co_attn_t(z_time, z_freq, z_freq) # [batch, win_len, d_model]
        co_attn2, f_attn = self.co_attn_f(z_freq, z_time, z_time)
        #print(f't_attn = {t_attn}\n')
        #print(f'f_attn = {f_attn}\n')
        t_attn = self.co_feed1(t_attn)
        f_attn = self.co_feed2(f_attn)

        ### co_attn1과 co_attn2 합치는 방법 실험들
        # 1. concatenation
        co_attn = torch.cat([co_attn1, co_attn2], dim=-1)

        # reconstruction decoder
        x_reconstructed = self.decoder(co_attn)

        # RevIN de-normalization (input의 non-stationary 정보를 output에 전달하기 위함)
        y = self.revin(x_reconstructed, 'denorm')

        return y, time_attn, t_attn, f_attn