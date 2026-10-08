import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PositionalEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # (batch_size, seq_len, num_feature-1) -> (batch_size, num_feature-1, seq_len)
        x = x.permute(0, 2, 1)
        # (batch_size, seq_len, num_mark_features) -> (batch_size, num_mark_features, seq_len)
        if x_mark is None:
            # 直接进行值嵌入 (batch_size, n_vars, seq_len) -> (batch_size, embed_dim, seq_len)
            x = self.value_embedding(x)
        else:
            x_mark = x_mark.permute(0, 2, 1)
            # 拼接: (batch_size, num_feature-1 + num_mark_features, seq_len) -> (batch_size, embed_dim, seq_len)
            x = self.value_embedding(torch.cat([x, x_mark], 1))
        return self.dropout(x)


class EnEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 去掉单变量维度 (B,1,L) → (B, L)
        x = x.squeeze(1)
        # 切分 patch (B, L) → (B, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # 值嵌入 + 位置嵌入  (B, n_patches, patch_len)
        x = self.value_embedding(x) + self.position_embedding(x)
        # 加入全局 token
        # (1, 1, d_model)
        glb = self.glb_token
        # (1, 1, d_model) -> (B, 1, d_model)
        glb = glb.repeat(x.shape[0], 1, 1)
        # (B, n_patches + 1, d_model)
        x = torch.cat([x, glb], dim=1)

        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # res = self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask,
        #     tau=tau, delta=None
        # )[0]
        res = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
        )[0]
        x = x + self.dropout(res)
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        # x_glb_attn = self.dropout(self.cross_attention(
        #     x_glb, cross, cross,
        #     attn_mask=cross_mask,
        #     tau=tau, delta=delta
        # )[0])
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        # 对不重复的Patch进行数量获取
        self.patch_num = int(configs.seq_len // configs.patch_len)
        # 内生变量
        self.en_embedding = EnEmbedding(configs.d_model, self.patch_len, configs.dropout)
        # 外生变量
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len + configs.pred_len, configs.d_model, configs.dropout)
        # self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    # seq_x, seq_y, seq_x_mark, seq_y_mark
    def forward(self, seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future, mask=None):
        # 1.RevIn    x_enc, x_mark_enc,x_dec,  x_mark_dec
        if self.configs.RevIn:
            means = seq_x.mean(1, keepdim=True).detach()
            seq_x = seq_x - means
            stdev = torch.sqrt(torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_x /= stdev

        # 内生变量嵌入。数据集中需要除时间列外，最后一个变量是内生变量(预测目标变量)
        # (batch_size, seq_len) -> (batch_size, seq_len, 1) -> (batch_size, 1, seq_len)
        en_input = seq_x[:, :, -1].unsqueeze(-1).permute(0, 2, 1)
        # (batch_size, 1, seq_len) -> (batch_size, n_patches + 1, d_model)
        en_embed = self.en_embedding(en_input)

        # 外生变量(协变量)嵌入。
        # (batch_size, seq_len, num_feature) -> (batch_size, seq_len, num_feature-1)
        ex_input = seq_x[:, :, 0:-1]
        # 历史外生变量+未来外生变量
        ex_input = torch.concat([ex_input, seq_future[:, :, 0:-1]], dim=1)
        # print(seq_future[:, :, -2]) 未来的功率全部设置为0了，保证信息不泄露
        ex_input_mark = torch.concat([seq_x_mark, seq_y_mark], dim=1)
        ex_embed = self.ex_embedding(ex_input, ex_input_mark)

        # 编码器处理
        # en_embed: (batch_size * 1, n_patches + 1, d_model)
        # ex_embed: (batch_size, embed_dim, seq_len)
        # enc_out -> (batch_size * 1, n_patches + 1, d_model)
        enc_out = self.encoder(en_embed, ex_embed)

        # 恢复batch和变量维度的分离
        # (batch_size * 1, n_patches + 1, d_model) -> (batch_size, 1, n_patches + 1, d_model)
        enc_out = torch.reshape(enc_out, (-1, 1, enc_out.shape[-2], enc_out.shape[-1]))

        # 调整维度顺序，准备送入预测头
        # (batch_size, 1, n_patches + 1, d_model) -> (batch_size, 1, d_model, n_patches + 1)
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 5.预测头处理：展平并线性映射到目标长度
        # (batch_size, 1, d_model, n_patches + 1) -> (batch_size, 1, pred_len)
        dec_out = self.head(enc_out)
        # (batch_size, 1, pred_len) -> (batch_size, pred_len, 1)
        dec_out = dec_out.permute(0, 2, 1)

        if self.configs.RevIn:
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len:, :]


def criterion(args, model):
    return nn.MSELoss()
