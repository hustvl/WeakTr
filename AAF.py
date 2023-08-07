from torch import nn
import torch


# 1. using attention feature to generate dynamic weight
class AAF(nn.Module):
    def __init__(self, channel, reduction=16, feats_channel=64, feat_reduction=8, pool="avg"):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool == "max":
            self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.attn_head_ffn = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),  # inplace=True sometimes slightly decrease the memory usage
            # nn.Sigmoid(),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )
        self.attn_feat_ffn = nn.Sequential(
                                    nn.Linear(feats_channel, int(feats_channel / feat_reduction)),
                                    nn.Linear(int(feats_channel / feat_reduction), 1),
                                )
    
    def forward_weight(self, x):
        b, c, n, m = x.size() # batchsize, attn heads num=72, class tokens + patch tokens, embedding_dim=64

        # 1. pooling for tokens
        x = x.permute(0, 1, 3, 2).contiguous().view(b, c*m, n, 1) 
        attn_feat_pool = self.avg_pool(x)

        # 2. FFN for channels, generate dynamic weight
        attn_feat_pool = attn_feat_pool.view(b*c, m)
        attn_weight = self.attn_feat_ffn(attn_feat_pool)

        # 3. FFN for attn heads generate last weight
        attn_weight = attn_weight.view(b, c)
        attn_weight = self.attn_head_ffn(attn_weight).view(b, c, -1, 1)

        return attn_weight

    def forward(self, attn_feat, x):
        weight = self.forward_weight(attn_feat)
        return x * weight.expand_as(x), x * weight.expand_as(x)


# 2. using randomly initialized weight to generate dynamic weight
class AAF_RandWeight(AAF):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = torch.randn(1, channel, requires_grad=False).cuda()
    
    def forward_weight(self, x):
        b, c, n, m = x.size() # batchsize, attn heads num=72, class tokens + patch tokens, embedding_dim=64

        attn_weight = self.attn_head_ffn(self.query.expand(b, -1)).unsqueeze(2).unsqueeze(3)

        return attn_weight