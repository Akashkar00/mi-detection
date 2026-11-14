"""Phases 4-5: paper LSTM / CNN branches, a strong 1D ResNet, and attention / concat fusion."""
import torch
import torch.nn as nn


class SignalLSTM(nn.Module):
    """Paper Table 2: 2 x LSTM(128), dropout 0.2 (torch has no recurrent dropout; noted in report)."""
    def __init__(self, in_ch=12, hid=128, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hid, 2, batch_first=True, dropout=dropout)
        self.out_dim = hid

    def forward(self, x):                      # (B, T, 12)
        return self.lstm(x)[0][:, -1]


class PaperCNN(nn.Module):
    """Paper Table 1: 3 conv (32/64/128, 3x3, maxpool 2, dropout .25) + dense 128."""
    def __init__(self, dropout=0.25):
        super().__init__()
        layers, c = [], 1
        for o in (32, 64, 128):
            layers += [nn.Conv2d(c, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(dropout)]
            c = o
        self.conv = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(128 * 16, 128), nn.ReLU())
        self.out_dim = 128

    def forward(self, x):                      # (B, 1, 224, 224); bf16 convs on CUDA (~1.5x), fp32 out
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.fc(self.conv(x)).float()


class _Block(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.c1 = nn.Conv1d(i, o, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(o)
        self.c2 = nn.Conv1d(o, o, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(o)
        self.sc = nn.Sequential(nn.Conv1d(i, o, 1, stride, bias=False), nn.BatchNorm1d(o))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(self.b2(self.c2(y)) + self.sc(x))


class ResNet1D(nn.Module):
    """Strong signal reference (resnet1d style) on (B, 1000, 12)."""
    def __init__(self, in_ch=12, width=64, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 15, 2, 7, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        chans = [width, width, width * 2, width * 2, width * 4]
        self.blocks = nn.Sequential(*[_Block(chans[i], chans[i + 1], 2) for i in range(4)])
        self.drop = nn.Dropout(dropout)
        self.out_dim = chans[-1]

    def forward(self, x):
        h = self.blocks(self.stem(x.transpose(1, 2)))
        return self.drop(h.mean(-1))


class Classifier(nn.Module):
    """Single branch + linear head -> 1 logit."""
    def __init__(self, branch):
        super().__init__()
        self.branch, self.head = branch, nn.Linear(branch.out_dim, 1)

    def forward(self, img, sig):
        x = sig if isinstance(self.branch, (SignalLSTM, ResNet1D)) else img
        return self.head(self.branch(x)).squeeze(-1)


class Fusion(nn.Module):
    """mode: r1 (MHA over 2 tokens, 4 heads x 32, residual+LN, flatten 256),
    r2 (Keras reading: 4 heads x key_dim 128, project to 256, mean-pool tokens),
    concat (f_img || f_sig -> 256)."""
    def __init__(self, mode="r1", dropout=0.0):
        super().__init__()
        self.mode, self.cnn, self.lstm = mode, PaperCNN(), SignalLSTM()
        d = 128
        if mode == "r1":
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.ln = nn.LayerNorm(d)
        elif mode == "r2":
            self.q, self.k, self.v = (nn.Linear(d, 4 * d) for _ in range(3))
            self.proj = nn.Linear(4 * d, 256)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(256, 1)
        self.last_attn = None

    def forward(self, img, sig):
        tok = torch.stack([self.cnn(img), self.lstm(sig)], 1)          # (B, 2, 128)
        if self.mode == "concat":
            z = tok.flatten(1)
        elif self.mode == "r1":
            o, w = self.attn(tok, tok, tok, need_weights=True, average_attn_weights=True)
            self.last_attn = w.detach()                                # (B, 2, 2)
            z = self.ln(tok + o).flatten(1)
        else:
            B = tok.shape[0]
            q, k, v = (f(tok).view(B, 2, 4, 128).transpose(1, 2) for f in (self.q, self.k, self.v))
            w = torch.softmax(q @ k.transpose(-1, -2) / 128 ** 0.5, -1)   # (B, 4, 2, 2)
            self.last_attn = w.mean(1).detach()
            z = self.proj((w @ v).transpose(1, 2).reshape(B, 2, 512)).mean(1)
        return self.head(self.drop(z)).squeeze(-1)


def build(name: str, dropout: float = 0.0) -> nn.Module:
    return {
        "B1_lstm": lambda: Classifier(SignalLSTM()),
        "B2_resnet1d": lambda: Classifier(ResNet1D()),
        "B3_cnn": lambda: Classifier(PaperCNN()),
        "F_r1": lambda: Fusion("r1", dropout),
        "F_r2": lambda: Fusion("r2", dropout),
        "F_concat": lambda: Fusion("concat", dropout),
    }[name]()
