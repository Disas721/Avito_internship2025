import math
import torch
import torch.nn as nn

# Собираем дефолтный набор символов, с которым работает токенизатор
def _build_default_charset():
    digits = list("0123456789")
    latin_lower = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    latin_upper = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    cyr_lower = [chr(c) for c in range(ord('а'), ord('я') + 1)] + ['ё']
    cyr_upper = [chr(c) for c in range(ord('А'), ord('Я') + 1)] + ['Ё']
    punct = list(" .,!?;:'\"()[]{}-%+«»")
    charset = []
    seen = set()
    for seq in ([' '], digits, latin_lower, latin_upper, cyr_lower, cyr_upper, punct):
        for ch in seq:
            if ch not in seen:
                seen.add(ch)
                charset.append(ch)
    return charset


# Предопределённый словарь символов и размер словаря с учётом служебных токенов
DEFAULT_CHARSET = _build_default_charset()
DEFAULT_VOCAB_SIZE = len(DEFAULT_CHARSET) + 4  # + PAD/BOS/EOS/UNK


# Символьный токенизатор с явными ID для служебных токенов
class CharTokenizer:
    """Токенизация по символам, а не по байтам."""

    def __init__(self, extra_chars=None):
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        charset = list(DEFAULT_CHARSET)
        if extra_chars:
            for ch in extra_chars:
                if ch not in charset:
                    charset.append(ch)
        self.char2id = {ch: idx for idx, ch in enumerate(charset, start=4)}
        self.id2char = {idx: ch for ch, idx in self.char2id.items()}
        self.vocab_size = len(self.char2id) + 4

    def encode(self, s: str):
        ids = [self.bos_id]
        for ch in s:
            ids.append(self.char2id.get(ch, self.unk_id))
        ids.append(self.eos_id)
        return ids

# Классическое синусоидальное позиционное кодирование для последовательностей
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Небольшой Transformer-энкодер, предсказывающий места для вставки пробелов
class TinyCharTransformer(nn.Module):
    def __init__(self, vocab_size=None, d_model=192, nhead=4, num_layers=4, dim_feedforward=384, dropout=0.1):
        super().__init__()
        if vocab_size is None:
            vocab_size = DEFAULT_VOCAB_SIZE
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.emb(x)
        h = self.pos(h)
        h = self.enc(h)
        logits = self.head(h).squeeze(-1)   # (B, T)
        return logits[:, 1:-1]              # positions between BOS..EOS
