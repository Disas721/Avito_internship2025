# train_char_tagger.py
import pandas as pd
import os, random, argparse
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_char_tagger import TinyCharTransformer, CharTokenizer

# Преобразует текст c пробелами в пару «строка без пробелов + метки вставок»
def make_pairs_from_line(line: str):
    line = " ".join(line.strip().split())
    if not line or len(line) < 3:
        return None
    no = line.replace(" ", "")
    labels = [0]*(len(no)-1)
    i_no = 0
    for i, ch in enumerate(line[:-1]):
        if ch == " ":
            continue
        if line[i+1] == " ":
            labels[i_no] = 1
        i_no += 1
    return no, labels

# Оборачивает пары строк в Dataset PyTorch и применяет токенизацию
class CharDataset(Dataset):
    def __init__(self, lines, tokenizer: CharTokenizer, max_len=512):
        self.items = []
        for ln in lines:
            pair = make_pairs_from_line(ln)
            if pair:
                s, y = pair
                if len(s) >= 3:
                    self.items.append((s, y))
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        s, y = self.items[idx]
        ids = self.tok.encode(s)
        return ids, y

# Ставит паддинги в батче и заполняет «пустые» метки значением -1
def collate(batch, pad_id=0):
    max_len = max(len(x[0]) for x in batch)
    X, Y = [], []
    for ids, y in batch:
        pad = max_len - len(ids)
        X.append(ids + [pad_id]*pad)
        yy = y + [-1]*(max_len-2-len(y))
        Y.append(yy)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# Унифицированное чтение корпусов из parquet или текстовых файлов
def read_lines(path, max_samples=None):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, engine="pyarrow")
        lines = df["text"].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    if max_samples and len(lines) > max_samples:
        lines = lines[:max_samples]
    return lines


# Точка входа CLI: парсит аргументы и запускает обучение модели
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--save", type=str, default="weights/char_tagger.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    lines = read_lines(args.corpus)
    random.shuffle(lines)
    # Держим небольшой валидационный хвост для отслеживания качества
    split = max(100, int(0.95*len(lines)))
    train_lines, val_lines = lines[:split], lines[split:]

    tok = CharTokenizer()
    train_ds = CharDataset(train_lines, tok, max_len=args.max_len)
    val_ds   = CharDataset(val_lines,   tok, max_len=args.max_len)

    # Обычные DataLoader'ы, но с кастомным collate для символьной длины
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCharTransformer(vocab_size=tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Положительные примеры (пробелы) редкие, поэтому повышаем их вес
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))

    best = 1e9
    for ep in range(1, args.epochs+1):
        model.train(); tot=0; n=0
        for X,Y in train_dl:
            X=X.to(device); Y=Y.to(device)
            logits = model(X)
            T = logits.size(1)
            # Маска отсекает паддинги, где нет целевых значений
            mask = (Y[:, :T] != -1)
            y = (Y[:, :T].clamp(min=0)).float()
            loss = loss_fn(logits[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        # Валидация на отложенной части корпуса
        model.eval(); vtot=0; m=0
        with torch.no_grad():
            for X,Y in val_dl:
                X=X.to(device); Y=Y.to(device)
                logits=model(X)
                T=logits.size(1)
                mask=(Y[:, :T] != -1)
                if mask.sum()==0: continue
                # При валидации используем ту же маску и веса, что и на обучении
                y=(Y[:, :T].clamp(min=0)).float()
                loss = loss_fn(logits[mask], y[mask])
                vtot+=loss.item(); m+=1
        vloss = vtot/max(1,m)
        print(f"epoch {ep}: train={tot/max(1,n):.4f}  val={vloss:.4f}")
        if vloss < best:
            best = vloss
            torch.save({"model_state": model.state_dict()}, args.save)
            print(f"  saved {args.save}")  # Сохраняем лучший чекпоинт по валидации

if __name__ == "__main__":
    main()
