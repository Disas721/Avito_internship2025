import argparse
import os
import numpy as np
import pandas as pd
from rules import postprocess_probs

# Загружает char-токенизатор и веса модели из чекпоинта
def load_tagger(weights_path):
    try:
        import torch
        from model_char_tagger import TinyCharTransformer, CharTokenizer
        ckpt = torch.load(weights_path, map_location="cpu")
        tok = CharTokenizer()
        model = TinyCharTransformer(vocab_size=tok.vocab_size)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()
        return model, tok
    except Exception as e:
        raise RuntimeError(f"Cannot load tagger: {e}")

# Прогоняет символьную модель и возвращает индексы пробелов выше порога
def predict_with_tagger(model, tok, s: str, thr: float = 0.5):
    import torch
    ids = torch.tensor([tok.encode(s)], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)  # (1, T-2)
        p_list = torch.sigmoid(logits)[0].cpu().float().view(-1).tolist()
    p = np.array(p_list, dtype=float)
    # Правила помогают отрезать очевидно неверные пробелы возле пунктуации и спецсимволов
    p = postprocess_probs(s, p, hard_clip=False)
    return [i+1 for i in range(len(p)) if p[i] > thr]

# Читает «грязный» CSV без строгого экранирования и приводит его к ожидаемым колонкам
def read_loose_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").rstrip("\r")
        if "," in header:
            h1, h2 = header.split(",", 1)
        else:
            h1, h2 = "id", "text"
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line: continue
            if "," in line:
                id_str, text = line.split(",", 1)
            else:
                id_str, text = line, ""
            id_str = id_str.strip()
            try:
                _id = int(id_str)
            except ValueError:
                continue
            rows.append((_id, text))
    df = pd.DataFrame(rows, columns=[h1.strip() or "id", h2.strip() or "text"])
    df = df.rename(columns={df.columns[0]:"id", df.columns[1]:"text"})
    return df

# CLI: читает датасет, прогоняет модель и сохраняет предсказания позиций пробелов
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="task_data.csv")
    ap.add_argument("--output", type=str, default="submission.csv")
    ap.add_argument("--weights", type=str, default="weights/char_tagger.pt")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--use_pandas", action="store_true",
                    help="Читать как обычный CSV (если файл корректно экранирован)")
    args = ap.parse_args()

    if args.use_pandas:
        df = pd.read_csv(args.input, sep=",", quotechar='"', escapechar="\\")
        # Автоматически определяем, какие колонки отвечают за текст и идентификаторы
        text_col = next((c for c in df.columns if "text" in c.lower()), None)
        if text_col is None:
            raise ValueError("Cannot find text column (needs 'text' in its name).")
        id_col = next((c for c in df.columns if c.lower() == "id"), df.columns[0])
        df = df.rename(columns={id_col:"id", text_col:"text"})
    else:
        df = read_loose_csv(args.input)

    # Подстраховка: без весов модель не запустится
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    model, tok = load_tagger(args.weights)

    preds = []
    for s in df["text"].fillna("").astype(str).tolist():
        # Модель работает по строке без дополнительных преобразований
        pos = predict_with_tagger(model, tok, s, thr=args.thr)
        preds.append("[" + ", ".join(map(str, pos)) + "]")

    # Формируем таблицу в требуемом формате сабмита
    out = df.copy()
    out["predicted_positions"] = preds
    out.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
