import argparse, json, random, os, sys
import numpy as np
import torch
import pandas as pd

# Модель и утилиты для токенизации и постобработки предсказаний
from model_char_tagger import TinyCharTransformer, CharTokenizer
from rules import postprocess_probs

# Читает и нормализует строки из parquet или текстового файла с фильтрами по длине
def read_lines(path, min_len=5, max_len=300, max_samples=None, column="text"):
    path = os.fspath(path)
    if isinstance(path, bytes):
        fs_enc = sys.getfilesystemencoding() or "utf-8"
        path = path.decode(fs_enc, errors="ignore")
    lines = []
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, engine="pyarrow")
        if column not in df.columns:
            raise KeyError(f"Колонка '{column}' не найдена в {path}. Доступно: {list(df.columns)}")
        series = df[column].dropna().astype(str)
        for s in series:
            s = " ".join(s.strip().split())
            if not s:
                continue
            if min_len <= len(s) <= max_len:
                lines.append(s)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                s = " ".join(ln.strip().split())
                if not s:
                    continue
                if min_len <= len(s) <= max_len:
                    lines.append(s)
    random.shuffle(lines)
    if max_samples:
        lines = lines[:max_samples]
    return lines

# Превращает исходный текст в строку без пробелов и множество истинных позиций пробелов
def to_squeezed_and_truth(s: str):
    """Возвращает (s_no_space, truth_set_of_positions)"""
    s = " ".join(s.strip().split())
    no = s.replace(" ", "")
    pos = set()
    i_no = 0
    for i, ch in enumerate(s[:-1]):
        if ch != " " and s[i+1] == " ":
            pos.add(i_no + 1)
        if ch != " ":
            i_no += 1
    return no, pos

# Считает F1 между множествами истинных и предсказанных позиций
def f1_sets(true_set, pred_set):
    tp = len(true_set & pred_set)
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0
    prec = tp / len(pred_set)
    rec  = tp / len(true_set)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

# Загружает веса, создаёт токенизатор и модель-символьный трансформер
def load_tagger(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    tok = CharTokenizer()
    model = TinyCharTransformer(vocab_size=tok.vocab_size)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, tok

# Применяет модель к строке без пробелов и возвращает множество позиций пробелов
@torch.no_grad()
def predict_positions(model, tok, s: str, thr: float):
    ids = torch.tensor([tok.encode(s)], dtype=torch.long)
    logits = model(ids)  # (1, T-2)
    p = torch.sigmoid(logits)[0].cpu().float().view(-1).numpy()
    # Смягчаем модельные вероятности правилом, чтобы учесть пунктуацию и другие эвристики
    p = postprocess_probs(s, p, hard_clip=False)
    return {i+1 for i, v in enumerate(p) if v > thr}

# CLI-точка входа: читает корпус, перебирает пороги и ищет лучший по F1
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="txt: одна строка — один чистый текст")
    ap.add_argument("--weights", required=True, help="weights/char_tagger.pt")
    ap.add_argument("--max_samples", type=int, default=20000, help="сколько строк взять для оценки (перемешиваются)")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=300)
    ap.add_argument("--grid", type=str, default="0.30:0.90:0.02", help="start:end:step для порога")
    ap.add_argument("--column", type=str, default="text", help="имя колонки с текстом (для parquet)")
    ap.add_argument("--save_json", type=str, default=None, help="путь для сохранения результатов (опц.)")
    args = ap.parse_args()

    start, end, step = map(float, args.grid.split(":"))
    thr_grid = np.arange(start, end + 1e-9, step)

    # Подготавливаем корпус: фильтруем и перемешиваем строки
    lines = read_lines(args.corpus, args.min_len, args.max_len, args.max_samples, args.column)
    if not lines:
        raise SystemExit("Корпус пуст после фильтров — проверь входные данные.")

    model, tok = load_tagger(args.weights)

    # Готовим пары (строка без пробелов, позиции настоящих пробелов)
    pairs = [to_squeezed_and_truth(s) for s in lines]

    results = []
    for thr in thr_grid:
        f1s = []
        for no, truth in pairs:
            # Прогоняем модель и считаем F1 для каждой строки
            pred = predict_positions(model, tok, no, thr)
            f1s.append(f1_sets(truth, pred))
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        results.append({"thr": round(float(thr), 4), "f1": round(mean_f1, 6)})

    # Лучший порог подбираем по средней F1
    best = max(results, key=lambda x: x["f1"]) if results else {"thr": 0.5, "f1": 0.0}

    # печать таблицы
    print("thr\tF1")
    for r in results:
        print(f"{r['thr']:.2f}\t{r['f1']:.4f}")
    print("\nBEST:")
    print(f"thr={best['thr']:.4f}\tF1={best['f1']:.4f}")

    if args.save_json:
        out = {"grid": results, "best": best}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.save_json}")

if __name__ == "__main__":
    main()
