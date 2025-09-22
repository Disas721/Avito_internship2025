# rules.py
import numpy as np

# Наборы пунктуации, возле которых пробелы чаще всего не нужны
_punct_right = set(",.:;!?»)]}")
_punct_left = set("«([{\"'")

# Эвристически корректирует вероятности пробелов с учётом соседних символов
def postprocess_probs(s: str, p: np.ndarray, hard_clip: bool = True) -> np.ndarray:
    L = len(s)
    if L <= 1: return p
    for i in range(L-1):
        if s[i+1] in _punct_right:
            p[i] = 0.0 if hard_clip else p[i]*0.2
    for i in range(L-1):
        if s[i] in _punct_left:
            p[i] = 0.0 if hard_clip else p[i]*0.2
    for i in range(L-1):
        if s[i].isdigit() and s[i+1].isalpha():
            p[i] = 0.0 if hard_clip else p[i]*0.2
        if s[i+1] == "%":
            p[i] = 0.0 if hard_clip else p[i]*0.2
    # Простейшая детекция алфавита помогает не ломать смешанные слова
    def script(c):
        if c.isdigit(): return "d"
        o = ord(c)
        if 0x0400 <= o <= 0x04FF: return "ru"
        if 0x0061 <= o <= 0x007A or 0x0041 <= o <= 0x005A: return "en"
        return "o"
    for i in range(L-1):
        if script(s[i]) != script(s[i+1]):
            # На стыке разных алфавитов разрешаем пробел с высокой вероятностью
            p[i] = p[i] * 0.5 + 0.5*(1.0 - 1e-6)
    return p
