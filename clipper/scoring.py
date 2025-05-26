import cv2
import numpy as np
import librosa
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re
from datasets import Dataset

# сидируем модель для эмбеддингов (нужна для novelty)
_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
hf_sent = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment",
    tokenizer="blanchefort/rubert-base-cased-sentiment",
    device=0,
    truncation=True,
    max_length=512,
)
cyrillic_re = re.compile(r'[\u0400-\u04FF]')

# 1. TF-IDF и Sentiment

def get_tfidf_scores(blocks, stop_words=None):
    texts = [b["text"] for b in blocks]
    vec = TfidfVectorizer(stop_words=stop_words)
    X = vec.fit_transform(texts)
    return X.mean(axis=1).A1

def get_sentiment_scores(blocks):
    sia = SentimentIntensityAnalyzer()
    texts = [b["text"] for b in blocks]
    ds = Dataset.from_dict({"text": texts})

    def batch_sentiment(batch):
        raw_scores = [None] * len(batch["text"])

        for idx, text in enumerate(batch["text"]):
            if not cyrillic_re.search(text):
                raw_scores[idx] = sia.polarity_scores(text)["compound"]

        rus_indices = [i for i, t in enumerate(batch["text"]) if cyrillic_re.search(t)]
        if rus_indices:
            rus_texts = [batch["text"][i] for i in rus_indices]
            results = hf_sent(rus_texts, batch_size=16)
            for i, res in zip(rus_indices, results):
                score = res["score"] if res["label"] == "POSITIVE" else -res["score"]
                raw_scores[i] = score

        return {"raw_sent": raw_scores}

    # Шаг 3. Применяем .map() с батчингом
    ds = ds.map(
        batch_sentiment,
        batched=True,
        batch_size=16,
        remove_columns=["text"]
    )

    # Шаг 4. Нормировка Min–Max в [0,1]
    raw = np.array(ds["raw_sent"], dtype=float).reshape(-1, 1)
    normed = MinMaxScaler((0, 1)).fit_transform(raw).ravel()

    return normed

# 2. Voice Energy и динамический диапазон

def get_voice_energy_score(video_path, start, end):
    try:
        y, sr = librosa.load(video_path, sr=None, offset=start, duration=end-start)
        return float(np.mean(y**2))
    except:
        return 0.0

def get_dynamic_range_scores(video_path, blocks):
    raw = []
    for b in blocks:
        try:
            y, sr = librosa.load(video_path, sr=None, offset=b["start"], duration=b["end"]-b["start"])
            raw.append(float(y.max() - y.min()))
        except:
            raw.append(0.0)
    return MinMaxScaler((0,1)).fit_transform(np.array(raw).reshape(-1,1)).ravel()

def normalize_audio_scores(raw_scores):
    return MinMaxScaler((0,1)).fit_transform(np.array(raw_scores).reshape(-1,1)).ravel()

# 3. Scene Change Score

def get_scene_change_score(video_path, start, end, threshold=0.3):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start*1000)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    changes = total = 0
    while cap.get(cv2.CAP_PROP_POS_MSEC) < end*1000:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        if cv2.countNonZero(diff) / diff.size > threshold:
            changes += 1
        total += 1
        prev_gray = gray
    cap.release()
    return changes / max(1, total)

# 4. Speech Rate

def get_speech_rate_scores(blocks):
    raw = []
    for b in blocks:
        wc = len(b["text"].split())
        dur = max(0.001, b["end"] - b["start"])
        raw.append(wc / dur)
    return MinMaxScaler((0,1)).fit_transform(np.array(raw).reshape(-1,1)).ravel()

# 5. Novelty (1 – макс косинус со схожестью предыдущих N)

def get_novelty_scores(blocks, window=5):
    texts = [b["text"] for b in blocks]
    embs = _embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    novelty = []
    for i in range(len(embs)):
        if i == 0:
            novelty.append(1.0)
        else:
            start = max(0, i - window)
            sims = cosine_similarity([embs[i]], embs[start:i])[0]
            novelty.append(1.0 - float(sims.max()))
    return MinMaxScaler((0,1)).fit_transform(np.array(novelty).reshape(-1,1)).ravel()

# 6. Keyword Density

def get_keyword_density_scores(blocks):
    raw = []
    for b in blocks:
        words = b["text"].lower().split()
        unique = len(set(words))
        total = len(words) or 1
        raw.append(unique / total)
    return MinMaxScaler((0,1)).fit_transform(np.array(raw).reshape(-1,1)).ravel()

# 7. Финальный скор (без LLM)

def compute_final_score(
    tfidf_score, sentiment_score, voice_score,
    scene_score, speech_rate, novelty_score,
    dynamic_range, keyword_density,
    weights=None
):
    if weights is None:
        weights = {
            "tfidf": 0.17,
            "sent": 0.06,
            "voice":0.10,
            "scene":0.07,
            "rate": 0.15,
            "novel":0.20,
            "dyn":  0.10,
            "kd":   0.15
        }
    return (
          tfidf_score    * weights["tfidf"]
        + sentiment_score* weights["sent"]
        + voice_score    * weights["voice"]
        + scene_score    * weights["scene"]
        + speech_rate    * weights["rate"]
        + novelty_score  * weights["novel"]
        + dynamic_range  * weights["dyn"]
        + keyword_density* weights["kd"]
    )

# 8. Сбор всех метрик в одну функцию

def score_blocks(blocks, video_path):
    tfidf     = get_tfidf_scores(blocks)
    sent      = get_sentiment_scores(blocks)
    voice_raw = [get_voice_energy_score(video_path, b["start"], b["end"]) for b in blocks]
    voice     = normalize_audio_scores(voice_raw)
    dyn       = get_dynamic_range_scores(video_path, blocks)
    scene     = [get_scene_change_score(video_path, b["start"], b["end"]) for b in blocks]
    rate      = get_speech_rate_scores(blocks)
    novel     = get_novelty_scores(blocks)
    kd        = get_keyword_density_scores(blocks)

    for i, b in enumerate(blocks):
        b["tfidf_score"]    = tfidf[i]
        b["sentiment_score"]= sent[i]
        b["voice_score"]    = voice[i]
        b["dynamic_range"]  = dyn[i]
        b["scene_score"]    = scene[i]
        b["speech_rate"]    = rate[i]
        b["novelty_score"]  = novel[i]
        b["keyword_density"]= kd[i]
        b["final_score"]    = compute_final_score(
            tfidf[i], sent[i], voice[i], scene[i],
            rate[i], novel[i], dyn[i], kd[i]
        )
    return blocks
