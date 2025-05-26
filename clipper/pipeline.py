import yaml
import librosa
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from clipper.scoring import (
    get_tfidf_scores,
    get_sentiment_scores,
    get_voice_energy_score,
    normalize_audio_scores,
    get_scene_change_score,
    get_speech_rate_scores,
    compute_final_score,
    get_novelty_scores,
    get_dynamic_range_scores,
    get_keyword_density_scores,
)

# Загружаем модель для семантического анализа текста
_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_config(path="configs/settings.yaml"):
    """Чтение конфигурационного YAML-файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_pauses(audio_path, min_silence, top_db):
    """Находит паузы в аудио, которые длиннее заданного порога."""
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=top_db)

    pauses = []
    last_end = 0
    for start, end in intervals:
        silence_duration = (start - last_end) / sr
        if silence_duration >= min_silence:
            pauses.append(last_end / sr)
        last_end = end

    print(f"найдено пауз: {len(pauses)}")
    return pauses


def detect_semantic_boundaries(segments, threshold):
    """Находит смысловые границы на основе схожести соседних сегментов."""
    texts = [seg["text"] for seg in segments]
    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    boundaries = []
    for i in range(len(texts) - 1):
        if sim_matrix[i, i + 1] < threshold:
            boundaries.append(segments[i + 1]["start"])

    print(f"найдено смысловых границ: {len(boundaries)}")
    return boundaries


def clean_boundaries(boundaries, merge_window):
    """Объединяет близкорасположенные границы, чтобы избежать лишней детализации."""
    if not boundaries:
        return []

    sorted_bounds = sorted(boundaries)
    result = []
    ref = sorted_bounds[0]

    for b in sorted_bounds[1:]:
        if b - ref >= merge_window:
            result.append(ref)
            ref = b

    result.append(ref)
    print(f"границ после очистки: {len(result)}")
    return result


def split_by_boundaries(segments, boundaries):
    """Разбивает сегменты на блоки по найденным границам,
    корректируя каждый блок так, чтобы он начинался и заканчивался
    на границах предложений."""
    if not segments:
        return []

    sentence_end_re = re.compile(r'[\.!?]$')
    ordered = sorted(segments, key=lambda s: s["start"])
    starts = [s["start"] for s in ordered]
    ends   = [s["end"]   for s in ordered]

    all_points = [starts[0]] + sorted(boundaries) + [ends[-1]]
    blocks = []

    for t_start, t_end in zip(all_points, all_points[1:]):
        i = next((idx for idx, s in enumerate(ordered) if s["start"] >= t_start), None)
        if i is None:
            continue

        valid_js = [idx for idx, s in enumerate(ordered) if s["end"] <= t_end]
        if not valid_js:
            continue

        j = max(valid_js)

        while i > 0 and not sentence_end_re.search(ordered[i - 1]["text"].strip()):
            i -= 1

        while j < len(ordered) - 1 and not sentence_end_re.search(ordered[j]["text"].strip()):
            j += 1

        if i > j:
            continue

        sub = ordered[i:j+1]
        block = {
            "start": sub[0]["start"],
            "end":   sub[-1]["end"],
            "text":  " ".join(s["text"] for s in sub).strip()
        }
        blocks.append(block)

    print(f"разбито на блоки: {len(blocks)}")
    return blocks


def pipeline(raw_segments, cfg):
    print(f"pipeline: получили {len(raw_segments)} raw_segments")

    # Получаем путь к видео из конфига
    video_path = cfg["paths"].get("input_video") or \
                 (cfg["paths"].get("videos") or [None])[0]

    if not video_path:
        raise ValueError("В конфигурации не указан путь до видео (input_video или videos)")

    print(f"обрабатываем видео: {video_path}")

    # --- Этап 1: Поиск границ ---
    print("Поиск пауз и смен тематики...")
    pauses = detect_pauses(
        video_path,
        cfg["pause"]["min_silence"],
        cfg["pause"]["top_db"]
    )

    semantics = detect_semantic_boundaries(
        raw_segments,
        cfg["semantic"]["threshold"]
    )

    boundaries = clean_boundaries(pauses + semantics, cfg["semantic"]["merge_window"])

    # --- Этап 2: Разделение на блоки ---
    print("Разделение на блоки...")
    blocks = split_by_boundaries(raw_segments, boundaries)

    # --- Этап 3: Удаление слишком коротких блоков ---
    min_dur = cfg["pipeline"]["min_duration"]
    blocks = [b for b in blocks if (b["end"] - b["start"]) >= min_dur]
    print(f"Отфильтровано по min_duration={min_dur}, осталось: {len(blocks)} блоков")

    if not blocks:
        print("Нет подходящих блоков после фильтрации")
        return []

    # --- Этап 4: Подсчет метрик для каждого блока ---
    print("Подсчет метрик...")
    tfidf_scores = get_tfidf_scores(blocks)
    sentiment_scores = get_sentiment_scores(blocks)
    voice_raw = [get_voice_energy_score(video_path, b["start"], b["end"]) for b in blocks]
    voice_scores = normalize_audio_scores(voice_raw)
    scene_scores = [get_scene_change_score(video_path, b["start"], b["end"]) for b in blocks]
    rate_scores = get_speech_rate_scores(blocks)
    novelty_scores = get_novelty_scores(blocks)
    dyn_scores = get_dynamic_range_scores(video_path, blocks)
    kd_scores = get_keyword_density_scores(blocks)
    print("Метрики успешно получены")

    # --- Этап 5: Подсчет итогового рейтинга ---
    print("Финальный скоринг...")
    for i, b in enumerate(blocks):
        b["tfidf_score"] = tfidf_scores[i]
        b["sentiment_score"] = sentiment_scores[i]
        b["voice_score"] = voice_scores[i]
        b["scene_score"] = scene_scores[i]
        b["rate_score"] = rate_scores[i]
        b["novelty_score"] = novelty_scores[i]
        b["dynamic_range"] = dyn_scores[i]
        b["keyword_density"] = kd_scores[i]
        b["final_score"] = compute_final_score(
            tfidf_score=b["tfidf_score"],
            sentiment_score=b["sentiment_score"],
            voice_score=b["voice_score"],
            scene_score=b["scene_score"],
            speech_rate=b["rate_score"],
            novelty_score=b["novelty_score"],
            dynamic_range=b["dynamic_range"],
            keyword_density=b["keyword_density"]
        )

    # --- Этап 6: Отбор лучших блоков ---
    top_n = cfg["pipeline"]["top_n"]
    top_blocks = sorted(blocks, key=lambda b: b["final_score"], reverse=True)[:top_n]
    print(f"Отобрано топ-{top_n} блоков по финальному скору")

    return top_blocks
