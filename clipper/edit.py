import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip
from ultralytics import YOLO
from cv2 import saliency
import moviepy.config as mpy_conf

# Для MoviePy–TextClip
mpy_conf.IMAGEMAGICK_BINARY = "magick"

# Детекторы инициализируем один раз
_face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_object_model = YOLO("yolov8n.pt").to("cuda")
print("YOLO device:", _object_model.device)  # убедиться, что модель на GPU
_saliency = saliency.StaticSaliencySpectralResidual_create()


def crop(input_video_path: str, output_video_path: str,
         start_time: float, end_time: float = None):
    clip = VideoFileClip(input_video_path).subclip(start_time, end_time)
    print("Segment created.")
    clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")


def caption(input_video_path: str, output_video_path: str, transcript: dict,
            max_chars_per_line: int = 30, max_words_per_phrase: int = 4,
            fontsize: int = 70, font: str = 'Arial-Bold',
            stroke_color: str = 'black', stroke_width: int = 1,
            position: tuple = ('center', 0.6)):
    import textwrap

    def wrap_text(text, max_chars, max_width):
        lines = textwrap.wrap(text, width=max_chars)
        wrapped = []
        for line in lines:
            if len(line) > max_width:
                wrapped.extend(textwrap.wrap(line, width=max_width))
            else:
                wrapped.append(line)
        return '\n'.join(wrapped)

    def split_text(text, max_words):
        words = text.split()
        return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    vid = VideoFileClip(input_video_path)
    max_w = vid.size[0]
    subs = []

    for seg in transcript.get('segments', []):
        start, end, text = seg['start'], seg['end'], seg['text']
        wrapped = wrap_text(text, max_chars_per_line, max_w)
        phrases = split_text(wrapped, max_words_per_phrase)
        dur = end - start

        for phr in phrases:
            pd = max(dur / len(phrases), 1)
            txt = (TextClip(phr, fontsize=fontsize, color='white', font=font,
                            stroke_color=stroke_color, stroke_width=stroke_width)
                   .set_position(position, relative=True)
                   .set_start(start).set_end(start + pd))
            subs.append(txt)
            start += pd
            dur -= pd

    out = CompositeVideoClip([vid] + subs)
    out.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    print("Caption done.")


def caption_words(input_video_path: str,
                  output_video_path: str,
                  transcript: dict,
                  start_offset: float = 0.0,
                  fontsize: int = 50,
                  font: str = "Arial-Bold",
                  color: str = "white",
                  stroke_color: str = "black",
                  stroke_width: int = 1,
                  position: tuple = ("center", 0.8)):
    video = VideoFileClip(input_video_path)
    clip_duration = video.duration
    subs = []

    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            w_start, w_end = w["start"], w["end"]
            if w_end <= start_offset or w_start >= start_offset + clip_duration:
                continue
            rel_start = max(0, w_start - start_offset)
            rel_end   = min(clip_duration, w_end - start_offset)

            txt = (TextClip(
                        w["word"],
                        fontsize=fontsize,
                        font=font,
                        color=color,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width
                    )
                    .set_position(position, relative=True)
                    .set_start(rel_start)
                    .set_end(rel_end))
            subs.append(txt)

    final = CompositeVideoClip([video] + subs)
    final.write_videofile(
        output_video_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    print("Word-by-word captions done.")


def _smooth_coordinates(curr, prev, alpha):
    if prev is None:
        return curr
    return (
        int(alpha * curr[0] + (1 - alpha) * prev[0]),
        int(alpha * curr[1] + (1 - alpha) * prev[1])
    )


def clip(input_video_path: str,
         output_video_path: str,
         face_check_interval: int = 90,
         clip_aspect_ratio: float = 9/16,
         jitter_thresh: int = 95,
         ma_window: int = 75):
    """
    Smart Reframe с двустадийным сглаживанием:
      1) базовая детекция раз в face_check_interval кадров
      2) скользящее среднее (ma_window кадров) по всей траектории
      3) порог джиттера для жесткой фильтрации мелких сдвигов
      4) проверка и клип координат, чтобы не выходить за края кадра
    """
    video = VideoFileClip(input_video_path)
    fps = video.fps
    w, h = video.size

    # Шаг 1: пробегаем по видео, детектим центры в ключевых кадрах и запоминаем их.
    raw_centers = []
    prev_center = (w // 2, h // 2)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Обходим кадры по времени, но не храним их — только рассчитываем центр на каждом кадре
    for i, t in enumerate(np.arange(0, video.duration, 1 / fps)):
        if i % face_check_interval == 0:
            frame = video.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                cand = (x + fw // 2, y + fh // 2)
            else:
                cand = prev_center
            # жёсткий порог джиттера
            if abs(cand[0] - prev_center[0]) < jitter_thresh and abs(cand[1] - prev_center[1]) < jitter_thresh:
                cand = prev_center
        else:
            cand = prev_center

        raw_centers.append(cand)
        prev_center = cand

    # Шаг 2: сглаживаем траекторию (скользящее среднее)
    half = ma_window // 2
    padded = [raw_centers[0]] * half + raw_centers + [raw_centers[-1]] * half
    centers = [
        (
            int(sum(c[0] for c in padded[i:i + ma_window]) / ma_window),
            int(sum(c[1] for c in padded[i:i + ma_window]) / ma_window),
        )
        for i in range(len(raw_centers))
    ]

    target_h = h
    target_w = int(h * clip_aspect_ratio)

    # Шаг 3: определяем функцию обрезки одного кадра
    def reframe(get_frame, t):
        frame = get_frame(t)
        idx = min(int(t * fps), len(centers) - 1)
        cx, cy = centers[idx]

        x1 = max(0, cx - target_w // 2)
        y1 = max(0, cy - target_h // 2)
        x2, y2 = x1 + target_w, y1 + target_h

        if x2 > w:
            x1, x2 = w - target_w, w
        if y2 > h:
            y1, y2 = h - target_h, h

        cropped = frame[y1:y2, x1:x2]
        if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
            cropped = cv2.resize(cropped, (target_w, target_h))
        return cropped

    # Шаг 4: применяем fl к видео и сразу пишем на диск
    transformed = video.fl(reframe, apply_to=['mask', 'video'])
    transformed.write_videofile(
        output_video_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )

def clip_static(input_video_path: str,
                output_video_path: str,
                face_check_interval: int = 10,
                padding: float = 0.1,
                clip_aspect_ratio: float = 9/16):
    """
    Плоский (static) reframe: единый bbox по всему видео.
    """
    vid = VideoFileClip(input_video_path)
    fps = vid.fps
    frames = list(vid.iter_frames(fps=fps, dtype="uint8"))
    h, w, _ = frames[0].shape

    # детект и union bbox
    boxes = []
    for idx, frame in enumerate(tqdm(frames, desc="Detect bboxes")):
        if idx % face_check_interval != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_detector.detectMultiScale(gray, 1.1, 4)
        if faces:
            x,y,fw,fh = max(faces, key=lambda b: b[2]*b[3])
            boxes.append((x, y, x+fw, y+fh))

    if not boxes:
        # fallback: центр
        cx, cy = w//2, h//2
        tw = int(h * clip_aspect_ratio)
        th = h
        x0 = cx - tw//2
        y0 = cy - th//2
        boxes = [(x0, y0, x0+tw, y0+th)]

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    # aspect ratio + padding
    box_w, box_h = x2-x1, y2-y1
    target_h = box_h
    target_w = int(box_h * clip_aspect_ratio)
    if box_w < target_w:
        delta = (target_w - box_w) // 2
        x1 = max(0, x1 - delta)
        x2 = min(w, x2 + (target_w - box_w - delta))

    pad_x = int((x2-x1)*padding)
    pad_y = int((y2-y1)*padding)
    x1, y1, x2, y2 = max(0, x1-pad_x), max(0, y1-pad_y), min(w, x2+pad_x), min(h, y2+pad_y)

    cropped = [frame[y1:y2, x1:x2] for frame in frames]
    out = ImageSequenceClip(cropped, fps=fps)
    out.audio = vid.audio
    out.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    print("Static Clip ready.")


def caption_highlight(
    input_video_path: str,
    output_video_path: str,
    transcript: dict,
    max_chars_per_line: int = 40,
    fontsize: int = 60,
    font: str = "Arial-Bold",
    dim_color: str = "gray",
    highlight_color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 1,
    position: tuple = ("center", 0.8),
):
    vid = VideoFileClip(input_video_path)
    clips = [vid]

    for seg in transcript.get("segments", []):
        # объединяем слова в фразу, при желании можно дополнительно wrap/textwrap
        words = seg.get("words", [])
        if not words:
            continue
        phrase = " ".join(w["word"] for w in words)
        # слой с фразой
        base = (TextClip(phrase, fontsize=fontsize, font=font,
                         color=dim_color, stroke_color=stroke_color,
                         stroke_width=stroke_width)
                .set_position(position, relative=True)
                .set_start(seg["start"])
                .set_end(seg["end"]))
        clips.append(base)

        x0, y0 = None, None

        for w in words:
            # время появления и скрытия слова относительно начала видео
            ws = w["start"]
            we = w["end"]
            highlight = (TextClip(w["word"], fontsize=fontsize, font=font,
                                  color=highlight_color, stroke_color=stroke_color,
                                  stroke_width=stroke_width)
                         .set_position(position, relative=True)
                         .set_start(ws)
                         .set_end(we))
            clips.append(highlight)

    final = CompositeVideoClip(clips)
    final.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
