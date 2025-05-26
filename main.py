import os
import json
import time
import argparse
import yaml

from clipper.transcript import Transcript
from clipper.pipeline import pipeline
from clipper.download import download_video
from clipper.utils import clean_tmp, save_shorts
from clipper.evaluation import log_block_metrics, generate_all_plots

def parse_args():
    parser = argparse.ArgumentParser(description="Генератор шортсов из видео (батч + очередь)")
    parser.add_argument("--config", "-c", type=str,
                        default="configs/settings.yaml",
                        help="Путь к YAML-конфигу")
    parser.add_argument("--input", "-i", type=str,
                        help="(Опционально) одиночный URL или путь к видео — заменит список")
    parser.add_argument("--min_duration", type=float,
                        help="(Опционально) переопределить минимальную длину клипа (с)")
    parser.add_argument("--top_n", type=int,
                        help="(Опционально) переопределить число шортсов")
    parser.add_argument("--center", type=str,
                        choices=["auto","face","object","saliency","motion"],
                        help="(Опционально) режим центрирования")
    parser.add_argument("--subtitles", action="store_true",
                        help="(Опционально) добавить субтитры")
    return parser.parse_args()

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ——— Override CLI ➞ config
    if not cfg.get("paths"):
        cfg["paths"] = {}
    if args.input:
        # если задан одиночный input, убираем старое поле и ставим список
        cfg["paths"]["videos"] = [args.input]
    if args.min_duration is not None:
        cfg.setdefault("pipeline", {})["min_duration"] = args.min_duration
    if args.top_n is not None:
        cfg.setdefault("pipeline", {})["top_n"] = args.top_n
    if args.center:
        cfg.setdefault("pipeline", {})["center_mode"] = args.center
    if args.subtitles:
        cfg.setdefault("pipeline", {})["subtitles"] = True

    # ——— Список видео: сначала новый список, иначе одиночный input_video
    paths_cfg = cfg["paths"]
    videos = paths_cfg.get("videos")
    if not videos:
        inp = paths_cfg.get("input_video")
        videos = [inp] if inp else []
    # ——— Очередь
    queue_items = (cfg.get("queue") or {}).get("items") or []

    sources = videos + queue_items

    for src in list(sources):
        print(f"\n▶ Обрабатываем: {src}")

        # 1) Автозагрузка или локальный файл
        download_dir = cfg.get("download", {}).get("output_dir", "downloads")
        local_path = download_video(src, output_dir=download_dir)
        # Имя видео без расширения
        video_id = os.path.splitext(os.path.basename(local_path))[0]

        # Папки из конфига (создадим, если надо)
        whisper_dir = cfg["paths"].get("whisper_dir", ".")
        transcript_dir = cfg["paths"].get("transcript_dir", ".")
        os.makedirs(whisper_dir, exist_ok=True)
        os.makedirs(transcript_dir, exist_ok=True)

        # Полные пути для этого видео
        seg_j = os.path.join(whisper_dir, f"{video_id}_segments.json")
        txt_j = os.path.join(transcript_dir, f"{video_id}_transcript.txt")

        # Обновляем cfg, чтобы downstream его использовал
        cfg["paths"]["whisper_segments"] = seg_j
        cfg["paths"]["transcript_txt"] = txt_j
        cfg["paths"]["input_video"] = local_path

        try:
            # 2) Whisper транскрипция

            print(f"Запускаем Whisper для {video_id} ({cfg['whisper']['model']})…")
            whisper_result = Transcript.get_whisper(
                input_video_path=local_path,
                whisper_model=cfg["whisper"]["model"],
                word_timestamps=cfg["whisper"].get("word_timestamps", False)
            )
            with open(seg_j, "w", encoding="utf-8") as f:
                json.dump(whisper_result, f, ensure_ascii=False, indent=2)
            txt = Transcript.get_text(whisper_result["segments"])
            with open(txt_j, "w", encoding="utf-8") as f:
                f.write(txt)
            print("Новый транскрипт сохранён.")

            # 3) Сегментация и скоринг
            t0 = time.perf_counter()
            blocks = pipeline(raw_segments=whisper_result["segments"], cfg=cfg)
            print(f"Пайплайн завершён за {time.perf_counter() - t0:.2f} сек.")

            # 4) Сохранение шортсов
            out_dir = paths_cfg.get("output_dir", "output")
            os.makedirs(out_dir, exist_ok=True)
            save_shorts(
                blocks,
                input_video_path=local_path,
                output_dir=out_dir,
                subtitles=cfg["pipeline"].get("subtitles", False),
                subtitle_params={
                    "transcript": whisper_result,
                    "start_offset": None,  # save_shorts сам перебирает блоки
                    "fontsize": cfg["pipeline"].get("sub_fontsize", 50),
                    "font": cfg["pipeline"].get("sub_font", "Arial-Bold"),
                    "color": cfg["pipeline"].get("sub_color", "white"),
                    "stroke_color": cfg["pipeline"].get("sub_stroke_color", "black"),
                    "stroke_width": cfg["pipeline"].get("sub_stroke_width", 1),
                    "position": tuple(cfg["pipeline"].get("sub_position", ("center", 0.8)))
                }
            )

            # 5) Логирование метрик
            vid_id = os.path.splitext(os.path.basename(local_path))[0]
            log_block_metrics(video_id=vid_id, blocks=blocks)

        finally:
            # 6) Автоочистка временных файлов
            clean_tmp(local_path, cfg)

        # 7) Если это элемент очереди — удаляем и сохраняем config
        if src in queue_items:
            queue_items.remove(src)
            save_config(cfg, args.config)

    # 8) Генерируем отчётные графики
    generate_all_plots()
    print(f"\n Все клипы в «{out_dir}», графики в «results/»")

if __name__ == "__main__":
    main()
