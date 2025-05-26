import os
import logging
from clipper.edit import crop, clip, caption_words, caption_highlight

# utils.py
# Сохраняет готовые шортсы и удаляет промежуточные файлы, чтобы в output_dir остались только финальные клипы с субтитрами

def save_shorts(blocks, input_video_path, output_dir, subtitles=False, subtitle_params=None):
    """
    Генерирует шортсы и сохраняет их в output_dir.
    Если subtitles=True, добавляет субтитры по параметрам из subtitle_params.
    :param blocks: список блоков с полями start, end, text и final_score
    :param input_video_path: путь к исходному видео
    :param output_dir: папка для сохранения шортсов
    :param subtitles: флаг добавления субтитров
    :param subtitle_params: словарь с параметрами для caption_words (кроме start_offset)
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, blk in enumerate(blocks, 1):
        s = blk.get("start")
        e = blk.get("end")
        title = "_".join(blk.get("text", "").split()[:5]).replace("?", "")
        seg_fn = f"{idx}_{title}_segment.mp4"
        clip_fn = f"{idx}_{title}_clip.mp4"
        seg_p = os.path.join(output_dir, seg_fn)
        clip_p = os.path.join(output_dir, clip_fn)

        # 1) Обрезаем сегмент и создаём клип
        crop(input_video_path, seg_p, s, e)
        clip(seg_p, clip_p)

        # 2) Добавляем субтитры
        if subtitles and subtitle_params:
            sub_fn = f"{idx}_{title}_subtitled.mp4"
            sub_p = os.path.join(output_dir, sub_fn)
            # Собираем параметры для caption_words
            params = {
                "input_video_path": clip_p,
                "output_video_path": sub_p,
                "transcript": subtitle_params.get("transcript"),
                "start_offset": s,
                "fontsize": subtitle_params.get("fontsize"),
                "font": subtitle_params.get("font"),
                "color": subtitle_params.get("color"),
                "stroke_color": subtitle_params.get("stroke_color"),
                "stroke_width": subtitle_params.get("stroke_width"),
                "position": subtitle_params.get("position"),
            }
            # Вызываем caption_words с корректными аргументами
            caption_words(**params)
            print(f"Субтитры добавлены: {os.path.basename(sub_p)}")

            # Удаляем промежуточные файлы сегмента и необработанного клипа
            for path in (seg_p, clip_p):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as err:
                    logging.warning(f"Не удалось удалить {path}: {err}")



def clean_tmp(local_path, cfg=None):
    """
    Безопасная очистка временных файлов пайплайна:
    - удаляет скачанное видео только если оно в download.output_dir
    - удаляет только транскрипты (whisper_segments, transcript_txt)
    """
    # 1) Удаляем видеофайл, если он в директории загрузок
    try:
        download_dir = cfg.get("download", {}).get("output_dir") if cfg else None
        if download_dir:
            abs_dl = os.path.abspath(download_dir)
            abs_lp = os.path.abspath(local_path)
            if abs_lp.startswith(abs_dl) and os.path.exists(abs_lp):
                os.remove(abs_lp)
                logging.info(f"Удалён временный видеофайл: {abs_lp}")
        else:
            # по умолчанию удаляем из 'downloads'
            if os.path.exists(local_path) and os.path.abspath(local_path).startswith(os.path.abspath("downloads")):
                os.remove(local_path)
                logging.info(f"Удалён временный видеофайл: {local_path}")
    except Exception as e:
        logging.warning(f"Не удалось удалить видео {local_path}: {e}")

    # 2) Удаляем только указанные файлы транскрипции
    if cfg and cfg.get("paths"):
        for key in ("whisper_segments", "transcript_txt"):
            fpath = cfg["paths"].get(key)
            try:
                if fpath and os.path.exists(fpath):
                    os.remove(fpath)
                    logging.info(f"Удалён файл транскрипции: {fpath}")
            except Exception as e:
                logging.warning(f"Не удалось удалить {fpath}: {e}")
