import os
import subprocess
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

def download_video(src, output_dir='downloads'):
    """
    Скачивает src как URL через yt-dlp, либо, если это локальный файл, возвращает его.
    Если yt-dlp не смог скачать (EOF/SSL и т.п.), пытается запустить ffmpeg напрямую.
    """
    # 1) Локальный файл?
    if not (src.startswith('http://') or src.startswith('https://')) and os.path.isfile(src):
        return src

    os.makedirs(output_dir, exist_ok=True)
    # имя в папке downloads, по video_id.mp4
    video_id = os.path.splitext(os.path.basename(src))[0]
    outpath = os.path.join(output_dir, f"{video_id}.mp4")

    # 2) Пытаемся yt-dlp
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'nocheckcertificate': True,
        'extractor_args': {
            'youtube': [
                'getpot_bgutil_baseurl=https://bgpot.b64.dev'
            ]
        }
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(src, download=True)
            filename = ydl.prepare_filename(info)
            # приводим имя к .mp4
            base, _ = os.path.splitext(filename)
            final = base + '.mp4'
            # переименуем, если надо
            if final != outpath:
                os.replace(final, outpath)
            return outpath
    except DownloadError:
        # 3) FALLBACK: ffmpeg напрямую
        print(f"⚠️ yt-dlp failed, falling back to ffmpeg for {src}")
        cmd = [
            'ffmpeg', '-y', '-i', src,
            '-c', 'copy',
            outpath
        ]
        # запустим внешний процесс
        subprocess.run(cmd, check=True)
        return outpath
