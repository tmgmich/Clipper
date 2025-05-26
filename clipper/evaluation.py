import os
import pandas as pd
import matplotlib.pyplot as plt

# Папка для хранения логов и графиков
RESULTS_DIR = 'results'
METRICS_CSV = os.path.join(RESULTS_DIR, 'metrics_log.csv')


def init_metrics_log():
    """
    Создает файл CSV с заголовками, если он не существует.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(METRICS_CSV):
        df = pd.DataFrame(columns=[
            'video_id', 'block_id',
            'tfidf', 'sentiment', 'voice', 'scene',
            'rate', 'novelty', 'dynamic_range', 'keyword_density',
            'final_score'
        ])
        df.to_csv(METRICS_CSV, index=False)


def log_block_metrics(video_id: str, blocks: list):
    """
    Добавляет строки с метриками каждого блока в лог-файл.
    :param video_id: Идентификатор или имя видео
    :param blocks: Список словарей с рассчитанными метриками
    """
    init_metrics_log()
    df = pd.read_csv(METRICS_CSV)
    rows = []
    for b in blocks:
        rows.append({
            'video_id': video_id,
            'block_id': b.get('id', ''),
            'tfidf': b['tfidf_score'],
            'sentiment': b['sentiment_score'],
            'voice': b['voice_score'],
            'scene': b['scene_score'],
            'rate': b['rate_score'],
            'novelty': b['novelty_score'],
            'dynamic_range': b['dynamic_range'],
            'keyword_density': b['keyword_density'],
            'final_score': b['final_score'],
        })
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(METRICS_CSV, index=False)


def plot_metric_distribution(metric_name: str):
    """
    Строит и сохраняет гистограмму распределения заданной метрики.
    """
    df = pd.read_csv(METRICS_CSV)
    plt.figure()
    df[metric_name].hist()
    plt.title(f'Distribution of {metric_name}')
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(RESULTS_DIR, f'{metric_name}_dist.png'))
    plt.close()


def plot_metric_scatter(x_metric: str, y_metric: str):
    """
    Строит и сохраняет scatter-диаграмму для двух метрик.
    """
    df = pd.read_csv(METRICS_CSV)
    plt.figure()
    df.plot.scatter(x=x_metric, y=y_metric)
    plt.title(f'{y_metric} vs {x_metric}')
    plt.savefig(os.path.join(RESULTS_DIR, f'{y_metric}_vs_{x_metric}.png'))
    plt.close()


def generate_all_plots():
    """
    Генерирует гистограммы для всех метрик и scatter для пар.
    """
    metrics = [
        'tfidf', 'sentiment', 'voice', 'scene', 'rate',
        'novelty', 'dynamic_range', 'keyword_density', 'final_score'
    ]
    for m in metrics:
        plot_metric_distribution(m)
    # Можно ограничить количество пар или выбирать наиболее интересные
    for x in metrics:
        for y in metrics:
            if x != y:
                plot_metric_scatter(x, y)
