import plotly.express as px
import os
import torch
from datetime import datetime

def store_results_data(results, snapshots):
    if not os.path.exists('history'):
        os.mkdir("history")
    if not os.path.exists('results'):
        os.mkdir("results")
    date_string = datetime.today().strftime('%Y-%m-%d')
    path_name_history = f'history/{date_string}'
    if not os.path.exists(path_name_history):
        os.mkdir(path_name_history)
    store_results_plots(results, path_name_history)
    store_results_plots(results, 'results')
    torch.save(snapshots, f'{path_name_history}/snapshots')


def store_results_plots(results, path):
    fig1 = px.line(results, 'episode', 'avg_reward')
    fig1.write_image(f'{path}/fig1.pdf')

    fig2 = px.line(results, 'episode', 'avg_ep_len')
    fig2.write_image(f'{path}/fig2.pdf')
