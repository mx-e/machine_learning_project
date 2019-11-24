import os
import plotly.graph_objects as go

def check_if_results_dir(path):
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, 'log.txt')):
            return True
    return False 

def generate_plot(results_dir):
    print(f'creating plot for {results_dir}...')
    ep_reward, no_frames = [], []
    with open(os.path.join(results_dir, 'log.txt'), 'r') as file:
        log = file.readlines()
    log = [line.rstrip('\n') for line in log] 
    for entry in log:
        if '50M frames' in entry:
            break
        if 'time' in entry:
            entry = entry.split(',')
            ep_reward.append(float(entry[3].split(' ')[3]))
            no_frames.append(float(entry[2].split(' ')[2].strip('M')) * 1e6)
    fig = go.Figure(data=go.Scatter(x=no_frames, y=ep_reward))
    fig.update_layout(
        #title = f'{results_dir.strip("/.")}',
        xaxis_title = 'no. of frames trained on',
        yaxis_title = 'avg. episode reward'
        )
    fig.write_image(os.path.join(results_dir, f'plot-{results_dir.strip("/.")}.pdf'))
    fig.write_image(os.path.join(results_dir, f'plot-{results_dir.strip("/.")}.png'))



listOfFiles = os.listdir('./')
for entry in listOfFiles:
        fullPath = os.path.join('./', entry)
        if check_if_results_dir(fullPath):
           generate_plot(fullPath)



