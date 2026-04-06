import os
import logging
import pandas as pd
from pathlib import Path
from bio_ml_agent.utils.config import get_config
import gradio as gr

log = logging.getLogger("bio_ml_agent")
config = get_config()

def update_file_list():
    work_dir = Path(config.workspace.base_dir).expanduser().resolve()
    if not work_dir.exists(): return gr.update(choices=[])
    allowed = ['.csv', '.json', '.txt', '.log', '.html', '.png', '.jpg', '.jpeg', '.py', '.md']
    files = [str(p.relative_to(work_dir)) for p in work_dir.rglob("*")
             if p.is_file() and p.suffix.lower() in allowed]
    return gr.update(choices=sorted(files))

def preview_file(filepath):
    if not filepath: return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    work_dir = Path(config.workspace.base_dir).expanduser().resolve()
    full_path = work_dir / filepath
    if not full_path.exists(): return gr.update(visible=False), gr.update(value="Dosya yok", visible=True), gr.update(visible=False), gr.update(visible=False)

    ext = full_path.suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(full_path, nrows=100)
            return gr.update(value=df, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=str(full_path), visible=True)
        else:
            with open(full_path, 'r', encoding='utf-8') as f:
                return gr.update(visible=False), gr.update(value=f.read(10000), visible=True), gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return gr.update(visible=False), gr.update(value=str(e), visible=True), gr.update(visible=False), gr.update(visible=False)

def list_xai_projects(current_project=None):
    work_dir = Path(config.workspace.base_dir).expanduser().resolve()
    if not work_dir.exists(): return gr.update(choices=[])
    projects = sorted([d.name for d in work_dir.iterdir() if d.is_dir()], reverse=True)
    return gr.update(choices=projects, value=current_project)

def load_xai_plots(project_name):
    if not project_name: return []
    work_dir = Path(config.workspace.base_dir).expanduser().resolve()
    project_dir = work_dir / project_name
    if not project_dir.exists(): return []
    return [str(p) for p in project_dir.rglob("*.png")]
