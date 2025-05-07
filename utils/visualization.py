# Визуализация прогресса
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path

class TrainingVisualizer:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.logger = logging.getLogger("TrainingVisualizer")
        self.log_dir.mkdir(exist_ok=True)

    def create_training_report(self, metrics: Dict[str, List], output_file=None):
        """Создание интерактивного отчета по обучению"""
        if not metrics:
            self.logger.warning("No metrics provided for visualization")
            return
            
        df = pd.DataFrame(metrics)
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Loss Over Time", 
            "Accuracy Metrics",
            "Learning Rate Schedule",
            "Resource Usage"
        ))
        
        # График потерь
        fig.add_trace(
            go.Scatter(x=df.index, y=df['loss'], name="Training Loss"),
            row=1, col=1
        )
        if 'val_loss' in df:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['val_loss'], name="Validation Loss"),
                row=1, col=1
            )
        
        # Метрики точности
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in df:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[metric], name=metric.capitalize()),
                    row=1, col=2
                )
        
        # График learning rate
        if 'lr' in df:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['lr'], name="Learning Rate"),
                row=2, col=1
            )
        
        # Использование ресурсов
        if 'gpu_usage' in df:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['gpu_usage'], name="GPU Usage (%)"),
                row=2, col=2
            )
        if 'memory_usage' in df:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['memory_usage'], name="Memory Usage (MB)"),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Training Metrics Report",
            showlegend=True,
            height=900,
            template="plotly_white"
        )
        
        if output_file:
            output_path = self.log_dir / output_file
            fig.write_html(str(output_path))
            self.logger.info(f"Saved report to {output_path}")
            return str(output_path)
        
        return fig

    def create_resource_dashboard(self, metrics: List[Dict]):
        """Создание дашборда использования ресурсов"""
        if not metrics:
            return None
            
        df = pd.DataFrame(metrics)
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "CPU Usage", 
            "Memory Usage",
            "GPU Memory",
            "Network Activity"
        ))
        
        # CPU
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu']['total'], name="Total CPU %"),
            row=1, col=1
        )
        
        # Memory
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory']['percent'], name="Memory %"),
            row=1, col=2
        )
        
        # GPU Memory
        if 'gpu' in df.iloc[0]:
            for gpu in df.iloc[0]['gpu'].keys():
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['gpu'].apply(lambda x: x[gpu]['memory']['allocated']),
                        name=f"{gpu} Allocated"
                    ),
                    row=2, col=1
                )
        
        # Network
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['network']['bytes_recv'], name="Bytes Received"),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['network']['bytes_sent'], name="Bytes Sent"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Resource Usage Dashboard",
            showlegend=True,
            height=900
        )
        
        return fig

    def save_plot(self, fig, filename):
        """Сохранение графика в файл"""
        output_path = self.log_dir / filename
        fig.write_html(str(output_path))
        self.logger.info(f"Saved plot to {output_path}")
        return str(output_path)
