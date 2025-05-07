import time
import threading
import psutil
import torch
from typing import Dict, Any
from datetime import datetime
import logging

class ResourceMonitor:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.metrics = []
        self._stop_event = threading.Event()
        self.logger = logging.getLogger("ResourceMonitor")
        
    def start(self):
        """Запуск мониторинга в отдельном потоке"""
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("Resource monitoring started")

    def stop(self):
        """Остановка мониторинга"""
        self._stop_event.set()
        self.thread.join()
        self.logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            metrics = self._collect_metrics()
            self.metrics.append(metrics)
            time.sleep(self.interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Сбор метрик системы"""
        timestamp = datetime.now().isoformat()
        cpu = psutil.cpu_percent(percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()
        
        metrics = {
            "timestamp": timestamp,
            "cpu": {
                "usage": cpu,
                "total": sum(cpu)/len(cpu)
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free
            },
            "network": {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv
            }
        }
        
        if torch.cuda.is_available():
            metrics["gpu"] = self._get_gpu_metrics()
            
        return metrics

    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Сбор метрик GPU"""
        import torch.cuda as cuda
        
        metrics = {}
        for i in range(cuda.device_count()):
            metrics[f"gpu_{i}"] = {
                "name": cuda.get_device_name(i),
                "memory": {
                    "total": cuda.get_device_properties(i).total_memory,
                    "allocated": cuda.memory_allocated(i),
                    "cached": cuda.memory_reserved(i)
                },
                "utilization": cuda.utilization(i)
            }
        return metrics

    def get_report(self, last_n: int = 10) -> Dict[str, Any]:
        """Получение отчета по последним метрикам"""
        if not self.metrics:
            return {}
            
        recent = self.metrics[-last_n:]
        return {
            "cpu_avg": sum(m['cpu']['total'] for m in recent)/len(recent),
            "memory_avg": sum(m['memory']['percent'] for m in recent)/len(recent),
            "gpu_usage": [m.get('gpu', {}) for m in recent]
        }
