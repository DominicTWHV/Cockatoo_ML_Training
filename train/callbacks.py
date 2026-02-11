import requests
from datetime import datetime
from transformers import TrainerCallback

from logging.context import model_training_logger as logger


class LiveMetricsWebhookCallback(TrainerCallback):
    # metrics hook callback to send training and evaluation metrics to a remote endpoint in real-time
    
    def __init__(self, endpoint_url: str, auth_token: str = None, experiment_id: str = None):
        self.endpoint_url = endpoint_url
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        self.experiment_id = experiment_id or "default-run"

    def _send_metrics(self, payload: dict):
        try:
            # dispatch metrics to api server
            r = requests.post(self.endpoint_url, json=payload, headers=self.headers, timeout=5)
            r.raise_for_status()

        except Exception as e:
            logger.warning(f"Failed to send metrics: {e}")

    def on_log(self, args, state, control, **kwargs):
        if state.log_history:
            # grab most recent log entry
            latest_metrics = state.log_history[-1].copy()
            
            payload = {
                "experiment_id": self.experiment_id,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "metrics": latest_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self._send_metrics(payload)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            payload = {
                "experiment_id": self.experiment_id,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "is_eval": True
            }
            self._send_metrics(payload)
