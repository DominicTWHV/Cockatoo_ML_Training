import requests
import json

from datetime import datetime, timezone

from transformers import TrainerCallback

from cockatoo_ml.registry import CallbackConfig
from cockatoo_ml.logger.context import model_training_logger as logger

class LiveMetricsWebhookCallback(TrainerCallback):
    # metrics hook callback to send training and evaluation metrics to remote endpoints in real-time
    
    def __init__(
        self,
        training_endpoint_url: str = None,
        validation_endpoint_url: str = None,

        auth_token: str = None,

        experiment_id: str = None,

        enable_training: bool = False,
        enable_validation: bool = False
    ):
        self.training_endpoint_url = training_endpoint_url
        self.validation_endpoint_url = validation_endpoint_url

        self.enable_training = enable_training
        self.enable_validation = enable_validation

        self.headers = {"Content-Type": "application/json"}

        if auth_token:
            self.headers["Authorization"] = auth_token
            
        self.experiment_id = experiment_id or CallbackConfig.DEFAULT_EXPERIMENT_ID

    def _send_metrics(self, endpoint_url: str, payload: dict):
        try:
            # dispatch metrics to api server

            json_payload = json.dumps(payload, default=str)

            logger.info(f"Posting data to {endpoint_url}: {json_payload}")
            
            r = requests.post(
                endpoint_url, 
                data=json_payload,
                headers=self.headers, 
                timeout=CallbackConfig.WEBHOOK_TIMEOUT
            )
            r.raise_for_status()

        except Exception as e:
            logger.warning(f"Failed to send metrics: {e}")

    def on_log(self, args, state, control, **kwargs):
        if not self.enable_training or not self.training_endpoint_url:
            return # only send metrics if training telemetry is enabled and endpoint is valid
        
        if state.log_history:
            # grab most recent log entry
            latest_metrics = state.log_history[-1].copy()
            
            payload = {
                "experiment_id": self.experiment_id,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "metrics": latest_metrics,
                "timestamp": str(datetime.now(timezone.utc).replace(tzinfo=None))
            }
            
            self._send_metrics(self.training_endpoint_url, payload)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.enable_validation or not self.validation_endpoint_url:
            return # only send metrics if validation telemetry is enabled and endpoint is valid
        
        if metrics:

            payload = {
                "experiment_id": self.experiment_id,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "metrics": metrics,
                "timestamp": str(datetime.now(timezone.utc).replace(tzinfo=None)),
                "is_eval": True
            }
            self._send_metrics(self.validation_endpoint_url, payload)
