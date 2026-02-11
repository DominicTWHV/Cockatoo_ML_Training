import os


class APIConfig:
    # api server settings (development defaults)
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    
    # model name
    MODEL_NAME = "constellation_one_text"
    
    # default prediction threshold (when not provided in request)
    DEFAULT_THRESHOLD = 0.5
    
    # threshold constraints (do not change unless you know what you're doing)
    MIN_THRESHOLD = 0.0
    MAX_THRESHOLD = 1.0
    
    # text length validation
    # prevents empty text inputs which can cause issues
    MIN_TEXT_LENGTH = 1


class WebhookConfig:
    enable = False

    # default POST endpoint to send training telemetry to
    DEFAULT_WEBHOOK_URL = "https://yourdomain.com/<path>"
    
    @classmethod
    def get_webhook_url(cls):
        return os.getenv("METRICS_WEBHOOK_URL", cls.DEFAULT_WEBHOOK_URL)
    
    @classmethod
    def get_api_key(cls):
        return os.getenv("METRICS_API_KEY")
