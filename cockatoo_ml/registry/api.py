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

    # text length validation
    # prevents empty text inputs which can cause issues
    MIN_TEXT_LENGTH = 1


class WebhookConfig:
    # enable training telemetry
    enable = False

    # enable validation telemetry (sent to a separate endpoint)
    enable_validation = False

    # default POST endpoint to send training telemetry to
    DEFAULT_WEBHOOK_URL = "https://yourdomain.com/<path>"

    # default POST endpoint to send validation telemetry to
    DEFAULT_VALIDATION_WEBHOOK_URL = "https://yourdomain.com/<validation-path>"
    
    @classmethod
    def get_webhook_url(cls):
        return os.getenv("METRICS_WEBHOOK_URL", cls.DEFAULT_WEBHOOK_URL)

    @classmethod
    def get_validation_webhook_url(cls):
        return os.getenv("METRICS_VALIDATION_WEBHOOK_URL", cls.DEFAULT_VALIDATION_WEBHOOK_URL)
    
    @classmethod
    def get_api_key(cls):
        return os.getenv("METRICS_API_KEY")
