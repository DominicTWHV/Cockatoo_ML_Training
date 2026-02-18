import asyncio
from quart import Quart, request, jsonify

from inference.model import ThreatClassifier
from inference.schemas import PredictionRequest, PredictionResponse
from cockatoo_ml.registry import APIConfig
from cockatoo_ml.logger.context import inference_api_server_logger as logger

app = Quart(__name__)
classifier = None


@app.before_serving
async def initialize():
    # init the model into global scope so it can be reused
    global classifier
    classifier = ThreatClassifier()
    logger.info("Model loaded and ready for inference")


@app.route("/health", methods=["GET"])
async def health():
    # server health check
    return jsonify({"status": "ok", "model": APIConfig.MODEL_NAME})


@app.route("/predict", methods=["POST"])
async def predict():
    # inference endpoint for single text input
    try:
        data = await request.get_json()
        req = PredictionRequest(**data)
        
        # run blocking inference in thread pool
        result = await asyncio.to_thread(classifier.predict, req.text)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        # determine thresholds: use provided or fall back to per-label defaults
        if req.threshold is None:
            # use per-label thresholds from column_mapping
            thresholds = classifier.get_label_thresholds()
        elif isinstance(req.threshold, dict):
            # use provided per-label thresholds
            thresholds = req.threshold
        else:
            # use single threshold for all labels
            thresholds = {label: req.threshold for label in result["predictions"].keys()}
        
        positive_labels = [
            label for label, score in result["predictions"].items()
            if float(score) >= thresholds.get(label, APIConfig.DEFAULT_THRESHOLD)
        ]
        
        # call into the inference helper to get predictions and format response
        response = PredictionResponse(
            text=req.text,
            predictions=result["predictions"],
            positive_labels=positive_labels,
            top_label=result["top_label"],
            max_score=float(result["max_score"])
        )
        
        return jsonify(response.model_dump())
    
    except ValueError as e:
        return jsonify({"error": f"Validation error: {str(e)}"}), 422
    
    except Exception as e:
        # catch errors
        return jsonify({"error": str(e)}), 500

@app.route("/batch", methods=["POST"])
async def batch_predict():
    try:
        data = await request.get_json()
        texts = data.get("texts", [])
        threshold = data.get("threshold", None)
        
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "texts must be a non-empty list"}), 400
        
        # determine thresholds: use provided or fall back to per-label defaults
        if threshold is None:
            # will use per-label thresholds from column_mapping
            use_per_label = True
            thresholds = classifier.get_label_thresholds()

        elif isinstance(threshold, dict):
            # use provided per-label thresholds
            use_per_label = True
            thresholds = threshold
            
        else:
            # use single threshold for all labels
            use_per_label = False
            single_threshold = threshold
        
        results = []
        for text in texts:
            result = await asyncio.to_thread(classifier.predict, text)
            if "error" not in result:
                if use_per_label:
                    positive_labels = [
                        label for label, score in result["predictions"].items()
                        if score >= thresholds.get(label, APIConfig.DEFAULT_THRESHOLD)
                    ]
                else:
                    positive_labels = [
                        label for label, score in result["predictions"].items()
                        if score >= single_threshold
                    ]
                results.append({
                    "text": text,
                    "predictions": result["predictions"],
                    "positive_labels": positive_labels,
                    "top_label": result["top_label"],
                    "max_score": result["max_score"]
                })
        
        return jsonify({"count": len(results), "results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # run with: hypercorn app:app --bind 0.0.0.0:8000

    # development server (not for production use)
    app.run(debug=APIConfig.DEBUG, host=APIConfig.HOST, port=APIConfig.PORT)
