import asyncio
from quart import Quart, request, jsonify

from inference.model import ThreatClassifier
from inference.schemas import PredictionRequest, PredictionResponse

app = Quart(__name__)
classifier = None


@app.before_serving
async def initialize():
    """Initialize model before handling requests"""
    global classifier
    classifier = ThreatClassifier()
    print("✓ Model loaded and ready for inference")


@app.route("/health", methods=["GET"])
async def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model": "constellation_one_text"})


@app.route("/predict", methods=["POST"])
async def predict():
    """Inference endpoint for threat classification"""
    try:
        data = await request.get_json()
        req = PredictionRequest(**data)
        
        # Run blocking inference in thread pool
        result = await asyncio.to_thread(classifier.predict, req.text)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        # FIX: Ensure score and threshold are compared as floats
        # This prevents the "type str doesn't define __round__ method" error
        positive_labels = [
            label for label, score in result["predictions"].items()
            if float(score) >= float(req.threshold)
        ]
        
        # Ensure numeric values in response are actual numbers
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
        # This will catch the type mismatch and return a 500
        return jsonify({"error": str(e)}), 500

@app.route("/batch", methods=["POST"])
async def batch_predict():
    """Batch inference endpoint"""
    try:
        data = await request.get_json()
        texts = data.get("texts", [])
        threshold = data.get("threshold", 0.5)
        
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "texts must be a non-empty list"}), 400
        
        results = []
        for text in texts:
            result = await asyncio.to_thread(classifier.predict, text)
            if "error" not in result:
                positive_labels = [
                    label for label, score in result["predictions"].items()
                    if score >= threshold
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
    # Run with: hypercorn app:app --bind 0.0.0.0:8000
    app.run(debug=False)
