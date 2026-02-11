import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
from pathlib import Path


class ThreatClassifier:
    def __init__(self, model_path: str = "constellation_one_text"):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Loading model on device: {'cuda:0' if self.device == 0 else 'cpu'}")

        # Option 1: Use pipeline (recommended - simplest & handles tokenization)
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                device=self.device,
                torch_dtype=torch.float16 if self.device == 0 else None,  # memory saving on GPU
                return_all_scores=True,
                batch_size=8,  # good default for inference
            )
            print("Pipeline loaded successfully")
        except Exception as e:
            print(f"Pipeline load failed: {e}")
            # Fallback: manual load
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            if self.device == 0:
                self.model = self.model.cuda()
            self.model.eval()
            self.classifier = None
            print("Fallback manual model loaded")

        # Get label mapping from config
        self.id2label = self._get_id2label(model_path)
        print(f"Detected labels: {list(self.id2label.values())}")

    def _get_id2label(self, model_path: str) -> Dict[int, str]:
        import json
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return config.get("id2label", {})
        return {}

    def predict(self, text: str) -> Dict:
        try:
            if self.classifier is not None:
                # Pipeline way (preferred)
                # With return_all_scores=True, returns [[{dict}, {dict}]] for single input
                raw_results = self.classifier(text, truncation=True, max_length=512)
                
                # Extract the list of label-score dicts (handle both batch and single input)
                if isinstance(raw_results, list) and len(raw_results) > 0:
                    scores_list = raw_results[0] if isinstance(raw_results[0], list) else raw_results
                    probs = {item['label']: round(item['score'], 4) for item in scores_list}
                else:
                    return {"error": "Unexpected pipeline output format"}
            else:
                # Manual way
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                if self.device == 0:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    logits = self.model(**inputs).logits

                probs_raw = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                probs = {self.id2label.get(i, f"LABEL_{i}"): round(float(score), 4) for i, score in enumerate(probs_raw)}

            # Build response
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_label = sorted_probs[0][0] if sorted_probs else None
            max_score = sorted_probs[0][1] if sorted_probs else None

            return {
                "predictions": probs,
                "top_label": top_label,
                "max_score": max_score
            }

        except Exception as e:
            return {"error": str(e)}