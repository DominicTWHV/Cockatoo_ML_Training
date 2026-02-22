# API Endpoints (inference server):

**Endpoint:** `/health`
**Method:** `GET`
**Description:** Checks if the server is responding and healthy. Returns a simple JSON response indicating the status and the active model version.

```json
{
  "status": "ok",
  "model": "constellation_one_text"
}
```

---

**Endpoint:** `/predict`
**Method:** `POST`
**Description:** Classifies a single string. Supports a global threshold or a per-label mapping.
**Request Body:**

```json
{
  "text": "The text to classify",
  "threshold": 0.5 
}

```

> **Note:** `threshold` is optional. It can be a **float** (0.5), a **dictionary** (`{"LABEL_1": 0.8, "LABEL_2": 0.4}`), or `null`. If `null`, the system defaults to per-label thresholds defined in the classifier configuration.

**Response Body:**

```json
{
  "text": "The text to classify",
  "predictions": {
    "LABEL_1": 0.12,
    "LABEL_2": 0.05,
    "LABEL_3": 0.9944
  },
  "positive_labels": ["LABEL_3"],
  "top_label": "LABEL_3",
  "max_score": 0.9944
}

```

> **Note:** The `predictions` field contains **all labels** from the model with their respective confidence scores, regardless of the threshold. The `positive_labels` field is a convenience filter showing only labels that meet the threshold criteria.

---

**Endpoint:** `/batch`
**Method:** `POST`
**Description:** Classifies a list of strings in a single request.
**Request Body:**

```json
{
  "texts": [
    "First text to classify",
    "Second text to classify"
  ],
  "threshold": {
    "LABEL_3": 0.90
  }
}

```

**Response Body:**

```json
{
  "count": 2,
  "results": [
    {
      "text": "First text to classify",
      "predictions": { "LABEL_1": 0.05, "LABEL_2": 0.02, "LABEL_3": 0.9944 },
      "positive_labels": ["LABEL_3"],
      "top_label": "LABEL_3",
      "max_score": 0.9944
    },
    {
      "text": "Second text to classify",
      "predictions": { "LABEL_1": 0.10, "LABEL_2": 0.05, "LABEL_3": 0.85 },
      "positive_labels": [],
      "top_label": "LABEL_3",
      "max_score": 0.85
    }
  ]
}

```

> **Note:** The `predictions` field in each result contains **all labels** from the model with their respective confidence scores, regardless of the threshold. The `positive_labels` field is a convenience filter showing only labels that meet the threshold criteria.