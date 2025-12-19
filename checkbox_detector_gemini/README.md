# Checkbox Detection using Gemini Vision LLM

This approach uses Google's Gemini multimodal LLM to detect checkboxes via zero-shot prompting. No training required. We support both Gemini 2.5 Pro and Gemini 3.0 Pro.

## Quick Start

### Setup

```bash
cd checkbox_detector_gemini
pip install -r requirements.txt
```

### Set API Key

Create a `.env` file in the repository root (already gitignored):

```bash
# In checkbox_detector/.env
GEMINI_API_KEY=your_api_key_here
```

### Run Detection

**Single image:**
```bash
# Auto-saves JSON + annotated image to runs/
python detect_checkboxes.py ../data/val/images/real.jpg

# Or specify output path
python detect_checkboxes.py ../data/val/images/real.jpg --output detected.jpg
```

**Batch prediction (for evaluation):**
```bash
# Generate predictions for all validation images
python predict.py --input ../data/val/images --output ../predictions/gemini --model gemini-2.5-pro
python predict.py --input ../data/val/images --output ../predictions/gemini30 --model gemini-3-pro-preview
```

## Models

### Gemini 3.0 Pro
- **Performance**: Near-perfect detection (99.4% recall, 100% precision at IoU=0.3)
- **mAP@50**: 0.944 (VOC standard)
- **Latency**: ~6s per image

### Gemini 2.5 Pro
- **Performance**: Good detection (86.9% recall, 86.9% precision at IoU=0.3)
- **mAP@50**: 0.570
- **Latency**: ~6s per image

Gemini 3.0 Pro shows dramatic improvement in spatial understanding and bounding box accuracy compared to 2.5 Pro.

## Output

Results are saved to `runs/` folder:
- `{input_name}_{timestamp}.json` — Detection results
- `{input_name}_{timestamp}_detected.jpg` — Annotated image

### JSON Format

```json
{
  "checkboxes": [
    {
      "box_2d": [124, 93, 140, 108],
      "label": "empty_checkbox"
    },
    {
      "box_2d": [124, 281, 140, 296],
      "label": "filled_checkbox"
    }
  ],
  "total_filled": 19,
  "total_empty": 24
}
```

Bounding boxes are normalized to 0-1000 in format `[y_min, x_min, y_max, x_max]`.

For evaluation, predictions are converted to standardized format with pixel coordinates `[x_min, y_min, x_max, y_max]`.

## How It Works

1. **Load Image**: Read image file as bytes
2. **Build Prompt**: Construct a structured prompt asking for bounding boxes
3. **Call Gemini API**: Send image + prompt to Gemini (2.5 Pro or 3.0 Pro)
4. **Parse Response**: Extract bounding box coordinates from JSON response
5. **Visualize**: Draw boxes on the image (green=empty, red=filled)

## Design Decisions

### Structured Output (JSON Schema)

We use Gemini's structured output feature with `response_mime_type="application/json"` and a JSON schema to ensure consistent, parseable responses. This prevents:
- **Parsing errors** from malformed JSON
- **Inconsistent output formats** that would require complex error handling
- **Manual string parsing** of free-form text responses

The schema enforces the exact structure we need: an array of checkboxes with bounding boxes and labels, plus total counts.

### Thinking Budget

We set the thinking budget to 0 as per Gemini documentation guidelines for object detection tasks. Object detection is a direct visual task that doesn't require extended reasoning, so disabling thinking reduces latency and cost while maintaining accuracy.

## Advantages

- **Zero training data** — Works out of the box
- **Zero GPU** — API-based, runs on any machine
- **Flexible** — Change behavior by modifying the prompt
- **Excellent accuracy** — Gemini 3.0 Pro achieves near-perfect detection

## Limitations

- **Latency** — ~6 seconds per image (API round-trip)
- **Rate limits** — API quotas apply
- **Internet required** — Needs API access

## Project Structure

```
checkbox_detector_gemini/
├── detect_checkboxes.py    # Main detection script (single image)
├── predict.py              # Batch prediction script (for evaluation)
├── evaluate.py             # Evaluation script
├── requirements.txt        # Python dependencies
├── runs/                   # Output folder (auto-created)
│   ├── *.json              # Detection results
│   └── *_detected.jpg      # Annotated images
└── README.md               # This file
```

## Comparison with Other Approaches

| Aspect | OpenCV | YOLO | Gemini |
|--------|--------|------|--------|
| Training Data | None | Required | None |
| GPU | No | Optional | No (API) |
| Latency | ~10ms | ~250ms | ~6s |
| Accuracy (mAP@50) | 0.408 | 0.681 | 0.944 (3.0 Pro) |
