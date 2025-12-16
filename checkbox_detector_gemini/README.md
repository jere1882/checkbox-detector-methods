# Checkbox Detection using Gemini Vision LLM

This approach uses Google's Gemini 2.5 Pro multimodal LLM to detect checkboxes via zero-shot prompting. No training required.

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

```bash
# Auto-saves JSON + annotated image to runs/
python detect_checkboxes.py ../data/val/images/real.jpg

# Or specify output path
python detect_checkboxes.py ../data/val/images/real.jpg output.jpg
```

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

## How It Works

1. **Load Image**: Read image file as bytes
2. **Build Prompt**: Construct a structured prompt asking for bounding boxes
3. **Call Gemini API**: Send image + prompt to Gemini 2.5 Pro
4. **Parse Response**: Extract bounding box coordinates from JSON response
5. **Visualize**: Draw boxes on the image (green=empty, red=filled)

## Advantages

- **Zero training data** — Works out of the box
- **Zero GPU** — API-based, runs on any machine
- **Flexible** — Change behavior by modifying the prompt

## Limitations

- **Cost** — ~$0.01-0.05 per image
- **Latency** — 2-5 seconds per image
- **Rate limits** — API quotas apply

## Project Structure

```
checkbox_detector_gemini/
├── detect_checkboxes.py    # Main detection script
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
| Latency | ~10ms | ~250ms | ~3s |
| Cost | Free | Free | ~$0.02/image |
