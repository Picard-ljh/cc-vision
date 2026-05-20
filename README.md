# cc-vision

PDF / image document parser using Qwen3-VL-235B via Aliyun DashScope API.

- Academic-grade LaTeX formula extraction
- BBOX-based chart/figure detection and cropping
- Markdown table output
- Batch page processing

## Quick Install

```bash
git clone https://github.com/Picard-ljh/cc-vision.git
cd cc-vision
bash install.sh
```

Set your DashScope API key:
```bash
export DASHSCOPE_API_KEY="your-key"
```

## Usage

```bash
cc-vision "document.pdf"
cc-vision "document.pdf" --pages 1-10
cc-vision "image.png"
```

Output: `.vision_parsed.md` (markdown) + `.vision_assets/` (cropped figures)
