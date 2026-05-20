#!/usr/bin/env python3
"""cc-vision — 全局多模态文档解析工具，基于 Qwen3-VL-235B-A22B-Instruct"""

import argparse
import base64
import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

API_KEY_ENV = "DASHSCOPE_API_KEY"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_ID = "qwen3-vl-235b-a22b-instruct"

SYSTEM_PROMPT = """你是一个高精度的学术文档解析助手，请对每一页执行完整的视觉识别：

**文字与公式提取**：
- 提取页面中所有文字，保持原文排版结构。
- 所有数学公式必须输出为最严谨的标准 LaTeX 源码。
- 严格保留所有多重上下标、嵌套括号、隐式变量声明。
- 涉及导数、度量张量等高阶推导时，必须保留完整展开形式，严禁使用任何简写符号或省略变量。
- 行内公式使用 $...$，块级公式使用 $$...$$。

**表格**：
- 输出为 Markdown 表格格式，保留所有行列结构。

**图表与插图（视觉定位）**：
- 对页面中的每一张图表、数据可视化、几何拓扑插图，必须先输出一个定位标记：
  <!-- BBOX: page=N x1=<左> y1=<上> x2=<右> y2=<下> -->
- 每个坐标值都是整数 0-999，x1,y1 为图表左上角，x2,y2 为图表右下角。
- 四个坐标值必须全部提供，缺一不可。
- 坐标取整个图表的最小外接矩形，**必须包含**标题、坐标轴标签、图例等外围文字。
- 定位标记后紧跟图表的完整数据级描述：
  * 图表类型（折线图/柱状图/散点图/饼图/流程图/拓扑图等）
  * 坐标轴标签与单位
  * 所有可见数据点的具体数值（按系列逐一列举）
  * 图例信息与对应数值
  * 关键趋势与异常点
- 此描述是纯文本模型理解图表的唯一依据，必须包含图中所有可见数据。
- **重要**：纯文字页面（标题页、摘要、目录、正文段落）、纯表格页面、纯公式页面绝不能输出 BBOX 标记。只有页面中确实存在独立的图表、数据可视化、几何插图、流程图、照片等视觉元素时，才输出 BBOX。封面、摘要、目录等排版元素不算图表。
- 如果页面没有图表/插图，不要输出 BBOX 标记。"""

# BBOX 正则 — 三种格式兼容：
#   新格式: <!-- BBOX: page=N x1=<左> y1=<上> x2=<右> y2=<下> -->
#   旧4坐标: <!-- BBOX: page=N,x1,y1,x2,y2 -->
#   旧3坐标: <!-- BBOX: page=N,x1,x2,y1 -->
BBOX_RE_LABELED = re.compile(
    r"<!--\s*BBOX:\s*page=(\d+)\s+"
    r"x1=[\"<]?(\d+)[\">]?\s+"
    r"y1=[\"<]?(\d+)[\">]?\s+"
    r"x2=[\"<]?(\d+)[\">]?\s+"
    r"y2=[\"<]?(\d+)[\">]?\s*-->"
)
BBOX_RE_COMMA_4 = re.compile(
    r"<!--\s*BBOX:\s*page=(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*-->"
)
BBOX_RE_COMMA_3 = re.compile(
    r"<!--\s*BBOX:\s*page=(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*-->"
)

# 自适应安全边距：最小 3%，最大 15%，但不超过与邻居的中点
CROP_MARGIN_MIN = 0.03
CROP_MARGIN_MAX = 0.15

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
PDF_EXTENSION = ".pdf"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def die(msg: str, code: int = 1):
    print(f"[cc-vision] 错误: {msg}", file=sys.stderr)
    sys.exit(code)


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        die(
            f"未设置环境变量 {API_KEY_ENV}\n"
            f"请执行: setx {API_KEY_ENV} \"your-api-key\"\n"
            f"获取 Key: https://bailian.console.alibabacloud.com/"
        )
    return key


def parse_pages(raw: str | None, total: int) -> list[int]:
    """将物理页码范围解析为 0-indexed 页号列表。"""
    if not raw:
        return list(range(total))

    pages: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a.strip()), int(b.strip())
        else:
            start = end = int(part)

        if start < 1 or end > total:
            die(f"页码 {start}-{end} 超出范围（共 {total} 页）")
        if start > end:
            die(f"页码范围无效: {start}-{end}")

        for p in range(start, end + 1):
            pages.append(p - 1)  # 物理页码 → 0-indexed

    return sorted(set(pages))


def encode_image(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/png")
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def render_pdf_page(doc, page_idx: int, dpi: int = 300):
    """渲染 PDF 页面为 PIL Image 和 PNG bytes。"""
    import io as _io
    import fitz as _fitz
    page = doc[page_idx]
    mat = _fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    buf = pix.tobytes("png")
    img = Image.open(_io.BytesIO(buf)).convert("RGB")
    return img, buf, pix.width, pix.height


# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------

def build_messages(page_specs: list[dict]) -> list[dict]:
    """
    构建单批次的 messages。
    page_specs: [{"page_num": int, "data_url": str}, ...]
    """
    content: list[dict] = []
    page_labels = []
    for spec in page_specs:
        content.append({
            "type": "image_url",
            "image_url": {"url": spec["data_url"]},
        })
        page_labels.append(f"第 {spec['page_num']} 页")

    n = len(page_specs)
    pages_str = "、".join(page_labels)
    user_text = (
        f"请解析以下 {n} 张页面图片（{pages_str}）。"
        f"对每一页严格按以下格式输出：\n\n"
        f"## 第 N 页\n\n"
        f"[本页文字内容与 LaTeX 公式]\n\n"
        f"<!-- BBOX: page=N x1=<左> y1=<上> x2=<右> y2=<下> -->\n"
        f"**图 N：** [图表完整数据级描述]\n\n"
        f"注意：纯文字页面、标题页、摘要页、目录页不要输出 BBOX 标记。"
        f"只有数据可视化图表（折线图、柱状图、饼图等）、几何插图、流程图或照片才需要 BBOX。"
    )
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def call_api(client: OpenAI, messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.1,
        max_tokens=30000,
    )
    content = resp.choices[0].message.content
    if content is None:
        die("模型返回为空，请检查 API Key 和配额")
    return content


# ---------------------------------------------------------------------------
# 后处理：BBOX 解析与切图
# ---------------------------------------------------------------------------

def process_bboxes(
    text: str,
    page_images: dict[int, Image.Image],
    page_dimensions: dict[int, tuple[int, int]],
    assets_dir: Path,
    page_order: list[int] | None = None,
) -> str:
    """
    扫描 BBOX 标记 → 收集所有边界框 → 自适应边距 → 物理裁切 → 替换为图片链接。
    兼容三种格式（优先级从高到低）：
      1. 带标签: <!-- BBOX: page=N x1=<左> y1=<上> x2=<右> y2=<下> -->
      2. 逗号4坐标: <!-- BBOX: page=N,x1,y1,x2,y2 -->
      3. 逗号3坐标: <!-- BBOX: page=N,x1,x2,y1 --> (y2 推断)

    自适应边距：每张图表向外扩展 3-10%，但不超过与同页相邻图表的中点。
    """
    assets_dir.mkdir(exist_ok=True)

    # --- BBOX 文本位置 → 文件物理页码 ---
    # 不信任模型输出的页号，按 ## 第 N 页 的出现顺序映射到 page_order
    section_header_re = re.compile(r"^##\s+第\s+(\d+)\s+页\s*$", re.MULTILINE)
    section_starts: list[int] = [m.start() for m in section_header_re.finditer(text)]

    if page_order:
        n_found = len(section_starts)
        n_expected = len(page_order)
        if n_found != n_expected:
            print(
                f"[cc-vision] 警告: 模型输出 {n_found} 个页面标题，"
                f"预期 {n_expected} 个，按位置匹配",
                file=sys.stderr,
            )

    def resolve_page(bbox_pos: int) -> int:
        section_idx = -1
        for i, pos in enumerate(section_starts):
            if bbox_pos >= pos:
                section_idx = i
            else:
                break
        if page_order and 0 <= section_idx < len(page_order):
            return page_order[section_idx]
        if section_idx >= 0:
            return section_idx + 1
        if page_order:
            return page_order[0]
        return 1

    # --- 阶段 1：收集所有 BBOX（归一化坐标）---
    # 每个条目: {"page": int, "x1": int, "y1": int, "x2": int, "y2": int, "start": int, "end": int}
    all_bboxes: list[dict] = []

    # Priority 1: 带标签格式
    for m in list(BBOX_RE_LABELED.finditer(text)):
        pn = resolve_page(m.start())
        x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        x1, x2 = sorted([max(0, min(999, x1)), max(0, min(999, x2))])
        y1, y2 = sorted([max(0, min(999, y1)), max(0, min(999, y2))])
        all_bboxes.append({"page": pn, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                           "start": m.start(), "end": m.end()})

    # Priority 2: 逗号 4 坐标
    for m in list(BBOX_RE_COMMA_4.finditer(text)):
        if any(b["start"] == m.start() for b in all_bboxes):
            continue
        pn = resolve_page(m.start())
        x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        x1, x2 = sorted([max(0, min(999, x1)), max(0, min(999, x2))])
        y1, y2 = sorted([max(0, min(999, y1)), max(0, min(999, y2))])
        all_bboxes.append({"page": pn, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                           "start": m.start(), "end": m.end()})

    # Priority 3: 逗号 3 坐标（同 page 内推断 y2）
    by_page_3: dict[int, list[re.Match]] = {}
    for m in list(BBOX_RE_COMMA_3.finditer(text)):
        if any(b["start"] == m.start() for b in all_bboxes):
            continue
        pn = resolve_page(m.start())
        by_page_3.setdefault(pn, []).append(m)

    for pn, matches in by_page_3.items():
        _, ph = page_dimensions.get(pn, (0, 999))
        for i, m in enumerate(matches):
            x1 = int(m.group(2))
            x2 = int(m.group(3))
            y1 = int(m.group(4))
            x1, x2 = sorted([max(0, min(999, x1)), max(0, min(999, x2))])
            y1 = max(0, min(999, y1))
            y2 = 999
            for j in range(i + 1, len(matches)):
                nx1 = int(matches[j].group(2))
                ny1 = int(matches[j].group(4))
                if abs(nx1 - x1) < 100:
                    y2 = ny1
                    break
            all_bboxes.append({"page": pn, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "start": m.start(), "end": m.end()})

    # --- 阶段 2：按页分组，计算自适应边距 ---
    by_page: dict[int, list[dict]] = {}
    for b in all_bboxes:
        by_page.setdefault(b["page"], []).append(b)

    for pn, bboxes in by_page.items():
        pw, ph = page_dimensions.get(pn, (0, 0))
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            # 计算每条边的自适应扩展量（归一化空间）
            def safe_expand(edge_val: float, dim: float, direction: str) -> float:
                """返回该方向的安全扩展量（归一化单位）。"""
                desired = dim * CROP_MARGIN_MAX
                minimum = dim * CROP_MARGIN_MIN
                limit = desired

                for other in bboxes:
                    if other is bbox:
                        continue
                    ox1, oy1, ox2, oy2 = other["x1"], other["y1"], other["x2"], other["y2"]

                    if direction == "down":
                        # 同列下方最近邻
                        if abs(ox1 - x1) < 150 and oy1 > y2:
                            gap = oy1 - y2
                            limit = min(limit, gap / 2)
                    elif direction == "up":
                        if abs(ox1 - x1) < 150 and oy2 < y1:
                            gap = y1 - oy2
                            limit = min(limit, gap / 2)
                    elif direction == "right":
                        if abs(oy1 - y1) < 150 and ox1 > x2:
                            gap = ox1 - x2
                            limit = min(limit, gap / 2)
                    elif direction == "left":
                        if abs(oy1 - y1) < 150 and ox2 < x1:
                            gap = x1 - ox2
                            limit = min(limit, gap / 2)

                return max(minimum, min(desired, limit))

            mx1 = safe_expand(x1, bw, "left")
            my1 = safe_expand(y1, bh, "up")
            mx2 = safe_expand(x2, bw, "right")
            my2 = safe_expand(y2, bh, "down")

            # 归一化 → 像素（非对称自适应边距裁切）
            crop_x1 = max(0, int((x1 - mx1) * pw / 1000))
            crop_y1 = max(0, int((y1 - my1) * ph / 1000))
            crop_x2 = min(pw, int((x2 + mx2) * pw / 1000))
            crop_y2 = min(ph, int((y2 + my2) * ph / 1000))

            # 原始 BBOX 像素坐标（无扩展）
            chart_x1 = int(x1 * pw / 1000)
            chart_y1 = int(y1 * ph / 1000)
            chart_x2 = int(x2 * pw / 1000)
            chart_y2 = int(y2 * ph / 1000)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            bbox["crop"] = (crop_x1, crop_y1, crop_x2, crop_y2)
            bbox["chart_bbox"] = (chart_x1, chart_y1, chart_x2, chart_y2)

    # --- 阶段 3：非对称裁切 + 对称白边填充居中 ---
    from PIL import ImageOps as _ImageOps
    replacements: dict[int, tuple[int, str]] = {}
    fig_counter = 1

    for bbox in all_bboxes:
        crop = bbox.get("crop")
        chart = bbox.get("chart_bbox")
        if not crop or not chart:
            continue
        pn = bbox["page"]
        if pn not in page_images:
            continue
        try:
            cropped = page_images[pn].crop(crop)
            cw, ch = cropped.size

            # 原始图表在裁切图内的位置
            cx1, cy1, cx2, cy2 = chart
            chart_left = cx1 - crop[0]
            chart_top = cy1 - crop[1]
            chart_right = cx2 - crop[0]
            chart_bottom = cy2 - crop[1]

            # 图表中心在裁切图中的位置
            chart_cx = (chart_left + chart_right) / 2
            chart_cy = (chart_top + chart_bottom) / 2

            # 距离裁切图中心的偏移
            offset_x = cw / 2 - chart_cx
            offset_y = ch / 2 - chart_cy

            if abs(offset_x) > 1 or abs(offset_y) > 1:
                pad_left = max(0, int(offset_x))
                pad_right = max(0, int(-offset_x))
                pad_top = max(0, int(offset_y))
                pad_bottom = max(0, int(-offset_y))
                cropped = _ImageOps.expand(
                    cropped,
                    border=(pad_left, pad_top, pad_right, pad_bottom),
                    fill=(255, 255, 255),
                )

            out_path = assets_dir / f"fig_{fig_counter}.png"
            cropped.save(str(out_path), "PNG")
        except Exception as e:
            print(f"[cc-vision] 警告: 裁切 fig_{fig_counter} 失败（第 {pn} 页）: {e}", file=sys.stderr)
            continue
        replacements[bbox["start"]] = (bbox["end"],
            f"![图 {fig_counter}]({assets_dir.name}/{out_path.name})")
        fig_counter += 1

    # --- 阶段 4：逆向拼接 ---
    result_parts = []
    cursor = 0
    for start in sorted(replacements.keys()):
        end, repl = replacements[start]
        if start < cursor:
            continue
        result_parts.append(text[cursor:start])
        result_parts.append(repl)
        cursor = end
    result_parts.append(text[cursor:])

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="cc-vision",
        description="全局多模态文档解析工具（Qwen3-VL-235B）",
    )
    parser.add_argument("file", help="图片或 PDF 文件路径")
    parser.add_argument(
        "--pages", default=None,
        help="物理页码范围，如 '3-5' 或 '1,3,5'（从 1 开始）"
    )
    parser.add_argument("--dpi", type=int, default=300, help="PDF 渲染 DPI（默认 300）")
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="每批最多页数（默认 5）"
    )
    args = parser.parse_args()

    filepath = Path(args.file).resolve()
    if not filepath.exists():
        die(f"文件不存在: {filepath}")
    if not filepath.is_file():
        die(f"不是有效文件: {filepath}")

    ext = filepath.suffix.lower()
    if ext not in IMAGE_EXTENSIONS and ext != PDF_EXTENSION:
        die(f"不支持的文件类型: {ext}")

    # 输出路径
    out_dir = Path.cwd()
    out_md = out_dir / ".vision_parsed.md"

    print(f"[cc-vision] 正在处理: {filepath.name} ...", file=sys.stderr)

    # ------------------------------------------------------------------
    # 渲染：PDF 逐页 → PIL Image + base64；图片直接 base64
    # ------------------------------------------------------------------

    page_specs: list[dict] = []          # {"page_num": int, "data_url": str}
    page_images: dict[int, Image.Image] = {}
    page_dimensions: dict[int, tuple[int, int]] = {}

    if ext == PDF_EXTENSION:
        import fitz
        doc = fitz.open(str(filepath))
        total_pages = doc.page_count
        selected = parse_pages(args.pages, total_pages)

        if not selected:
            die("没有匹配的页面")

        print(
            f"[cc-vision] PDF 共 {total_pages} 页，"
            f"处理第 {selected[0] + 1}–{selected[-1] + 1} 页 "
            f"({len(selected)} 页，{args.dpi} DPI，批次大小 {args.batch_size})",
            file=sys.stderr,
        )

        for idx in selected:
            img, png_bytes, w, h = render_pdf_page(doc, idx, args.dpi)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"
            page_num = idx + 1  # 物理页码
            page_specs.append({"page_num": page_num, "data_url": data_url})
            page_images[page_num] = img
            page_dimensions[page_num] = (w, h)

        doc.close()
    else:
        data_url = encode_image(str(filepath))
        page_specs.append({"page_num": 1, "data_url": data_url})
        img = Image.open(str(filepath)).convert("RGB")
        page_images[1] = img
        page_dimensions[1] = img.size
        print(
            f"[cc-vision] 图片 {img.size[0]}×{img.size[1]}，"
            f"直接送入模型",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # API 调用（分批）
    # ------------------------------------------------------------------

    api_key = get_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    all_results: list[str] = []
    total_batches = (len(page_specs) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        batch = page_specs[start:end]

        pages_range = f"{batch[0]['page_num']}–{batch[-1]['page_num']}"
        if len(batch) == 1:
            pages_range = f"{batch[0]['page_num']}"

        print(
            f"[cc-vision] 批次 {batch_idx + 1}/{total_batches} "
            f"（第 {pages_range} 页）→ API 调用中...",
            file=sys.stderr,
        )

        messages = build_messages(batch)
        try:
            result = call_api(client, messages)
        except Exception as e:
            die(f"API 调用失败（第 {pages_range} 页）: {e}")

        all_results.append(result)
        print(
            f"[cc-vision] 批次 {batch_idx + 1}/{total_batches} 完成",
            file=sys.stderr,
        )

    raw_text = "\n\n".join(all_results)

    # ------------------------------------------------------------------
    # 后处理：BBOX → 裁切
    # ------------------------------------------------------------------

    assets_dir = out_dir / ".vision_assets"
    # 清理上次运行的旧切图
    if assets_dir.exists():
        for f in assets_dir.glob("fig_*.png"):
            f.unlink()
    # 构建物理页码顺序列表（用于 BBOX 页码映射）
    page_order = [s["page_num"] for s in page_specs]
    processed_text = process_bboxes(raw_text, page_images, page_dimensions, assets_dir, page_order)

    # 文档标题行
    doc_title = filepath.stem

    final_output = f"# {doc_title}\n\n{processed_text}\n"

    out_md.write_text(final_output, encoding="utf-8")
    print(f"[cc-vision] 完成 → {out_md}", file=sys.stderr)

    # 统计
    chart_count = len(list(assets_dir.glob("fig_*.png"))) if assets_dir.exists() else 0
    if chart_count:
        print(
            f"[cc-vision] 裁切图表 {chart_count} 张 → {assets_dir}/",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
