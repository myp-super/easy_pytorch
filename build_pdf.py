#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a compact PDF book from all markdown chapters via Pandoc -> HTML -> WeasyPrint.
Optimized for printing: small margins, small fonts, minimal whitespace.
"""

import subprocess, os, sys, re
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
CHAPTERS = ROOT / "chapters"
OUTPUT_PDF = ROOT / "PyTorch_学习笔记.pdf"
MERGED_MD = ROOT / ".build_merged.md"
HTML_FILE = ROOT / ".build_output.html"

# ── Chapter ordering ──────────────────────────────────────────────

ORDER = [
    ("README.md", "前言", True),
    ("chapters/part1_python/01/开发环境配置与基础语法.md", "第〇一章 开发环境配置与基础语法", False),
    ("chapters/part1_python/01/Homework_Answer.md", "第〇一章 课后答案", True),
    ("chapters/part1_python/02/条件与循环语句.md", "第〇二章 条件与循环语句", False),
    ("chapters/part1_python/02/Homework_Answer.md", "第〇二章 课后答案", True),
    ("chapters/part1_python/03/函数与模块.md", "第〇三章 函数与模块", False),
    ("chapters/part1_python/03/Homework_Answer.md", "第〇三章 课后答案", True),
    ("chapters/part1_python/04/数据结构.md", "第〇四章 数据结构", False),
    ("chapters/part1_python/04/Homework_Answer.md", "第〇四章 课后答案", True),
    ("chapters/part1_python/05/文件操作与异常处理.md", "第〇五章 文件操作与异常处理", False),
    ("chapters/part1_python/05/Homework_Answer.md", "第〇五章 课后答案", True),
    ("chapters/part1_python/05_oop/面向对象编程基础.md", "第〇五章补充 面向对象编程基础", False),
    ("chapters/part1_python/05_oop/Homework_Answer.md", "第〇五章补充 课后答案", True),
    ("chapters/part1_python/06/综合实践.md", "第〇六章 综合实践", False),
    ("chapters/part1_python/06/Homework_Answer.md", "第〇六章 课后答案", True),
    ("chapters/part2_basics/00_environment.md", "第〇章 环境搭建与工具准备", False),
    ("chapters/part2_basics/01_tensor.md", "第一章 Tensor — 深度学习的基本语言", False),
    ("chapters/part2_basics/02_autograd.md", "第二章 自动求导 autograd", False),
    ("chapters/part2_basics/03_oop_basics.md", "第三章 Python 面向对象基础", False),
    ("chapters/part2_basics/04_nn_module.md", "第四章 nn.Module — 神经网络类的骨架", False),
    ("chapters/part2_basics/05_linear_layer.md", "第五章 单层网络 — 理解数据流动", False),
    ("chapters/part2_basics/06_activation.md", "第六章 激活函数", False),
    ("chapters/part2_basics/07_loss.md", "第七章 损失函数", False),
    ("chapters/part2_basics/08_optimizer.md", "第八章 优化器", False),
    ("chapters/part2_basics/09_train_loop.md", "第九章 训练循环", False),
    ("chapters/part2_basics/10_dataloader.md", "第十章 Dataset 与 DataLoader", False),
    ("chapters/part2_basics/11_mnist_training.md", "第十一章 完整训练 — MNIST", False),
    ("chapters/part2_basics/12_inference.md", "第十二章 推理、保存与加载", False),
    ("chapters/part2_basics/13_gpu.md", "第十三章 GPU 训练", False),
    ("chapters/part2_basics/14_projects.md", "第十四章 实战项目", False),
    ("chapters/part3_advanced/15_cnn.md", "第十五章 CNN — 卷积神经网络", False),
    ("chapters/part3_advanced/15b_resnet.md", "第十五章补充 ResNet 与跳跃连接", False),
    ("chapters/part3_advanced/16_batchnorm_dropout.md", "第十六章 Batch Normalization 与 Dropout", False),
    ("chapters/part3_advanced/17_augmentation_scheduler.md", "第十七章 数据增强与学习率调度", False),
    ("chapters/part3_advanced/18_rnn_lstm.md", "第十八章 RNN / LSTM", False),
    ("chapters/part3_advanced/19_transformer.md", "第十九章 Transformer", False),
    ("chapters/part3_advanced/20_engineering.md", "第二十章 PyTorch 工程化实践", False),
    ("chapters/part3_advanced/21_advanced_projects.md", "第二十一章 进阶实战项目", False),
    ("git-guide.md", "附录：Git 使用指南", True),
]

# ── HTML template with compact print CSS ──────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyTorch 真正初学者系统教程</title>
<style>
  /* ── Reset & Base ── */
  @page {
    size: A4;
    margin: 0.45in 0.42in 0.42in 0.42in;
  }

  @page :first {
    @top-center { content: none; }
    @top-right { content: none; }
    @bottom-center { content: none; }
  }

  * { box-sizing: border-box; }

  body {
    font-family: "SimSun", "Noto Serif CJK SC", "Songti SC", serif;
    font-size: 7pt;
    line-height: 1.28;
    color: #1a1a1a;
    max-width: 100%;
    margin: 0;
    padding: 0;
    orphans: 2;
    widows: 2;
  }

  /* ── Title page ── */
  .title-page {
    page-break-after: always;
    text-align: center;
    padding-top: 25%;
  }
  .title-page h1 {
    font-family: "SimHei", "Microsoft YaHei", sans-serif;
    font-size: 28pt;
    margin: 0 0 4pt 0;
    letter-spacing: 4pt;
  }
  .title-page h2 {
    font-family: "SimHei", sans-serif;
    font-size: 16pt;
    font-weight: normal;
    margin: 4pt 0 20pt 0;
    color: #333;
  }
  .title-page .info {
    font-size: 9pt;
    color: #555;
    line-height: 2;
  }
  .title-page .repo {
    font-family: "Courier New", monospace;
    font-size: 8pt;
    color: #777;
  }

  /* ── TOC ── */
  .toc {
    page-break-after: always;
  }
  .toc h2 {
    font-family: "SimHei", sans-serif;
    font-size: 12pt;
    border-bottom: 1px solid #333;
    padding-bottom: 4pt;
    margin-bottom: 8pt;
  }
  .toc ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .toc li {
    font-size: 8pt;
    line-height: 1.5;
    padding: 0;
    margin: 0;
  }
  .toc li a {
    color: #1a1a1a;
    text-decoration: none;
    display: inline;
  }
  .toc li a::after {
    content: leader(".") target-counter(attr(href), page);
  }
  .toc .part-title {
    font-family: "SimHei", sans-serif;
    font-weight: bold;
    font-size: 8.5pt;
    margin-top: 8pt;
    padding-top: 3pt;
    border-top: 0.5pt solid #ccc;
  }

  /* ── Headings ── */
  h1 {
    font-family: "SimHei", "Microsoft YaHei", sans-serif;
    font-size: 11pt;
    page-break-before: auto;
    page-break-after: avoid;
    margin: 0 0 3pt 0;
    padding: 4pt 0 2pt 0;
    border-bottom: 0.8pt solid #333;
  }
  h1.no-break { page-break-before: avoid; }
  h2 {
    font-family: "SimHei", sans-serif;
    font-size: 9pt;
    margin: 5pt 0 2pt 0;
    padding: 0;
  }
  h3 {
    font-family: "SimHei", sans-serif;
    font-size: 8pt;
    margin: 4pt 0 1pt 0;
    padding: 0;
  }
  h4 {
    font-family: "SimHei", sans-serif;
    font-size: 7.5pt;
    margin: 3pt 0 1pt 0;
    padding: 0;
  }

  /* ── Paragraphs ── */
  p {
    margin: 1pt 0 2pt 0;
    text-indent: 0;
  }

  /* ── Code ── */
  code {
    font-family: "Courier New", "Consolas", monospace;
    font-size: 6pt;
    background: #f5f5f5;
    padding: 0 2pt;
    border-radius: 1pt;
  }
  pre {
    font-family: "Courier New", "Consolas", monospace;
    font-size: 5.5pt;
    line-height: 1.15;
    background: #f8f8f8;
    border: 0.2pt solid #ddd;
    border-left: 1.5pt solid #4a90d9;
    padding: 2pt 4pt;
    margin: 2pt 0;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
  pre code {
    background: none;
    padding: 0;
    font-size: 5.5pt;
  }
  div.sourceCode {
    margin: 2pt 0;
    font-size: 5.5pt;
    line-height: 1.15;
  }
  div.sourceCode pre {
    margin: 0;
  }

  /* ── Tables ── */
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 2pt 0;
    font-size: 6.5pt;
  }
  th {
    font-family: "SimHei", sans-serif;
    background: #f0f0f0;
    padding: 1pt 3pt;
    text-align: left;
    border: 0.2pt solid #bbb;
  }
  td {
    padding: 1pt 3pt;
    border: 0.2pt solid #ddd;
    vertical-align: top;
  }
  tr { page-break-inside: avoid; }

  /* ── Lists ── */
  ul, ol {
    margin: 0.5pt 0 1pt 0;
    padding-left: 14pt;
  }
  li {
    margin: 0;
    padding: 0;
    line-height: 1.25;
  }
  li p { margin: 0; }

  /* ── Blockquote ── */
  blockquote {
    margin: 2pt 0 2pt 6pt;
    padding: 2pt 4pt;
    border-left: 1.5pt solid #4a90d9;
    background: #f9f9f9;
    font-size: 6.5pt;
  }
  blockquote p { margin: 0.5pt 0; }

  /* ── Horizontal rule ── */
  hr {
    border: none;
    border-top: 0.5pt solid #ddd;
    margin: 8pt 0;
  }

  /* ── Images ── */
  img {
    max-width: 100%;
    max-height: 9cm;
    display: block;
    margin: 3pt auto;
  }
  figure {
    margin: 3pt 0;
    text-align: center;
  }
  figcaption {
    font-size: 7pt;
    color: #666;
    margin-top: 1pt;
  }

  /* ── Links ── */
  a { color: #2a6496; text-decoration: none; }

  /* ── Part separator ── */
  .part-header {
    page-break-before: always;
    text-align: center;
    padding-top: 30%;
  }
  .part-header h1 {
    font-size: 18pt;
    border: none;
    page-break-before: avoid;
    string-set: doctitle content();
  }
  .part-header .subtitle {
    font-size: 10pt;
    color: #555;
  }

  /* ── Answers section ── */
  .answers-header {
    background: #fff3cd;
    padding: 4pt 8pt;
    margin: 8pt 0 4pt 0;
    font-family: "SimHei", sans-serif;
    font-size: 8pt;
  }

  /* ── Compact helpers ── */
  .no-break { page-break-before: avoid; }
  .page-break { page-break-before: always; }

  /* ── Floating ── */
  strong { font-family: "SimHei", sans-serif; color: #000; }
  em { color: #333; }
</style>
</head>
<body>

<!-- Title page -->
<div class="title-page">
  <h1>PyTorch</h1>
  <h2>真正初学者系统教程</h2>
  <div class="info">
    从 Python 到深度学习工程<br>
    打印复习版<br><br>
    共三部分 · 37 章 · 覆盖从零基础到 Transformer
  </div>
  <div class="repo">github.com/myp-super/easy_pytorch</div>
</div>

<!-- TOC -->
<div class="toc">
<h2>目录</h2>
<nav id="TOC">
$toc$
</nav>
</div>

<!-- Content -->
$body$

</body>
</html>
"""


def fix_image_paths(content: str) -> str:
    """Fix image paths and handle unsupported formats."""
    content = content.replace('](easy_pytorch/', '](')
    # Replace SVG images with text notes
    content = re.sub(
        r'!\[([^\]]*)\]\(([^)]+\.svg)\)',
        r'<p class="img-note"><em>[\1]</em> — 图片: <code>\2</code> (SVG)</p>',
        content)
    return content


def merge_chapters() -> str:
    """Merge all chapters in order."""
    parts = {
        "chapters/part1_python/": ("第〇部分：Python 基础", "为写 PyTorch 代码所需的 Python 知识"),
        "chapters/part2_basics/": ("第一部分：PyTorch 基础篇", "从零理解 PyTorch 工程代码"),
        "chapters/part3_advanced/": ("第二部分：PyTorch 进阶篇", "掌握 CNN、RNN、Transformer 等现代架构"),
    }

    merged = []
    current_part = None
    ok, skip = 0, 0

    for rel_path, display_title, is_appendix in ORDER:
        fpath = ROOT / rel_path
        if not fpath.exists():
            print(f"  [SKIP] {rel_path}")
            skip += 1
            continue

        # Part boundary
        for prefix, (ptitle, psubtitle) in parts.items():
            if rel_path.startswith(prefix) and prefix != current_part:
                current_part = prefix
                merged.append(f'\n<div class="part-header">')
                merged.append(f'<h1>{ptitle}</h1>')
                merged.append(f'<p class="subtitle">{psubtitle}</p>')
                merged.append(f'</div>\n')
                break

        content = fpath.read_text(encoding="utf-8")
        content = fix_image_paths(content)

        if is_appendix:
            merged.append(f'\n<div class="page-break"></div>\n')
            merged.append(f'<h1 class="answers-header">{display_title}</h1>\n')
            merged.append(content)
        else:
            merged.append(content)

        ok += 1

    print(f"  Merged {ok}, skipped {skip}")
    return "\n\n".join(merged)


def build_pdf():
    """Main build routine."""
    print("=" * 50)
    print("  PyTorch Study Notes -> Compact PDF Builder (HTML engine)")
    print("=" * 50)

    # 1. Merge chapters
    print("\n[1/4] Merging chapters...")
    merged = merge_chapters()
    MERGED_MD.write_text(merged, encoding="utf-8")
    lc = len(merged.splitlines())
    cc = len(merged)
    print(f"  OK: {MERGED_MD.name} ({lc:,} lines, {cc:,} chars)")

    # 2. Pandoc -> HTML
    print("\n[2/4] Converting to HTML with Pandoc...")
    template_path = ROOT / ".html_template.html"
    template_path.write_text(HTML_TEMPLATE, encoding="utf-8")
    # Write merged md with a YAML frontmatter for pandoc template vars
    merged_with_header = "---\ntitle: PyTorch Study Notes\nauthor: myp-super\n---\n\n" + merged
    MERGED_MD.write_text(merged_with_header, encoding="utf-8")

    cmd = [
        "pandoc",
        str(MERGED_MD),
        "-o", str(HTML_FILE),
        "-t", "html5",
        f"--template={template_path}",
        "--toc", "--toc-depth=1",
        "--highlight-style=tango",
        "--resource-path", str(ROOT),
        "--self-contained",
        "--metadata", "title=PyTorch Study Notes",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(ROOT), timeout=300)

    if result.returncode != 0 or not HTML_FILE.exists():
        print("  ERROR: Pandoc HTML failed:")
        for line in result.stderr.splitlines()[-20:]:
            print(f"    {line}")
        return False

    html_size = HTML_FILE.stat().st_size / (1024 * 1024)
    print(f"  OK: {HTML_FILE.name} ({html_size:.1f} MB)")

    # 3. HTML -> PDF via MS Edge headless mode
    print("\n[3/4] Rendering PDF with MS Edge (headless)...")
    edge_paths = [
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    ]
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break

    if not edge_exe:
        print("  ERROR: Microsoft Edge not found.")
        return False

    # Edge headless print-to-pdf
    result = subprocess.run(
        [edge_exe, "--headless", "--disable-gpu",
         f"--print-to-pdf={OUTPUT_PDF}",
         "--no-pdf-header-footer",
         "--print-to-pdf-no-header",
         f"file:///{HTML_FILE.as_posix()}"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)

    if result.returncode != 0:
        print(f"  ERROR: Edge headless failed (exit {result.returncode})")
        for line in result.stderr.splitlines()[-10:]:
            print(f"    {line}")
        # Fallback: try without --no-pdf-header-footer
        result2 = subprocess.run(
            [edge_exe, "--headless", "--disable-gpu",
             f"--print-to-pdf={OUTPUT_PDF}",
             f"file:///{HTML_FILE.as_posix()}"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        if result2.returncode != 0:
            return False
        print("  (fallback used, headers may appear)")

    if OUTPUT_PDF.exists():
        size_mb = OUTPUT_PDF.stat().st_size / (1024 * 1024)
        print(f"  OK: {OUTPUT_PDF.name} ({size_mb:.1f} MB)")

        # Try to count pages
        try:
            import io
            from weasyprint import HTML as WHTML
        except Exception:
            pass
        try:
            from PyPDF2 import PdfReader
            r = PdfReader(str(OUTPUT_PDF))
            print(f"  PAGES: {len(r.pages)}")
        except Exception:
            try:
                import fitz  # pymupdf
                doc = fitz.open(str(OUTPUT_PDF))
                print(f"  PAGES: {doc.page_count}")
            except Exception:
                pass
    else:
        print("  ERROR: PDF not found")
        return False

    # 4. Cleanup
    print("\n[4/4] Cleaning up...")
    MERGED_MD.unlink(missing_ok=True)
    HTML_FILE.unlink(missing_ok=True)
    template_path.unlink(missing_ok=True)
    print("  Done")

    print(f"\n{'=' * 50}")
    print(f"  PDF: {OUTPUT_PDF}")
    print(f"{'=' * 50}")
    return True


if __name__ == "__main__":
    os.chdir(str(ROOT))
    success = build_pdf()
    sys.exit(0 if success else 1)
