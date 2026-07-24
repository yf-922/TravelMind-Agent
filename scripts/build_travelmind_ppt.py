"""Generate Experiment 2 design deck using ASCII-only source text."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path(__file__).resolve().parents[1] / "docs" / "TravelMind_\u5b9e\u9a8c\u4e8c_\u6982\u8981\u8bbe\u8ba1\u4e0e\u8be6\u7ec6\u8bbe\u8ba1.pptx"
FONT = "Microsoft YaHei"
BG = RGBColor(250, 247, 240)
INK = RGBColor(40, 34, 27)
MUTED = RGBColor(100, 92, 80)
LINE = RGBColor(221, 211, 194)
WHITE = RGBColor(255, 253, 248)
RED = RGBColor(185, 72, 34)
GREEN = RGBColor(48, 105, 78)
BLUE = RGBColor(55, 95, 135)
GOLD = RGBColor(196, 137, 35)
PINK = RGBColor(247, 226, 217)
MINT = RGBColor(225, 240, 230)
SKY = RGBColor(225, 235, 246)
YELLOW = RGBColor(249, 239, 213)


def text(slide, value, x, y, w, h, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT):
    item = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = item.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.08)
    frame.margin_top = Inches(0.04)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def box(slide, x, y, w, h, fill=WHITE, line=LINE, rounded=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    item = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    item.fill.solid()
    item.fill.fore_color.rgb = fill
    item.line.color.rgb = line
    return item


def line(slide, x1, y1, x2, y2, color=INK, width=1.4):
    item = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    item.line.color.rgb = color
    item.line.width = Pt(width)
    item.line.end_arrowhead = True


def slide(prs, kicker, title, subtitle, page):
    page_slide = prs.slides.add_slide(prs.slide_layouts[6])
    page_slide.background.fill.solid()
    page_slide.background.fill.fore_color.rgb = BG
    text(page_slide, kicker, 0.65, 0.38, 4, 0.24, 10, RED, True)
    text(page_slide, title, 0.63, 0.68, 12, 0.48, 25, INK, True)
    text(page_slide, subtitle, 0.65, 1.24, 12, 0.28, 10.5, MUTED)
    divider = page_slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.62), Inches(12.0), Inches(0.02))
    divider.fill.solid()
    divider.fill.fore_color.rgb = INK
    divider.line.fill.background()
    text(page_slide, f"TravelMind \u00b7 \u5b9e\u9a8c\u4e8c\uff1a\u6982\u8981\u8bbe\u8ba1\u4e0e\u8be6\u7ec6\u8bbe\u8ba1                         {page:02d}", 0.65, 7.08, 12, 0.2, 8.3, MUTED)
    return page_slide


def tag(slide, value, x, y, color):
    box(slide, x, y, 1.0, 0.28, color, color)
    text(slide, value, x + 0.03, y + 0.06, 0.94, 0.13, 8.2, WHITE, True, PP_ALIGN.CENTER)


def card(slide, number, title, body, x, y, color=RED, w=3.75, h=1.55):
    box(slide, x, y, w, h)
    text(slide, number, x + 0.17, y + 0.14, 0.45, 0.2, 10, color, True)
    text(slide, title, x + 0.17, y + 0.43, w - 0.34, 0.26, 12.5, INK, True)
    text(slide, body, x + 0.17, y + 0.8, w - 0.34, h - 0.9, 9.5, MUTED)


def cards(slide, items, top=2.05, height=1.7):
    for index, item in enumerate(items):
        card(
            slide,
            item[0],
            item[1],
            item[2],
            0.7 + (index % 3) * 4.08,
            top + (index // 3) * (height + 0.38),
            item[3],
            h=height,
        )


def status_card(slide, state, title, body, x, y, color, fill):
    box(slide, x, y, 3.85, 1.15, fill, fill)
    tag(slide, state, x + 0.16, y + 0.14, color)
    text(slide, title, x + 1.32, y + 0.13, 2.28, 0.22, 11.5, INK, True)
    text(slide, body, x + 0.16, y + 0.54, 3.48, 0.42, 9.3, MUTED)


def agent_card(slide, name, goal, boundary, tools, memory, scope, x, y, color):
    box(slide, x, y, 3.86, 3.75)
    box(slide, x, y, 3.86, 0.48, color, color)
    text(slide, name, x + 0.15, y + 0.12, 3.5, 0.2, 13, WHITE, True)
    rows = [
        ("\u76ee\u6807", goal),
        ("\u80fd\u529b\u8fb9\u754c", boundary),
        ("\u5de5\u5177", tools),
        ("\u8bb0\u5fc6", memory),
        ("\u8303\u56f4", scope),
    ]
    for index, (label, value) in enumerate(rows):
        yy = y + 0.64 + index * 0.6
        text(slide, label, x + 0.15, yy, 0.68, 0.18, 9.2, color, True)
        text(slide, value, x + 0.88, yy, 2.77, 0.36, 9.1, INK)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "TravelMind \u5b9e\u9a8c\u4e8c - \u6982\u8981\u8bbe\u8ba1\u4e0e\u8be6\u7ec6\u8bbe\u8ba1"
    prs.core_properties.author = "yf"

    # 1 Cover
    cover = prs.slides.add_slide(prs.slide_layouts[6])
    cover.background.fill.solid()
    cover.background.fill.fore_color.rgb = BG
    box(cover, 0.7, 0.68, 0.52, 0.52, RED, RED)
    text(cover, "\u9014", 0.78, 0.76, 0.35, 0.26, 17, WHITE, True, PP_ALIGN.CENTER)
    text(cover, "TravelMind", 1.38, 0.73, 4, 0.3, 18, INK, True)
    text(cover, "AI \u4e2a\u6027\u5316\u65c5\u884c\u89c4\u5212\u52a9\u624b", 1.38, 1.08, 4.5, 0.22, 10.5, MUTED, True)
    text(cover, "\u5b9e\u9a8c\u4e8c\uff1a\u6982\u8981\u8bbe\u8ba1\u4e0e\u8be6\u7ec6\u8bbe\u8ba1", 0.9, 2.3, 10.8, 0.55, 29, INK, True)
    text(cover, "\u591a\u667a\u80fd\u4f53\u534f\u4f5c\u3001\u6d88\u606f\u7ed3\u6784\u4e0e\u8bb0\u5fc6\u9694\u79bb\u65b9\u6848", 0.92, 3.05, 8.9, 0.3, 16, RED, True)
    box(cover, 0.9, 4.15, 11.45, 1.45)
    text(cover, "\u8bbe\u8ba1\u5b9a\u4f4d", 1.18, 4.45, 1.15, 0.2, 11, RED, True)
    text(cover, "\u57fa\u4e8e\u73b0\u6709 FloatTrip \u5b9e\u73b0\uff0c\u5c06\u9690\u5f0f\u7684\u56fa\u5b9a\u5de5\u4f5c\u6d41\u91cd\u6784\u4e3a\u201cSupervisor \u8c03\u5ea6 + Worker Agent \u5206\u5de5 + \u72ec\u7acb\u8bb0\u5fc6\u201d\u7684\u53ef\u843d\u5730\u8bbe\u8ba1\u3002", 2.45, 4.28, 8.9, 0.55, 14, INK, True)
    text(cover, "\u6c47\u62a5\u4eba / yf     \u65e5\u671f / 2026 \u5e74 7 \u6708 24 \u65e5", 0.92, 6.5, 7, 0.24, 11, MUTED)
    text(cover, "TravelMind \u00b7 \u5b9e\u9a8c\u4e8c                         01", 0.65, 7.08, 12, 0.2, 8.3, MUTED)

    # 2 Requirements mapping
    page = slide(prs, "\u5b9e\u9a8c\u76ee\u6807\u4e0e\u8bbe\u8ba1\u8303\u56f4", "\u672c\u6b21\u91cd\u8bbe\u8ba1\u4ea4\u4ed8\u4ec0\u4e48\uff1f", "\u4ee5\u201c\u8c01\u8c03\u5ea6\u8c01\u3001\u4fe1\u606f\u600e\u4e48\u6d41\u3001\u8bb0\u5fc6\u5b58\u54ea\u91cc\u201d\u4e3a\u7ed3\u6784\u5316\u8bbe\u8ba1\u4e3b\u7ebf", 2)
    items = [
        ("01", "\u6982\u8981\u67b6\u6784", "\u8bbe\u8ba1\u663e\u5f0f Supervisor\uff0c\u8fde\u63a5 Worker Agent\u3001\u5916\u90e8\u5de5\u5177\u548c\u6570\u636e\u5c42\u3002", RED),
        ("02", "Agent \u8be6\u7ec6\u8bbe\u8ba1", "\u4e3a\u6bcf\u4e2a Agent \u5b9a\u4e49\u76ee\u6807\u3001\u80fd\u529b\u8fb9\u754c\u3001\u5de5\u5177\u548c\u72ec\u7acb\u8bb0\u5fc6\u3002", GREEN),
        ("03", "\u6d88\u606f\u534f\u8bae", "\u7edf\u4e00 task_type\u3001from\u3001to\u3001content\u3001status\uff0c\u5b9e\u73b0\u53ef\u8ffd\u8e2a\u4ea4\u63a5\u3002", BLUE),
        ("04", "\u8bb0\u5fc6\u7ea2\u7ebf", "\u7528\u6237\u957f\u671f\u8bb0\u5fc6\u3001\u5f53\u524d\u4efb\u52a1\u72b6\u6001\u3001\u5ba1\u6838\u8bb0\u5fc6\u5206\u5c42\u3001\u6309\u6743\u9650\u8bfb\u53d6\u3002", GOLD),
        ("05", "\u52a8\u6001\u8c03\u6574", "\u6307\u5b9a\u5730\u70b9\u3001\u5929\u6c14\u3001\u665a\u51fa\u53d1\u7b49\u53d8\u66f4\u89e6\u53d1\u5c40\u90e8\u91cd\u89c4\u5212\u3002", RED),
        ("06", "\u5b9e\u65bd\u8fb9\u754c", "\u672c\u6b21\u4ea4\u4ed8\u8bbe\u8ba1\u6587\u6863\u4e0e\u67b6\u6784\u56fe\uff1b\u672a\u5b8c\u6210\u80fd\u529b\u660e\u786e\u6807\u6ce8\u4e3a\u540e\u7eed\u5b9e\u73b0\u3002", GOLD),
    ]
    for index, item in enumerate(items):
        card(page, item[0], item[1], item[2], 0.7 + (index % 3) * 4.08, 2.02 + (index // 3) * 2.05, item[3], h=1.68)

    # 3 Architecture
    page = slide(prs, "\u6982\u8981\u8bbe\u8ba1", "\u591a Agent \u67b6\u6784\u56fe\uff1aSupervisor \u8c03\u5ea6\u5168\u5c40\u5de5\u4f5c\u6d41", "\u5f53\u524d LangGraph \u5df2\u6709\u56fa\u5b9a\u6d41\u7a0b\uff1b\u672c\u8bbe\u8ba1\u5c06\u5176\u663e\u5f0f\u5316\u4e3a Supervisor \u4e0e\u6d88\u606f\u8c03\u5ea6", 3)
    box(page, 0.75, 2.7, 1.6, 0.72, SKY, SKY)
    text(page, "\u7528\u6237\u8bf7\u6c42", 0.89, 2.93, 1.32, 0.2, 13, BLUE, True, PP_ALIGN.CENTER)
    box(page, 3.02, 2.38, 2.35, 1.34, RED, RED)
    text(page, "Supervisor\n\u8c03\u5ea6\u4e2d\u67a2", 3.25, 2.68, 1.9, 0.55, 16, WHITE, True, PP_ALIGN.CENTER)
    line(page, 2.35, 3.05, 3.02, 3.05, BLUE)
    workers = [
        ("Intent / \u9700\u6c42\u89e3\u6790", 6.1, 1.95, GREEN),
        ("POI / \u4e8b\u5b9e\u68c0\u7d22", 9.25, 1.95, BLUE),
        ("Planner / \u884c\u7a0b\u751f\u6210", 6.1, 3.48, GOLD),
        ("Reviewer / \u8def\u7ebf\u5ba1\u6838", 9.25, 3.48, RED),
        ("TimeCheck / \u65f6\u95f4\u6838\u9a8c", 6.1, 5.01, GREEN),
        ("Memory / \u504f\u597d\u53ec\u56de", 9.25, 5.01, BLUE),
    ]
    for label, x, y, color in workers:
        box(page, x, y, 2.55, 0.72, WHITE, color)
        text(page, label, x + 0.14, y + 0.23, 2.25, 0.22, 11, color, True, PP_ALIGN.CENTER)
        line(page, 5.37, 3.05, x, y + 0.36, color, 1.1)
    text(page, "\u5916\u90e8\u5de5\u5177\uff1a\u9ad8\u5fb7 POI / \u5929\u6c14 / \u8def\u7ebf / \u9910\u5385", 1.08, 4.75, 4.2, 0.22, 11, MUTED, True)
    box(page, 0.9, 5.15, 4.35, 0.88, YELLOW, YELLOW)
    text(page, "\u6570\u636e\u5c42\uff1aSQLite \u7528\u6237\u753b\u50cf / \u5386\u53f2\u884c\u7a0b\uff1bChroma \u8bed\u4e49\u8bb0\u5fc6", 1.08, 5.43, 3.98, 0.26, 11, INK, True, PP_ALIGN.CENTER)
    text(page, "\u4fe1\u606f\u6d41\uff1a\u7528\u6237\u2192 Supervisor \u2192 Worker Agent \u2192 Supervisor \u2192 \u7528\u6237\u7ed3\u679c\n\u63a7\u5236\u6d41\uff1aSupervisor \u6839\u636e\u72b6\u6001\u9009\u62e9\u4e0b\u4e00\u4e2a Agent\uff0c\u5931\u8d25\u65f6\u91cd\u8bd5\u6216\u964d\u7ea7\u3002", 0.86, 6.25, 11.6, 0.5, 10.5, MUTED, True, PP_ALIGN.CENTER)

    # 4 Current vs future
    page = slide(prs, "\u73b0\u72b6\u8bc4\u4f30", "\u54ea\u4e9b\u5df2\u5b9e\u73b0\uff0c\u54ea\u4e9b\u662f\u672c\u6b21\u8bbe\u8ba1\u7684\u540e\u7eed\u4efb\u52a1\uff1f", "\u4e0d\u628a\u8bbe\u8ba1\u5199\u6210\u5df2\u5b8c\u6210\u529f\u80fd\uff0c\u660e\u786e\u4ea4\u4ed8\u7269\u4e0e\u672a\u6765\u5b9e\u73b0\u7684\u8fb9\u754c", 4)
    status_card(page, "\u5df2\u5b9e\u73b0", "\u56fa\u5b9a Agent \u7f16\u6392", "Intent\u3001Planner\u3001Reviewer\u3001Time Check\u3001\u9ad8\u5fb7\u68c0\u7d22\u7b49\u8282\u70b9\u5df2\u5728 LangGraph \u4e2d\u4e32\u8054\u8fd0\u884c\u3002", 0.8, 2.0, GREEN, MINT)
    status_card(page, "\u5df2\u5b9e\u73b0", "\u57fa\u7840\u4e2a\u6027\u5316\u8bb0\u5fc6", "SQLite \u4fdd\u5b58\u53ef\u7f16\u8f91\u753b\u50cf\uff1bChroma \u6309 user_id \u9694\u79bb\u53ec\u56de\u8bed\u4e49\u504f\u597d\u3002", 4.75, 2.0, GREEN, MINT)
    status_card(page, "\u5df2\u5b9e\u73b0", "\u6709\u9650\u8f6e\u6b21\u5ba1\u6838", "Planner \u4e0e Reviewer \u53ef\u4ee5\u56de\u8def\u4fee\u6b63\uff0c\u964d\u4f4e\u5019\u9009\u6c60\u4ee5\u5916\u666f\u70b9\u5e7b\u89c9\u3002", 8.7, 2.0, GREEN, MINT)
    status_card(page, "\u672a\u5b9e\u73b0", "\u663e\u5f0f Supervisor \u6d88\u606f\u8c03\u5ea6", "\u5f53\u524d\u4e3a\u56fa\u5b9a\u56fe\u7f16\u6392\uff0c\u5c1a\u672a\u5b9e\u73b0\u6839\u636e task_type \u52a8\u6001\u8c03\u7528 Worker \u7684\u72ec\u7acb\u8c03\u5ea6\u5668\u3002", 0.8, 3.55, RED, PINK)
    status_card(page, "\u672a\u5b9e\u73b0", "\u6307\u5b9a\u5730\u70b9\u6d88\u6b67 + \u52a8\u6001\u5019\u9009\u6c60", "\u5bf9\u201c\u5317\u4eac\u5927\u5b66\u201d\u3001\u201c\u79c0\u5c71\u5174\u9686\u5768\u201d\u7b49\u957f\u5c3e\u5730\u70b9\u8fd8\u7f3a\u7cbe\u786e\u641c\u7d22\u548c\u7528\u6237\u786e\u8ba4\u673a\u5236\u3002", 4.75, 3.55, RED, PINK)
    status_card(page, "\u672a\u5b9e\u73b0", "\u4ea4\u901a\u9a71\u52a8\u7684\u5c40\u90e8\u91cd\u89c4\u5212", "\u672a\u5c06\u5730\u94c1\u3001\u6253\u8f66\u7b49\u901a\u52e4\u8017\u65f6\u4f5c\u4e3a\u786c\u7ea6\u675f\uff0c\u4fee\u6539\u6d41\u7a0b\u4e5f\u9700\u8865\u5168 Time Check\u3002", 8.7, 3.55, RED, PINK)

    # 5 Supervisor detailed design
    page = slide(prs, "\u8be6\u7ec6\u8bbe\u8ba1 \u00b7 \u8c03\u5ea6\u4e2d\u67a2", "Supervisor Agent \u8bbe\u8ba1", "Supervisor \u53ea\u8d1f\u8d23\u8c03\u5ea6\u548c\u72b6\u6001\u8f6c\u79fb\uff0c\u4e0d\u81ea\u5df1\u7f16\u9020\u666f\u70b9\u3001\u4e0d\u76f4\u63a5\u751f\u6210\u884c\u7a0b", 5)
    agent_card(page, "Supervisor Agent", "\u63a5\u6536\u7528\u6237\u8bf7\u6c42\uff0c\u521b\u5efa task_id\uff0c\u6309\u72b6\u6001\u8c03\u5ea6 Worker\uff0c\u6c47\u603b\u7ed3\u679c\u3002", "\u4e0d\u751f\u6210\u65c5\u884c\u7ed3\u8bba\uff1b\u4e0d\u76f4\u63a5\u4fee\u6539\u7528\u6237\u957f\u671f\u8bb0\u5fc6\u3002", "LangGraph\u3001\u6d88\u606f\u961f\u5217\u3001\u4efb\u52a1\u72b6\u6001\u5b58\u50a8", "task_id\u3001\u5f53\u524d\u4efb\u52a1\u72b6\u6001\u3001\u5931\u8d25\u6b21\u6570\u3001\u8282\u70b9\u8f93\u51fa\u5f15\u7528", "\u4efb\u52a1\u7ea7\u79c1\u6709\uff1b\u53ea\u8bfb\u5171\u4eab\u7528\u6237\u8bf7\u6c42", 0.75, 2.0, RED)
    card(page, "\u8c03\u5ea6\u89c4\u5219 1", "\u9700\u6c42\u4e0d\u5168\uff1a\u8f6c Intent Agent", "\u7f3a\u5c11\u76ee\u7684\u5730\u3001\u65e5\u671f\u7b49\u5173\u952e\u5b57\u6bb5\u65f6\uff0c\u5efa\u7acb\u8865\u5145\u4fe1\u606f\u5b50\u4efb\u52a1\u3002", 5.0, 2.0, GREEN, 3.45, 1.25)
    card(page, "\u8c03\u5ea6\u89c4\u5219 2", "\u6307\u5b9a\u5730\u70b9\uff1a\u8f6c POI Resolver", "\u5148\u9a8c\u8bc1 POI \u540d\u79f0\u3001\u57ce\u5e02\u3001\u5750\u6807\uff1b\u591a\u4e2a\u7ed3\u679c\u8f6c\u4eba\u5de5\u786e\u8ba4\u3002", 8.82, 2.0, BLUE, 3.45, 1.25)
    card(page, "\u8c03\u5ea6\u89c4\u5219 3", "\u5ba1\u6838\u4e0d\u901a\u8fc7\uff1a\u56de\u9000 Planner", "\u6700\u591a\u4e09\u8f6e\u4fee\u6539\uff1b\u8d85\u8fc7\u4e0a\u9650\u8f93\u51fa\u7ea6\u675f\u51b2\u7a81\u4e0e\u5907\u9009\u65b9\u6848\u3002", 5.0, 3.55, GOLD, 3.45, 1.25)
    card(page, "\u8c03\u5ea6\u89c4\u5219 4", "\u5de5\u5177\u5931\u8d25\uff1a\u91cd\u8bd5 / \u964d\u7ea7", "\u6709\u9650\u91cd\u8bd5\uff1b\u4fdd\u7559 error_code\uff1b\u63d0\u793a\u7528\u6237\u54ea\u4e9b\u4e8b\u5b9e\u672a\u9a8c\u8bc1\u3002", 8.82, 3.55, RED, 3.45, 1.25)
    text(page, "\u5b9e\u73b0\u72b6\u6001\uff1a\u5f53\u524d LangGraph \u8d1f\u8d23\u56fa\u5b9a\u8282\u70b9\u8df3\u8f6c\uff1b\u672c\u9875 Supervisor \u4e3a\u672a\u6765\u9700\u8981\u5b9e\u73b0\u7684\u663e\u5f0f\u8c03\u5ea6\u5c42\u3002", 5.1, 5.35, 7.0, 0.35, 10.2, RED, True, PP_ALIGN.CENTER)

    # 6 Worker cards one
    page = slide(prs, "\u8be6\u7ec6\u8bbe\u8ba1 \u00b7 Worker Agent", "\u9700\u6c42\u89e3\u6790\u3001\u4e8b\u5b9e\u68c0\u7d22\u3001\u884c\u7a0b\u89c4\u5212", "\u5c06\u201c\u7406\u89e3\u7528\u6237\u201d\u3001\u201c\u67e5\u8be2\u4e8b\u5b9e\u201d\u548c\u201c\u751f\u6210\u65b9\u6848\u201d\u62c6\u5206\uff0c\u907f\u514d\u4e00\u4e2a Agent \u4e0a\u4e0b\u6587\u8fc7\u8f7d", 6)
    agent_card(page, "Intent Agent", "\u5c06\u81ea\u7136\u8bed\u8a00\u8f6c\u4e3a\u76ee\u7684\u5730\u3001\u65e5\u671f\u3001\u504f\u597d\u7b49\u7ed3\u6784\u5316\u9700\u6c42\u3002", "\u4e0d\u641c\u7d22 POI\uff1b\u4e0d\u751f\u6210\u884c\u7a0b\uff1b\u4e0d\u66f4\u65b0\u957f\u671f\u8bb0\u5fc6\u3002", "LLM \u7ed3\u6784\u5316\u8f93\u51fa\u3001\u53ea\u8bfb\u7528\u6237\u753b\u50cf", "\u672c\u8f6e\u5bf9\u8bdd\u4e0a\u4e0b\u6587\u3001\u7f3a\u5931\u5b57\u6bb5", "\u4efb\u52a1\u7ea7\u79c1\u6709\uff1b\u8bfb\u5168\u5c40\u9700\u6c42", 0.75, 2.0, GREEN)
    agent_card(page, "POI Resolver Agent", "\u7cbe\u786e\u68c0\u7d22\u7528\u6237\u6307\u5b9a\u5730\u70b9\uff0c\u5c06\u9a8c\u8bc1\u540e\u7684 POI \u52a0\u5165\u5019\u9009\u6c60\u3002", "\u4e0d\u63a8\u6d4b\u4e0d\u5b58\u5728\u7684\u5730\u70b9\uff1b\u591a\u4e49\u7ed3\u679c\u4e0d\u81ea\u884c\u9009\u62e9\u3002", "\u9ad8\u5fb7 POI \u641c\u7d22\u3001\u5730\u7406\u7f16\u7801", "\u672c\u6b21 POI \u5019\u9009\u7ed3\u679c\u3001\u6d88\u6b67\u7ed3\u679c", "\u4efb\u52a1\u7ea7\u79c1\u6709\uff1b\u5411 Planner \u53d1\u9001\u7ed3\u679c", 4.75, 2.0, BLUE)
    agent_card(page, "Planner Agent", "\u5728\u5df2\u9a8c\u8bc1\u5019\u9009\u6c60\u548c\u7ea6\u675f\u4e0b\u751f\u6210\u6309\u5929\u7684\u884c\u7a0b\u8349\u6848\u3002", "\u4e0d\u8bbf\u95ee\u5176\u4ed6\u7528\u6237\u8bb0\u5fc6\uff1b\u4e0d\u628a\u672a\u9a8c\u8bc1 POI \u5199\u5165\u7ed3\u679c\u3002", "LLM\u3001\u5019\u9009\u6c60\u3001\u5929\u6c14\u3001\u4ea4\u901a\u7ea6\u675f\uff08\u5f85\u63a5\u5165\uff09", "\u5f53\u524d route\u3001\u4fee\u6539\u610f\u89c1\u3001\u672c\u8f6e\u5ba1\u6838\u53cd\u9988", "\u4efb\u52a1\u7ea7\u79c1\u6709\uff1b\u8bfb\u53ea\u8bfb\u5168\u5c40\u7ea6\u675f", 8.75, 2.0, GOLD)

    # 7 Worker cards two
    page = slide(prs, "\u8be6\u7ec6\u8bbe\u8ba1 \u00b7 Worker Agent", "\u5ba1\u6838\u3001\u65f6\u95f4\u6838\u9a8c\u3001\u4e2a\u6027\u5316\u8bb0\u5fc6", "\u5ba1\u6838\u89d2\u8272\u4e0d\u53c2\u4e0e\u751f\u6210\uff1b\u8bb0\u5fc6\u89d2\u8272\u4e0d\u53c2\u4e0e\u8def\u7ebf\u5199\u5165\uff0c\u4ee5\u4fdd\u8bc1\u804c\u8d23\u72ec\u7acb", 7)
    agent_card(page, "Reviewer Agent", "\u5ba1\u6838\u666f\u70b9\u771f\u5b9e\u6027\u3001\u7ea6\u675f\u5339\u914d\u3001\u5730\u7406\u8de8\u5ea6\uff0c\u8f93\u51fa\u53ef\u6267\u884c\u7684\u4fee\u6539\u610f\u89c1\u3002", "\u4e0d\u91cd\u5199\u884c\u7a0b\uff1b\u4e0d\u76f4\u63a5\u4fee\u6539 Planner \u7684\u79c1\u6709\u72b6\u6001\u3002", "\u5019\u9009\u6c60\u3001\u8def\u7ebf\u8349\u6848\u3001\u89c4\u5219\u6821\u9a8c", "\u5ba1\u6838\u7ed3\u8bba\u3001\u95ee\u9898\u5217\u8868\u3001\u8bc4\u5206", "\u5ba1\u6838\u79c1\u6709\uff1b\u5411 Supervisor \u53d1\u9001 review_result", 0.75, 2.0, RED)
    agent_card(page, "Time Check Agent", "\u7528\u5f00\u653e\u65f6\u95f4\u548c\u901a\u52e4\u8017\u65f6\uff08\u5f85\u63a5\u5165\uff09\u68c0\u67e5\u65f6\u95f4\u7ebf\u3002", "\u4e0d\u8c03\u6574\u666f\u70b9\u987a\u5e8f\uff1b\u4ec5\u8f93\u51fa\u51b2\u7a81\u548c\u53d7\u5f71\u54cd\u8282\u70b9\u3002", "\u9ad8\u5fb7\u5f00\u653e\u65f6\u95f4\u3001\u8def\u7ebf\u89c4\u5212", "\u672c\u8f6e\u7684\u65f6\u95f4\u51b2\u7a81\u5217\u8868", "\u4efb\u52a1\u7ea7\u79c1\u6709\uff1b\u5411 Supervisor \u53d1\u9001 time_violation", 4.75, 2.0, GREEN)
    agent_card(page, "Memory Agent", "\u53ec\u56de\u4e0e\u5199\u5165\u7a33\u5b9a\u504f\u597d\uff0c\u8ba9\u89c4\u5212\u5177\u5907\u8de8\u4f1a\u8bdd\u4e2a\u6027\u5316\u80fd\u529b\u3002", "\u4e0d\u4fdd\u5b58\u6574\u6bb5\u5bf9\u8bdd\uff1b\u4e0d\u4f7f\u7528\u5176\u4ed6\u7528\u6237\u7684\u5411\u91cf\u8bb0\u5fc6\u3002", "SQLite \u753b\u50cf\u3001Chroma \u5411\u91cf\u5e93\u3001LLM \u63d0\u70bc", "\u53ef\u7f16\u8f91\u753b\u50cf\u3001\u8bed\u4e49\u504f\u597d\u3001\u6765\u6e90\u4e0e\u7f6e\u4fe1\u5ea6\uff08\u5f85\u5b8c\u5584\uff09", "\u6309 user_id \u4e25\u683c\u79c1\u6709\uff1b\u4ec5\u5411 Planner \u63d0\u4f9b Top-K \u53ec\u56de", 8.75, 2.0, BLUE)

    # 8 Message schema
    page = slide(prs, "\u6d88\u606f\u7ed3\u6784\u8bbe\u8ba1", "\u7edf\u4e00 Agent \u95f4\u6d88\u606f\u534f\u8bae", "\u7528 Pydantic \u5b9a\u4e49\u7ed3\u6784\u5316\u6d88\u606f\uff1b\u5f53\u524d\u72b6\u6001 dict \u53ef\u9010\u6b65\u8fc1\u79fb\u5230\u8fd9\u4e2a\u534f\u8bae", 8)
    box(page, 0.8, 1.98, 5.5, 3.95, RGBColor(35, 39, 44), RGBColor(35, 39, 44))
    code = '''{
  "task_id": "trip-20260724-001",
  "task_type": "poi_resolve",
  "from": "supervisor",
  "to": "poi_resolver",
  "content": {
    "query": "重庆秀山兴隆坳",
    "city_hint": "重庆"
  },
  "status": "pending",
  "attempt": 0,
  "trace_id": "...",
  "created_at": "ISO-8601"
}'''
    text(page, code, 1.02, 2.22, 5.08, 3.43, 11, WHITE)
    fields = [
        ("task_id", "\u540c\u4e00\u6b21\u4efb\u52a1\u7684\u5168\u94fe\u8def\u8ffd\u8e2a ID"),
        ("task_type", "\u89e6\u53d1\u8c03\u5ea6\u7684\u4efb\u52a1\u7c7b\u578b\uff0c\u5982 poi_resolve / review"),
        ("from / to", "\u6d88\u606f\u53d1\u8d77\u65b9\u4e0e\u63a5\u6536\u65b9\uff0c\u660e\u786e\u8d23\u4efb\u8fb9\u754c"),
        ("content", "\u7ed3\u6784\u5316\u8f93\u5165\u6216\u8f93\u51fa\uff0c\u7981\u6b62\u53ea\u4f20\u81ea\u7531\u6587\u672c"),
        ("status", "pending / running / done / failed / retrying"),
        ("attempt / trace_id", "\u652f\u6301\u91cd\u8bd5\u4e0e\u94fe\u8def\u8c03\u8bd5\u3002"),
    ]
    for index, (name, detail) in enumerate(fields):
        y = 2.02 + index * 0.62
        box(page, 6.75, y, 5.65, 0.48, WHITE, LINE)
        text(page, name, 6.92, y + 0.13, 1.28, 0.17, 10.2, BLUE, True)
        text(page, detail, 8.28, y + 0.1, 3.9, 0.25, 9.5, INK)

    # 9 Flow
    page = slide(prs, "\u534f\u4f5c\u6d41\u7a0b", "\u52a8\u6001\u8c03\u6574\u7684\u591a Agent \u4ea4\u4e92\u6d41\u7a0b", "\u793a\u4f8b\uff1a\u7528\u6237\u8bf7\u6c42\u5c06\u201c\u5317\u4eac\u5927\u5b66\u201d\u52a0\u5165\u884c\u7a0b\uff0c\u7cfb\u7edf\u53ea\u91cd\u89c4\u5212\u53d7\u5f71\u54cd\u7684\u65e5\u671f", 9)
    flow = [
        ("01", "\u7528\u6237 \u2192 Supervisor", "modify_plan\uff1a\u52a0\u5165\u5317\u4eac\u5927\u5b66"),
        ("02", "Supervisor \u2192 Intent", "\u8bc6\u522b\uff1a\u6307\u5b9a\u5730\u70b9 + \u884c\u7a0b\u4fee\u6539"),
        ("03", "Supervisor \u2192 POI Resolver", "\u9ad8\u5fb7\u7cbe\u786e\u68c0\u7d22\u3001\u6d88\u6b67\u3001\u8fd4\u56de\u5750\u6807"),
        ("04", "Supervisor \u2192 Planner", "\u4ec5\u4f20\u5165\u53d7\u5f71\u54cd\u65e5\u671f\u7684\u8def\u7ebf\u4e0e\u65b0 POI"),
        ("05", "Reviewer + Time Check", "\u5ba1\u6838\u5019\u9009\u6c60\u3001\u5f00\u653e\u65f6\u95f4\u3001\u901a\u52e4\u8017\u65f6\uff08\u5f85\u63a5\u5165\uff09"),
        ("06", "Supervisor \u2192 \u7528\u6237", "\u8f93\u51fa\u8c03\u6574\u540e\u884c\u7a0b\u3001\u53d8\u66f4\u539f\u56e0\u3001\u5907\u9009\u65b9\u6848"),
    ]
    for index, (number, name, body) in enumerate(flow):
        x = 0.75 + (index % 3) * 4.1
        y = 2.08 + (index // 3) * 2.0
        card(page, number, name, body, x, y, RED if index < 3 else GREEN, w=3.62, h=1.45)
    text(page, "\u8fdb\u9636\u5931\u8d25\u8def\u5f84\uff1aPOI \u591a\u4e49\u2192 \u4eba\u5de5\u786e\u8ba4\uff1b\u5de5\u5177\u8d85\u65f6\u2192 \u6709\u9650\u91cd\u8bd5\u2192 \u6a21\u62df\u6570\u636e\u964d\u7ea7\uff1b\u5ba1\u6838\u8d85\u8fc7\u8f6e\u6b21\u2192 \u8f93\u51fa\u7ea6\u675f\u51b2\u7a81\u3002", 0.94, 5.6, 11.4, 0.35, 11, MUTED, True, PP_ALIGN.CENTER)

    # 10 memory isolation table
    page = slide(prs, "\u8bb0\u5fc6\u9694\u79bb\u65b9\u6848", "\u8c01\u8bb0\u4ec0\u4e48\uff0c\u8c01\u80fd\u770b\uff1f", "\u6838\u5fc3\u7ea2\u7ebf\uff1a\u4e0d\u5141\u8bb8\u6240\u6709 Agent \u5171\u7528\u65e0\u9650\u4e0a\u4e0b\u6587\uff1b\u5171\u4eab\u7684\u5fc5\u987b\u662f\u6709\u8303\u56f4\u3001\u53ef\u8ffd\u8e2a\u7684\u53ea\u8bfb\u6570\u636e", 10)
    columns = [("\u8bb0\u5fc6\u9879", 0.75, 1.75), ("\u5f52\u5c5e", 3.4, 1.75), ("\u5171\u4eab\u8303\u56f4", 5.25, 1.75), ("\u8bf4\u660e / \u7ea2\u7ebf", 8.15, 1.75)]
    for label, x, y in columns:
        box(page, x, y, 2.35 if x < 3 else (1.6 if x < 5 else (2.5 if x < 8 else 4.4)), 0.44, INK, INK)
        text(page, label, x + 0.08, y + 0.11, 2.1, 0.16, 10.2, WHITE, True, PP_ALIGN.CENTER)
    rows = [
        ("\u7528\u6237\u539f\u59cb\u8bf7\u6c42", "Supervisor", "\u5168\u5c40\u53ea\u8bfb", "\u5404 Agent \u53ea\u8bfb\u5f53\u524d task_id \u7684\u8bf7\u6c42\u3002"),
        ("\u5019\u9009 POI \u4e0e\u5929\u6c14\u4e8b\u5b9e", "POI Resolver", "\u5411 Planner / Reviewer \u53ea\u8bfb", "\u4ec5\u5df2\u9a8c\u8bc1\u7ed3\u679c\u53ef\u8fdb\u5165\u5171\u4eab\u72b6\u6001\u3002"),
        ("Planner \u8def\u7ebf\u8349\u6848", "Planner", "\u5411 Reviewer / TimeCheck \u53ea\u8bfb", "Reviewer \u4e0d\u5f97\u76f4\u63a5\u4fee\u6539\u8349\u6848\u3002"),
        ("\u5ba1\u6838\u53cd\u9988\u4e0e\u95ee\u9898\u5217\u8868", "Reviewer", "\u5411 Supervisor / Planner \u53ea\u8bfb", "\u8bb0\u5fc6\u4e0e Planner \u7684\u751f\u6210\u7ea0\u7f20\u5206\u5f00\u3002"),
        ("\u65f6\u95f4\u51b2\u7a81\u5217\u8868", "Time Check", "\u5411 Supervisor / Planner \u53ea\u8bfb", "\u4ec5\u8bb0\u5f55\u5f53\u524d\u4efb\u52a1\uff0c\u4efb\u52a1\u7ed3\u675f\u5373\u8fc7\u671f\u3002"),
        ("\u7528\u6237\u957f\u671f\u504f\u597d", "Memory Agent", "\u5411 Intent / Planner Top-K \u53ea\u8bfb", "\u6309 user_id \u4e25\u683c\u9694\u79bb\uff0c\u7528\u6237\u53ef\u67e5\u770b\u548c\u5220\u9664\u3002"),
    ]
    for index, row in enumerate(rows):
        y = 2.19 + index * 0.62
        fills = [WHITE, WHITE, WHITE, WHITE] if index % 2 == 0 else [RGBColor(247, 244, 237)] * 4
        widths = [2.35, 1.6, 2.5, 4.4]
        xs = [0.75, 3.4, 5.25, 8.15]
        for col, value in enumerate(row):
            box(page, xs[col], y, widths[col], 0.58, fills[col], LINE, False)
            text(page, value, xs[col] + 0.08, y + 0.12, widths[col] - 0.16, 0.3, 9.25, INK, col == 0)
    text(page, "\u6ce8\uff1a\u5f53\u524d SQLite \u753b\u50cf + Chroma \u8bed\u4e49\u8bb0\u5fc6\u5df2\u5b9e\u73b0 user_id \u9694\u79bb\uff1bPlanner / Reviewer / TimeCheck \u7684\u79c1\u6709\u72b6\u6001\u5206\u533a\u662f\u672c\u6b21\u91cd\u6784\u7684\u540e\u7eed\u5b9e\u73b0\u91cd\u70b9\u3002", 0.82, 6.25, 11.6, 0.35, 10.3, RED, True, PP_ALIGN.CENTER)

    # 11 Design steps and reflection
    page = slide(prs, "\u8bbe\u8ba1\u63a8\u6f14\u4e0e\u603b\u7ed3", "\u4e3a\u4ec0\u4e48\u8fd9\u6837\u5206\u5c42\u3001\u8fd9\u6837\u9694\u79bb\uff1f", "\u8bbe\u8ba1\u7684\u76ee\u6807\u4e0d\u662f\u5c06 Agent \u6570\u91cf\u5806\u5f97\u66f4\u591a\uff0c\u800c\u662f\u8ba9\u51b3\u7b56\u8d23\u4efb\u6e05\u6670\u3001\u95ee\u9898\u53ef\u5b9a\u4f4d\u3001\u4e2a\u4eba\u6570\u636e\u53ef\u63a7", 11)
    cards(page, [
        ("01", "\u5148\u5206\u79bb\u201c\u4e8b\u5b9e\u201d\u4e0e\u201c\u751f\u6210\u201d", "POI\u3001\u5929\u6c14\u3001\u8def\u7ebf\u8c03\u7528\u786e\u5b9a\u6027\u5de5\u5177\uff1bLLM \u8d1f\u8d23\u7406\u89e3\u548c\u7ec4\u7ec7\u3002", RED),
        ("02", "\u518d\u5206\u79bb\u201c\u751f\u6210\u201d\u4e0e\u201c\u5ba1\u6838\u201d", "Planner \u4e0d\u81ea\u5ba1\uff1bReviewer \u4e0d\u91cd\u5199\u8def\u7ebf\uff0c\u907f\u514d\u65e2\u5f53\u8fd0\u52a8\u5458\u53c8\u5f53\u88c1\u5224\u3002", GREEN),
        ("03", "\u8bb0\u5fc6\u6309\u751f\u547d\u5468\u671f\u5206\u5c42", "\u5f53\u524d\u4efb\u52a1\u72b6\u6001\u4e0e\u7528\u6237\u957f\u671f\u504f\u597d\u5206\u5f00\uff0c\u907f\u514d\u4e0a\u4e0b\u6587\u6c61\u67d3\u3002", BLUE),
        ("04", "\u5148\u8bbe\u8ba1\u5931\u8d25\u8def\u5f84", "\u5de5\u5177\u8d85\u65f6\u3001POI \u6d88\u6b67\u5931\u8d25\u3001\u5ba1\u6838\u8d85\u8f6e\u6b21\u90fd\u6709\u660e\u786e\u964d\u7ea7\u4e0e\u63d0\u793a\u3002", GOLD),
        ("05", "\u540e\u7eed\u5b9e\u73b0\u987a\u5e8f", "\u663e\u5f0f Supervisor \u2192 \u6307\u5b9a POI \u52a8\u6001\u52a0\u5165 \u2192 \u4ea4\u901a\u7ea6\u675f \u2192 \u8bb0\u5fc6\u53ef\u89c6\u5316\u4e0e\u5220\u9664\u3002", RED),
        ("06", "\u5fc3\u5f97\u4f1a", "\u72ec\u7acb\u8bb0\u5fc6\u7684\u610f\u4e49\u662f\u201c\u53ea\u628a\u5fc5\u8981\u4fe1\u606f\u4ea4\u7ed9\u5fc5\u8981\u7684\u89d2\u8272\u201d\uff0c\u5b83\u540c\u65f6\u4fdd\u62a4\u4e2a\u4eba\u6570\u636e\u548c\u5ba1\u6838\u72ec\u7acb\u6027\u3002", GOLD),
    ])
    box(page, 0.7, 6.18, 11.95, 0.55, INK, INK)
    text(page, "\u7ed3\u8bba\uff1a\u5f53\u524d\u9879\u76ee\u5df2\u9a8c\u8bc1\u57fa\u7840\u591a Agent \u95ed\u73af\u548c\u4e2a\u6027\u5316\u8bb0\u5fc6\u53ef\u884c\uff1b\u672c\u8bbe\u8ba1\u8fdb\u4e00\u6b65\u628a\u5b83\u843d\u4e3a\u53ef\u8c03\u5ea6\u3001\u53ef\u5ba1\u6838\u3001\u53ef\u6269\u5c55\u7684\u5b9e\u73b0\u84dd\u56fe\u3002", 0.9, 6.34, 11.5, 0.18, 11, WHITE, True, PP_ALIGN.CENTER)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
