"""Generate a 10-minute, 15-slide TravelMind defense deck."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TravelMind_项目综合答辩.pptx"
IMG = ROOT / "static" / "images"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

NAVY = RGBColor(17, 38, 72)
BLUE = RGBColor(31, 91, 168)
TEAL = RGBColor(18, 143, 139)
ORANGE = RGBColor(232, 126, 55)
RED = RGBColor(196, 71, 71)
INK = RGBColor(28, 39, 57)
MUTED = RGBColor(94, 108, 126)
LIGHT = RGBColor(244, 247, 251)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(218, 226, 235)
GREEN = RGBColor(39, 150, 99)


def add_bg(slide, color=WHITE):
    bg = slide.background.fill
    bg.solid(); bg.fore_color.rgb = color


def textbox(slide, text, x, y, w, h, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, font="Microsoft YaHei", valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.name = font; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    tf.margin_left = tf.margin_right = Inches(.03); tf.margin_top = tf.margin_bottom = Inches(.02)
    return shape


def rect(slide, x, y, w, h, fill=LIGHT, line=LINE, radius=True):
    typ = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    s = slide.shapes.add_shape(typ, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    return s


def title(slide, kicker, heading, sub=None):
    textbox(slide, kicker.upper(), .58, .32, 4.2, .25, 9, BLUE, True)
    textbox(slide, heading, .58, .62, 12.1, .55, 26, NAVY, True)
    if sub:
        textbox(slide, sub, .6, 1.2, 12, .34, 11, MUTED)
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(.58), Inches(1.62), Inches(1.05), Inches(.055))
    line.fill.solid(); line.fill.fore_color.rgb = ORANGE; line.line.fill.background()


def footer(slide, page):
    textbox(slide, "TravelMind · AI 多智能体旅行规划助手", .6, 7.15, 5, .18, 8.5, MUTED)
    textbox(slide, str(page), 12.25, 7.12, .45, .2, 8.5, MUTED, align=PP_ALIGN.RIGHT)


def bullet_box(slide, x, y, w, h, heading, bullets, accent=BLUE):
    rect(slide, x, y, w, h)
    rect(slide, x, y, .08, h, accent, accent, False)
    textbox(slide, heading, x+.25, y+.18, w-.45, .26, 14, NAVY, True)
    tf = slide.shapes.add_textbox(Inches(x+.26), Inches(y+.56), Inches(w-.45), Inches(h-.68)).text_frame
    tf.clear(); tf.word_wrap = True
    for i, value in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = value; p.level = 0; p.font.name = "Microsoft YaHei"; p.font.size = Pt(11.5); p.font.color.rgb = INK; p.space_after = Pt(7)
        p.text = "• " + p.text


def connector(slide, x1, y1, x2, y2, color=BLUE):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(1.6); c.line.end_arrowhead = True
    return c


def image(slide, filename, x, y, w, h):
    path = IMG / filename
    if path.exists():
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        rect(slide, x, y, w, h, RGBColor(235, 238, 243)); textbox(slide, f"缺少截图：{filename}", x+.15, y+.15, w-.3, .3, 11, MUTED)


# 1 cover
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s, NAVY)
rect(s, .62, .62, .11, 5.86, ORANGE, ORANGE, False)
textbox(s, "AI 多智能体开发与应用 · 项目综合答辩", .98, .82, 8, .35, 14, RGBColor(181, 205, 238), True)
textbox(s, "TravelMind", .97, 1.52, 8.8, .75, 38, WHITE, True)
textbox(s, "面向真实出行需求的多智能体旅行规划助手", .99, 2.37, 9.6, .45, 20, RGBColor(223, 232, 247))
textbox(s, "Supervisor 调度 · 独立 Agent 记忆 · RAG / 工具调用 · 评测闭环", .99, 3.07, 10.5, .35, 13, RGBColor(187, 208, 237))
rect(s, .99, 4.38, 5.7, 1.15, RGBColor(27, 55, 99), RGBColor(73, 107, 157))
textbox(s, "汇报人：请填写    学号：请填写\n项目仓库：Trip_Agent", 1.25, 4.67, 5.1, .56, 13, WHITE)
for i, (label, color) in enumerate([("Intent", BLUE), ("Planner", TEAL), ("Reviewer", ORANGE), ("RAG", RED)]):
    rect(s, 9.25+(i%2)*1.6, 1.65+(i//2)*1.2, 1.32, .7, color, color)
    textbox(s, label, 9.25+(i%2)*1.6, 1.88+(i//2)*1.2, 1.32, .22, 11, WHITE, True, PP_ALIGN.CENTER)
textbox(s, "2026.07", 1, 6.55, 2, .3, 11, RGBColor(181, 205, 238))

# 2 problem
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "01 · Background", "问题与目标：把“想去哪”变成可执行行程", "目标用户：需要快速获得景点、餐饮、交通、预算建议的自由行用户")
bullet_box(s,.65,1.95,3.9,3.55,"用户痛点",["信息分散：景点、餐饮、天气、交通要多次搜索","需求有约束：日期、预算、节奏、饮食偏好难同时满足","结果不可验证：模型容易编造地点或忽略路线可行性"],ORANGE)
bullet_box(s,4.75,1.95,3.9,3.55,"系统目标",["多 Agent 分工完成检索、规划、审核","调用真实高德 POI / 路线 / 天气等工具","将测试、日志、评估纳入交付，而不是只生成文本"],TEAL)
bullet_box(s,8.85,1.95,3.85,3.55,"一句话定位",["TravelMind 是一个由 Supervisor 集中编排的旅行规划系统","从用户自然语言请求出发，生成带路线、预算和依据的行程"],BLUE); footer(s,2)

# 3 solution overview
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"02 · Solution","核心方案与可展示能力")
items=[("真实 POI","候选池约束，减少景点幻觉",BLUE),("规划与审核","Planner 草案 → Reviewer 复核 → 有限返工",TEAL),("用户记忆","用户画像、任务状态、Agent 私有记忆分层",ORANGE),("工程防护","超时、重试、降级、权限边界、日志",RED)]
for i,(a,b,c) in enumerate(items):
    x=.7+(i%2)*6.1; y=1.9+(i//2)*1.78; rect(s,x,y,5.7,1.38,RGBColor(248,250,253)); rect(s,x+.24,y+.23,.56,.56,c,c)
    textbox(s,str(i+1),x+.24,y+.36,.56,.2,13,WHITE,True,PP_ALIGN.CENTER); textbox(s,a,x+.98,y+.23,4.4,.27,15,NAVY,True); textbox(s,b,x+.98,y+.62,4.35,.45,11.5,MUTED)
textbox(s,"答辩聚焦：不是“一个 Prompt 切换角色”，而是可审计的独立 Agent + 调度 + 记忆 + 评测闭环。",.75,5.85,11.7,.45,15,NAVY,True,PP_ALIGN.CENTER); footer(s,3)

# 4 architecture
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"03 · Architecture","系统架构：数据流、控制流与记忆边界")
nodes=[("用户请求",.65,2.55,1.3,BLUE),("Supervisor",2.3,2.55,1.45,NAVY),("Worker Agents",4.2,1.7,2.1,TEAL),("工具 / RAG",6.85,1.7,1.75,ORANGE),("规划结果",10.8,2.55,1.5,GREEN)]
for label,x,y,w,c in nodes: rect(s,x,y,w,.72,c,c); textbox(s,label,x,y+.23,w,.24,12,WHITE,True,PP_ALIGN.CENTER)
connector(s,1.95,2.91,2.3,2.91); connector(s,3.75,2.91,4.2,2.12); connector(s,6.3,2.06,6.85,2.06); connector(s,8.6,2.06,10.8,2.91)
for label,x,y,w,c in [("Intent Agent",4.25,2.75,1.55,TEAL),("POI Research",6.1,2.75,1.55,TEAL),("Planner / Reviewer",7.95,2.75,2.15,TEAL)]: rect(s,x,y,w,.62,RGBColor(232,247,246),c); textbox(s,label,x,y+.2,w,.2,10.5,INK,True,PP_ALIGN.CENTER)
textbox(s,"控制流：Supervisor 决定下一位执行者；数据流：结构化消息在 Agent 间传递；工具数据不直接写入所有 Agent 记忆。",.78,4.35,11.7,.45,13,MUTED,False,PP_ALIGN.CENTER)
bullet_box(s,.7,5.15,3.7,1.28,"外部数据",["高德 POI、路线、天气；旅行知识 RAG"],ORANGE); bullet_box(s,4.82,5.15,3.7,1.28,"存储",["SQLite 行程 / 用户画像；Agent 独立记忆"],TEAL); bullet_box(s,8.94,5.15,3.7,1.28,"前端",["React + SSE 阶段播报 + 地图与预算"],BLUE); footer(s,4)

# 5 agent + memory
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"04 · Multi-Agent","角色分工与记忆隔离：为什么这是真多 Agent")
agents=[("Intent", "识别目的地、日期与偏好", "私有：意图抽取记录", BLUE), ("POI Research", "只调用 POI 工具并返回候选池", "私有：检索记录", TEAL), ("Planner", "基于候选池生成行程", "私有：规划草案", ORANGE), ("Reviewer", "检查候选池、质量与返工", "私有：审核意见", RED)]
for i,(name,job,mem,c) in enumerate(agents):
    x=.7+i*3.15; rect(s,x,1.95,2.75,3.15,RGBColor(249,251,253)); rect(s,x,1.95,2.75,.55,c,c); textbox(s,name,x,2.12,2.75,.2,12,WHITE,True,PP_ALIGN.CENTER)
    textbox(s,"职责",x+.2,2.72,2.2,.22,10,c,True); textbox(s,job,x+.2,3.0,2.25,.55,11,INK)
    textbox(s,"记忆边界",x+.2,3.82,2.2,.22,10,c,True); textbox(s,mem,x+.2,4.1,2.25,.44,11,INK)
textbox(s,"共享的是“当前任务的结构化消息”；不共享完整 messages 列表。每个 Agent 有独立系统提示、独立实例、独立 memory。",.75,5.7,11.8,.45,14,NAVY,True,PP_ALIGN.CENTER); footer(s,5)

# 6 collaboration
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"05 · Orchestration","一次请求如何协作：固定主链路 + 条件路由")
flow=[("1. Intent",.75,2.35,BLUE),("2. POI 检索",3.08,2.35,TEAL),("3. Planner",5.41,2.35,ORANGE),("4. Reviewer",7.74,2.35,RED),("5. 输出",10.07,2.35,GREEN)]
for label,x,y,c in flow: rect(s,x,y,1.75,.72,c,c); textbox(s,label,x,y+.23,1.75,.23,11,WHITE,True,PP_ALIGN.CENTER)
for i in range(4): connector(s,2.5+i*2.33,2.71,3.08+i*2.33,2.71)
connector(s,8.62,3.1,6.28,3.85,RED); textbox(s,"未通过 → 最多一次返工",6.15,3.92,2.6,.25,10,RED,True)
rect(s,.8,5.0,11.8,.78,RGBColor(255,246,237),RGBColor(250,206,157)); textbox(s,"异常路径：某 Agent 超时 → Supervisor 有限重试 → 结构化失败结果 / 前端友好提示；不会让整个系统直接崩溃。",1.06,5.25,11.1,.24,12,INK,True,PP_ALIGN.CENTER)
footer(s,6)

# 7 tools/RAG
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"06 · Grounding","工具、RAG 与防幻觉设计")
bullet_box(s,.65,1.9,3.85,3.75,"实时工具",["高德 POI：候选景点、餐厅、酒店","高德路线：公交 / 地铁 / 步行等路线补充","天气：按日期提示出行风险","票价：页面实时查询，失败时如实标注未计价"],BLUE)
bullet_box(s,4.75,1.9,3.85,3.75,"RAG",["旅行知识库检索后供 Agent 参考","把检索与生成分离：先看召回，再看回答","RAG 不存实时票价，避免过期事实伪装成实时数据"],TEAL)
bullet_box(s,8.85,1.9,3.85,3.75,"约束",["Planner 只能从候选池选择景点","工具权限按 Agent 区分","保存行程等写操作需要用户确认","Prompt Injection / 参数错误纳入测试"],ORANGE); footer(s,7)

# 8 UI Demo screenshots
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"07 · Demo","端到端演示：输入、实时播报、可执行行程")
image(s,"规划页.png",.65,1.75,3.85,4.45); image(s,"规划进度.png",4.75,1.75,3.85,4.45); image(s,"规划详情.png",8.85,1.75,3.85,4.45)
for label,x in [("① 自然语言需求",.65),("② Agent 阶段播报",4.75),("③ 行程 / 地图 / 预算",8.85)]: textbox(s,label,x,6.33,3.85,.25,11,NAVY,True,PP_ALIGN.CENTER)
footer(s,8)

# 9 engineering improvements
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"08 · Engineering","工程化改进：从“能生成”到“可使用")
items=[("候选池扩展", "用户点名景点不在池中时，重新检索并核验后重规划"),("交通与费用", "路线、餐饮、门票、酒店费用分别标注实时 / 估算来源"),("历史行程", "先快速打开历史记录，再在后台实时刷新价格，避免页面卡死"),("可观测性", "SSE 显示每个阶段的处理结果，便于 Demo 和故障定位")]
for i,(h,b) in enumerate(items):
    x=.75+(i%2)*6.05; y=1.9+(i//2)*1.65; rect(s,x,y,5.65,1.27,RGBColor(248,250,253)); textbox(s,h,x+.25,y+.22,5,.25,14,NAVY,True); textbox(s,b,x+.25,y+.62,5.05,.38,11.5,MUTED)
textbox(s,"原则：实时数据缺失时“明确降级”，而不是编造；所有可量化数字由脚本、日志或报告复现。",.75,5.75,11.8,.38,14,ORANGE,True,PP_ALIGN.CENTER); footer(s,9)

# 10 testing
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"09 · Testing & Conclusion","三层测试、可复现评估与项目结论")
cols=[("单元测试","Intent / POI Research / Planner\n字段、私有记忆、候选池约束","Agent 函数正确",BLUE),("集成测试","Supervisor 调度、审核返工、工具权限、失败重试、SQLite 隔离","链路正确传递",TEAL),("端到端测试","10 条 Golden Set：用户请求 → 多 Agent → 行程 → Reviewer","业务结果可交付",ORANGE)]
for i,(h,b,c,co) in enumerate(cols):
    x=.7+i*4.16; rect(s,x,1.95,3.75,3.15,RGBColor(249,251,253)); rect(s,x,1.95,3.75,.58,co,co); textbox(s,h,x,2.15,3.75,.25,13,WHITE,True,PP_ALIGN.CENTER); textbox(s,b,x+.27,2.86,3.18,.75,12,INK,True); textbox(s,c,x+.27,4.3,3.18,.28,11,MUTED)
textbox(s,"自动化：68 passed；端到端：10 / 10；LLM-as-Judge：10 条评分完成、通过率 80%、平均 3.00 秒。",.75,5.55,11.8,.36,14,GREEN,True,PP_ALIGN.CENTER)
textbox(s,"结论：TravelMind 将独立角色、受控工具、记忆隔离和可复现评估组合为一条可演示、可验证的 Agent 工程链路。",.75,6.08,11.8,.34,12,NAVY,True,PP_ALIGN.CENTER); footer(s,10)

# 11 eval
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"10 · Evaluation","可复现评估与 LLM-as-Judge")
rect(s,.72,1.92,3.0,3.35,RGBColor(239,247,255),RGBColor(188,215,244)); textbox(s,"固定基线",1.0,2.2,2.45,.28,15,BLUE,True,PP_ALIGN.CENTER); textbox(s,"10 条 Golden Set\n固定 Fixture POI\n固定 Prompt 版本\ntemperature = 0",1.08,2.8,2.25,1.25,15,NAVY,True,PP_ALIGN.CENTER)
rect(s,5.15,1.92,3.0,3.35,RGBColor(239,250,247),RGBColor(181,229,213)); textbox(s,"Judge 维度",5.42,2.2,2.45,.28,15,TEAL,True,PP_ALIGN.CENTER); textbox(s,"候选池一致性\n需求满足\n行程完整性\nReviewer 一致性\n表达清晰度",5.55,2.76,2.22,1.55,13,INK,True,PP_ALIGN.CENTER)
rect(s,9.58,1.92,3.0,3.35,RGBColor(255,246,237),RGBColor(250,206,157)); textbox(s,"首轮真实结果",9.85,2.2,2.45,.28,15,ORANGE,True,PP_ALIGN.CENTER); textbox(s,"10 / 10 完成评分\nJudge 通过率 80%\n平均延迟 3.00 秒\n失败：constraint_miss ×2",9.88,2.8,2.4,1.35,14,NAVY,True,PP_ALIGN.CENTER)
textbox(s,"报告保存：模型提供方、Prompt 版本、每条输入、分项分数、失败分类与延迟；支持后续版本对比。",.82,5.72,11.7,.36,13,MUTED,False,PP_ALIGN.CENTER); footer(s,11)

# 12 failures
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"11 · Failure Analysis","失败案例与下一轮优化：用数据驱动迭代")
bullet_box(s,.7,1.9,3.7,3.65,"观察到的问题",["LLM-as-Judge 发现 2 条 constraint_miss","“慢节奏 / 餐饮偏好”等自然语言约束在离线 Fixture 行程中只得到基本满足","该问题不是模型崩溃，而是约束表达和可验证字段不足"],RED)
bullet_box(s,4.82,1.9,3.7,3.65,"已具备的防护",["Reviewer 审核与有限返工","Worker 超时有重试与结构化失败返回","真实工具失败时降级提示，不伪造实时数据"],TEAL)
bullet_box(s,8.94,1.9,3.7,3.65,"下一步（3项）",["将用户偏好显式传入 Planner 的结构化约束","为慢节奏、亲子、雨天、餐饮添加可自动验证字段","收集线上失败请求，按缺日期 / 工具失败 / 候选不足分类回归"],ORANGE); footer(s,12)

# 13 comparison
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"12 · Design Decision","为什么不是单 Agent：取舍与收益")
headers=["维度","单 Agent","TravelMind 多 Agent"]
rows=[("职责","检索、规划、审核混在同一上下文","职责分离，失败可定位到具体 Worker"),("数据依据","容易直接生成并混入幻觉","候选池 + 工具边界 + Reviewer 校验"),("记忆","全局 messages 易污染","独立实例 / 私有 memory / 结构化消息"),("工程维护","难复现、难测单点","可做单元、集成、端到端与 Judge 评估")]
xs=[.75,3.1,7.5]; ws=[2.2,4.15,5.05]
for i,h in enumerate(headers): rect(s,xs[i],1.85,ws[i],.5,NAVY,NAVY); textbox(s,h,xs[i],2.02,ws[i],.2,12,WHITE,True,PP_ALIGN.CENTER)
for r,row in enumerate(rows):
    y=2.38+r*.8
    for i,value in enumerate(row): rect(s,xs[i],y,ws[i],.75,RGBColor(249,251,253)); textbox(s,value,xs[i]+.13,y+.17,ws[i]-.26,.4,11,INK, i==0, PP_ALIGN.CENTER if i==0 else PP_ALIGN.LEFT)
textbox(s,"代价：链路更长、模型调用更多。因此必须配合有限轮次、并发、缓存、评测与降级策略。",.85,5.9,11.6,.33,13,ORANGE,True,PP_ALIGN.CENTER); footer(s,13)

# 14 demo script
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s,"13 · Demo Plan","现场 Demo 脚本与备份方案")
steps=[("0:00","输入：北京 2 日文化游，节奏慢，想吃本地小吃"),("0:30","展示 SSE 播报：Intent → POI 候选池 → Planner / Reviewer"),("1:30","展示行程：酒店起点、地图路线、门票 / 餐饮 / 交通预算"),("2:30","展示异常：日期缺失提示 或 工具失败后的降级信息"),("3:10","展示评测报告：三层测试 + Judge 分项结果")]
for i,(t,v) in enumerate(steps):
    y=1.8+i*.78; rect(s,.85,y,1.0,.48,BLUE,BLUE); textbox(s,t,.85,y+.14,1,.18,10,WHITE,True,PP_ALIGN.CENTER); textbox(s,v,2.12,y+.1,9.75,.3,13,INK, i==0)
rect(s,.85,5.95,11.7,.55,RGBColor(255,246,237),RGBColor(250,206,157)); textbox(s,"Plan B：准备本地截图 / 录屏与离线 Fixture 测试；网络或 API 异常时仍能演示相同调度逻辑与评测结果。",1.1,6.14,11.2,.2,11.5,INK,True,PP_ALIGN.CENTER); footer(s,14)

# 15 close
s=prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s,NAVY)
textbox(s,"结论",.85,.86,2,.3,14,RGBColor(181,205,238),True)
textbox(s,"TravelMind：让旅行规划\n从一次生成走向可验证的协作系统",.85,1.45,10.6,1.3,30,WHITE,True)
textbox(s,"我们交付的不只是路线文本，而是：独立角色、受控工具、可追溯记忆、失败防护与可复现评估。",.88,3.18,11.2,.45,16,RGBColor(210,224,244))
rect(s,.9,4.45,11.4,.95,RGBColor(27,55,99),RGBColor(73,107,157)); textbox(s,"谢谢各位老师，欢迎提问",.9,4.74,11.4,.3,20,WHITE,True,PP_ALIGN.CENTER)
textbox(s,"备用材料：架构图 · 评测报告 · 失败日志 · 项目源码",.9,6.2,11.4,.28,11,RGBColor(181,205,238),False,PP_ALIGN.CENTER)

# 答辩要求控制在约 10 页：后续备用页保留在脚本中，生成时仅输出前 10 页。
while len(prs.slides) > 10:
    rel_id = prs.slides._sldIdLst[-1].rId
    prs.part.drop_rel(rel_id)
    del prs.slides._sldIdLst[-1]

prs.save(OUT)
print(OUT)
