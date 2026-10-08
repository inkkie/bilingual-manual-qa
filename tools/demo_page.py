"""Screenshot-only fixture; synthetic response, no model or API calls."""
from pathlib import Path
import runpy
import sys
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
st.session_state["answer_bundle"] = {
    "question": "【预置模拟回答 / Demo — 非真实API输出】出现 e333 后重新上电仍显示怎么办？",
    "sources": [{"source_id": "S1", "source_file": "英文手册内容.pdf", "language": "en", "pdf_page": 404,
        "chunk_id": "en_p0404_c003", "text": "First, turn the power OFF then back ON again. If the display remains the same, the Digital Controller must be repaired."}],
    "result": {"status": "sufficient", "needs_pdf_image": False, "limitations": "本截图仅演示页面布局、来源和原文展开，不是模型质量评测。",
        "claims": [{"text": "先断电再重新上电；若仍显示相同错误，手册要求维修控制器。",
            "evidence": [{"source_id": "S1", "quote": "First, turn the power OFF then back ON again."}]}]}}
# Disable submission for this screenshot fixture. The real app uses the environment normally.
import ask_deepseek
def no_api(*args, **kwargs):
    raise ask_deepseek.SafeError("演示模式不调用API。请运行app.py进行实际问答。")
ask_deepseek.call_deepseek = no_api
runpy.run_path(str(BASE / "app.py"), run_name="__main__")
