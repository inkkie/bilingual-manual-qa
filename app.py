"""本地页面：python -m streamlit run app.py --server.address 127.0.0.1"""
import os
import threading

import streamlit as st
import ask_deepseek as qa


@st.cache_resource(show_spinner=False)
def cached_retriever():
    # 仅缓存本地模型/索引及锁，不缓存Key、请求或API结果。
    embedder, engine = qa.load_retriever()
    return embedder, engine, threading.Lock()


def scrub(value, key):
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, list):
        return [scrub(v, key) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v, key) for k, v in value.items()}
    return value


def main():
    st.set_page_config(page_title="手册问答", page_icon="📖", layout="wide")
    st.title("手册问答 / Manual Q&A")
    st.caption("支持中英文提问 · 跨语言检索 · 引用页码为 PDF 物理页码，从 1 开始")
    st.caption("提交后，仅将本题及检索到的 5 个片段发送给 DeepSeek。每题独立，不沿用上题上下文。")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        st.warning("未读取到 DEEPSEEK_API_KEY。请在已设置该变量的同一个 PowerShell 终端启动页面。")
    with st.form("question_form"):
        question = st.text_area("问题 / Question", placeholder="例如：出现 e333 后重新上电仍显示怎么办？", max_chars=4000)
        submitted = st.form_submit_button("检索并回答 / Ask", disabled=not bool(key))
    if submitted:
        # 清除上一题结果，避免请求失败时把旧回答展示为新回答。
        st.session_state.pop("answer_bundle", None)
        question = question.strip()
        if not question:
            st.warning("请先输入问题。")
        elif key in question:
            st.error("问题包含凭据，已拒绝发送。请仅输入手册问题。")
        else:
            try:
                with st.spinner("正在本地检索；首次加载模型需要稍候……"):
                    embedder, engine, lock = cached_retriever()
                    with lock:
                        encoded, _ = embedder.encode([question], "query")
                        hits = engine.search(question, encoded[0], top_k=5, candidate_k=100)["hits"]
                    sources = qa.sources_from_hits(hits)
                if not sources:
                    result = {"status": "insufficient", "needs_pdf_image": False, "claims": [],
                              "limitations": "没有检索到片段，未调用 API。"}
                else:
                    with st.spinner("DeepSeek 正在根据片段生成回答……"):
                        result = qa.call_deepseek(question, sources, key, "deepseek-flash", 120)
                # 复用CLI的引用、逐字摘录和依据状态校验；失败则不展示回答。
                qa.render_answer(result, sources)
                st.session_state["answer_bundle"] = scrub(
                    {"question": question, "result": result, "sources": sources}, key)
            except qa.SafeError as exc:
                st.error(scrub(str(exc), key))
            except Exception:
                # 不使用st.exception/logger/traceback，避免错误携带请求头或Key。
                st.error("运行失败。请检查本地模型、索引、依赖及网络；未输出可能含凭据的异常详情。")
    bundle = st.session_state.get("answer_bundle")
    if not bundle:
        return
    bundle = scrub(bundle, key)
    result, sources = bundle["result"], bundle["sources"]
    by_id = {s["source_id"]: s for s in sources}
    st.subheader("回答 / Answer")
    st.text(bundle["question"])
    if result["needs_pdf_image"]:
        st.warning("依据不足：需查看 PDF 原图确认具体对应关系，不得据混排文字推断端子或连线。")
    elif result["status"] == "sufficient":
        st.success("依据状态：模型判断有充分依据")
    elif result["status"] == "partial":
        st.warning("依据不足：仅能回答部分内容")
    else:
        st.warning("依据不足：本次检索无法确认答案")
    cited = set()
    for claim in result["claims"]:
        # 模型文本按纯文本显示，避免外链/HTML等被解释为网页内容。
        st.text(claim["text"])
        for sid in dict.fromkeys(e["source_id"] for e in claim["evidence"]):
            s = by_id[sid]
            cited.add(sid)
            with st.expander(f"[{sid}] {s['source_file']} · PDF第{s['pdf_page']}页 · {s['language']} · 查看原始片段"):
                st.caption(s["chunk_id"])
                st.text(s["text"])
    if result["limitations"]:
        st.text("限制：" + result["limitations"])
    remaining = [s for s in sources if s["source_id"] not in cited]
    if remaining:
        st.subheader("其他检索片段（未作为回答依据）")
        for s in remaining:
            with st.expander(f"[{s['source_id']}] {s['source_file']} · PDF第{s['pdf_page']}页 · {s['language']}"):
                st.caption(s["chunk_id"])
                st.text(s["text"])
    st.caption("引用编号与摘录已核验；引用存在不等于结论一定成立，请结合原文判断。")


if __name__ == "__main__":
    main()
