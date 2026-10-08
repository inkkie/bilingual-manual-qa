"""页面离线测试：只使用虚构Key与模拟API。"""
import os
import unittest
from unittest.mock import Mock, patch
import streamlit as st
from streamlit.testing.v1 import AppTest


class PageTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-ui-test-token"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.embedder, self.engine = Mock(), Mock()
        self.embedder.encode.return_value = ([[1.0]], {})
        self.engine.search.return_value = {"hits": [{"chunk_id": "en_p0404_c003", "source_file": "英文手册内容.pdf",
            "pdf_page": 404, "language": "en", "text": "e333: power OFF then ON."}]}
        self.result = {"status": "partial", "needs_pdf_image": True, "limitations": "需查看 PDF 原图。",
            "claims": [{"text": "部分依据。", "evidence": [{"source_id": "S1", "quote": "power OFF then ON."}]}]}

    def test_bilingual_submission_cache_and_expanders(self):
        with patch("ask_deepseek.load_retriever", return_value=(self.embedder, self.engine)) as load, \
             patch("ask_deepseek.call_deepseek", return_value=self.result) as api:
            page = AppTest.from_file("app.py").run()
            for question in ["出现 e333 怎么办？", "What does e333 mean?"]:
                page.text_area[0].set_value(question)
                page.button[0].click().run()
                self.assertFalse(page.exception)
                self.assertIn("PDF 原图", page.warning[0].value)
                self.assertIn("PDF第404页", page.expander[0].label)
                self.assertTrue(any("power OFF" in t.value for t in page.text))
            self.assertEqual(load.call_count, 1)
            self.assertEqual(api.call_count, 2)
            page.run()
            self.assertEqual(api.call_count, 2)  # 展开/重跑不会重复计费调用

    def test_missing_key_and_safe_error(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            page = AppTest.from_file("app.py").run()
            self.assertTrue(page.button[0].disabled)
        with patch("ask_deepseek.load_retriever", return_value=(self.embedder, self.engine)), \
             patch("ask_deepseek.call_deepseek", side_effect=RuntimeError("fake-ui-test-token")):
            page = AppTest.from_file("app.py").run()
            page.text_area[0].set_value("test")
            page.button[0].click().run()
            self.assertFalse(page.exception)
            self.assertNotIn("fake-ui-test-token", page.error[0].value)
            self.assertNotIn("answer_bundle", page.session_state)


if __name__ == "__main__":
    unittest.main()
