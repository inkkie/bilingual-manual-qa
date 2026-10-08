"""离线测试：不读取真实Key，不连接DeepSeek。"""
import io
import json
import unittest
from unittest.mock import Mock, patch
import urllib.error

from ask_deepseek import SafeError, call_deepseek, render_answer


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"source_id": "S1", "source_file": "英文手册内容.pdf",
                         "pdf_page": 404, "language": "en", "text": "e333: turn power OFF then ON."}]
        self.answer = {"status": "sufficient", "needs_pdf_image": False, "limitations": "",
                       "claims": [{"text": "重新上电。", "evidence": [{"source_id": "S1", "quote": "turn power OFF then ON."}]}]}

    def test_citation_comes_from_metadata(self):
        self.assertIn("PDF第404页", render_answer(self.answer, self.sources))

    def test_unknown_reference_and_fabricated_quote_rejected(self):
        for evidence in ({"source_id": "S9", "quote": "e333"}, {"source_id": "S1", "quote": "invented"}):
            self.answer["claims"][0]["evidence"] = [evidence]
            with self.assertRaises(SafeError):
                render_answer(self.answer, self.sources)

    def test_abstention_and_image_requirement(self):
        answer = {"status": "insufficient", "claims": [], "needs_pdf_image": True,
                  "limitations": "文字没有极性对应关系。"}
        text = render_answer(answer, self.sources)
        self.assertIn("依据不足", text)
        self.assertIn("PDF 原图", text)

    @patch("ask_deepseek.urllib.request.build_opener")
    def test_request_and_response_offline(self, build):
        response = io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(self.answer)}}]}).encode())
        build.return_value.open.return_value = response
        self.assertEqual(call_deepseek("test", self.sources, "fake-test-key", "deepseek-flash", 120), self.answer)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        payload = json.loads(request.data)
        self.assertNotIn("fake-test-key", json.dumps(payload))
        self.assertIn("sources", payload["messages"][1]["content"])

    @patch("ask_deepseek.urllib.request.build_opener")
    def test_http_error_does_not_echo_body_or_key(self, build):
        build.return_value.open.side_effect = urllib.error.HTTPError("https://api.deepseek.com", 401,
            "fake-test-key", {}, io.BytesIO(b"fake-test-key"))
        with self.assertRaises(SafeError) as error:
            call_deepseek("test", self.sources, "fake-test-key", "deepseek-flash", 120)
        self.assertNotIn("fake-test-key", str(error.exception))


if __name__ == "__main__":
    unittest.main()
