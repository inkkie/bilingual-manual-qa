"""提取两份手册的前 10 页；仅使用本地 PyMuPDF，不调用模型 API。"""

from pathlib import Path
import sys
import unicodedata

BASE_DIR = Path(__file__).resolve().parent
# 支持安装在当前项目中的依赖，也兼容常规 pip 安装。
LOCAL_PACKAGES = BASE_DIR / ".python-packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import pymupdf


FILES = (
    ("中文手册内容.pdf", "中文前10页.txt"),
    ("英文手册内容.pdf", "英文前10页.txt"),
)


def page_warning(text: str) -> str | None:
    """启发式检测，不能识别所有乱码；无文字也可能是扫描页。"""
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "空白页或未提取到文字（可能是扫描图片页）"
    suspicious = sum(
        char == "\ufffd" or unicodedata.category(char) in {"Cc", "Co", "Cs", "Cn"}
        for char in visible
    )
    if suspicious / len(visible) >= 0.05 or "(cid:" in text:
        return "疑似乱码页（异常字符占比至少 5%，或含 CID 占位符）"
    return None


def extract_pdf(source_name: str, target_name: str) -> None:
    source = BASE_DIR / source_name
    target = BASE_DIR / "output" / target_name
    with pymupdf.open(source) as document:
        if document.needs_pass:
            raise ValueError("PDF 已加密，需要密码")
        page_count = min(10, len(document))
        print(f"\n{source_name}：共 {len(document)} 页，处理前 {page_count} 页")
        with target.open("w", encoding="utf-8") as output:
            for index in range(page_count):
                text = document[index].get_text("text")
                page_number = index + 1
                print(f"  PDF 第 {page_number} 页：{len(text)} 个字符（含空白字符）")
                warning = page_warning(text)
                if warning:
                    print(f"  提示：{source_name} PDF 第 {page_number} 页：{warning}")
                output.write(f"===== PDF 第 {page_number} 页 =====\n")
                output.write(text)
                output.write("\n\n")
    print(f"已保存：{target}")


def main() -> int:
    (BASE_DIR / "output").mkdir(exist_ok=True)
    failed = False
    for source_name, target_name in FILES:
        try:
            extract_pdf(source_name, target_name)
        except Exception as error:
            failed = True
            print(f"错误：处理 {source_name} 失败：{error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
