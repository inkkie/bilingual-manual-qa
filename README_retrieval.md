# 本地手册向量检索

当前已建库并完成20题评测，无需重新下载或建库即可查询。未执行OCR，没有调用远程嵌入、生成或评判API。

## 当前推荐入口：混合检索

打开 `output/retrieval_hybrid/comparison_report.html` 查看同一20题的新旧Top-5全文、来源语言、手册、PDF页码、判定及变化。JSON、Markdown与CSV在同目录。原来的 `output/retrieval_eval/` 保留作为基线，不覆盖。

复测充分依据由9题变为15题（45%→75%），部分4题、未找到1题。9题改善、10题不变、1题退步（Q07）；Q08仍未找到完整依据。Q12/Q15仍需查看PDF原图，未推断端子对应关系。Q11的新Top-5有逐型号明确的电源端子文字，计为充分。这是参与词表和排序调整的20题开发集复测，不是独立测试集成绩。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-retrieval\Scripts\python.exe hybrid_retrieval.py query --text "出现 e333 后重新上电仍显示怎么办？"
```

默认跨中英文检索。`--language en` 或 `--language zh` 仅在用户主动需要过滤时使用。查询输出完整片段、来源语言、PDF页码、向量/BM25分数与排序诊断；不会自动生成答案。

实现与检查：

- 修正单独分词 `query: ` / `passage: ` 尾空格多出token的问题，标准前缀在1596个文档和20个查询上均与整句分词一致。派生向量单独保存于 `output/vector_index_hybrid/`，原索引不变。`LocalEmbedder(canonical_prefix=True)`用于新入口；旧入口默认保留原分词以复现基线。
- 长片段仍按512-token窗口、64-token重叠编码，315219个正文token全部覆盖，50个长片段没有静默截断；聚合为一片段一向量仍可能稀释局部信息。
- 每路取100个向量/BM25/精确标识符候选，另补每种语言各50个向量和BM25候选，去重后融合排序。双语术语表只扩展查询，没有改问题、答案要求或索引文字。
- BM25按语言归一化，加入术语、参数代码和故障码匹配；精确型号区分E5CC、E5CC-U、E5CC-B。目录/索引点引线密度识别后分数乘0.35，不删除页面；每页最多2条进入Top-5。无点引线的目录可能漏识别，参数附录不一律降权。
- Q01新第1名为英文155页，原zh_p0398_c001虽进入候选仍未进Top-5；Q06中文175页与Q18英文404页目标片段均排第1。

复现改进后的评测与验证（完全离线，无需新依赖）：

```powershell
.\.venv-retrieval\Scripts\python.exe hybrid_retrieval.py evaluate
.\.venv-retrieval\Scripts\python.exe report_hybrid.py
.\.venv-retrieval\Scripts\python.exe verify_hybrid.py
```

`hybrid_assessments.json` 是本次会话逐题阅读Top-5后的评判，绑定检索结果哈希。若检索结果发生变化，报告脚本拒绝自动套用旧判断，需要先重新阅读内容。`embedding_audit.json` 记录前缀、分窗与重点题排名检查，`validation.json` 记录原始数据不变、准入隔离、语言过滤、元数据/报告一致性等验证。

## 数据范围

- 原始1995个片段及原始PDF均保留不变。
- 仅纳入 `output/chunk_audit/retrieval_manifest.jsonl` 中 `eligible: true` 的1596条：中文276、英文1320。
- 399条损坏/待复核/自然语言内容不足的片段不入索引；没有重新放宽筛选条件。
- `output/vector_index/metadata.jsonl` 逐条保留 `source_file`、`manual_name`、`language`、`pdf_page`、`chunk_id`、`text`，以及对应向量行号。
- 页码始终是从1开始的PDF物理页码，不是正文页脚的章节页码。

## 索引与本地模型

使用 [intfloat/multilingual-e5-small 官方模型](https://huggingface.co/intfloat/multilingual-e5-small)，固定版本 `614241f622f53c4eeff9890bdc4f31cfecc418b3`。遵循模型文档的 `query: ` / `passage: ` 前缀、掩码均值池化和L2归一化。

索引为1596×384的float32向量矩阵（`vectors.npy`），查询用NumPy精确计算余弦相似度，不依赖数据库服务。默认同时搜索两种语言，也可按语言过滤。

模型单窗口最多512 tokens。50个超过窗口的片段采用64-token重叠分窗，再按新增token数加权合并为一个向量，最终仍是一片段一向量。全部1646个窗口均在本地计算，未静默截断；原片段文本完全保留。

模型文件位于 `models/multilingual-e5-small/`。运行环境在 `.venv-retrieval/`，复用机器已有的 PyTorch 2.8.0+cu126、NumPy 2.1.3，安装了 Transformers 4.57.6。支持CPU，当前建库使用本机RTX 4060 Laptop GPU。CPU查询可能慢一些。

## 原向量基线查询（保留用于对照）

在当前项目目录的PowerShell中运行：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-retrieval\Scripts\python.exe local_vector_index.py query --text "显示 s.err 时应该检查什么？" --top-k 5
```

只检索英文：

```powershell
.\.venv-retrieval\Scripts\python.exe local_vector_index.py query --text "How can I clear an alarm latch?" --language en --top-k 5
```

返回每个命中的片段全文、手册名、语言、PDF页码、chunk_id和相似度。查询只检索，不生成答案，也不会自动判断任意新问题是否有充分依据。

## 原向量基线20题评测

问题在 `retrieval_questions.json`：中英文各10题，参数设置、报警、接线、故障排查各5题。问题及判定要求在运行检索前固定。

本次采用双语全库Top-5，不翻译、不做关键词混合、不重排。逐题阅读实际检索内容后，评为充分9题、部分2题、未找到9题；充分依据率45%。这是小规模功能评测，不代表全库准确率或召回率。

“是否找到有效依据”仅在充分时为true。相似度高不等于有依据；目录、不同型号的接线说明、不同故障的处理建议不会被计作有效答案。

阅读文件：

- `output/retrieval_eval/evaluation_report.html`：可按语言、类别、判定筛选，并展开片段全文。
- `output/retrieval_eval/evaluation_report.md`：20题的实际Top-5全文、来源页码与逐题解释。
- `output/retrieval_eval/evaluation_report.json`：含逐题布尔值、三类判断、实际命中与依据chunk_id。
- `output/retrieval_eval/evaluation_summary.csv`：逐题结果表。
- `output/retrieval_eval/retrieval_results.json`：未添加评判的原始检索结果。
- `retrieval_assessments.json`：本次会话阅读结果后记录的判断，绑定结果文件哈希；不是自动裁判。

中文题本次只有2题充分、1题部分，英文题7题充分、1题部分。部分问题未命中库内已有答案，另有目录抢占排名、接线图结构丢失等问题。不能把所有失败都归因为OCR缺失，也不能直接用当前检索结果自动回答所有问题。

## 原向量基线复现与验证

```powershell
.\.venv-retrieval\Scripts\python.exe local_vector_index.py build
.\.venv-retrieval\Scripts\python.exe local_vector_index.py evaluate
.\.venv-retrieval\Scripts\python.exe verify_vector_index.py
.\.venv-retrieval\Scripts\python.exe report_retrieval.py
```

`build` / `evaluate` 分别重写派生索引/原始评测结果，不触碰PDF、原chunks或准入清单。若库、模型、问题、硬件或结果发生变化，旧评判不得自动复用，`report_retrieval.py` 会因哈希不符拒绝生成报告，需要重新阅读新结果并更新判断。

`verify_vector_index.py` 核验准入集合、排除集合、每条原文及元数据、向量归一化、自身最近邻以及20题结果完整性。当前核验已通过，记录见 `output/vector_index/validation.json`。

另一台机器需先安装依赖和下载模型（仅下载阶段联网）：

```powershell
python -m venv --system-site-packages .venv-retrieval
.\.venv-retrieval\Scripts\python.exe -m pip install -r requirements-retrieval.txt
.\.venv-retrieval\Scripts\python.exe download_embedding_model.py
```

此后建库、检索和评测强制离线，仅加载本地文件；不需要API Key。下载脚本不读取任何手册或片段。
