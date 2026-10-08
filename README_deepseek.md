# DeepSeek手册问答

## 本地网页

页面入口为 `app.py`，在已经设置Key的同一个PowerShell终端运行：

```powershell
Set-Location 'D:\项目训练'
.\.venv-retrieval\Scripts\python.exe -m streamlit run .\app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false --server.fileWatcherType none
```

浏览器打开 `http://127.0.0.1:8501`，终端按Ctrl+C停止服务。当前环境已有Streamlit 1.45.1，无需另行安装；新环境可安装 `requirements-app.txt`。

支持中英文提问、依据状态、手册名、来源语言和PDF物理页码，每条引用可展开查看片段全文。接线关系不明确时保留查看PDF原图提示。复用 `ask_deepseek.py` 的检索器、API调用及引用校验；使用 `st.cache_resource` 缓存模型与索引，并加锁保护共享推理。仅提交表单时调用API，展开引用或页面重跑不会自动重新请求。

Key只从服务进程的 `DEEPSEEK_API_KEY` 读取，无网页Key输入框，不存入session_state或缓存，不写日志。网页只在会话内保存经过凭据遮蔽的最近一次回答与片段，不写回答文件。若启动后才设置环境变量，需要从设置变量的终端重启服务。

页面测试使用Streamlit官方 [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)，模拟API验证交互，不使用真实Key调用DeepSeek：

```powershell
.\.venv-retrieval\Scripts\python.exe -m unittest test_app.py test_ask_deepseek.py
```

## 命令行

脚本：`ask_deepseek.py`。复用1596条通过审核片段的混合索引，跨中英文取Top 5，再调用DeepSeek。无需安装新依赖，HTTP使用Python标准库。

在已经设置`DEEPSEEK_API_KEY`的同一个VS Code PowerShell终端运行：

```powershell
Set-Location 'D:\项目训练'
.\.venv-retrieval\Scripts\python.exe .\ask_deepseek.py
```

输入问题后回车；`exit`退出。每题独立，不保留对话上下文。也可单次提问：

```powershell
.\.venv-retrieval\Scripts\python.exe .\ask_deepseek.py --question "出现 e333 后重新上电仍显示怎么办？"
```

`--show-sources`显示片段全文；`--dry-run --question "问题"`仅本地检索、不读取Key或调用API。`--timeout 180`可调整超时，`--model`可指定账户支持的模型名。

默认模型`deepseek-flash`、端点`https://api.deepseek.com/chat/completions`，依据[DeepSeek官方调用文档](https://api-docs.deepseek.com/zh-cn/)。关闭思考模式，请求JSON格式回答。

Key仅通过环境变量读取，不放命令行、不写文件、不打印、不传入模型正文。只发送当前问题及检索到的5条片段（含来源元数据），不上传整个PDF或全库。不自动记录回答或请求日志；错误只显示固定提示，不回显服务端错误正文或请求头，不自动重试收费请求。

回答按“充分/部分/不足”声明依据状态；页码为从1开始的PDF物理页码。模型选择来源ID与逐字摘录，本地核验后从索引元数据拼接引用。无效引用、伪造摘录或不完整输出会被拒绝展示。接线图缺少明确对应文字时要求查看PDF原图，不推断端子号。引用存在不等于结论一定被证据支持，充分性仍由生成模型判断，重要结论需要核对原文。

已运行离线检索与模拟HTTP测试；没有使用真实Key验证远程认证、账户余额或模型生成效果。真实调用由用户在已设置Key的终端运行。

```powershell
.\.venv-retrieval\Scripts\python.exe -m unittest test_ask_deepseek.py
```
