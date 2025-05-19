📘 项目功能介绍：基于 Streamlit + DeepSeek + MCP 多工具集成的智能问答系统

本项目构建了一个集 文档问答（RAG）、外部工具调用（MCP Server） 和 对话上下文记忆 于一体的智能问答系统，支持用户通过 Web 页面与模型交互，执行复杂任务并获取个性化回复。
🎯 项目核心功能概览
1. Streamlit 前端界面交互

    基于 Streamlit 构建简洁直观的对话界面。

    支持上传本地文档（.txt, .pdf, .docx, .md），用于构建私有知识库。

    提供侧边栏按钮清空对话记录，便于重新提问。

2. 文档问答功能（RAG）

    文档上传后自动解析并分块嵌入，构建临时向量库（Chroma）。

    通过封装的 CustomRetrieverTool，支持对用户上传文档进行内容检索。

    模型根据检索结果和用户问题生成准确回复，实现私有文档问答功能。

3. MCP Server 多工具远程调用

    支持通过 MCP（Multi-Component Protocol）协议连接多个外部服务端工具，默认支持以下工具：

        🗺️ 高德地图（amap_maps）

        🔎 Tavily 在线搜索（tavily）

        📁 本地文件系统访问（filesystem）

        🧠 链式推理工具（sequential_thinking）

        🕒 当前时间查询（time）

    工具元数据由模型自动加载并识别，模型会在必要时主动调用对应工具完成任务。

    所有工具参数与返回结果自动封装处理，简化开发维护。

4. DeepSeek 大模型调用

    使用 OpenAI 接口方式对接 DeepSeek 模型（需设置 .env 中 DEEPSEEK_API_KEY、BASE_URL 和 MODEL）。

    支持 Function Calling 方式与工具无缝协作，根据任务自动决定是否调用工具。

    支持注入不同的系统提示词模板，适配是否启用文档问答功能（RAG）。

5. 对话记忆管理

    集成 LangChain 的 ConversationBufferMemory 与 StreamlitChatMessageHistory，实现连续上下文记忆。

    支持对用户和 AI 历史对话进行跟踪，并在模型生成回复时作为上下文参考。


🧩 环境与配置说明

    .env 文件中需要设置以下内容：

    DEEPSEEK_API_KEY=<your-deepseek-api-key>
    BASE_URL=https://api.deepseek.com
    MODEL=deepseek-chat

    AMAP_MAPS_SERVER_COMMAND=python
    AMAP_MAPS_SERVER_ARGS=amap_server.py
    AMAP_MAPS_SERVER_ENV=AMAP_API_KEY=your-key

    TAVILY_SERVER_COMMAND=python
    TAVILY_SERVER_ARGS=tavily_server.py
    TAVILY_SERVER_ENV=TAVILY_API_KEY=your-token

    # ... 其他 MCP 工具配置

📎 使用方式

    配置 MCP Server 工具服务（确保配置和路径正确）

    运行 streamlit run app.py 启动前端问答界面

    通过侧边栏上传文档（可选）

    在对话框中输入问题，即可开始与 AI 聊天，并享受工具增强能力

✅ 已实现的实用场景示例

    查询上传 PDF 报告中某段数据摘要

    调用高德地图返回城市坐标信息

    联网搜索当前新闻并总结

    使用本地文件系统列出目录文件

    自动完成链式逻辑推理任务

如需添加更多工具服务，仅需编写 MCP 兼容服务端并在 .env 中添加配置，即可动态注册新工具，具备良好的可拓展性与模块化设计。
