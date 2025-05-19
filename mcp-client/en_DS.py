# ------------------ 环境与依赖配置 ------------------
import sys
import os
import json
import asyncio
import tempfile
from dotenv import load_dotenv

# Windows 平台兼容
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()

# ------------------ 第三方依赖导入 ------------------
import streamlit as st
from contextlib import AsyncExitStack
from openai import OpenAI

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------------ 自定义rag工具包装结构 ------------------
class CustomRetrieverTool:
    def __init__(self, retriever):
        self.name = "rag"
        self.description = "用于从上传文档中检索答案"
        self.retriever = retriever

    def run(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        result = "\n\n".join([doc.page_content for doc in docs[:3]]) if docs else "没有找到相关内容。"

        # 模拟 .content[0].text 的结构
        class MockResponse:
            def __init__(self, text):
                self.content = [type("Text", (), {"text": text})()]

        return MockResponse(result)

# ------------------ 文档向量检索器构建 ------------------
@st.cache_resource(ttl="1h")
def configure_retriever(files):
    if not files:
        return None

    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir="D:\\")
    for file in files:
        file_path = os.path.join(temp_dir.name, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        ext = os.path.splitext(file.name)[1].lower()
        try:
            if ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                st.warning(f"不支持的文件类型: {file.name}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"加载文件 {file.name} 失败：{e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = DashScopeEmbeddings()
    vectordb = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
    return vectordb.as_retriever()

# ------------------ MCP 客户端封装 ------------------
class MCPClient:
    def __init__(self, memory=None):
        # 初始化会话和客户端对象
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")

        if not self.deepseek_api_key:
            raise ValueError("❌ 未找到 DeepSeek API Key，请在 .env 文件中设置 DEEPSEEK_API_KEY")

        self.client = OpenAI(api_key=self.deepseek_api_key, base_url=self.base_url)
        self.exit_stack = AsyncExitStack()
        self.memory = memory
        self.sessions = {}  # 存储多个服务端会话
        self.tools_map = {}  # 工具映射：工具名称 -> {"server_id": server_id, "tool_obj": tool}
        self.custom_tools = {}  # 存储自定义工具，如rag

    def add_custom_tool(self, name, tool):
        self.custom_tools[name] = tool

    async def connect_to_server(self, server_id):
        """连接到 MCP 服务器,从 .env 加载服务器配置"""
        command = os.getenv(f"{server_id.upper()}_SERVER_COMMAND")
        args_str = os.getenv(f"{server_id.upper()}_SERVER_ARGS")
        args = [arg.strip() for arg in args_str.split(",")]
        env_str = os.getenv(f"{server_id.upper()}_SERVER_ENV")
        if env_str:
            env = dict(item.strip().split("=", 1) for item in env_str.split(",") if "=" in item)
            server_params = StdioServerParameters(command=command, args=args, env=env)
        else:
            server_params = StdioServerParameters(command=command, args=args)
        # 启动 MCP 服务器并建立通信
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        self.sessions[server_id] = {"session": session, "stdio": stdio, "write": write}

        # 更新工具映射
        for tool in (await session.list_tools()).tools:
            self.tools_map[tool.name] = {
                "server_id": server_id,
                "tool_obj": tool
            }

    async def process_query(self, query):
        """使用 DeepSeek 和可用的工具处理查询"""
        # 根据是否加载了RAG工具动态生成 system prompt
        if "rag" in self.custom_tools:
            instruction = '''你是一个结合了文档检索（rag）与mcp工具调用的智能助手。请根据以下行为准则进行工作：
        1. 知识库处理：
        用户已上传知识库文档，无论用户输入为什么，请使用rag工具获取相关信息，并基于检索结果进行回答。
        2. 工具调用原则：
        无论是否使用RAG工具，你都可以调用可用的MCP工具来辅助完成任务。
        3. 回答要求：
        - 结合检索信息和工具结果进行清晰、有逻辑的回答。
        - 如检索结果无关紧要，请依靠MCP工具或自身知识作答。
        - 若无有效答案，也请坦率说明并提出合理建议。
        '''
        else:
            instruction = '''你是一个结合了外部mcp工具调用能力的智能助手。请根据以下行为准则进行工作：
        1. 工具调用原则：
        你可以调用你所能访问的MCP工具来辅助回答问题。
        2. 回答要求：
        - 主动判断并调用可用工具完成任务。
        - 回答需清晰、有逻辑，尽可能准确。
        - 若无有效信息，也请说明情况并提出合理建议。
        '''

        history = []
        if self.memory:
            for msg in self.memory.chat_memory.messages:
                if msg.type == "human":
                    history.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    history.append({"role": "assistant", "content": msg.content})

        messages = [{"role": "system", "content": instruction}] + history + [{"role": "user", "content": query}]

        # 构建统一的工具列表,分别添加mcp工具和自定义工具
        available_tools = []
        for tool_name, tool_info in self.tools_map.items():
            tool = tool_info["tool_obj"]
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        for tool in self.custom_tools.values():
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "用户要检索的问题"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })

        # 循环处理工具调用
        while True:
            # 初始 DeepSeek API 调用
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )

            # 处理返回的内容
            content = response.choices[0]
            if content.finish_reason == "tool_calls":
                # 执行工具调用
                for tool_call in content.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # 根据工具名称调用对应的服务端/自定义工具
                    if tool_name in self.tools_map:
                        session = self.sessions[self.tools_map[tool_name]["server_id"]]["session"]
                        result = await session.call_tool(tool_name, tool_args)
                        print(f"\n[ <{tool_args}> 调用工具 <{tool_name}>]\n")
                        tool_obj = self.tools_map[tool_name]["tool_obj"]
                        print(f"[注册工具] {tool_obj.name} 参数结构: {json.dumps(tool_obj.inputSchema, ensure_ascii=False, indent=2)}")
                        print(result)
                    elif tool_name in self.custom_tools:
                        result = self.custom_tools[tool_name].run(tool_args["query"])
                        print(f"\n[ <{tool_args}> 调用工具 <{tool_name}>]\n")
                        print(result)
                    # 将 tool 的结果作为一条消息回传
                    messages.append({"role": "assistant", "tool_calls": [tool_call]})
                    messages.append({
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_call_id": tool_call.id
                    })
            else:
                # 如果没有工具调用，写入记忆并返回最终回复
                if self.memory:
                    self.memory.chat_memory.add_user_message(query)
                    self.memory.chat_memory.add_ai_message(content.message.content)

                return content.message.content

# ------------------ MCP 初始化 ------------------
@st.cache_resource
def init_mcp(_memory):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    mcp = MCPClient(memory=memory)
    # 此处添加新的mcp server
    loop.run_until_complete(mcp.connect_to_server("amap_maps"))
    print("Connected to server: amap_maps")
    loop.run_until_complete(mcp.connect_to_server("tavily"))
    print("Connected to server: tavily")
    loop.run_until_complete(mcp.connect_to_server("filesystem"))
    print("Connected to server: filesystem")
    loop.run_until_complete(mcp.connect_to_server("sequential_thinking"))
    print("Connected to server: sequential_thinking")
    loop.run_until_complete(mcp.connect_to_server("time"))
    print("Connected to server: time")


    return mcp, loop

# ------------------ Streamlit 页面逻辑 ------------------
st.set_page_config(page_title="DeepSeek增强版", layout="wide")
st.title("📞 DeepSeek增强版")

uploaded_files = st.sidebar.file_uploader(
    "上传 RAG知识库文档（可选）",
    type=["txt", "pdf", "docx", "md"],
    accept_multiple_files=True,
)

retriever = configure_retriever(uploaded_files)

# 创建聊天消息历史记录和对话缓冲区内存
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

# 如果session_state中没有消息记录或用户点击了清空聊天记录按钮，则初始化消息记录和记忆
if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，我是文档问答助手。 有什么可以帮助你的？"}]
    msgs.clear()  # 清空历史对话

# 展示历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

mcp_client, event_loop = init_mcp(memory)

if 'rag' in mcp_client.custom_tools:
    del mcp_client.custom_tools['rag']
# 若用户上传知识库，调用知识库工具
if retriever:
    rag_tool = CustomRetrieverTool(retriever)
    mcp_client.add_custom_tool(rag_tool.name, rag_tool)

# 处理用户查询
user_input = st.chat_input("请开始提问吧!")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        with st.spinner("正在思考中..."):
            try:
                mcp_response = event_loop.run_until_complete(mcp_client.process_query(user_input))
                st.session_state.messages.append({"role": "assistant", "content": mcp_response})
                st.write(mcp_response)
            except Exception as e:
                st.error(f"MCP工具调用失败：{e}")
