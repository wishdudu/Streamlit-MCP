import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek

# 初始化 DeepSeek 大模型客户端
llm = ChatDeepSeek(
    model="deepseek-chat",  # 指定 DeepSeek 的模型名称
    api_key="sk-e508ba61639640848060a1a2c1ee7b17"  # 替换为您自己的 DeepSeek API 密钥
)


# 解析并输出结果
def print_optimized_result(agent_response):
    """
    解析代理响应并输出优化后的结果。
    :param agent_response: 代理返回的完整响应
    """
    messages = agent_response.get("messages", [])
    steps = []  # 用于记录计算步骤
    final_answer = None  # 最终答案

    for message in messages:
        if hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
            # 提取工具调用信息
            tool_calls = message.additional_kwargs["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                steps.append(f"调用工具: {tool_name}({tool_args})")
        elif message.type == "tool":
            # 提取工具执行结果
            tool_name = message.name
            tool_result = message.content
            steps.append(f"{tool_name} 的结果是: {tool_result}")
        elif message.type == "ai":
            # 提取最终答案
            final_answer = message.content

    # 打印优化后的结果
    print("\n计算过程:")
    for step in steps:
        print(f"- {step}")
    if final_answer:
        print(f"\n最终答案: {final_answer}")


# 定义异步主函数
async def main():
    # 创建服务器参数
    server_params = StdioServerParameters(
        command="python",
        # 确保更新为 math_server.py 文件路径
        args=["./math_server.py"],
    )

    # 使用 stdio_client 进行连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            print("成功连接到 Math 服务")

            # 加载工具
            tools = await load_mcp_tools(session)
            print("加载工具完成: ", [tool.name for tool in tools])

            # 创建代理
            agent = create_react_agent(llm, tools)

            # 循环接收用户输入
            while True:
                try:
                    # 提示用户输入问题
                    user_input = input("\n请输入您的问题（或输入 'exit' 退出）：")
                    if user_input.lower() == "exit":
                        print("感谢使用！再见！")
                        break

                    # 调用代理处理问题
                    agent_response = await agent.ainvoke({"messages": user_input})

                    # 打印完整响应（调试用）
                    # print("\n完整响应:", agent_response)

                    # 调用抽取的方法处理输出结果
                    print_optimized_result(agent_response)

                except Exception as e:
                    print(f"发生错误：{e}")
                    continue


# 使用 asyncio 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())