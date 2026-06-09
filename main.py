from bedrock_agentcore import BedrockAgentCoreApp
from mcp.client.streamable_http import streamablehttp_client # add this
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient

app = BedrockAgentCoreApp()
defaultAgent = Agent(model="global.anthropic.claude-haiku-4-5-20251001-v1:0")

@app.entrypoint
def invoke(payload):
    """Your AI agent function"""
    user_message = payload.get("prompt", "Hello! How can I help you today?")
    qlikToken = user_message[:4]
    result = None
    if (qlikToken == "/qat"):
        parts = user_message.split(" ", 1)
        access_token = parts[0][4:]
        user_query = parts[1] if len(parts) > 1 else "Hello! How can I help you today?"
        # setup Qlik MCP
        headers = {}
        headers["Authorization"] = f"Bearer {access_token}"
        qlikClient = MCPClient(lambda: streamablehttp_client("https://presales-showcase.us.qlikcloud.com/api/ai/mcp", headers=headers, timeout=120))

        with qlikClient:
            tools = qlikClient.list_tools_sync()

            qlikAgent = Agent(
                model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                tools=tools,
                system_prompt="""You are a data analytics assistant with access to Qlik Cloud.

                When a user asks an analytical question or wants to explore data, always default 
                to using the qlik_create_data_object tool to retrieve and analyze data directly.

                Only use the following tools if the user explicitly asks for them:
                - qlik_create_sheet
                - qlik_create_measure
                - qlik_create_dimension
                - qlik_add_chart
                - qlik_add_filter

                Do not create sheets, measures, dimensions, charts, or filters unless the user 
                specifically requests it. Focus on answering questions using data objects only.
                
                RESPONSE FORMATTING:
                Format all responses using Markdown. Use headers to organize sections, 
                bullet or numbered lists for enumerations, bold for key terms, tables 
                for structured data, and fenced code blocks for any code or query syntax.

                """
            )
            result = qlikAgent(user_query)
    else:
        result = defaultAgent(user_message)

    return {"result": result.message}

if __name__ == "__main__":
    app.run()
