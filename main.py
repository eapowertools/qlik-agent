import boto3
import json
import time
from bedrock_agentcore import BedrockAgentCoreApp
from datetime import datetime, timezone
from mcp.client.streamable_http import streamablehttp_client # add this
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient

DEFAULT_PROMPT = "Hello!"
QLIK_MCP_TENANT_URL = "https://presales-showcase.us.qlikcloud.com/api/ai/mcp"
AGENT_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
TABLE_NAME = "agentcore-sessions-eps"
SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours

app = BedrockAgentCoreApp()

def get_Qlik_token(session_id: str) -> str:
    table = dynamodb.Table(TABLE_NAME)
    try:
        response = table.get_item(Key={"session_id": session_id+"-token"})
        item = response.get("Item")
        if item:
            return item.get("tokenValue", "")
    except Exception as e:
        print(f"[session] Failed to load Qlik token for {session_id}: {e}")
    return ""

def set_Qlik_token(session_id: str, token: str):
    """Persist Qlik token to DynamoDB."""
    table = dynamodb.Table(TABLE_NAME)
    ttl = int(time.time()) + SESSION_TTL_SECONDS
    try:
        table.put_item(Item={
            "session_id": session_id+"-token",
            "tokenValue": token,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "ttl": ttl,
        })
    except Exception as e:
        print(f"[session] Failed to save token for {session_id}: {e}")

def get_history(session_id: str) -> list:
    """Load conversation history from DynamoDB."""
    table = dynamodb.Table(TABLE_NAME)
    try:
        response = table.get_item(Key={"session_id": session_id})
        item = response.get("Item")
        if item:
            return json.loads(item.get("history", "[]"))
    except Exception as e:
        print(f"[session] Failed to load history for {session_id}: {e}")
    return []

def save_history(session_id: str, history: list):
    """Persist updated conversation history to DynamoDB."""
    table = dynamodb.Table(TABLE_NAME)
    ttl = int(time.time()) + SESSION_TTL_SECONDS
    try:
        table.put_item(Item={
            "session_id": session_id,
            "history": json.dumps(history),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "ttl": ttl,
        })
    except Exception as e:
        print(f"[session] Failed to save history for {session_id}: {e}")

@app.entrypoint
def invoke(payload):
    """AgentCore invoke — stateful via DynamoDB session store."""
    qlik_access_token = payload.get("qlik_access_token", "")
    user_message = payload.get("prompt", DEFAULT_PROMPT)
    session_id = payload.get("session_id", "default")  # caller must provide this
    if session_id == "default":
        defaultAgent = Agent(model=AGENT_MODEL)
        result = defaultAgent(user_message)
        return {"result": result.message}
    elif qlik_access_token:
        set_Qlik_token(session_id, qlik_access_token)
        return {"result": "Set Qlik access token for MCP server."}

    qlik_access_token = get_Qlik_token(session_id)
    if not qlik_access_token:
        return {"result": "No Qlik access token was found for session."}

    # Load existing history for this session
    history = get_history(session_id)

    result_message = None


    headers = {"Authorization": f"Bearer {qlik_access_token}"}

    qlik_client = MCPClient(
        lambda: streamablehttp_client(
            QLIK_MCP_TENANT_URL,
            headers=headers,
            timeout=120,
        )
    )

    with qlik_client:
        tools = qlik_client.list_tools_sync()

        agent = Agent(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            tools=tools,
            # Inject prior history into the system prompt or messages —
            # depends on how your Agent class handles multi-turn context.
            # Option A: pass messages directly (preferred if Agent supports it)
            messages=history,
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
            """,
        )

        result = agent(user_message)
        result_message = result.message

        # Persist the updated messages back to DynamoDB
        updated_history = agent.messages  # contains full turn history
        save_history(session_id, updated_history)

    return {
        "result": result_message,
        "session_id": session_id,  # echo back so client can reuse it
    }

if __name__ == "__main__":
    app.run()
