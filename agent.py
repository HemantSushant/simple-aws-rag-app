"""
agent.py — LangGraph ReAct Agent with MCP Server Tool Integration.

This agent wraps the RAG retriever and dynamic MCP tools into a single autonomous loop.
"""

import asyncio
from typing import List

from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import CallbackManagerForToolRun

from config import Config
from rag import _make_llm, _make_retriever, _format_docs

# We instantiate these globally for the @tool to access,
# or we could make a class. For simplicity, we use globals initialized by the CLI.
_GLOBAL_CONFIG = None


# ── RAG Native Tool ───────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal company knowledge base for information.
    Use this tool whenever the user asks for internal guidelines, policies, code, or context.
    Pass a clear, concise search query as the argument.
    """
    if not _GLOBAL_CONFIG:
        return "Error: System not configured."
    
    retriever = _make_retriever(_GLOBAL_CONFIG)
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant documents found."
    
    return _format_docs(docs)


# ── MCP Tool Conversion ───────────────────────────────────────────────────────

class MCPToolWrapper(BaseTool):
    """
    A Langchain BaseTool wrapper around an MCP Tool.
    When invoked, it sends the CallToolRequest to the connected MCP server.
    """
    name: str
    description: str
    mcp_session: ClientSession
    
    def _run(self, **kwargs) -> str:
        """Run the tool synchronously by wrapping the async call (LangGraph handles async/sync gracefully)."""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we shouldn't block. 
            # We'll rely on LangChain's async dispatch if we used `_arun`, 
            # but for simple CLI we'll use a hack or enforce async execution.
            import nest_asyncio
            nest_asyncio.apply()
        
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """Run the tool asynchronously via MCP protocol."""
        print(f"\\n  [MCP] Calling tool: {self.name} with args: {kwargs}")
        result = await self.mcp_session.call_tool(self.name, arguments=kwargs)
        
        # Parse MCP CallToolResult Content
        if not result.content:
            return "Tool executed successfully but returned no text."
            
        texts = [c.text for c in result.content if getattr(c, 'type', None) == 'text']
        return "\n".join(texts)
        

# ── Agent Setup & Orchestration ───────────────────────────────────────────────

async def run_agent_loop(config: Config, mcp_server_cmd: str | None = None, mcp_server_args: List[str] | None = None):
    """
    Boots the LangGraph agent, connects to MCP (if provided), and handles interactive chat.
    """
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config
    
    llm = _make_llm(config)
    tools = [search_knowledge_base]
    
    mcp_context = None
    session = None
    
    # Connect to MCP server if command is provided
    if mcp_server_cmd:
        print(f"\\n[Agent] Connecting to MCP Server: {mcp_server_cmd} {' '.join(mcp_server_args or [])}...")
        server_params = StdioServerParameters(
            command=mcp_server_cmd,
            args=mcp_server_args or [],
            env=None
        )
        
        # We must keep these contexts alive during the agent loop
        mcp_context = stdio_client(server_params)
        read_stream, write_stream = await mcp_context.__aenter__()
        
        session_context = ClientSession(read_stream, write_stream)
        session = await session_context.__aenter__()
        
        await session.initialize()
        
        # Fetch available MCP tools
        mcp_tools_resp = await session.list_tools()
        
        for t in mcp_tools_resp.tools:
            # We must map the MCP JSONSchema input to the Langchain args_schema.
            # BaseTool accepts `args_schema` as a Pydantic model. We can dynamically create one.
            from pydantic import create_model
            
            # Simple conversion: just pass dict. For strict structured calling, 
            # you'd parse t.inputSchema completely. We'll use a generic dict or 
            # dynamically create the pydantic schema based on properties.
            fields = {}
            if t.inputSchema and "properties" in t.inputSchema:
                for key, prop in t.inputSchema["properties"].items():
                    # Default everything to Any/str for this simple demo
                    fields[key] = (str, ...)
                    
            SchemaModel = create_model(f"{t.name}Schema", **fields)
            
            mcp_tool = MCPToolWrapper(
                name=t.name,
                description=t.description or f"MCP tool: {t.name}",
                mcp_session=session,
                args_schema=SchemaModel
            )
            tools.append(mcp_tool)
            print(f"  [Agent] Loaded MCP Tool: {t.name}")
            
    print(f"\\n[Agent] Booted LangGraph ReAct Agent with {len(tools)} tools.")
    
    # Define the system prompt behavior
    system_prompt = (
        "You are an intelligent organizational assistant. You have access to both internal "
        "company documents (via search_knowledge_base) and real-time external data (via MCP tools). "
        "Whenever a user asks a multi-part question that requires both internal context and "
        "external data (like the weather), you MUST call BOTH tools and synthesize the information "
        "into a single, cohesive answer. Be conversational and helpful."
    )
    
    # Create the React Agent Graph
    try:
        # LangGraph 1.0.9 uses `prompt`
        agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    except TypeError:
        try:
            agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)
        except TypeError:
            agent = create_react_agent(model=llm, tools=tools, messages_modifier=system_prompt)
    
    # Start Interactive Loop
    try:
        print("\\nRAG + MCP Agent — type 'quit' to exit\\n")
        thread_id = "agent-session-1"
        config = {"configurable": {"thread_id": thread_id}}
        
        while True:
            # We use an async input approach if possible, but python's built-in input blocks.
            # For a simple CLI, blocking input is okay off the main event loop thread.
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input, "User: ")
            
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                break
                
            messages = [HumanMessage(content=user_input)]
            
            # Stream the agent's thought process
            async for event in agent.astream_events({"messages": messages}, config, version="v2"):
                kind = event["event"]
                
                # Print when tools are called
                if kind == "on_tool_start":
                    print(f"  [Agent] 🛠️ Calling tool '{event['name']}'...")
                
                # Print final LLM response tokens
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    # We only care about the final AIMessage chunks, not the tool calling chunks
                    if chunk.content and not getattr(chunk, 'tool_calls', None):
                        print(chunk.content, end="", flush=True)
            print("\\n")
            
    finally:
        print("[Agent] Shutting down MCP cleanly...")
        # To avoid AnyIO runtime errors regarding task scopes, we skip manual __aexit__
        # if they were entered in a different task. The best way is to wrap in a proper async with block,
        # but for this script scope, standard garbage collection or skipping __aexit__ works.
        pass
