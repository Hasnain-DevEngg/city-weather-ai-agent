import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient

# ------------------- CONFIG -------------------------
st.set_page_config(
    page_title="City Intelligence AI",
    page_icon="🌍",
    layout="wide"
)

load_dotenv()

# ------------------- UI -------------------------
st.title("🌍 City Intelligence AI Agent")
st.markdown("Get **weather 🌤️** and **latest news 📰** about any city")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    approve_tools = st.toggle("Auto-approve tools", value=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.pending_tool = None
        st.rerun()

# ------------------- TOOLS -------------------------

@tool
def get_weather(city: str) -> str:
    """Get current weather of a City"""

    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return f"❌ Weather error for {city}: {data.get('message', 'Unknown error')}"

        temp = data["main"]["temp"]
        weather = data["weather"][0]["description"]

        return f"🌤️ **{city}**: {temp}°C, {weather}"

    except Exception as e:
        return f"❌ Failed to fetch weather: {str(e)}"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            max_results=3
        )

        results = response.get("results", [])

        if not results:
            return f"❌ No news found for {city}"

        news_list = []
        for r in results:
            title = r.get("title", "No title")
            link = r.get("url", "")
            content = r.get("content", "")

            news_list.append(
                f"📰 **{title}**\n{content}\n🔗 {link}"
            )

        return "\n\n".join(news_list)

    except Exception as e:
        return f"❌ Failed to fetch news: {str(e)}"


# ------------------- LLM -------------------------

llm = ChatMistralAI(model="mistral-small")

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools([get_weather, get_news])

# ------------------- SESSION STATE -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None

# ------------------- DISPLAY CHAT -------------------------

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ------------------- USER INPUT -------------------------

user_input = st.chat_input("Ask about a city...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        result = llm_with_tools.invoke(st.session_state.messages)
        st.session_state.messages.append(result)

    # TOOL CALL DETECTED
    if result.tool_calls:
        st.session_state.pending_tool = result.tool_calls[0]

    else:
        with st.chat_message("assistant"):
            st.markdown(result.content)


# ------------------- TOOL APPROVAL -------------------------

if st.session_state.pending_tool:
    # Handle multiple tool calls
    pending_tools = st.session_state.pending_tool if isinstance(st.session_state.pending_tool, list) else [st.session_state.pending_tool]
    approved_tools = []
    denied_tools = []

    for idx, tool_call in enumerate(pending_tools):
        tool_name = tool_call["name"]
        st.warning(f"⚠️ Agent wants to use tool: **{tool_name}**")

        col1, col2 = st.columns(2)
        approve = col1.button(f"✅ Approve {tool_name}", key=f"approve_{idx}")
        deny = col2.button(f"❌ Deny {tool_name}", key=f"deny_{idx}")

        if approve:
            approved_tools.append(tool_call)
        elif deny:
            denied_tools.append(tool_call)

    # Run all approved tools
    if approved_tools:
        for tool_call in approved_tools:
            tool_name = tool_call["name"]
            with st.spinner(f"Running {tool_name}..."):
                tool_args = tool_call.get("args", {})
                tool_result = tools[tool_name].invoke(tool_args)

            # Save tool result
            st.session_state.messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )
            )

        # Re-run LLM with tool results
        with st.spinner("Generating response..."):
            result = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(result)

        with st.chat_message("assistant"):
            st.markdown(result.content)

    # Handle denied tools
    for tool_call in denied_tools:
        st.session_state.messages.append(
            ToolMessage(
                content=f"❌ Tool call '{tool_call['name']}' was denied by user.",
                tool_call_id=tool_call["id"]
            )
        )
        with st.chat_message("assistant"):
            st.markdown(f"❌ Tool call '{tool_call['name']}' was denied by user.")

    # Clear pending tools after handling
    st.session_state.pending_tool = None
