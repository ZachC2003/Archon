import streamlit as st
import sys
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import asyncio
from openai import AsyncOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import json
import platform

# Set Windows event loop policy for subprocess support
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Add paths and load environment variables
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "FinDataMCP"))
env_path = os.path.join(os.path.dirname(__file__), "FinDataMCP", ".env")
load_dotenv(env_path)

# OpenAI client setup
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MCP client setup
mcp_client = MultiServerMCPClient()

# Set page config
st.set_page_config(
    page_title="Stock Data Dashboard",
    page_icon="📈",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary-color: #00CC99;
        --secondary-color: #EB2D8C;
        --text-color: #262730;
    }
    h1, h2, h3 { color: var(--primary-color); }
    .stButton > button {
        color: white;
        border: 2px solid var(--primary-color);
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        border: 2px solid var(--secondary-color);
    }
    </style>
""", unsafe_allow_html=True)

# Start MCP client and connect to servers
async def start_mcp_client():
    try:
        await mcp_client.__aenter__()
        st.write("Starting MCP client...")  # Debug output
        # Use venv's Python executable
        venv_python = os.path.join(os.path.dirname(sys.executable), "python.exe" if platform.system() == "Windows" else "python")
        await mcp_client.connect_to_server(
            "stock",
            command=venv_python,
            args=[os.path.join(os.path.dirname(__file__), "FinDataMCP", "stock_server.py")]
        )
        await mcp_client.connect_to_server(
            "analyzer",
            command=venv_python,
            args=[os.path.join(os.path.dirname(__file__), "FinDataMCP", "financial_analyzer.py")]
        )
        st.session_state.mcp_connected = True
        st.write("MCP client connected successfully.")  # Debug output
    except Exception as e:
        st.error(f"Failed to start MCP client: {str(e)}")
        st.session_state.mcp_connected = False

async def stop_mcp_client():
    if st.session_state.get("mcp_connected", False):
        await mcp_client.__aexit__(None, None, None)
        st.session_state.mcp_connected = False

async def _call_mcp_tool(server: str, tool_name: str, args: dict) -> dict:
    tools = mcp_client.get_tools()
    tool = next((t for t in tools if t.name == tool_name and t.server == server), None)
    if tool:
        return await tool.ainvoke(args)
    raise ValueError(f"Tool {tool_name} not found on server {server}")

# Fetch and Analyze Stock Data
async def fetch_stock_data(ticker: str) -> dict:
    """Fetch and analyze stock data using MCP servers."""
    if not st.session_state.get("mcp_connected", False):
        st.error("MCP client not connected. Cannot fetch data.")
        return None
    results = {}
    try:
        results["stock_info"] = await _call_mcp_tool("stock", "get_stock_info", {"ticker": ticker})
        results["financial_snapshot"] = await _call_mcp_tool("stock", "get_financial_snapshot", {"ticker": ticker})
        results["insider_trades"] = await _call_mcp_tool("stock", "get_insider_trades", {"ticker": ticker})
        results["institutional_ownership"] = await _call_mcp_tool("stock", "get_institutional_ownership", {"ticker": ticker})
        results["news"] = await _call_mcp_tool("stock", "get_news", {"ticker": ticker})

        results["financial_analysis"] = {
            "metrics": await _call_mcp_tool("analyzer", "analyze_financial_metrics", {"ticker": ticker}),
            "sentiment": await _call_mcp_tool("analyzer", "analyze_news_sentiment", {"ticker": ticker}),
            "institutional": await _call_mcp_tool("analyzer", "analyze_institutional_activity", {"ticker": ticker})
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
    return results

# Display Functions
def display_stock_data(data: dict, ticker: str):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Company Facts", "Financial Snapshot", "Insider Trades",
        "Institutional Ownership", "News"
    ])
    with tab1:
        st.subheader(f"Company Facts for {ticker}")
        if data["stock_info"]["success"]:
            st.json(data["stock_info"]["data"])
        else:
            st.error(f"Failed to fetch company facts: {data['stock_info']['error']}")
    with tab2:
        st.subheader(f"Financial Snapshot for {ticker}")
        if data["financial_snapshot"]["success"]:
            snapshot = data["financial_snapshot"]["data"]
            df = pd.DataFrame([snapshot])
            st.dataframe(df)
            valuation_metrics = {
                "P/E Ratio": snapshot.get("pe_ratio"),
                "P/B Ratio": snapshot.get("pb_ratio"),
                "EV/EBITDA": snapshot.get("ev_to_ebitda")
            }
            fig = px.bar(
                x=list(valuation_metrics.keys()),
                y=[val for val in valuation_metrics.values() if val is not None],
                title="Valuation Metrics",
                labels={"x": "Metric", "y": "Value"}
            )
            st.plotly_chart(fig)
        else:
            st.error(f"Failed to fetch financial snapshot: {data['financial_snapshot']['error']}")
    with tab3:
        st.subheader(f"Insider Trades for {ticker}")
        if data["insider_trades"]["success"]:
            trades = data["insider_trades"]["data"]
            if trades:
                df = pd.DataFrame(trades)
                display_cols = ["name", "title", "transaction_date", "transaction_shares",
                                "transaction_price_per_share", "transaction_value", "security_title"]
                df = df[[col for col in display_cols if col in df.columns]]
                df.columns = ["Name", "Title", "Date", "Shares", "Price/Share", "Value", "Security Type"]
                st.dataframe(df)
            else:
                st.info("No insider trades data available.")
        else:
            st.error(f"Failed to fetch insider trades: {data['insider_trades']['error']}")
    with tab4:
        st.subheader(f"Institutional Ownership for {ticker}")
        if data["institutional_ownership"]["success"]:
            ownership = data["institutional_ownership"]["data"]
            if ownership:
                df = pd.DataFrame(ownership)
                display_cols = ["investor", "report_period", "shares", "market_value", "price"]
                df = df[[col for col in display_cols if col in df.columns]]
                df.columns = ["Institution", "Report Date", "Shares", "Market Value", "Price"]
                st.dataframe(df)
            else:
                st.info("No institutional ownership data available.")
        else:
            st.error(f"Failed to fetch institutional ownership: {data['institutional_ownership']['error']}")
    with tab5:
        st.subheader(f"Recent News for {ticker}")
        if data["news"]["success"]:
            news = data["news"]["data"]
            if news:
                for article in news:
                    with st.expander(article.get("title", "No Title")):
                        st.write(f"Source: {article.get('source', 'N/A')}")
                        st.write(f"Date: {article.get('publication_date', 'N/A')}")
                        st.write(f"URL: {article.get('url', 'N/A')}")
                        st.write(f"Sentiment: {article.get('sentiment', 'N/A')}")
            else:
                st.info("No recent news available.")
        else:
            st.error(f"Failed to fetch news: {data['news']['error']}")

# AI Agents
@dataclass
class FinancialDeps:
    openai_api_key: str | None

financial_agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt=(
        "You are a financial analysis assistant. "
        "Analyze financial data and provide insights based on metrics, news sentiment, and institutional activity."
    ),
    deps_type=FinancialDeps,
    retries=2,
)

@financial_agent.tool
async def analyze_financial_data(ctx: RunContext[FinancialDeps], ticker: str, data: dict) -> str:
    enriched_input = (
        f"Financial Data for {ticker}:\n"
        f"Metrics: {json.dumps(data['financial_analysis']['metrics']['data'], indent=2)}\n"
        f"News Sentiment: {json.dumps(data['financial_analysis']['sentiment']['data'], indent=2)}\n"
        f"Institutional Activity: {json.dumps(data['financial_analysis']['institutional']['data'], indent=2)}\n"
        f"Snapshot: {json.dumps(data['financial_snapshot']['data'], indent=2)}\n"
        "Provide a detailed financial analysis."
    )
    response = await financial_agent.run(enriched_input, deps=ctx.deps)
    return response.data

@dataclass
class ArchonDeps:
    openai_api_key: str | None

archon_agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt=(
        "You are Archon, a meta-agent coordinating financial analysis. "
        "Work with the financial agent to synthesize data, provide strategic insights, and generate a final report."
    ),
    deps_type=ArchonDeps,
    retries=2,
)

@archon_agent.tool
async def coordinate_analysis(ctx: RunContext[ArchonDeps], ticker: str, financial_insights: str, data: dict) -> str:
    prompt = (
        f"Financial Insights from Agent for {ticker}:\n{financial_insights}\n\n"
        f"Raw Data:\n{json.dumps(data, indent=2)}\n\n"
        "Synthesize these insights with the raw data to provide a strategic analysis and recommendations."
    )
    response = await archon_agent.run(prompt, deps=ctx.deps)
    return response.data

async def run_ai_analysis(ticker: str, data: dict) -> str:
    financial_deps = FinancialDeps(openai_api_key=os.getenv("OPENAI_API_KEY"))
    archon_deps = ArchonDeps(openai_api_key=os.getenv("OPENAI_API_KEY"))
    financial_insights = await financial_agent.analyze_financial_data(ticker, data, deps=financial_deps)
    final_report = await archon_agent.coordinate_analysis(ticker, financial_insights, data, deps=archon_deps)
    return final_report

def ai_insights_tab(ticker: str, data: dict):
    st.subheader(f"AI Insights for {ticker}")
    st.write("Insights generated by Archon and financial agents working together.")
    if st.button("Generate AI Insights"):
        with st.spinner("Generating insights..."):
            insights = asyncio.run(run_ai_analysis(ticker, data))
            st.session_state.ai_insights = insights
            st.session_state.ai_insights_generated = True
    if "ai_insights_generated" in st.session_state and st.session_state.ai_insights_generated:
        st.markdown(st.session_state.ai_insights)

# Main App Logic
def main():
    st.title("Stock Data Dashboard")
    st.write("Enter a stock ticker to view detailed financial data and AI-driven insights.")

    # Initialize MCP client on first run with error handling
    if "mcp_connected" not in st.session_state:
        st.session_state.mcp_connected = False
        try:
            asyncio.run(start_mcp_client())
        except Exception as e:
            st.error(f"Failed to initialize MCP client: {str(e)}")
            st.write("Continuing without MCP connection. Some features may be unavailable.")

    # Debug output to confirm reaching sidebar
    st.write("Rendering sidebar...")

    with st.sidebar:
        st.header("Stock Selection")
        ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL").upper()
        if st.button("Fetch Data", use_container_width=True):
            st.session_state.ticker = ticker
            with st.spinner(f"Fetching data for {ticker}..."):
                data = asyncio.run(fetch_stock_data(ticker))
                if data:
                    st.session_state.data = data
                    st.session_state.ai_insights_generated = False

    if "data" in st.session_state:
        ticker = st.session_state.ticker
        data = st.session_state.data
        tab1, tab2 = st.tabs(["Stock Data", "AI Insights"])
        with tab1:
            display_stock_data(data, ticker)
        with tab2:
            ai_insights_tab(ticker, data)

if __name__ == "__main__":
    try:
        main()
    finally:
        if st.session_state.get("mcp_connected", False):
            asyncio.run(stop_mcp_client())