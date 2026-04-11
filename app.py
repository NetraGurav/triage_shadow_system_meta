import os
import sys
import io

# Fix Windows encoding issues for emojis in console
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Tuple
import uvicorn
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# ── local imports ──────────────────────────────────────────────────────────────
from model.sft_model import TriageModel
from tasks.environment import TicketTriageEnv
from tasks.models import (
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    TriageAction,
)

load_dotenv()

# ─── LOGGING UTILITY ──────────────────────────────────────────────────────────
console = Console()

class AppLogger:
    @staticmethod
    def header(text: str):
        console.rule(f"[bold cyan]{text}[/bold cyan]")

    @staticmethod
    def triage_event(source: str, category: str, priority: str, route: str):
        table = Table(title="🎫 Triage Processed", title_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Source", f"[bold yellow]{source}[/bold yellow]")
        table.add_row("Category", category)
        table.add_row("Priority", f"[bold]{priority}[/bold]")
        table.add_row("Route", route)
        console.print(table)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
api_key  = os.getenv("GROQ_API_KEY")
model_id = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# ─── SFT BASELINE MODEL ───────────────────────────────────────────────────────
model_path = "model/saved_model.pkl"

# Load the model
try:
    sft_triage = TriageModel.load(model_path)
except (MemoryError, Exception) as e:
    console.print(f"[bold red]Critical Error:[/bold red] Failed to load/train SFT model: {e}")
    console.print("[yellow]Falling back to LLM-only mode. Expect SFT features to be disabled.[/yellow]")
    sft_triage = None

# ─── LLM AGENT ──────────────────────────────────────────────────────────────
_llm_agent = None

def _get_llm_agent():
    global _llm_agent
    if _llm_agent is not None:
        return _llm_agent
    try:
        from model.llm_agent import LLMTriageAgent
        _llm_agent = LLMTriageAgent(use_fallback=False)
        if _llm_agent._client is not None or _llm_agent._using_gemini or _llm_agent._using_groq:
            return _llm_agent
        _llm_agent = None
        return None
    except Exception as e:
        console.print(f"[bold red]LLM init failed:[/bold red] {e}")
        return None

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; padding-top: 2rem; }
.header-box { text-align: center; margin-bottom: 2rem; }
.triage-card { 
    background: rgba(255, 255, 255, 0.05); 
    border-radius: 12px; 
    padding: 1.5rem; 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.metric-row { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
.metric-value { font-weight: bold; color: #4F46E5; }
.priority-p1 { color: #EF4444 !important; font-weight: 800; }
.priority-p2 { color: #F59E0B !important; font-weight: 800; }
.priority-p3 { color: #10B981 !important; font-weight: 700; }
.priority-p4 { color: #6B7280 !important; font-weight: 700; }
"""

# ─── CORE TRIAGE FUNCTION ────────────────────────────────────────────────────

def integrated_triage(user_input: str) -> Tuple[str, str, str]:
    """Process ticket and return formatted results."""
    AppLogger.header("New Triage Request")
    
    if sft_triage:
        shadow_result = sft_triage.predict_with_confidence(user_input)
        shadow_action = shadow_result["action"]
        conf = shadow_result["confidence"]
        avg_conf = (conf["category"] + conf["priority"] + conf["route"]) / 3
        reward = 1.0 if avg_conf > 0.6 else 0.4
        regret = round(1.0 - avg_conf, 4)
        
        category = shadow_action["category"].upper()
        priority = shadow_action["priority"]
        route = shadow_action["route"]
        reasoning = "SFT baseline (no LLM available)"
    else:
        # Fallback values if SFT failed
        avg_conf = 0.0
        reward = 0.0
        regret = 1.0
        category = "UNKNOWN"
        priority = "P4"
        route = "unassigned"
        reasoning = "SFT failed to load; using LLM if available."

    agent = _get_llm_agent()
    llm_used = False

    if agent is not None:
        try:
            result = agent.predict(user_input)
            action = result.get("action", {})
            category = action.get("category", category).upper()
            priority = action.get("priority", priority)
            route = action.get("route", route)
            reasoning = result.get("reasoning", "LLM decision")
            llm_used = True
        except Exception as e:
            reasoning = f"LLM call failed ({e}); using SFT fallback."
            console.print(f"[bold red]LLM Error:[/bold red] {e}")

    source_tag = "LLM Agent" if llm_used else "SFT Baseline"
    AppLogger.triage_event(source_tag, category, priority, route)

    # UI Formatting
    p_class = f"priority-{priority.lower()}"
    
    decision_html = f"""
    <div class="triage-card">
        <h3 style="margin-top:0">🎯 Agent Decision <span style="font-size: 0.8em; color: #888;">[{source_tag}]</span></h3>
        <p><b>Category:</b> <span style="color: #6366F1;">{category}</span></p>
        <p><b>Priority:</b> <span class="{p_class}">{priority}</span></p>
        <p><b>Routing:</b> <span style="color: #6366F1;">{route}</span></p>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0;">
        <p style="font-style: italic; color: #9CA3AF;">"{reasoning}"</p>
    </div>
    """

    status_icon = "✅" if regret < 0.3 else "⚠️"
    status_color = "#10B981" if regret < 0.3 else "#F59E0B"
    
    # LLM Diagnostics
    if agent:
        b_status = agent.get_backend_status()
        active = b_status["active_backend"]
        conn_icon = "🟢" if b_status["connected"] else "🔴"
        
        missing = []
        if not b_status["groq_key"]: missing.append("GROQ_API_KEY")
        if not b_status["gemini_key"]: missing.append("GEMINI_API_KEY")
        if not b_status["hf_token"]: missing.append("HF_TOKEN")
        
        libs = b_status.get("libs", {})
        lib_status = []
        for l_name, l_avail in libs.items():
            icon = "✅" if l_avail else "❌"
            lib_status.append(f"{icon} {l_name.capitalize()}")
        
        libs_str = f"<br><span style='font-size:0.8em;'>Libs: {', '.join(lib_status)}</span>"
        missing_str = f"<br><span style='font-size:0.8em; color:#ef4444;'>Missing: {', '.join(missing)}</span>" if missing else ""
        
        backend_html = f"""
        <div style="font-size: 0.85em; margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.25); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
            <b>📡 LLM Status:</b> {conn_icon} {active.upper()} ({b_status['model']})
            {libs_str}
            {missing_str}
            {f'<br><span style="font-size:0.8em; color:#ef4444;">Error: {b_status["last_error"]}</span>' if b_status.get("last_error") else ""}
        </div>
        """
    else:
        # Check environment variables directly to see what's missing
        is_hf = os.environ.get("SPACES_ZERO_GPU") or os.environ.get("HF_SPACE_ID")
        setup_msg = "Add these in Space Settings > Variables and secrets" if is_hf else "Check your local .env file"
        
        backend_html = f"""
        <div style="font-size: 0.85em; margin-top: 10px; padding: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2);">
            <b style="color: #ef4444;">⚠️ LLM Unconfigured</b>
            <p style="margin: 4px 0; font-size: 0.9em; color: #9CA3AF;">The LLM agent is inactive because no API keys were found.</p>
            <p style="margin: 4px 0; font-size: 0.8em; color: #D1D5DB;"><b>Needed:</b> GROQ_API_KEY or GEMINI_API_KEY</p>
            <hr style="border: 0; border-top: 1px solid rgba(239, 68, 68, 0.2); margin: 8px 0;">
            <p style="margin: 0; font-size: 0.8em; font-style: italic;">{setup_msg}</p>
        </div>
        """

    stats_html = f"""
    <div class="triage-card" style="margin-top: 1rem;">
        <h3 style="margin-top:0">🤖 Shadow RL Stats</h3>
        <div class="metric-row"><span>Status:</span> <span style="color: {status_color}; font-weight: bold;">{status_icon} {"Aligned" if regret < 0.3 else "High Regret"}</span></div>
        <div class="metric-row"><span>Confidence:</span> <span class="metric-value">{avg_conf*100:.1f}%</span></div>
        <div class="metric-row"><span>Reward:</span> <span class="metric-value">{reward:.4f}</span></div>
        <div class="metric-row"><span>Regret:</span> <span class="metric-value">{regret:.4f}</span></div>
        <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; margin-top: 10px;">
            <div style="background: {status_color}; width: {avg_conf*100}%; height: 100%; border-radius: 4px;"></div>
        </div>
        {backend_html}
    </div>
    """

    return decision_html, stats_html

# ─── GRADIO UI (Blocks) ───────────────────────────────────────────────────────

with gr.Blocks() as demo:
    with gr.Group(elem_classes=["container"]):
        with gr.Group(elem_classes=["header-box"]):
            gr.Markdown("# 🎫 IT Triage: Shadow RL Agent")
            gr.Markdown("Agentic triage system with real-time regret-based RL signals and multi-backend fallback.")

        with gr.Row():
            with gr.Column(scale=2):
                ticket_input = gr.Textbox(
                    lines=8,
                    placeholder="Describe the IT issue in detail...",
                    label="Ticket Description",
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Analyze Ticket", variant="primary")
                
                gr.Examples(
                    examples=[
                        ["My laptop screen is black and won't turn on. Client presentation in 2 hours."],
                        ["Can't log in to corporate email — wrong password error for 20 minutes."],
                        ["Ransomware note appeared on file server. All shared drives show .locked files."],
                    ],
                    inputs=ticket_input,
                )

            with gr.Column(scale=3):
                decision_out = gr.HTML(label="Agent Decision")
                stats_out = gr.HTML(label="Shadow RL Stats")

        submit_btn.click(
            fn=integrated_triage,
            inputs=ticket_input,
            outputs=[decision_out, stats_out],
        )
        clear_btn.click(lambda: [None, None, None], None, [ticket_input, decision_out, stats_out])

# ─── FASTAPI ──────────────────────────────────────────────────────────────────

app_fastapi = FastAPI(
    title="IT Triage OpenEnv API",
    description="OpenEnv-compliant IT support ticket triage environment",
    version="1.1.0",
)

global_env = TicketTriageEnv()

@app_fastapi.post("/reset", response_model=ResetResponse)
def reset_env(req: ResetRequest = ResetRequest()):
    obs = global_env.reset(task=req.task, ticket_id=req.ticket_id)
    safe_obs = dict(obs)
    return ResetResponse(observation=safe_obs)

@app_fastapi.post("/step", response_model=StepResponse)
def step_env(req: StepRequest):
    action = {"category": req.category, "priority": req.priority, "route": req.route}
    obs, reward, done, info = global_env.step(action)
    safe_info = {k: v for k, v in info.items() if k not in ("shadow_scores",)}
    return StepResponse(reward=reward, done=done, info=safe_info)

@app_fastapi.get("/state")
def get_state():
    s = global_env.state()
    return s


@app_fastapi.get("/health")
def health():
    return {"status": "ok"}

# ─── MOUNT GRADIO ─────────────────────────────────────────────────────────────

app = gr.mount_gradio_app(app_fastapi, demo, path="/", theme=gr.themes.Soft(), css=CUSTOM_CSS)

if __name__ == "__main__":
    AppLogger.header("STARTING TRIAGE SERVER")
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)