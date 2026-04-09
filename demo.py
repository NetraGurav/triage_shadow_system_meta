"""
demo.py
────────
Standalone demo script for IT Ticket Triage.
Demonstrates the LLM Agent's ability to classify, prioritize, and route tickets.
"""

import os
from rich.console import Console
from rich.table import Table
from model.llm_agent import LLMTriageAgent

console = Console()

def run_demo():
    console.rule("[bold cyan]IT Ticket Triage Demo[/bold cyan]")
    
    # Initialize the agent
    # It will automatically pick up API keys from environment or hardcoded fallback
    agent = LLMTriageAgent(use_fallback=True)
    status = agent.get_backend_status()
    
    console.print(f"📡 [bold]LLM Backend:[/bold] {status['active_backend'].upper()} ({status['model']})")
    console.print(f"📡 [bold]Connected:[/bold] {'🟢 Yes' if status['connected'] else '🔴 No (Using SFT Fallback)'}")
    print("\n")

    # Sample tickets for demonstration
    test_tickets = [
        {
            "id": "E001",
            "text": "My laptop screen is completely black and won't turn on. Urgent presentation in 1 hour."
        },
        {
            "id": "M005",
            "text": "I can't log in to the corporate email. It says 'password incorrect' but I'm sure it's right. Trying to access HR portal."
        },
        {
            "id": "H003",
            "text": "The entire office WiFi is down and the server room is making a loud buzzing sound. Multiple users reporting cloud app timeouts."
        }
    ]

    for ticket in test_tickets:
        console.print(f"[bold yellow]Analyzing Ticket {ticket['id']}:[/bold yellow]")
        console.print(f"[italic]'{ticket['text']}'[/italic]")
        
        try:
            result = agent.predict(ticket["text"])
            action = result["action"]
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Field", style="dim", width=12)
            table.add_column("Decision", style="bold")
            
            table.add_row("Category", action["category"])
            table.add_row("Priority", action["priority"])
            table.add_row("Route", action["route"])
            table.add_row("Source", result["source"])
            
            console.print(table)
            console.print(f"[dim]Reasoning: {result['reasoning']}[/dim]")
            print("-" * 50)
            
        except Exception as e:
            console.print(f"[bold red]Error analyzing ticket:[/bold red] {e}")

    console.rule("[bold cyan]Demo Complete[/bold cyan]")

if __name__ == "__main__":
    run_demo()
