"""
Command-line interface for interacting with the LLM service.

This script serves as a simple way to test the `generate_summary` function
from a terminal, isolating it from the UI and other services.
"""

from services.llm.engine import generate_summary, pipe

def run_cli():
    """
    Starts an interactive command-line loop to get user input and show summaries.
    """
    print("-" * 50)
    print("LLM Service Runner (CLI)")
    print("Enter medical text below. Type 'quit' to exit.")
    print("-" * 50)

    while True:
        try:
            # Use a multi-line input method for easier pasting of long texts.
            print("Medical Text (Press Ctrl+D or Ctrl+Z on Windows then Enter to submit) >")
            lines = []
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            user_input = "\n".join(lines)

        if user_input.lower().strip() in ['quit', 'exit']:
            break
        if not user_input.strip():
            print("\n")
            continue

        summary = generate_summary(user_input)
        print("\n--- Generated Summary ---")
        print(summary)
        print("-------------------------\n")

    print("👋 Exiting.")

if __name__ == "__main__":
    if pipe is not None:
        run_cli()
    else:
        print("🔥 Cannot start CLI because the LLM pipeline failed to initialize.")
        print("   Please check the logs above for initialization errors.")