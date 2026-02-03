import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

try:
    from Agents import Research_Paper_Summarizer as rps
except Exception as exc:
    raise SystemExit(
        "Failed to import project dependencies. "
        "Run this inside your project environment (e.g. activate conda env) "
        "and ensure requirements are installed."
    ) from exc


def main() -> None:
    os.makedirs("assets", exist_ok=True)
    output_path = os.path.join("assets", "langgraph.png")
    compiled = rps.app
    graph = compiled.get_graph()
    png_bytes = graph.draw_mermaid_png()
    with open(output_path, "wb") as handle:
        handle.write(png_bytes)
    print(f"Saved graph to {output_path}")


if __name__ == "__main__":
    main()

