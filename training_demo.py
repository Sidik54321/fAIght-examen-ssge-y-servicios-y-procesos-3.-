import os

from ollama_api import chat


def ask(model: str, prompt: str) -> str:
    return chat(
        [{"role": "system", "content": "Responde en español."}, {"role": "user", "content": prompt}],
        model=model,
        temperature=0.2,
    ).strip()


def main() -> int:
    base_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
    tuned_model = os.getenv("OLLAMA_TUNED_MODEL", "ceacfp-tuned")
    prompts = [
        "Define MCP en una frase, como si fuera para un examen.",
        "Explica RAG de forma muy simple y con un ejemplo.",
        "Convierte esta intención en un JSON de petición: \"Dame las cuotas pendientes\".",
    ]

    for p in prompts:
        print("=" * 80)
        print("PROMPT:", p)
        print("\n--- ANTES (base) ---")
        print(ask(base_model, p))
        print("\n--- DESPUÉS (tuned) ---")
        print(ask(tuned_model, p))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
