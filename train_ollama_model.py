import json
import os
import subprocess
from pathlib import Path


def main() -> int:
    base_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
    new_model = os.getenv("OLLAMA_TUNED_MODEL", "ceacfp-tuned")
    data_path = Path(os.getenv("TRAINING_DATA", "training_data.json"))
    modelfile_path = Path(os.getenv("MODELFILE_OUT", "Modelfile.ceacfp"))

    pairs = json.loads(data_path.read_text(encoding="utf-8"))
    system = (
        "Eres un asistente para estudiar SSGG/E y Programación de Servicios y Procesos con temática de boxeo.\n"
        "Respondes en español claro, estilo examen, sin inventar.\n"
        "Cuando te pidan una petición para el MiniSaaS del club de boxeo, devuelves JSON válido."
    )

    lines = [f"FROM {base_model}", f'SYSTEM """{system}"""']
    for item in pairs:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a:
            continue
        lines.append(f'MESSAGE user """{q}"""')
        lines.append(f'MESSAGE assistant """{a}"""')
    modelfile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = ["ollama", "create", new_model, "-f", str(modelfile_path)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("No se encontró el comando 'ollama'. Instala Ollama y asegúrate de que esté en el PATH.")
        print(f"Modelfile generado en: {modelfile_path}")
        return 2
    except subprocess.CalledProcessError as e:
        print("Falló 'ollama create'.")
        print(f"Comando: {' '.join(cmd)}")
        print(f"Modelfile generado en: {modelfile_path}")
        print(f"Error: {e}")
        return 3
    print(f"Modelo creado: {new_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
