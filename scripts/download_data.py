"""
download_data.py — Baixa o dataset real "Give Me Some Credit" do Kaggle
para data/cs-training.csv. Requer credenciais da API do Kaggle (~/.kaggle/kaggle.json).
Se a CLI ou as credenciais faltarem, imprime instrucoes manuais.
"""

import glob
import os
import subprocess
import zipfile

DEST_DIR = "data"
DEST = os.path.join(DEST_DIR, "cs-training.csv")


def _manual():
    print("Passo manual:")
    print("  1. Aceite as regras em https://www.kaggle.com/c/GiveMeSomeCredit/data")
    print("  2. Baixe cs-training.csv e coloque em data/cs-training.csv")
    print("  3. Rode: DATA_PATH=data/cs-training.csv python main.py")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(DEST):
        print(f"{DEST} ja existe."); return

    try:
        import kaggle  # noqa: F401
    except Exception:
        print("Pacote kaggle nao instalado. Rode: pip install kaggle")
        print("E configure as credenciais: https://www.kaggle.com/docs/api")
        _manual(); return

    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "GiveMeSomeCredit", "-p", DEST_DIR],
            check=True,
        )
        for z in glob.glob(os.path.join(DEST_DIR, "*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(DEST_DIR)
        if os.path.exists(DEST):
            print(f"Pronto: {DEST}. Rode: DATA_PATH=data/cs-training.csv python main.py")
        else:
            print("Download concluido, mas cs-training.csv nao foi encontrado no zip.")
            _manual()
    except Exception as e:
        print(f"Download falhou: {e}")
        _manual()


if __name__ == "__main__":
    main()
