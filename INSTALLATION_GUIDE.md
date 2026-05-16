# Installation Guide — Modules 5 & 7

Install the tools below **before** the session. Each section ends with a verify command — run it to confirm the install worked.

---

## 1 · Git

| | |
|---|---|
| **Mac** | Download from [git-scm.com/download/mac](https://git-scm.com/download/mac) |
| **Windows** | Download from [git-scm.com/download/win](https://git-scm.com/download/win) — keep all defaults |

```bash
# Verify
git --version
```

---

## 2 · Python 3.12

| | |
|---|---|
| **Mac** | Download from [python.org/downloads](https://www.python.org/downloads/) — select **3.12.x** |
| **Windows** | Download from [python.org/downloads](https://www.python.org/downloads/) — select **3.12.x** · **tick "Add python.exe to PATH"** before installing |

```bash
# Verify
python3 --version   # Mac
python --version    # Windows
```

---

## 3 · Virtual Environment & Dependencies

Run these commands once from the **repo root**:

**Mac**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Project/Deployment/requirements_server.txt
```

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r Project\Deployment\requirements_server.txt
```

```bash
# Verify — you should see fastapi, uvicorn, scikit-learn, pandas in the list
pip list
```

---

## 4 · Docker Desktop (Module 7)

| | |
|---|---|
| **Mac** | Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| **Windows** | Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) — WSL2 backend is enabled automatically during install |

After installing, **open Docker Desktop** and wait until the status bar shows "Engine running".

```bash
# Verify
docker --version
```

---

## 5 · Code Editor

**VS Code** is recommended: [code.visualstudio.com](https://code.visualstudio.com/)

Any IDE works — PyCharm, Cursor, or whatever you already use.
