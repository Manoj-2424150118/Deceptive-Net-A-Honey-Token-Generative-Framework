# Contributing to Deceptive-Net

Thanks for your interest in improving Deceptive-Net — a generative-AI framework for
real-time financial fraud detection and cyber deception (honey tokens). Contributions
of all sizes are welcome: bug fixes, detection/deception improvements, tests, and
documentation.

> **Research scope:** Deceptive-Net is an academic / defensive-security research
> project. All transaction data is synthetically generated — contributions must never
> add real financial data, real PII, or offensive tooling. See the *Academic & Research
> Notice* in the [README](README.md).

## Ways to Contribute

- **Report a bug** — open an issue with steps to reproduce, expected vs. actual
  behaviour, and your environment (OS, Docker / Python version, relevant logs).
- **Suggest a feature** — open an issue first so the design can be discussed before
  you invest time in a large pull request.
- **Improve the ML ensemble** — e.g. LightGBM / PyTorch autoencoder tuning, SHAP
  explanations, or synthetic-data quality. Include before/after metrics in your PR.
- **Improve deception** — honey-token (`HTXN`) generation, watermarking, audit/alert
  handling, or the simulated network-topology visualisation.
- **Add tests or docs** — `test_suite.py`, `docs/ARCHITECTURE.md`, and
  `docs/SECURITY_AUDIT.md` are good places to start.

## Development Setup

### Prerequisites

- Python **3.10** (the version used by both Dockerfiles)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Docker
  Compose, for the full stack
- Git

### Option A — Full stack with Docker (recommended)

```bash
git clone https://github.com/Manoj-2424150118/Deceptive-Net-A-Honey-Token-Generative-Framework.git
cd Deceptive-Net-A-Honey-Token-Generative-Framework
docker compose up --build
```

- Dashboard: <http://localhost:8080>
- Backend API docs (Swagger): <http://localhost:8000/docs>
- Demo credentials: `admin` / `admin123` (administrator), `analyst` / `analyst123` (analyst) — seeded only because the bundled compose file sets `DECEPTIVE_NET_ENABLE_DEMO_USERS=true` for the local demo; the backend default is off.

On Windows you can instead run the one-click launcher, `start_deceptive_net.bat`,
which checks Docker, builds the containers, and opens the dashboard.

### Option B — Backend only, for Python work

```bash
cd Deceptive-Net-A-Honey-Token-Generative-Framework
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs>. The frontend in `frontend/` is static
HTML/CSS/JS and can be served by any static file server pointed at the API.

## Project Layout

| Path | What lives there |
| --- | --- |
| `backend/` | FastAPI service — JWT auth & RBAC, rate limiting (SlowAPI), audit log, deception / honey-token logic |
| `frontend/` | Framework-free HTML/CSS/JS dashboard (WebSocket live transaction stream) |
| `ml/` | Model training (`train_model.py`) and inference (`predict.py`) — LightGBM + PyTorch autoencoder |
| `docs/` | `ARCHITECTURE.md` and `SECURITY_AUDIT.md` |
| `test_suite.py` | Test suite for data validation and the detection pipeline |
| `docker-compose.yml` | Backend (`:8000`) + frontend (`:8080`) services |

## Pull Request Process

1. **Fork** the repository and create a topic branch from `main`:
   `git checkout -b fix/short-description` (use `feat/`, `docs/`, `test/` as appropriate).
2. **Keep the change focused** — one bug fix or feature per PR. Update the README or
   `docs/` if behaviour, endpoints, or setup steps change.
3. **Run the tests** before opening the PR (see below) and note the result in the PR
   description.
4. **Write a clear PR description** — what changed, why, how you tested it, and any
   screenshots for dashboard/UI changes.
5. **Never commit secrets or generated artefacts** — no real API keys, JWT secrets
   (set `JWT_SECRET_KEY` via the environment, as in `docker-compose.yml`), datasets
   with real PII, trained-model dumps, or logs. Demo credentials in the README are
   for local evaluation only.
6. A maintainer will review for correctness, security impact, and scope. Please be
   responsive to review comments; PRs may be asked to split or add tests.

## Running Tests

```bash
pip install -r backend/requirements.txt
python test_suite.py
```

If you touch the ML pipeline, also re-run training (`python ml/train_model.py`) and
report the resulting detection metrics in your PR.

## Code Style

- **Python:** PEP 8, type hints on public functions, and a module docstring that
  summarises endpoints/responsibilities (follow the style of `backend/main.py`).
  Prefer small, single-purpose functions and explicit error handling over silent
  `except: pass`.
- **Security-sensitive code** (auth, token generation, audit logging): default to
  deny, log security-relevant events, and never weaken JWT/RBAC checks or rate
  limits for convenience.
- **Frontend:** vanilla HTML/CSS/JS only — please don't introduce a framework or a
  build step without opening an issue first.
- **Commits:** imperative, scoped messages, e.g. `Fix honey-token export audit log`
  or `Add SHAP endpoint test`.

## Questions

Open an issue with the `question` label, or start from `docs/ARCHITECTURE.md` for a
tour of how the simulation, ensemble engine, and deception layers fit together.

By contributing, you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
