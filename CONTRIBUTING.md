# Contributing to COVID-19 Hospitalization Prediction

Thank you for your interest in contributing! This project follows MIT-grade software engineering standards.

## 🔧 Setup

```bash
git clone https://github.com/Aranya2801/COVID19-Hospitalization-Prediction.git
cd COVID19-Hospitalization-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term
```

All PRs must maintain **≥ 90% test coverage** and pass `flake8` linting.

## 📋 Pull Request Checklist

- [ ] New tests added for new features
- [ ] All existing tests pass
- [ ] Docstrings updated
- [ ] `configs/config.yaml` updated if hyperparameters change
- [ ] PR description explains the clinical motivation

## 🏗️ Branch Strategy

- `main` — stable, production-ready
- `develop` — integration branch
- `feature/xxx` — new features
- `fix/xxx` — bug fixes

## 📧 Contact

Open a GitHub Issue or email the maintainer.
