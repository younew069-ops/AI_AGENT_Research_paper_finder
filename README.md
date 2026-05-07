# 📚 Research Paper Agent

AI-powered academic paper search for structural engineering and all fields.
**100% free to run.**

---

## What it does

- Searches **Semantic Scholar** + **OpenAlex** simultaneously (200M+ papers)
- Ranks each paper by **citations · Journal Impact Factor · recency · publisher tier**
- Prioritises top publishers: **Springer, Elsevier, ASCE, Wiley, ICE**
- Checks **Unpaywall** for a free legal PDF for every paper
- Uses **Claude AI** to write a plain-English summary + 3 key findings per paper
- Suggests **related searches** based on your results
- Full **chat interface** — ask questions about the papers you found

---

## Setup (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run locally
```bash
streamlit run app.py
```

### 3. Add your Claude API key
- Open the sidebar in the app
- Paste your key from https://console.anthropic.com
- Free tier is enough for typical academic use

---

## Deploy free on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo → select `app.py`
4. Add `ANTHROPIC_API_KEY` as a secret in the dashboard
5. Deploy — live in ~2 minutes

---

## Free API summary

| API | Purpose | Cost | Limit |
|-----|---------|------|-------|
| Semantic Scholar | Paper search | Free | 100 req/5 min |
| OpenAlex | Metadata + Impact Factor | Free | Unlimited (polite) |
| Unpaywall | Free PDF links | Free | Unlimited |
| CrossRef | DOI metadata | Free | Unlimited |
| Claude API | Summaries + chat | Free tier | ~$5 free credit |

---

## Folder structure

```
research_agent/
├── app.py            ← main Streamlit app (everything in one file)
├── requirements.txt  ← 3 dependencies only
└── README.md
```
