import streamlit as st
import httpx
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Optional
import anthropic

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.paper-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
    background: #fafafa;
}
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
}
.score-high { background: #d4edda; color: #155724; }
.score-mid  { background: #fff3cd; color: #856404; }
.score-low  { background: #f8d7da; color: #721c24; }
.publisher-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    background: #e8f4fd;
    color: #0c447c;
    font-size: 12px;
    font-weight: 500;
    margin-left: 8px;
}
.finding-item {
    padding: 6px 12px;
    margin: 4px 0;
    border-left: 3px solid #534AB7;
    background: #f5f4fe;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
}
.chat-message {
    padding: 14px 18px;
    border-radius: 10px;
    margin: 8px 0;
    font-size: 15px;
}
.chat-user   { background: #EEF2FF; border-left: 4px solid #534AB7; }
.chat-agent  { background: #F0FDF4; border-left: 4px solid #1D9E75; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE         = "https://api.openalex.org"
CROSSREF_BASE         = "https://api.crossref.org"
UNPAYWALL_BASE        = "https://api.unpaywall.org/v2"
UNPAYWALL_EMAIL       = "research.agent@example.com"  # Required by Unpaywall ToS

PUBLISHER_TIERS = {
    "springer": 10, "elsevier": 10, "wiley": 9, "asce": 10,
    "ice": 9, "taylor & francis": 8, "sage": 7, "emerald": 7,
    "mdpi": 5, "hindawi": 5, "nature": 10, "science": 10,
    "oxford": 9, "cambridge": 9,
}

STRUCTURAL_KEYWORDS = [
    "structural engineering", "civil engineering", "concrete", "steel structure",
    "seismic", "earthquake", "foundation", "geotechnical", "bridge", "beam",
    "column", "reinforcement", "finite element", "structural analysis",
    "load bearing", "structural dynamics", "fatigue", "fracture mechanics",
]

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "web"  # "web" or "chat"


# ══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

async def search_semantic_scholar(query: str, limit: int = 15) -> list:
    """Search Semantic Scholar — free, no key needed."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,authors,externalIds,venue,publicationVenue,openAccessPdf,url",
    }
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                return data.get("data", [])
    except Exception:
        pass
    return []


async def search_openalex(query: str, limit: int = 15) -> list:
    """Search OpenAlex — fully open, includes Impact Factor data."""
    url = f"{OPENALEX_BASE}/works"
    params = {
        "search": query,
        "per-page": limit,
        "filter": "language:en",
        "select": "id,title,abstract_inverted_index,publication_year,cited_by_count,authorships,primary_location,open_access,doi,concepts",
        "mailto": UNPAYWALL_EMAIL,
    }
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                return r.json().get("results", [])
    except Exception:
        pass
    return []


async def check_unpaywall(doi: str) -> Optional[str]:
    """Check if a free legal PDF exists via Unpaywall."""
    if not doi:
        return None
    clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    url = f"{UNPAYWALL_BASE}/{clean_doi}"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, params={"email": UNPAYWALL_EMAIL})
            if r.status_code == 200:
                data = r.json()
                if data.get("is_oa") and data.get("best_oa_location"):
                    return data["best_oa_location"].get("url_for_pdf") or data["best_oa_location"].get("url")
    except Exception:
        pass
    return None


async def get_crossref_metadata(doi: str) -> dict:
    """Fetch extra metadata (Impact Factor proxy via citations) from CrossRef."""
    if not doi:
        return {}
    clean_doi = doi.replace("https://doi.org/", "")
    url = f"{CROSSREF_BASE}/works/{clean_doi}"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.json().get("message", {})
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_paper(paper: dict) -> int:
    """Score a paper 0-100 based on citations, publisher, recency, field relevance."""
    score = 0

    # Citations (35 pts) — log-scale
    cites = paper.get("citations", 0)
    if cites >= 500:   score += 35
    elif cites >= 200: score += 28
    elif cites >= 100: score += 22
    elif cites >= 50:  score += 16
    elif cites >= 20:  score += 10
    elif cites >= 5:   score += 5

    # Publisher tier (25 pts)
    publisher = (paper.get("publisher") or "").lower()
    venue = (paper.get("venue") or "").lower()
    combined = publisher + " " + venue
    pub_score = 0
    for pub, pts in PUBLISHER_TIERS.items():
        if pub in combined:
            pub_score = max(pub_score, pts)
    score += int(pub_score * 2.5)

    # Recency (20 pts)
    year = paper.get("year", 0)
    current_year = datetime.now().year
    age = current_year - year if year else 20
    if age <= 2:    score += 20
    elif age <= 5:  score += 15
    elif age <= 10: score += 10
    elif age <= 15: score += 5

    # Has free PDF (10 pts)
    if paper.get("pdf_url"):
        score += 10

    # Has abstract (5 pts)
    if paper.get("abstract"):
        score += 5

    # Field relevance — structural engineering bonus (5 pts)
    title_abstract = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    for kw in STRUCTURAL_KEYWORDS:
        if kw in title_abstract:
            score += 5
            break

    return min(score, 100)


# ══════════════════════════════════════════════════════════════════════════════
# DATA NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_semantic_scholar(raw: dict) -> dict:
    doi = (raw.get("externalIds") or {}).get("DOI", "")
    venue = raw.get("venue") or ""
    pub_venue = raw.get("publicationVenue") or {}
    oa = raw.get("openAccessPdf") or {}
    authors = raw.get("authors") or []
    return {
        "title":     raw.get("title", "Untitled"),
        "abstract":  raw.get("abstract", ""),
        "year":      raw.get("year", 0),
        "citations": raw.get("citationCount", 0),
        "authors":   [a.get("name", "") for a in authors[:4]],
        "doi":       doi,
        "venue":     pub_venue.get("name") or venue,
        "publisher": pub_venue.get("name") or "",
        "pdf_url":   oa.get("url", ""),
        "source":    "Semantic Scholar",
    }


def reconstruct_abstract(inverted: Optional[dict]) -> str:
    """OpenAlex stores abstracts as inverted index — reconstruct it."""
    if not inverted:
        return ""
    try:
        max_pos = max(pos for positions in inverted.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inverted.items():
            for pos in positions:
                words[pos] = word
        return " ".join(w for w in words if w)
    except Exception:
        return ""


def normalize_openalex(raw: dict) -> dict:
    loc = raw.get("primary_location") or {}
    source = loc.get("source") or {}
    oa = raw.get("open_access") or {}
    authorships = raw.get("authorships") or []
    doi = raw.get("doi", "")
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))
    return {
        "title":     raw.get("title", "Untitled"),
        "abstract":  abstract,
        "year":      raw.get("publication_year", 0),
        "citations": raw.get("cited_by_count", 0),
        "authors":   [
            (a.get("author") or {}).get("display_name", "")
            for a in authorships[:4]
        ],
        "doi":       doi.replace("https://doi.org/", "") if doi else "",
        "venue":     source.get("display_name", ""),
        "publisher": source.get("host_organization_name", ""),
        "pdf_url":   oa.get("oa_url", "") or "",
        "source":    "OpenAlex",
    }


def deduplicate(papers: list) -> list:
    seen_titles = set()
    seen_dois   = set()
    unique = []
    for p in papers:
        title_key = p["title"].lower()[:60]
        doi       = p.get("doi", "")
        if doi and doi in seen_dois:
            continue
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        if doi:
            seen_dois.add(doi)
        unique.append(p)
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# AI FUNCTIONS (Claude)
# ══════════════════════════════════════════════════════════════════════════════

def get_claude_client() -> Optional[anthropic.Anthropic]:
    api_key = st.session_state.get("anthropic_api_key", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def expand_query(query: str) -> str:
    """Use Claude to expand the query with academic synonyms."""
    client = get_claude_client()
    if not client:
        return query
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": (
                    f"Expand this research query with 3-4 relevant academic keywords/synonyms. "
                    f"Output only the expanded query string, nothing else.\n\nQuery: {query}"
                )
            }]
        )
        return msg.content[0].text.strip()
    except Exception:
        return query


def ai_summarize_paper(title: str, abstract: str) -> dict:
    """Generate summary + key findings using Claude."""
    client = get_claude_client()
    if not client or not abstract:
        return {"summary": abstract[:300] + "..." if len(abstract) > 300 else abstract, "findings": []}
    try:
        prompt = f"""Paper title: {title}
Abstract: {abstract}

Respond in JSON only (no markdown). Format:
{{
  "summary": "2-3 sentence plain-English summary of problem, method, and conclusion",
  "findings": ["finding 1", "finding 2", "finding 3"]
}}"""
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {
            "summary": abstract[:300] + "..." if len(abstract) > 300 else abstract,
            "findings": []
        }


def ai_chat_response(user_msg: str, papers: list) -> str:
    """Chat with context of found papers."""
    client = get_claude_client()
    if not client:
        return "Please add your Claude API key in the sidebar to enable chat."

    paper_context = ""
    for i, p in enumerate(papers[:5], 1):
        paper_context += f"\n[{i}] {p['title']} ({p['year']}) — {p.get('venue','')}\nSummary: {p.get('summary','')}\n"

    history = st.session_state.chat_history[-6:]  # last 3 turns
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": user_msg})

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=(
                "You are a research assistant specializing in structural engineering and academic papers. "
                f"The user has just searched for papers. Here are the top results:\n{paper_context}\n"
                "Answer questions about these papers or related topics concisely."
            ),
            messages=messages,
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI error: {str(e)}"


def suggest_related(papers: list) -> list:
    """Return related paper titles suggested by AI."""
    client = get_claude_client()
    if not client or not papers:
        return []
    titles = [p["title"] for p in papers[:5]]
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Based on these papers:\n" + "\n".join(f"- {t}" for t in titles) +
                    "\n\nSuggest 3 related research topics or paper titles to search next. "
                    "Output a JSON array of 3 strings only."
                )
            }]
        )
        raw = msg.content[0].text.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SEARCH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

async def run_search_pipeline(query: str, max_results: int, use_ai: bool) -> list:
    """Full async pipeline: search → deduplicate → score → PDF check → AI."""

    expanded = expand_query(query) if use_ai else query

    # Parallel search
    ss_task  = search_semantic_scholar(expanded, limit=max_results)
    oa_task  = search_openalex(expanded, limit=max_results)
    ss_raw, oa_raw = await asyncio.gather(ss_task, oa_task)

    papers = (
        [normalize_semantic_scholar(p) for p in ss_raw] +
        [normalize_openalex(p) for p in oa_raw]
    )
    papers = deduplicate(papers)

    # PDF check for top papers (limit to 10 to stay fast)
    async def enrich_pdf(p):
        if not p.get("pdf_url") and p.get("doi"):
            p["pdf_url"] = await check_unpaywall(p["doi"]) or ""
        return p

    enriched = await asyncio.gather(*[enrich_pdf(p) for p in papers[:20]])
    papers[:20] = list(enriched)

    # Score + sort
    for p in papers:
        p["score"] = score_paper(p)
    papers.sort(key=lambda x: x["score"], reverse=True)
    papers = papers[:max_results]

    # AI summaries (sequential to respect rate limits)
    if use_ai:
        for p in papers:
            ai = ai_summarize_paper(p["title"], p.get("abstract", ""))
            p["summary"]  = ai.get("summary", "")
            p["findings"] = ai.get("findings", [])

    return papers


def search_sync(query, max_results, use_ai):
    return asyncio.run(run_search_pipeline(query, max_results, use_ai))


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def score_color(s):
    if s >= 70: return "score-high"
    if s >= 40: return "score-mid"
    return "score-low"


def render_paper_card(paper: dict, idx: int):
    score = paper.get("score", 0)
    sc    = score_color(score)
    doi_url = f"https://doi.org/{paper['doi']}" if paper.get("doi") else ""
    pdf_url = paper.get("pdf_url", "")

    authors_str = ", ".join(paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors_str += " et al."

    publisher = paper.get("publisher") or paper.get("venue") or ""
    pub_badge = f'<span class="publisher-badge">{publisher[:40]}</span>' if publisher else ""

    summary  = paper.get("summary", paper.get("abstract", ""))[:350]
    findings = paper.get("findings", [])

    pdf_btn = ""
    if pdf_url:
        pdf_btn = f'<a href="{pdf_url}" target="_blank" style="margin-left:10px;padding:4px 12px;background:#1D9E75;color:white;border-radius:6px;text-decoration:none;font-size:13px;">⬇ Free PDF</a>'
    doi_btn = ""
    if doi_url:
        doi_btn = f'<a href="{doi_url}" target="_blank" style="margin-left:8px;padding:4px 12px;background:#534AB7;color:white;border-radius:6px;text-decoration:none;font-size:13px;">DOI</a>'

    findings_html = ""
    for f in findings:
        findings_html += f'<div class="finding-item">• {f}</div>'

    st.markdown(f"""
<div class="paper-card">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
    <div style="flex:1">
      <span style="font-size:13px;color:#888">#{idx} · {paper.get('source','')}</span>
      <h4 style="margin:4px 0 6px;font-size:16px;line-height:1.4">{paper['title']}</h4>
      <div style="font-size:13px;color:#555;margin-bottom:6px">
        {authors_str} &nbsp;·&nbsp; <b>{paper.get('year','')}</b>
        {pub_badge}
      </div>
    </div>
    <div style="text-align:right;white-space:nowrap">
      <span class="score-badge {sc}">{score}/100</span>
      <div style="font-size:12px;color:#888;margin-top:4px">📝 {paper.get('citations',0)} citations</div>
    </div>
  </div>
  <p style="font-size:14px;color:#444;margin:8px 0">{summary}</p>
  {findings_html}
  <div style="margin-top:10px">{pdf_btn}{doi_btn}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "Claude API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get free key at console.anthropic.com — needed for AI summaries & chat",
        value=st.session_state.get("anthropic_api_key", ""),
    )
    if api_key:
        st.session_state["anthropic_api_key"] = api_key
        st.success("API key saved ✓")

    st.divider()
    st.subheader("Search options")
    max_results = st.slider("Max results", 5, 20, 10)
    use_ai = st.toggle("AI summaries & query expansion", value=bool(api_key))

    st.divider()
    st.subheader("Field filter")
    field_focus = st.multiselect(
        "Boost results from",
        ["Structural Engineering", "Civil Engineering", "Geotechnical", "Earthquake/Seismic", "All fields"],
        default=["Structural Engineering"],
    )

    st.divider()
    st.caption("Free APIs used:")
    st.caption("• Semantic Scholar (search)")
    st.caption("• OpenAlex (metadata + IF)")
    st.caption("• CrossRef (DOI)")
    st.caption("• Unpaywall (free PDFs)")
    st.caption("• Claude API (AI features)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — TABS
# ══════════════════════════════════════════════════════════════════════════════

st.title("📚 Research Paper Agent")
st.caption("Find, rank, and summarize academic papers — free, high-quality sources only")

tab_search, tab_chat = st.tabs(["🔍 Search papers", "💬 Chat about results"])

# ── TAB 1: SEARCH ─────────────────────────────────────────────────────────────
with tab_search:
    with st.form("search_form", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Research topic",
                placeholder="e.g. seismic retrofitting of reinforced concrete frames",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Search", use_container_width=True, type="primary")

    if submitted and query.strip():
        with st.spinner("Searching Semantic Scholar + OpenAlex · Checking free PDFs · Ranking by quality…"):
            results = search_sync(query.strip(), max_results, use_ai)
            st.session_state.last_results = results
            st.session_state.search_query = query.strip()

    if st.session_state.last_results:
        results = st.session_state.last_results

        # Stats row
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Papers found", len(results))
        col_b.metric("With free PDF", sum(1 for p in results if p.get("pdf_url")))
        col_c.metric("Avg score", f"{sum(p['score'] for p in results) // len(results)}/100")
        col_d.metric("Top citations", max(p.get('citations', 0) for p in results))

        st.divider()

        # Sort control
        sort_by = st.selectbox("Sort by", ["Quality score", "Citations", "Year (newest)"], label_visibility="visible")
        if sort_by == "Citations":
            results = sorted(results, key=lambda x: x.get("citations", 0), reverse=True)
        elif sort_by == "Year (newest)":
            results = sorted(results, key=lambda x: x.get("year", 0), reverse=True)

        for i, paper in enumerate(results, 1):
            render_paper_card(paper, i)

        # Related suggestions
        if use_ai and api_key:
            st.divider()
            st.subheader("🔗 Suggested next searches")
            suggestions = suggest_related(results)
            if suggestions:
                cols = st.columns(len(suggestions))
                for col, sug in zip(cols, suggestions):
                    with col:
                        if st.button(f"🔎 {sug}", use_container_width=True):
                            st.session_state["prefill_query"] = sug
                            st.rerun()


# ── TAB 2: CHAT ───────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.last_results:
        st.info("Search for papers first — then chat with the agent about the results.")
    else:
        st.caption(f"Chatting about: **{st.session_state.get('search_query', 'your last search')}** ({len(st.session_state.last_results)} papers)")

        # Render history
        for msg in st.session_state.chat_history:
            css_class = "chat-user" if msg["role"] == "user" else "chat-agent"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="chat-message {css_class}">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        # Quick prompts
        st.caption("Quick questions:")
        q_cols = st.columns(3)
        quick_prompts = [
            "What are the main research gaps in these papers?",
            "Which paper is best for structural engineers in practice?",
            "Summarize the key methods used across these papers",
        ]
        for col, qp in zip(q_cols, quick_prompts):
            if col.button(qp, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": qp})
                reply = ai_chat_response(qp, st.session_state.last_results)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

        # Chat input
        user_input = st.chat_input("Ask anything about these papers…")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                reply = ai_chat_response(user_input, st.session_state.last_results)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat history", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
