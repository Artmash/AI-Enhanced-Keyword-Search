import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    @staticmethod
    def generate_report(results: list[dict], session_stats: dict, session_id: str, last_query: str = "", indexed_doc_count: int = 0, output_dir: str = "./output") -> dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"forensic_report_{session_id}_{timestamp}"

        report_data = {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "last_query": last_query,
            "indexed_documents": indexed_doc_count,
            "session_stats": session_stats,
            "results": results,
        }

        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report_data, fh, indent=2, ensure_ascii=False)

        html_path = os.path.join(output_dir, f"{base_name}.html")
        html = ReportGenerator._build_html(report_data)
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        return {"json_path": json_path, "html_path": html_path}

    @staticmethod
    def _score_class(score: float) -> str:
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _build_html(data: dict) -> str:
        stats = data["session_stats"]
        results = data["results"]
        positive_rate = stats.get("positive_rate", 0) * 100

        rows_html = ""
        for r in results:
            cls = ReportGenerator._score_class(r["similarity_score"])
            personalized = f"{r['personalized_score']:.3f}" if r.get("personalized_score") else "—"
            preview = r.get("preview", "").replace("<", "&lt;").replace(">", "&gt;")
            rows_html += f"""
            <tr>
              <td class="rank">#{r['rank']}</td>
              <td class="docname">{r['document_name']}</td>
              <td class="score {cls}">{r['similarity_score']:.3f}</td>
              <td>{personalized}</td>
              <td class="source">{r.get('source','unknown')}</td>
              <td class="preview">{preview}</td>
            </tr>"""

        if not rows_html:
            rows_html = '<tr><td colspan="6" style="text-align:center;color:#888;">No results recorded in this session.</td></tr>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Forensic Investigation Report – {data['session_id']}</title>
<style>
  :root {{ --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --green: #3fb950; --orange: #d29922; --red: #f85149; --accent: #58a6ff; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 32px; line-height: 1.6; }}
  header {{ border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 28px; }}
  header h1 {{ font-size: 1.8rem; color: var(--accent); }}
  header p {{ color: var(--muted); font-size: 0.9rem; margin-top: 4px; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .meta-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .meta-card .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .meta-card .value {{ font-size: 1.2rem; font-weight: 600; margin-top: 4px; }}
  h2 {{ font-size: 1.1rem; color: var(--accent); margin-bottom: 14px; margin-top: 32px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  th {{ background: var(--surface); padding: 10px 12px; text-align: left; border-bottom: 2px solid var(--border); color: var(--muted); text-transform: uppercase; font-size: 0.75rem; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:hover td {{ background: rgba(88,166,255,0.04); }}
  .rank {{ font-weight: 700; color: var(--muted); }}
  .docname {{ font-weight: 600; }}
  .score.high {{ color: var(--green); font-weight: 700; }}
  .score.medium {{ color: var(--orange); font-weight: 700; }}
  .score.low {{ color: var(--red); font-weight: 700; }}
  .source {{ font-size: 0.8rem; color: var(--muted); }}
  .preview {{ font-size: 0.8rem; color: var(--muted); max-width: 420px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--muted); text-align: center; }}
  .bar-wrap {{ background: var(--border); border-radius: 99px; height: 8px; margin-top: 8px; }}
  .bar {{ height: 8px; border-radius: 99px; background: var(--green); width: {positive_rate:.0f}%; transition: width 0.4s; }}
</style>
</head>
<body>
<header>
  <h1>🔍 Forensic Investigation Report</h1>
  <p>Session: <strong>{data['session_id']}</strong> &nbsp;·&nbsp; Generated: {data['generated_at']}</p>
</header>
<div class="meta-grid">
  <div class="meta-card"><div class="label">Query</div><div class="value" style="font-size:1rem">{data['last_query'] or '—'}</div></div>
  <div class="meta-card"><div class="label">Indexed Documents</div><div class="value">{data['indexed_documents']}</div></div>
  <div class="meta-card"><div class="label">Results Returned</div><div class="value">{len(results)}</div></div>
  <div class="meta-card"><div class="label">Total Feedback</div><div class="value">{stats.get('total_feedback', 0)}</div></div>
  <div class="meta-card"><div class="label">Relevance Rate</div><div class="value">{positive_rate:.0f}%</div><div class="bar-wrap"><div class="bar"></div></div></div>
</div>
<h2>📄 Search Results</h2>
<table>
  <thead><tr><th>Rank</th><th>Document</th><th>Similarity</th><th>Personalised</th><th>Source</th><th>Preview</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<footer>Forensic Investigation Platform &nbsp;·&nbsp; AI-Powered Semantic Search with Active Learning</footer>
</body>
</html>"""