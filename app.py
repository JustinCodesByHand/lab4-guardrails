import uuid

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import storage
from config import DB_PATH, RATE_LIMITS
from signals.llm import llm_score
from signals.stylometry import stylometry_score
from scoring import combine, classify, label_for

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

storage.init_db(DB_PATH)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Provenance Guard</title>
<style>
  :root { font-family: system-ui, sans-serif; }
  body { max-width: 900px; margin: 0 auto; padding: 0; color: #1f2937; background: #f3f4f6; }
  .container { margin: 40px 16px; }
  h1 { margin: 0 0 4px; }
  p.sub { color: #6b7280; margin: 0 0 24px; }
  .tabs { display: flex; gap: 0; border-bottom: 2px solid #e5e7eb; margin-bottom: 24px; }
  .tab-btn { padding: 12px 16px; background: none; border: none; border-bottom: 3px solid transparent; cursor: pointer;
    font-weight: 500; color: #6b7280; margin-bottom: -2px; }
  .tab-btn.active { color: #4f46e5; border-color: #4f46e5; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  label { display: block; font-weight: 600; margin: 16px 0 6px; }
  textarea, input { width: 100%; box-sizing: border-box; padding: 10px;
    border: 1px solid #d1d5db; border-radius: 8px; font: inherit; }
  textarea { min-height: 140px; resize: vertical; }
  button { padding: 10px 22px; border: 0; border-radius: 8px;
    background: #4f46e5; color: #fff; font-weight: 600; font-size: 1em; cursor: pointer; margin-top: 16px; }
  button:disabled { opacity: .6; cursor: progress; }
  .result { margin-top: 24px; }
  .card { border-left: 5px solid #999; background: #fff; border-radius: 0 8px 8px 0;
    padding: 14px 18px; margin-bottom: 12px; }
  .badge { display: inline-block; color: #fff; font-weight: 700; font-size: .8em;
    letter-spacing: .05em; padding: 3px 12px; border-radius: 12px; }
  .label-text { margin: 10px 0 0; }
  .scores { color: #6b7280; font-size: .85em; margin-top: 8px; }
  .err { color: #b91c1c; }
  .success { color: #16a34a; }
  .ai     { border-color:#dc2626 } .ai .badge     { background:#dc2626 }
  .unc    { border-color:#d97706 } .unc .badge    { background:#d97706 }
  .human  { border-color:#16a34a } .human .badge  { background:#16a34a }
  .log-table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }
  .log-table th { background: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; font-size: .9em; }
  .log-table td { padding: 12px; border-top: 1px solid #e5e7eb; font-size: .9em; }
  .log-table tr:hover { background: #f9fafb; }
  .id-mono { font-family: monospace; font-size: .85em; color: #6b7280; }
</style>
</head>
<body>
<div class="container">
  <h1>Provenance Guard</h1>
  <p class="sub">AI content detection & appeal system</p>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('analyze')">Analyze</button>
    <button class="tab-btn" onclick="switchTab('log')">Log</button>
    <button class="tab-btn" onclick="switchTab('appeal')">Appeal</button>
  </div>

  <div id="analyze" class="tab-content active">
    <label for="text">Text</label>
    <textarea id="text" placeholder="Paste a poem, story excerpt, or blog post..."></textarea>

    <label for="creator">Creator ID</label>
    <input id="creator" value="demo-user">

    <button onclick="analyze()">Analyze</button>

    <div id="result" class="result"></div>
  </div>

  <div id="log" class="tab-content">
    <button onclick="loadLog()">Load Log</button>
    <div id="log-result" class="result"></div>
  </div>

  <div id="appeal" class="tab-content">
    <label for="appeal-id">Content ID</label>
    <input id="appeal-id" placeholder="UUID of flagged content">

    <label for="appeal-reason">Why do you disagree?</label>
    <textarea id="appeal-reason" placeholder="Explain why this content was misclassified..."></textarea>

    <button onclick="submitAppeal()">Submit Appeal</button>

    <div id="appeal-result" class="result"></div>
  </div>
</div>

<script>
const CLASS = { likely_ai: "ai", uncertain: "unc", likely_human: "human" };

function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tab).classList.add('active');
  document.querySelector(`button[onclick="switchTab('${tab}')"]`).classList.add('active');
}

async function analyze() {
  const btn = event.target;
  const out = document.getElementById("result");
  const text = document.getElementById("text").value.trim();
  const creator_id = document.getElementById("creator").value.trim() || "demo-user";
  if (!text) { out.innerHTML = '<p class="err">Please enter some text.</p>'; return; }

  btn.disabled = true; btn.textContent = "Analyzing...";
  out.innerHTML = "";
  try {
    const r = await fetch("/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, creator_id })
    });
    if (r.status === 429) { out.innerHTML = '<p class="err">Rate limit reached (10/min). Wait a moment.</p>'; return; }
    const d = await r.json();
    if (!r.ok) { out.innerHTML = '<p class="err">' + (d.error || "Error") + '</p>'; return; }
    const cls = CLASS[d.attribution] || "";
    out.innerHTML =
      '<div class="card ' + cls + '">' +
        '<span class="badge">' + d.attribution.replace("_", " ").toUpperCase() + '</span>' +
        '<p class="label-text">' + d.label + '</p>' +
        '<p class="scores">confidence ' + d.confidence +
          ' &middot; llm ' + d.llm_score + ' &middot; stylometry ' + d.stylometry_score +
          '<br><span class="id-mono">id: ' + d.content_id + '</span></p>' +
      '</div>';
  } catch (e) {
    out.innerHTML = '<p class="err">Request failed: ' + e + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = "Analyze";
  }
}

async function loadLog() {
  const btn = event.target;
  const out = document.getElementById("log-result");
  btn.disabled = true; btn.textContent = "Loading...";
  out.innerHTML = "";
  try {
    const r = await fetch("/log");
    const d = await r.json();
    if (!r.ok) { out.innerHTML = '<p class="err">Failed to load log</p>'; return; }

    const entries = d.entries || [];
    if (entries.length === 0) { out.innerHTML = '<p>No entries yet.</p>'; return; }

    let html = '<table class="log-table"><thead><tr><th>Status</th><th>Creator</th><th>Attribution</th><th>Confidence</th><th>Content ID</th></tr></thead><tbody>';
    entries.forEach(e => {
      const cls = CLASS[e.attribution] || "";
      html += '<tr><td>' + e.status + '</td><td>' + e.creator_id + '</td>' +
              '<td><span class="badge ' + cls + '">' + e.attribution.replace("_", " ") + '</span></td>' +
              '<td>' + (e.confidence || 'N/A').toFixed(4) + '</td>' +
              '<td class="id-mono">' + e.content_id + '</td></tr>';
    });
    html += '</tbody></table>';
    out.innerHTML = html;
  } catch (e) {
    out.innerHTML = '<p class="err">Request failed: ' + e + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = "Load Log";
  }
}

async function submitAppeal() {
  const btn = event.target;
  const out = document.getElementById("appeal-result");
  const content_id = document.getElementById("appeal-id").value.trim();
  const reasoning = document.getElementById("appeal-reason").value.trim();
  if (!content_id || !reasoning) { out.innerHTML = '<p class="err">Please fill all fields.</p>'; return; }

  btn.disabled = true; btn.textContent = "Submitting...";
  out.innerHTML = "";
  try {
    const r = await fetch("/appeal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content_id, creator_reasoning: reasoning })
    });
    const d = await r.json();
    if (!r.ok) { out.innerHTML = '<p class="err">' + (d.error || "Error") + '</p>'; return; }
    out.innerHTML = '<p class="success">✓ ' + d.message + '</p>';
    document.getElementById("appeal-id").value = "";
    document.getElementById("appeal-reason").value = "";
  } catch (e) {
    out.innerHTML = '<p class="err">Request failed: ' + e + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = "Submit Appeal";
  }
}
</script>
</div>
</body>
</html>"""


@app.route("/", methods=["GET"])
def index():
    return INDEX_HTML


@app.route("/submit", methods=["POST"])
@limiter.limit(RATE_LIMITS)
def submit():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    creator_id = (body.get("creator_id") or "").strip()
    if not text or not creator_id:
        return jsonify({"error": "text and creator_id are required"}), 400

    llm = llm_score(text)
    stylo = stylometry_score(text)
    confidence = combine(llm, stylo)
    attribution = classify(confidence)
    label = label_for(attribution)
    content_id = str(uuid.uuid4())

    record = {
        "content_id": content_id, "creator_id": creator_id, "text": text,
        "llm_score": llm, "stylometry_score": stylo, "confidence": confidence,
        "attribution": attribution, "status": "classified",
    }
    storage.save_submission(record, DB_PATH)
    storage.append_audit({**record, "appeal_reasoning": None}, DB_PATH)

    return jsonify({
        "content_id": content_id, "attribution": attribution,
        "confidence": round(confidence, 4), "label": label,
        "llm_score": round(llm, 4), "stylometry_score": round(stylo, 4),
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": storage.recent_log(50, DB_PATH)})


@app.route("/appeal", methods=["POST"])
def appeal():
    body = request.get_json(silent=True) or {}
    content_id = (body.get("content_id") or "").strip()
    reasoning = (body.get("creator_reasoning") or "").strip()
    if not content_id or not reasoning:
        return jsonify({"error": "content_id and creator_reasoning are required"}), 400

    original = storage.get_content(content_id, DB_PATH)
    if original is None:
        return jsonify({"error": "content_id not found"}), 404

    storage.update_status(content_id, "under_review", DB_PATH)
    storage.append_audit({
        "content_id": content_id, "creator_id": original["creator_id"],
        "attribution": original["attribution"], "confidence": original["confidence"],
        "llm_score": original["llm_score"],
        "stylometry_score": original["stylometry_score"],
        "status": "under_review", "appeal_reasoning": reasoning,
    }, DB_PATH)

    return jsonify({
        "content_id": content_id, "status": "under_review",
        "message": "Appeal received. This content is now under human review.",
    })


@app.route("/content/<content_id>", methods=["GET"])
def content(content_id):
    rec = storage.get_content(content_id, DB_PATH)
    if rec is None:
        return jsonify({"error": "content_id not found"}), 404
    return jsonify(rec)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
