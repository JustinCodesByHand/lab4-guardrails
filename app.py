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
  body { max-width: 680px; margin: 40px auto; padding: 0 16px; color: #1f2937; }
  h1 { margin-bottom: 4px; }
  p.sub { color: #6b7280; margin-top: 0; }
  label { display: block; font-weight: 600; margin: 16px 0 6px; }
  textarea, input { width: 100%; box-sizing: border-box; padding: 10px;
    border: 1px solid #d1d5db; border-radius: 8px; font: inherit; }
  textarea { min-height: 140px; resize: vertical; }
  button { margin-top: 16px; padding: 10px 22px; border: 0; border-radius: 8px;
    background: #4f46e5; color: #fff; font-weight: 600; font-size: 1em; cursor: pointer; }
  button:disabled { opacity: .6; cursor: progress; }
  #result { margin-top: 24px; }
  .card { border-left: 5px solid #999; background: #f9fafb; border-radius: 0 8px 8px 0;
    padding: 14px 18px; }
  .badge { display: inline-block; color: #fff; font-weight: 700; font-size: .8em;
    letter-spacing: .05em; padding: 3px 12px; border-radius: 12px; }
  .label-text { margin: 10px 0 0; }
  .scores { color: #6b7280; font-size: .85em; margin-top: 8px; }
  .err { color: #b91c1c; }
  .ai     { border-color:#dc2626 } .ai .badge     { background:#dc2626 }
  .unc    { border-color:#d97706 } .unc .badge    { background:#d97706 }
  .human  { border-color:#16a34a } .human .badge  { background:#16a34a }
</style>
</head>
<body>
  <h1>Provenance Guard</h1>
  <p class="sub">Paste text to check whether it reads as human- or AI-written.</p>

  <label for="text">Text</label>
  <textarea id="text" placeholder="Paste a poem, story excerpt, or blog post..."></textarea>

  <label for="creator">Creator ID</label>
  <input id="creator" value="demo-user">

  <button id="go" onclick="analyze()">Analyze</button>

  <div id="result"></div>

<script>
const CLASS = { likely_ai: "ai", uncertain: "unc", likely_human: "human" };

async function analyze() {
  const btn = document.getElementById("go");
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
          '<br>content_id: ' + d.content_id + '</p>' +
      '</div>';
  } catch (e) {
    out.innerHTML = '<p class="err">Request failed: ' + e + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = "Analyze";
  }
}
</script>
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
