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
