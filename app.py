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


if __name__ == "__main__":
    app.run(port=5000, debug=True)
