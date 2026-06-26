"""Run the four reference inputs through the full pipeline and print scores.
Manual M4 calibration aid: python scripts/calibrate.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.llm import llm_score
from signals.stylometry import stylometry_score
from scoring import combine, classify

CASES = {
    "clear_ai": "Artificial intelligence represents a transformative paradigm shift in "
        "modern society. It is important to note that while the benefits of AI are "
        "numerous, it is equally essential to consider the ethical implications.",
    "clear_human": "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in it and i "
        "was thirsty for like three hours after.",
    "formal_human": "The relationship between monetary policy and asset price inflation "
        "has been extensively studied in the literature. Central banks face a "
        "fundamental tension between price stability and prolonged low interest rates.",
    "edited_ai": "I've been thinking a lot about remote work lately. There are genuine "
        "tradeoffs — flexibility and no commute on one side, isolation and blurred "
        "work-life boundaries on the other.",
}

if __name__ == "__main__":
    print(f"{'case':<14}{'llm':>6}{'stylo':>8}{'conf':>7}  attribution")
    for name, text in CASES.items():
        llm = llm_score(text)
        stylo = stylometry_score(text)
        conf = combine(llm, stylo)
        print(f"{name:<14}{llm:>6.2f}{stylo:>8.2f}{conf:>7.2f}  {classify(conf)}")
