"""Recompute all published QVHighlights summary metrics from saved predictions."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone_eval.eval import eval_submission

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def main():
    predictions = read_jsonl(ROOT/'evaluation/full_val_predictions.jsonl')
    truth = read_jsonl(ROOT/'evaluation/highlight_val_release.jsonl')
    expected = json.loads((ROOT/'evaluation/best_audio_metrics.json').read_text())['brief']
    actual = eval_submission(predictions, truth, verbose=False, match_number=True)['brief']
    errors = {key: {'expected':value, 'actual':actual.get(key)} for key,value in expected.items() if abs(actual.get(key, -999)-value) > 0.005}
    if errors:
        raise SystemExit(json.dumps(errors, indent=2))
    print(json.dumps({'queries':len(predictions), 'metrics_matched':len(expected), 'metrics':actual}, indent=2))

if __name__ == '__main__':
    main()
