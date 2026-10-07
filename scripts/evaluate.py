"""Run the fixed OCS-DETR model on sample or full QVHighlights features."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from qd_detr.model import build_model
from qd_detr.start_end_dataset_audio import StartEndDataset_audio, start_end_collate_audio
from qd_detr.inference import compute_mr_results
from standalone_eval.eval import eval_submission

def run(args):
    torch.set_num_threads(8)
    cfg = json.loads((ROOT/'configs/best_audio.json').read_text())
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    feature_root = Path(args.feature_root).resolve() if args.feature_root else ROOT/'features'
    cfg.update(device=device, pin_memory=device.startswith('cuda'), debug=False)
    cfg['v_feat_dirs'] = [str(feature_root/'slowfast_features'), str(feature_root/'clip_features')]
    cfg['t_feat_dir'] = str(feature_root/'clip_text_features')
    cfg['a_feat_dir'] = str(feature_root/'umt_pann_features')
    if 'tef' in cfg['ctx_mode']:
        cfg['v_feat_dim'] += 2
    options = SimpleNamespace(**cfg)
    asset = json.loads((ROOT/'assets-manifest.json').read_text())['assets'][0]
    checkpoint_path = ROOT/asset['destination']
    if not checkpoint_path.exists():
        raise SystemExit('Run python scripts/download_assets.py first.')
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != asset['sha256']:
        raise SystemExit('Checkpoint SHA-256 mismatch.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model, _ = build_model(options)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device).eval()
    data_path = ROOT/('data/smoke_val.jsonl' if args.smoke else 'evaluation/highlight_val_release.jsonl')
    dataset = StartEndDataset_audio(
        dset_name='hl', data_path=str(data_path), v_feat_dirs=cfg['v_feat_dirs'],
        q_feat_dir=cfg['t_feat_dir'], a_feat_dir=cfg['a_feat_dir'], q_feat_type='last_hidden_state',
        max_q_l=cfg['max_q_l'], max_v_l=cfg['max_v_l'], ctx_mode=cfg['ctx_mode'],
        normalize_v=not cfg['no_norm_vfeat'], normalize_t=not cfg['no_norm_tfeat'],
        load_labels=False, clip_len=cfg['clip_length'], max_windows=cfg['max_windows'],
        span_loss_type=cfg['span_loss_type'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=start_end_collate_audio)
    started = time.perf_counter()
    predictions, _ = compute_mr_results(model, loader, options)
    elapsed = time.perf_counter()-started
    metrics = eval_submission(predictions, dataset.data, verbose=False, match_number=True)
    report = {'epoch':checkpoint['epoch'], 'strict_load':True, 'samples':len(predictions),
              'parameters':sum(p.numel() for p in model.parameters()), 'device':device,
              'torch':torch.__version__, 'inference_seconds':round(elapsed,3), 'metrics':metrics['brief']}
    if not args.smoke:
        historical = json.loads((ROOT/'evaluation/best_audio_metrics.json').read_text())['brief']
        report['matching_summary_metrics'] = sum(abs(metrics['brief'][k]-v) <= 0.005 for k,v in historical.items())
    output = ROOT/'outputs';output.mkdir(exist_ok=True)
    name = 'smoke' if args.smoke else 'full_val'
    (output/(name+'_run.json')).write_text(json.dumps(report,indent=2))
    (output/(name+'_predictions.jsonl')).write_text('\n'.join(json.dumps(x) for x in predictions)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--feature-root')
    parser.add_argument('--device')
    parser.add_argument('--batch-size',type=int,default=100)
    run(parser.parse_args())
