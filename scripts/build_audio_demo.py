#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SETS = [
    ("baseline", Path("runs/arctic_demand/listening_eval")),
    ("spatial_c120", Path("runs/arctic_demand_spatial_c120/listening_eval")),
    ("c116_loud_gate", Path("runs/arctic_demand_spatial_c116_loud/listening_eval_loud")),
    ("c116_quiet_gate", Path("runs/arctic_demand_spatial_c116_loud/listening_eval_quiet")),
    ("c116_oracle_blend_balanced", Path("runs/arctic_demand_spatial_c116_loud/listening_eval_oracle_blend_balanced")),
    ("c116_postfilter_balanced", Path("runs/arctic_demand_spatial_c116_loud/listening_eval_postfilter_balanced")),
    ("c116_noise_first_balanced", Path("runs/arctic_demand_spatial_c116_loud/listening_eval_noise_first_balanced")),
    ("c116_noise_aggressive_best", Path("runs/arctic_demand_spatial_c116_loud/listening_eval_noise_aggressive_best")),
    ("c120_psm_gain_sisdr_m1.50", Path("runs/arctic_demand_spatial_c120_psm_gain_sisdr/listening_eval_gate_m1.50_best")),
    ("c120_psm_noise_loss_m1.50", Path("runs/arctic_demand_spatial_c120_psm_gain_noise_loss/listening_eval_gate_m1.50_best")),
]


def parse_set(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("set label cannot be empty")
        return label, Path(path)
    path = Path(value)
    return path.parent.name or path.name, path


def read_index(directory: Path) -> dict[str, Any]:
    index_path = directory / "index.json"
    if not index_path.exists():
        return {"summary": {}, "items": infer_items(directory)}
    return json.loads(index_path.read_text(encoding="utf-8"))


def infer_items(directory: Path) -> list[dict[str, Any]]:
    samples = sorted({path.name[:10] for path in directory.glob("sample_*_*.wav")})
    rows = []
    for sample in samples:
        files = {}
        for kind in ("noisy", "offline", "realtime", "clean"):
            name = f"{sample}_{kind}.wav"
            if (directory / name).exists():
                files[kind] = name
        if files:
            rows.append({"sample": sample, "files": files})
    return rows


def rel_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()


def fmt_number(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def metric_value(item: dict[str, Any], primary: str, fallback: str | None = None) -> float:
    value = item.get(primary)
    if isinstance(value, (int, float)):
        return float(value)
    if fallback:
        value = item.get(fallback)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def build_set_html(label: str, directory: Path, payload: dict[str, Any], out_dir: Path, repo_root: Path) -> str:
    summary = payload.get("summary", {})
    items = payload.get("items", [])
    cards = []
    for item in items:
        noisy = metric_value(item, "noisy_si_sdr")
        offline_imp = metric_value(item, "offline_improvement", "si_sdr_improvement")
        realtime_imp = metric_value(item, "realtime_improvement", "si_sdr_improvement")
        best_imp = max(offline_imp, realtime_imp)
        is_regression = best_imp < 0.0
        files = item.get("files", {})
        players = []
        for kind, title in (
            ("noisy", "Noisy"),
            ("offline", "Offline"),
            ("realtime", "Realtime"),
            ("clean", "Clean"),
        ):
            filename = files.get(kind)
            if not filename:
                continue
            audio_path = rel_path(directory / filename, out_dir)
            players.append(
                f"""
                <div class="player">
                  <div class="player-title">{html.escape(title)}</div>
                  <audio controls preload="metadata" src="{html.escape(audio_path)}"></audio>
                </div>
                """
            )
        cards.append(
            f"""
            <article class="sample {'regression' if is_regression else ''}"
              data-noisy="{noisy:.6f}"
              data-offline-improvement="{offline_imp:.6f}"
              data-realtime-improvement="{realtime_imp:.6f}"
              data-best-improvement="{best_imp:.6f}"
              data-regression="{str(is_regression).lower()}">
              <div class="sample-head">
                <div>
                  <h3>{html.escape(item.get("sample", "sample"))}{' <span>Regression</span>' if is_regression else ''}</h3>
                  <p>{html.escape(item.get("source_mix", ""))}</p>
                </div>
                <div class="score-grid">
                  <span>Noisy <b>{fmt_number(item.get("noisy_si_sdr"))}</b></span>
                  <span>Offline +<b>{fmt_number(item.get("offline_improvement"))}</b></span>
                  <span>Realtime +<b>{fmt_number(item.get("realtime_improvement"))}</b></span>
                </div>
              </div>
              <div class="players">
                {''.join(players)}
              </div>
            </article>
            """
        )

    source_dir = rel_path(directory, repo_root)
    return f"""
    <section class="run">
      <div class="run-head">
        <div>
          <h2>{html.escape(label)}</h2>
          <p>{html.escape(source_dir)}</p>
        </div>
        <div class="summary">
          <span>Items <b>{html.escape(str(summary.get("items", len(items))))}</b></span>
          <span>Noisy <b>{fmt_number(summary.get("mean_noisy_si_sdr"))}</b></span>
          <span>Offline +<b>{fmt_number(summary.get("mean_offline_improvement"))}</b></span>
          <span>Realtime +<b>{fmt_number(summary.get("mean_realtime_improvement"))}</b></span>
        </div>
      </div>
      <div class="samples">
        {''.join(cards)}
      </div>
    </section>
    """


def build_html(sets: list[tuple[str, Path, dict[str, Any]]], out_dir: Path, repo_root: Path) -> str:
    set_html = "\n".join(build_set_html(label, directory, payload, out_dir, repo_root) for label, directory, payload in sets)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TinyHear Audio Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #ffffff;
      --accent: #16655a;
      --accent-soft: #e4f3f0;
      --warn: #8a4b10;
      --warn-soft: #fff0dc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 249, 0.94);
      backdrop-filter: blur(10px);
    }}
    .topbar {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 720;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 6px 10px;
      border-radius: 8px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 650;
      white-space: nowrap;
    }}
    .toolbar {{
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    .toolbar-inner {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 10px 24px;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 10px;
      flex-wrap: wrap;
    }}
    select, label.toggle {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    select {{
      padding: 6px 28px 6px 10px;
    }}
    label.toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      cursor: pointer;
      user-select: none;
    }}
    input[type="checkbox"] {{
      width: 15px;
      height: 15px;
      accent-color: var(--accent);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px 24px 48px;
    }}
    .run {{
      margin: 0 0 30px;
    }}
    .run-head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 12px;
    }}
    h2 {{
      margin: 0 0 4px;
      font-size: 18px;
      line-height: 1.25;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .summary, .score-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .summary span, .score-grid span {{
      min-height: 30px;
      padding: 6px 9px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    b {{
      color: var(--ink);
      font-weight: 720;
    }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
    }}
    .sample {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .sample.regression {{
      border-color: #f0c78d;
      background: #fffaf3;
    }}
    .sample.hidden {{
      display: none;
    }}
    .sample-head {{
      min-height: 78px;
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    h3 {{
      margin: 0 0 4px;
      font-size: 16px;
      line-height: 1.25;
    }}
    h3 span {{
      display: inline-flex;
      min-height: 24px;
      margin-left: 8px;
      padding: 3px 7px;
      border-radius: 8px;
      background: var(--warn-soft);
      color: var(--warn);
      font-size: 12px;
      vertical-align: 1px;
    }}
    .players {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 0;
    }}
    .player {{
      padding: 12px 14px;
      display: grid;
      grid-template-columns: 82px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      border-bottom: 1px solid var(--line);
    }}
    .player:last-child {{ border-bottom: 0; }}
    .player-title {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }}
    audio {{
      width: 100%;
      height: 36px;
    }}
    .empty {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--warn-soft);
      color: var(--warn);
      padding: 18px;
    }}
    @media (max-width: 720px) {{
      .topbar, .toolbar-inner, main {{ padding-left: 14px; padding-right: 14px; }}
      .topbar, .run-head {{ grid-template-columns: 1fr; display: grid; }}
      .toolbar-inner {{ justify-content: flex-start; }}
      .summary, .score-grid {{ justify-content: flex-start; }}
      .samples {{ grid-template-columns: 1fr; }}
      .player {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>TinyHear Audio Demo</h1>
      <div class="badge">A/B Listening</div>
    </div>
  </header>
  <div class="toolbar">
    <div class="toolbar-inner">
      <select id="sort-mode" aria-label="Sort samples">
        <option value="sample">Sample order</option>
        <option value="worst">Worst improvement</option>
        <option value="best">Best improvement</option>
        <option value="noisy-low">Lowest noisy SI-SDR</option>
        <option value="noisy-high">Highest noisy SI-SDR</option>
      </select>
      <label class="toggle">
        <input id="regression-only" type="checkbox">
        Regression only
      </label>
    </div>
  </div>
  <main>
    {set_html or '<div class="empty">No listening samples found.</div>'}
  </main>
  <script>
    const sortMode = document.getElementById('sort-mode');
    const regressionOnly = document.getElementById('regression-only');

    function numberValue(node, name) {{
      return Number.parseFloat(node.dataset[name] || '0');
    }}

    function applyView() {{
      document.querySelectorAll('.samples').forEach((container) => {{
        const samples = Array.from(container.querySelectorAll('.sample'));
        const sorted = samples.sort((a, b) => {{
          switch (sortMode.value) {{
            case 'worst':
              return numberValue(a, 'bestImprovement') - numberValue(b, 'bestImprovement');
            case 'best':
              return numberValue(b, 'bestImprovement') - numberValue(a, 'bestImprovement');
            case 'noisy-low':
              return numberValue(a, 'noisy') - numberValue(b, 'noisy');
            case 'noisy-high':
              return numberValue(b, 'noisy') - numberValue(a, 'noisy');
            default:
              return 0;
          }}
        }});
        sorted.forEach((sample) => {{
          const hide = regressionOnly.checked && sample.dataset.regression !== 'true';
          sample.classList.toggle('hidden', hide);
          container.appendChild(sample);
        }});
      }});
    }}

    sortMode.addEventListener('change', applyView);
    regressionOnly.addEventListener('change', applyView);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        action="append",
        type=parse_set,
        dest="sets",
        help="Listening set as LABEL=DIR. Can be passed multiple times.",
    )
    parser.add_argument("--out", default="runs/audio_demo/index.html")
    args = parser.parse_args()

    repo_root = Path.cwd()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    requested_sets = args.sets or DEFAULT_SETS
    loaded = []
    for label, directory in requested_sets:
        if not directory.exists():
            continue
        payload = read_index(directory)
        loaded.append((label, directory, payload))
    out_path.write_text(build_html(loaded, out_path.parent, repo_root), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
