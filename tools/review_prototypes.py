from __future__ import annotations

"""行为原型人工终审工具（ROADMAP Stage 2，§5.3 人工确认进环）。

三步工作流:
  1. name   python tools/review_prototypes.py name
       对 pending_review 的簇调 VLM 归纳建议名与特征（填空式 prompt）
  2. page   python tools/review_prototypes.py page [--out results/prototypes/review.html]
       生成审阅页：建议名/特征/成员描述/证据图 + 各原型的 verdict 填写说明
  3. apply  python tools/review_prototypes.py apply verdicts.json
       verdicts.json: [{"proto_id": "proto_001", "action": "approve", "name": "反复挥臂"}, ...]
       action ∈ {approve, rename(=approve+name), reject}
       批准后输出可直接创建的 NL 规则文本；引擎运行中时用
       POST /api/v1/alert-rules {"text": "<rule_text>"} 热生效（工具进程改文件不会热加载）

另: python tools/review_prototypes.py list [--status pending_review|active|...]
"""

import argparse
import json
import sys
from html import escape
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.behavior_prototypes import BehaviorPrototypes
from src.core.novel_behavior_memo import NovelBehaviorMemo


def load_vlm_cfg() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "vlm_config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _load_memo_texts(protos: BehaviorPrototypes, memo: NovelBehaviorMemo) -> None:
    """把备忘正文补进原型（命名 prompt 与审阅页用）。"""
    records = {f"{r.get('task_id', '')}/novel_{r.get('frame_id')}_{r.get('track_id', '')}.json": r
               for r in memo.iter_memos()}
    for proto in protos.list_prototypes():
        texts = [records[k]["memo"].get("summary", k) if k in records else k
                 for k in proto.get("member_memo_keys", [])]
        protos.set_member_texts(proto["proto_id"], texts)


def run_name(protos: BehaviorPrototypes) -> None:
    named = protos.name_pending()
    print(f"已生成 {len(named)} 个建议名：")
    for p in named:
        print(f"  {p['proto_id']}: {p['proposed_name']} — {p['features']}（{p['occurrences']} 次出现）")


def run_list(protos: BehaviorPrototypes, status: str | None) -> None:
    protos_list = protos.list_prototypes(status)
    if not protos_list:
        print("（无原型）")
        return
    print(f"{'ID':<10}{'状态':<16}{'次数':<6}名称/建议名  特征")
    for p in protos_list:
        name = p.get("name") or p.get("proposed_name") or "—"
        print(f"{p['proto_id']:<10}{p['status']:<16}{p['occurrences']:<6}{name}  {p.get('features', '')[:36]}")


def run_page(protos: BehaviorPrototypes, out: Path) -> None:
    protos_list = [p for p in protos.list_prototypes() if p["status"] in ("pending_review", "active")]
    cards = []
    for p in protos_list:
        imgs = "".join(f'<img src="../{escape(e)}" style="max-width:180px;border-radius:6px;margin:4px">'
                       for e in p.get("evidence", [])[:4])
        members = "".join(f"<li>{escape(t)}</li>" for t in p.get("member_texts", [])[:8])
        cards.append(
            f"<div class='card'><h3>{escape(p['proto_id'])} — {escape(p.get('name') or p.get('proposed_name') or '待命名')}"
            f" <small>({p['status']}, {p['occurrences']} 次)</small></h3>"
            f"<p><b>特征:</b> {escape(p.get('features', '—'))}</p>"
            f"<div>{imgs}</div><ul>{members}</ul></div>"
        )
    out.write_text(
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:system-ui;background:#12121e;color:#eee;margin:20px}"
        ".card{background:#1c2333;border-radius:10px;padding:14px;margin-bottom:16px;max-width:900px}"
        "</style></head><body><h1>行为原型终审</h1>"
        "<p>对每个原型在 verdicts.json 中填写 "
        '{"proto_id": "...", "action": "approve|reject", "name": "批准时可选改名"}，'
        "然后执行 apply。</p>" + "".join(cards) + "</body></html>",
        encoding="utf-8",
    )
    print(f"审阅页已生成: {out}")


def run_apply(protos: BehaviorPrototypes, verdicts_file: str) -> None:
    verdicts = json.loads(Path(verdicts_file).read_text(encoding="utf-8"))
    created_rules = []
    for v in verdicts:
        pid, action = str(v.get("proto_id", "")), str(v.get("action", ""))
        if action in ("approve", "rename"):
            proto = protos.approve(pid, name=v.get("name"))
            if proto:
                created_rules.append(proto["rule_text"])
                print(f"✓ {pid} 已批准为「{proto['name']}」")
        elif action == "reject":
            protos.reject(pid)
            print(f"✗ {pid} 已驳回")
        elif action == "merge_into":
            dst = protos.merge(pid, str(v.get("target", "")))
            print(f"⇒ {pid} 已并入 {v.get('target')}" if dst else f"? {pid} 合并失败")
        else:
            print(f"? {pid} 未知 action: {action}")
    if created_rules:
        print("\n批准原型的 NL 规则文本（引擎运行中用 REST 热生效）:")
        for rt in created_rules:
            print(f'  curl -X POST http://localhost:8000/api/v1/alert-rules -H "Content-Type: application/json" '
                  f'-d \'{{"text": "{rt}"}}\'')
            print(f'  规则文本: {rt}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["name", "list", "page", "apply"])
    ap.add_argument("verdicts", nargs="?")
    ap.add_argument("--status", default=None)
    ap.add_argument("--out", default="results/prototypes/review.html")
    args = ap.parse_args()

    vlm_cfg = load_vlm_cfg()
    protos = BehaviorPrototypes(vlm_cfg, output_dir="results")
    memo = NovelBehaviorMemo(vlm_cfg, output_dir="results")
    _load_memo_texts(protos, memo)

    if args.cmd == "name":
        run_name(protos)
    elif args.cmd == "list":
        run_list(protos, args.status)
    elif args.cmd == "page":
        run_page(protos, Path(args.out))
    else:
        if not args.verdicts:
            raise SystemExit("apply 需要 verdicts.json 路径")
        run_apply(protos, args.verdicts)


if __name__ == "__main__":
    main()
