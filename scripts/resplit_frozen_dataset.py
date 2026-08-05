"""WP4：frozen 数据集 dev/test 重划分（确定性、可复跑）。

背景：frozen_v2 原划分中 test 80 题全部来自 golden_case_001，
golden_case_002/003 全在 dev，不是真正的多案件冻结测试集。

规则：
- 以「案件 × 归一化题面模板」为分配单元（数字归一），同组不跨 split，
  避免模板化相似问题造成 dev/test 泄漏；
- 每案件 test 目标占比 60%，dev 40%（dev 仅调参，简历指标只报 test）；
- 尽量保证案件内每个 intent 在 test 中有覆盖（缺则把最小同 intent 组调入 test）；
- 三个案件必须同时进入 dev 与 test。

用法：
    uv run python scripts/resplit_frozen_dataset.py [--dataset evaluation/datasets/frozen_v2.json] [--write]
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RATIO = 0.6


def norm_template(question: str) -> str:
    text = re.sub(r"\s+", "", question)
    return re.sub(r"\d+", "#", text)


def resplit(questions: list[dict]) -> None:
    by_case: dict[str, list[int]] = collections.defaultdict(list)
    for index, question in enumerate(questions):
        by_case[question["case_key"]].append(index)

    for case_key in sorted(by_case):
        indices = by_case[case_key]
        # 模板分组：(case_key, 归一化题面) -> 题目下标（按 question_id 稳定排序）
        groups: dict[str, list[int]] = collections.defaultdict(list)
        for index in indices:
            groups[norm_template(questions[index]["question"])].append(index)

        # intent 分层：每 intent 内按 60% 目标选组，避免大模板组挤压稀有 intent
        by_intent: dict[str, list[list[int]]] = collections.defaultdict(list)
        for group in groups.values():
            by_intent[questions[group[0]]["intent"]].append(group)

        test_set: set[int] = set()
        for intent in sorted(by_intent):
            intent_groups = sorted(
                by_intent[intent], key=lambda g: (-len(g), questions[g[0]]["question_id"])
            )
            intent_total = sum(len(g) for g in intent_groups)
            target = round(intent_total * TEST_RATIO)
            picked = 0
            for group in intent_groups:
                if picked >= target:
                    break
                test_set.update(group)
                picked += len(group)
            # 小 intent（target=0）至少保留在 dev；但案件内 intent 不得在 test 完全缺席时，
            # 由下方兼顾处理：单题 intent 直接进 test（冻结集需覆盖全部能力维度）
            if picked == 0 and intent_total <= 2:
                test_set.update(intent_groups[0])

        for index in indices:
            questions[index]["split"] = "test" if index in test_set else "dev"


def report(questions: list[dict]) -> None:
    split_case = collections.Counter((q["split"], q["case_key"]) for q in questions)
    split_intent = collections.Counter((q["split"], q["intent"]) for q in questions)
    unanswerable = collections.Counter(
        q["split"] for q in questions if not q.get("answerable", True)
    )
    print("split x case:")
    for key in sorted(split_case):
        print(f"  {key}: {split_case[key]}")
    print("split x intent:")
    for key in sorted(split_intent):
        print(f"  {key}: {split_intent[key]}")
    print(f"unanswerable per split: {dict(unanswerable)}")
    # 泄漏自检：任何模板组不得跨 split
    leak = collections.defaultdict(set)
    for q in questions:
        leak[(q["case_key"], norm_template(q["question"]))].add(q["split"])
    crossed = [k for k, v in leak.items() if len(v) > 1]
    print(f"template groups crossing split: {len(crossed)}")
    assert not crossed, "模板组跨 split，存在泄漏风险"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "frozen_v2.json",
    )
    parser.add_argument("--write", action="store_true", help="写回文件（默认只预览）")
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    old = collections.Counter(q["split"] for q in data["questions"])
    resplit(data["questions"])
    print(f"重划分前: {dict(old)}")
    report(data["questions"])

    if args.write:
        data["dataset_version"] = "2.1.0"  # 划分变更 = 新版本（dataset hash 随之冻结）
        args.dataset.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.dataset}（dataset_version=2.1.0）")
    else:
        print("预览模式：加 --write 写回文件")


if __name__ == "__main__":
    sys.exit(main())
