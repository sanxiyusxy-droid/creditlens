"""CreditLens 面试演示页（任务 30）。

启动（先起 API）：
    uv run uvicorn apps.api.main:app --port 8000
    uv run streamlit run apps/demo/streamlit_app.py

或一条命令：powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1

演示编排（8–12 分钟）见 docs/演示脚本.md：
时点切换 → 完整预审 DAG → 证据回原文页 → HITL 复核与报告 → Trace 审计。
"""

import hashlib

import requests
import streamlit as st

st.set_page_config(page_title="CreditLens 授信预审演示", layout="wide")

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_CASE = "00000000-0000-0000-0000-000000000201"

with st.sidebar:
    st.title("CreditLens")
    st.caption("小微企业授信尽调与审查 · RAG + Multi-Agent 原型")
    api_base = st.text_input("API 地址", DEFAULT_API)
    case_id = st.text_input("案件 ID", DEFAULT_CASE)
    try:
        ready = requests.get(f"{api_base}/health/ready", timeout=5).json()
        st.success(f"API {ready.get('status', '?')}（RLS 业务角色连接）")
    except Exception:
        st.error("API 未就绪：请先运行 scripts/start_demo.ps1")
    st.divider()
    st.caption(
        "安全边界：服务端派生 ACL / Case Snapshot 冻结 / RLS 行级隔离 /"
        " 硬过滤先于召回 / 回表复核 / 独立 Leakage 审计 = 0"
    )


def _post(path: str, payload: dict) -> dict:
    resp = requests.post(f"{api_base}{path}", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, **params) -> dict:
    resp = requests.get(f"{api_base}{path}", params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _show_evidence_locators(title: str, locators: list[dict], claim_id: str, polarity: str) -> None:
    """在 HITL 复核区展示可回原文的结构化证据定位。"""
    st.markdown(f"**{title}（{len(locators)} 条）**")
    if not locators:
        st.caption("无可回原文的文档证据定位")
        return
    for index, locator in enumerate(locators):
        page = locator.get("page_number")
        st.caption(
            f"{locator.get('evidence_type', 'EVIDENCE')} · "
            f"第 {page if page is not None else '?'} 页 · "
            f"hash {str(locator.get('content_hash') or '')[:12]}…"
        )
        required = (
            "section_id",
            "document_version_id",
            "parse_run_id",
            "page_number",
            "content_hash",
        )
        if not all(locator.get(field) is not None for field in required):
            st.caption("定位字段不完整，不能安全打开原文页")
            continue
        if st.button(
            "打开原文页",
            key=f"preview-{claim_id}-{polarity}-{index}",
        ):
            response = requests.get(
                f"{api_base}/api/v1/evidence/preview",
                params={"case_id": case_id, **{field: locator[field] for field in required}},
                timeout=60,
            )
            if response.status_code == 200:
                st.image(response.content, caption=f"{title} · 原始 PDF 页")
            else:
                st.error(f"{response.status_code}: {response.text}")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "① 政策时点切换",
        "② 完整预审 (Multi-Agent)",
        "③ 证据回原文页",
        "④ 人工复核 HITL",
        "⑤ Trace 审计",
    ]
)

# ---------------------------------------------------------------- ① 时点切换
with tab1:
    st.subheader("同一问题，不同审查时点 → 不同版本政策条款")
    st.caption(
        "政策适用性由 as_of_date 决定（2026-01-01 起新版政策生效：负债率上限 65%→70%、"
        "流动比率 1.2→1.0）。检索前硬过滤 + Snapshot 冻结，时点错误的版本根本不进入候选。"
    )
    with st.form("qa_form"):
        question = st.text_input(
            "问题（输入后按回车或点下方按钮，将同时以两个时点检索对比）",
            "流动资金贷款对资产负债率的要求是多少？",
        )
        submitted = st.form_submit_button("🔍 双时点对比检索", type="primary")
    if submitted and question.strip():
        for as_of, cutoff in (
            ("2026-06-30", None),
            ("2025-06-30", "2025-06-30T15:59:59+00:00"),
        ):
            payload = {"question": question, "top_k": 3, "as_of_date": as_of}
            if cutoff:
                payload["decision_cutoff_at"] = cutoff
            with st.spinner(f"以 {as_of} 冻结 Snapshot + 硬过滤检索中..."):
                st.session_state[f"qa-{as_of}"] = _post(
                    f"/api/v1/cases/{case_id}/questions", payload
                )
        st.session_state["qa-question"] = question

    shown_question = st.session_state.get("qa-question")
    if shown_question:
        st.caption(f"当前结果对应问题：**{shown_question}**")
    col_a, col_b = st.columns(2)
    for col, as_of, label in (
        (col_a, "2026-06-30", "审查日 2026-06-30（新政策 v2026 生效）"),
        (col_b, "2025-06-30", "审查日 2025-06-30（旧政策 v2024 生效）"),
    ):
        with col:
            st.markdown(f"**{label}**")
            data = st.session_state.get(f"qa-{as_of}")
            if data:
                for c in data["candidates"]:
                    with st.container(border=True):
                        st.markdown(f"**{' > '.join(c['heading_path'])}**（第 {c['page']} 页）")
                        st.write(c["text"][:300])

# ---------------------------------------------------------------- ② 完整预审
with tab2:
    st.subheader("Supervisor 固定 DAG：Policy → Financial → Challenger → Auditor")
    st.caption(
        "POST /runs 立即返回 202 + run_id，DAG 后台执行并按阶段提交；"
        "Agent 只交换结构化 Artifact，Claim 必须绑定证据或确定性计算。"
    )
    if st.button("🚀 启动完整预审", type="primary"):
        data = _post(f"/api/v1/cases/{case_id}/runs", {"run_type": "FULL_REVIEW"})
        st.session_state["run_id"] = data["run_id"]
    run_id = st.session_state.get("run_id", "")
    run_id = st.text_input("Run ID", run_id)
    if run_id:
        st.session_state["run_id"] = run_id
        if st.button("刷新状态"):
            pass  # 触发重跑即可
        run = _get(f"/api/v1/runs/{run_id}")
        terminal = {"COMPLETED", "FAILED", "HUMAN_REVIEW"}
        status = run["status"]
        (st.success if status in terminal else st.info)(
            f"状态：{status}（HUMAN_REVIEW = 等待人工复核，转 ④）"
        )
        execution = run.get("execution") or {}
        if execution.get("degraded"):
            st.warning(
                "本次运行存在降级覆盖："
                + "、".join(execution.get("degraded_agents") or ["未知 Agent"])
            )
        claims = run.get("claims", [])
        if claims:
            st.markdown(f"**Claims（{len(claims)} 条）**")
            st.dataframe(
                [
                    {
                        "类别": c["category"],
                        "结论": c["verdict"],
                        "陈述": c["statement"],
                        "复核状态": c["review_status"],
                        "正证据": len((c.get("evidence") or {}).get("supporting_evidence_ids", [])),
                        "反证据": len((c.get("evidence") or {}).get("opposing_evidence_ids", [])),
                        "计算": len((c.get("evidence") or {}).get("calculation_ids", [])),
                    }
                    for c in claims
                ],
                use_container_width=True,
            )
            st.session_state["claims"] = claims

# ---------------------------------------------------------------- ③ 证据回原文
with tab3:
    st.subheader("每条证据可回到原始 PDF 页（含案件授权 + 哈希校验）")
    st.caption(
        "EvidenceRef = section + document_version + parse_run + 页码 + text_hash；"
        "服务端校验 Membership、案件绑定、解析批次一致后才渲染。"
    )
    source = st.radio("证据来源", ["从 ① 的检索候选", "手动输入"], horizontal=True)
    ref = None
    if source == "从 ① 的检索候选":
        options = []
        for key in ("qa-2026-06-30", "qa-2025-06-30"):
            data = st.session_state.get(key)
            if data:
                for c in data["candidates"]:
                    options.append((f"{' > '.join(c['heading_path'])} (p{c['page']})", c))
        if options:
            label = st.selectbox("选择候选", [o[0] for o in options])
            ref = dict(next(o[1] for o in options if o[0] == label))
        else:
            st.info("请先在 ① 执行一次检索")
    if ref and st.button("打开原文页"):
        resp = requests.get(
            f"{api_base}/api/v1/evidence/preview",
            params={
                "case_id": case_id,
                "section_id": ref["section_id"],
                "document_version_id": ref["document_version_id"],
                "parse_run_id": ref["parse_run_id"],
                "page_number": ref["page"],
                "text_hash": ref["text_hash"],
            },
            timeout=60,
        )
        if resp.status_code == 200:
            st.image(resp.content, caption="原始 PDF 页渲染（非重排文本）")
        else:
            st.error(f"{resp.status_code}: {resp.text}")

# ---------------------------------------------------------------- ④ HITL
with tab4:
    st.subheader("人工复核：blocking Claim 全部解决后 Run 才能 COMPLETED")
    st.caption(
        "人工决定追加写（不覆盖 Agent 输出）；报告版本持久化是 COMPLETED 前置条件；"
        "幂等键防重复提交。"
    )
    run_id = st.session_state.get("run_id", "")
    if not run_id:
        st.info("请先在 ② 启动一次预审")
    else:
        run = _get(f"/api/v1/runs/{run_id}")
        pending = [
            c for c in run.get("claims", []) if c["review_status"] in ("PENDING", "NEEDS_REWORK")
        ]
        st.write(f"Run 状态：**{run['status']}**，blocking Claims：**{len(pending)}**")
        if pending:
            st.caption("请先在同屏核对支持证据与反证；下方按钮会回到经案件授权校验的原始 PDF 页。")
            for claim in pending:
                evidence = claim.get("evidence") or {}
                with st.expander(f"[{claim['category']}] {claim['statement'][:80]}"):
                    left, right = st.columns(2)
                    with left:
                        _show_evidence_locators(
                            "支持证据",
                            evidence.get("supporting_locators", []),
                            claim["claim_id"],
                            "supporting",
                        )
                    with right:
                        _show_evidence_locators(
                            "反证",
                            evidence.get("opposing_locators", []),
                            claim["claim_id"],
                            "opposing",
                        )
            approved = st.multiselect(
                "选择要批准的 Claim",
                [c["claim_id"] for c in pending],
                format_func=lambda cid: next(
                    c["statement"][:60] for c in pending if c["claim_id"] == cid
                ),
            )
            if st.button("批准所选（APPROVE_CLAIM）", type="primary") and approved:
                result = _post(
                    f"/api/v1/runs/{run_id}/review-decisions",
                    {
                        "action": "APPROVE_CLAIM",
                        "target_claim_ids": approved,
                        "reason_code": "DEMO",
                        "reason": "演示：正反证据已核对",
                        # 幂等键 + 乐观锁为必填契约：键随所选 Claim 稳定，
                        # 重复点击不会重复应用；版本不符返回 409
                        "idempotency_key": (
                            f"demo-approve-{run['state_version']}-"
                            f"{hashlib.sha256(','.join(sorted(approved)).encode()).hexdigest()[:12]}"
                        ),
                        "expected_state_version": run["state_version"],
                    },
                )
                st.success(f"提交后状态：{result['status']}（部分批准会保持 HUMAN_REVIEW）")
        if run["status"] == "COMPLETED":
            report = _get(f"/api/v1/runs/{run_id}/report")
            st.markdown(
                f"**报告 v{report['version_no']}** · {report['status']} · "
                f"hash `{report['content_hash'][:16]}…`"
            )
            st.warning(report["content"]["disclaimer"])
            for claim in report["content"]["claims"]:
                with st.container(border=True):
                    st.markdown(
                        f"**[{claim['category']}] {claim['verdict']}**（{claim['review_status']}）"
                    )
                    st.write(claim["statement"])

# ---------------------------------------------------------------- ⑤ Trace
with tab5:
    st.subheader("Run Trace：已持久化的状态审计事件")
    st.caption(
        "当前 MVP 持久化 run_events 的阶段状态与脱敏载荷；完整 Tool/Model Trace "
        "尚未作为独立审计表落库，事件记录也不等同于不可篡改存证。"
    )
    run_id = st.session_state.get("run_id", "")
    if run_id and st.button("加载 Trace"):
        trace = _get(f"/api/v1/runs/{run_id}/trace")
        st.dataframe(
            [
                {
                    "#": e["sequence_no"],
                    "事件": e["event_type"],
                    "内容": str(e["payload"]),
                    "时间": e["occurred_at"],
                }
                for e in trace["events"]
            ],
            use_container_width=True,
            height=560,
        )
