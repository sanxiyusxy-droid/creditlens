"""CreditLens 面试演示页（任务 30）。

启动（先起 API）：
    uv run uvicorn apps.api.main:app --port 8000
    uv run streamlit run apps/demo/streamlit_app.py

或一条命令：powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1

演示编排（8–12 分钟）见 docs/演示脚本.md：
时点切换 → 完整预审 DAG → 证据回原文页 → HITL 复核与报告 → Trace 审计。
"""

import hashlib
import uuid
from pathlib import Path

import streamlit as st
from apps.demo.http_client import DemoHTTPError, get_binary, get_json, post_json
from apps.demo.presenter import present_retrieval, present_run_trace
from pydantic import ValidationError

from creditlens.evaluation.failure_cases import FailureCaseReport

st.set_page_config(page_title="CreditLens 授信预审演示", layout="wide")

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_CASE = "00000000-0000-0000-0000-000000000201"

with st.sidebar:
    st.title("CreditLens")
    st.caption("小微企业授信尽调与审查 · RAG + Multi-Agent 原型")
    api_base = st.text_input("API 地址", DEFAULT_API)
    case_id = st.text_input("案件 ID", DEFAULT_CASE)
    try:
        ready = get_json(f"{api_base}/health/ready", timeout=5)
        st.success(f"API {ready.get('status', '?')}（RLS 业务角色连接）")
    except DemoHTTPError:
        st.error("API 未就绪：请先运行 scripts/start_demo.ps1")
    st.divider()
    st.caption(
        "安全边界：服务端派生 ACL / Case Snapshot 冻结 / RLS 行级隔离 /"
        " 硬过滤先于召回 / 回表复核 / 独立 Leakage 审计 = 0"
    )


def _show_http_error(error: DemoHTTPError) -> None:
    if error.status_code is None:
        st.error(f"API 请求失败（{error.code}）：服务不可达、超时或响应格式无效。")
    else:
        st.error(f"API 请求失败（HTTP {error.status_code}，{error.code}）。")


def _post(path: str, payload: dict) -> dict:
    try:
        return post_json(f"{api_base}{path}", payload=payload)
    except DemoHTTPError as exc:
        _show_http_error(exc)
        st.stop()


def _get(path: str, **params) -> dict:
    try:
        return get_json(f"{api_base}{path}", params=params)
    except DemoHTTPError as exc:
        _show_http_error(exc)
        st.stop()


def _render_deep_rag(data: dict, *, view_key: str) -> None:
    """展示 API 已返回的深 RAG 事实；旧响应缺字段时只提示、不阻断。"""
    view = present_retrieval(data)
    with st.expander("深 RAG Trace：Rewrite → 多路召回 → RRF → 精排 → Packing"):
        for warning in view["warnings"]:
            st.info(warning)
        if not view["available"]:
            return

        query = view["query"]
        st.markdown("**QuerySpec / 术语归一**")
        query_cols = st.columns(4)
        query_cols[0].metric("意图", query["intent"])
        query_cols[1].metric("Rewrite 置信度", query["confidence"])
        query_cols[2].metric("不可变数字", len(query["immutable_numbers"]))
        query_cols[3].metric("精确词", len(query["exact_terms"]))
        st.caption(f"原问题：{query['original']} · 独立问题：{query['standalone']}")
        st.caption(
            f"产品：{query['product_code']} · as_of_date：{query['as_of_date']} · "
            f"decision_cutoff_at：{query['decision_cutoff_at']}"
        )
        if query["normalized_terms"]:
            st.write("术语归一/词法扩展：" + "、".join(query["normalized_terms"]))
        else:
            st.caption("本次问题未触发规则词典中的术语归一。")
        if view["normalization_rows"]:
            st.dataframe(view["normalization_rows"], use_container_width=True, hide_index=True)
        if query["must_not_assume"]:
            st.caption("禁止假设：" + "；".join(query["must_not_assume"]))

        st.markdown("**Query Variants**")
        if view["variant_rows"]:
            st.dataframe(view["variant_rows"], use_container_width=True, hide_index=True)
        else:
            st.caption("无 Query Variant 明细。")

        st.markdown("**Dense / Sparse / Summary / Exact 多路召回与拒绝**")
        st.dataframe(view["route_rows"], use_container_width=True, hide_index=True)
        if view["rejection_rows"]:
            st.dataframe(view["rejection_rows"], use_container_width=True, hide_index=True)
        else:
            st.caption("Trace 未记录候选拒绝；未执行的路由与执行后 0 命中已分别标注。")

        fusion = view["fusion"]
        st.markdown("**RRF 融合与精排**")
        fusion_cols = st.columns(4)
        fusion_cols[0].metric("RRF k", fusion["rrf_k"])
        fusion_cols[1].metric("输入排名表", len(fusion["input_lists"]))
        fusion_cols[2].metric("融合候选", fusion["fused_count"])
        fusion_cols[3].metric("最终候选", fusion["final_count"])
        if fusion["input_lists"]:
            st.caption("输入列表：" + "、".join(fusion["input_lists"]))
        rerank = view["rerank"]
        if rerank["degraded"]:
            st.warning(
                f"精排降级：{rerank['reason']}；系统保留 RRF 顺序继续，"
                "但不会把降级结果伪装成已完成精排。"
            )
        elif rerank["applied"]:
            st.success(f"精排已执行 · 版本 {rerank['version']}")
        else:
            st.info("本次未应用精排（这与精排调用失败的 degraded 状态不同）。")

        packing = view["packing"]
        st.markdown("**Context Packing**")
        packing_cols = st.columns(4)
        packing_cols[0].metric("选入段落", packing["selected_count"])
        packing_cols[1].metric(
            "Token 预算",
            f"{packing['total_tokens_est']} / {packing['budget']}" if packing["budget"] else "—",
        )
        packing_cols[2].metric("相邻扩展", packing["expanded_count"])
        packing_cols[3].metric("Packing 丢弃", len(packing["dropped"]))
        if packing["rows"]:
            st.dataframe(
                packing["rows"],
                use_container_width=True,
                hide_index=True,
                key=f"packing-{view_key}",
            )
        elif packing["available"]:
            st.caption("Packing 已执行，但没有段落被选入。")
        if packing["dropped"]:
            st.caption(
                "Packing 只返回丢弃 Section ID；来源可能是预算/文档配额，"
                "也可能是相邻扩展重新回表校验未通过。"
            )
            st.caption("被丢弃 Section ID：" + "、".join(packing["dropped"]))


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
            preview_params = {
                "case_id": case_id,
                "section_id": locator["section_id"],
                "document_version_id": locator["document_version_id"],
                "parse_run_id": locator["parse_run_id"],
                "page_number": locator["page_number"],
                "text_hash": locator["content_hash"],
            }
            try:
                content = get_binary(
                    f"{api_base}/api/v1/evidence/preview",
                    params=preview_params,
                )
                st.image(content, caption=f"{title} · 原始 PDF 页")
            except DemoHTTPError as exc:
                _show_http_error(exc)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "① 可审计问答 / 政策时点",
        "② 完整预审 (Multi-Agent)",
        "③ 证据回原文页",
        "④ 人工复核 HITL",
        "⑤ Trace 审计",
        "⑥ Fail-Closed 案例",
    ]
)

# ---------------------------------------------------------------- ① 时点切换
with tab1:
    st.subheader("Grounded Answer：同一问题，不同审查时点 → 不同版本政策条款")
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
            payload = {
                "idempotency_key": f"demo-qa-{uuid.uuid4()}",
                "question": question,
                "top_k": 3,
                "as_of_date": as_of,
            }
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
                status = data.get("answer_status")
                st.caption(
                    f"答案状态：{status} · 生成模式：{data.get('generation_mode', '?')} · "
                    f"Run {data.get('run_id', '')[:8]}…"
                )
                if status == "ABSTAINED":
                    st.warning(data.get("abstention_reason") or "证据不足，系统拒答")
                    st.caption(
                        f"稳定拒答码：{data.get('refusal_reason_code') or 'UNSPECIFIED_REFUSAL'}"
                    )
                elif status == "NEEDS_REVIEW":
                    st.warning("当前结果只完成结构化证据校验，需人工复核后才能作为正式答案。")
                elif data.get("answer"):
                    st.success(data["answer"])
                if data.get("missing_information"):
                    st.info("缺失信息：" + "；".join(data["missing_information"]))
                if data.get("conflicts"):
                    st.warning("证据冲突：" + "；".join(data["conflicts"]))
                for claim in data.get("claims", []):
                    with st.expander(f"已审计 Claim：{claim['statement'][:50]}"):
                        st.write(claim["statement"])
                        _show_evidence_locators(
                            "正式引用",
                            claim.get("citations", []),
                            claim["claim_id"],
                            f"qa-{as_of}-supporting",
                        )
                        _show_evidence_locators(
                            "反向引用",
                            claim.get("opposing_citations", []),
                            claim["claim_id"],
                            f"qa-{as_of}-opposing",
                        )
                st.markdown("**检索候选（答案只能引用其中已通过审计的段落）**")
                for c in data["candidates"]:
                    with st.container(border=True):
                        st.markdown(f"**{' > '.join(c['heading_path'])}**（第 {c['page']} 页）")
                        st.write(c["text"][:300])
                _render_deep_rag(data, view_key=as_of)

# ---------------------------------------------------------------- ② 完整预审
with tab2:
    st.subheader("Supervisor 固定 DAG：Policy → Financial → Risk → Challenger → Auditor → Report")
    st.caption(
        "POST /runs 立即返回 202 + run_id，DAG 后台执行并按阶段提交；"
        "当前专业 Agent 在进程内顺序执行；Agent 只交换结构化 Artifact，"
        "Claim 必须绑定证据或确定性计算。"
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
        try:
            content = get_binary(
                f"{api_base}/api/v1/evidence/preview",
                params={
                    "case_id": case_id,
                    "section_id": ref["section_id"],
                    "document_version_id": ref["document_version_id"],
                    "parse_run_id": ref["parse_run_id"],
                    "page_number": ref["page"],
                    "text_hash": ref["text_hash"],
                },
            )
            st.image(content, caption="原始 PDF 页渲染（非重排文本）")
        except DemoHTTPError as exc:
            _show_http_error(exc)

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
    st.subheader("Run Trace：Invocation Ledger + Telemetry Outbox + 状态事件")
    st.caption(
        "MODEL/TOOL 的四类终态（SUCCESS / FAILED / DENIED / CANCELLED）已写入"
        "持久调用账本，并与 Telemetry Outbox 同事务提交；Trace 会重新校验载荷、哈希、"
        "投影及 Run/Outbox 绑定。该机制提供可校验完整性，不等同于外部不可篡改存证。"
    )
    run_id = st.session_state.get("run_id", "")
    if run_id and st.button("加载 Trace"):
        st.session_state["loaded_trace"] = _get(f"/api/v1/runs/{run_id}/trace")
        st.session_state["loaded_trace_run_id"] = run_id
    trace = (
        st.session_state.get("loaded_trace")
        if st.session_state.get("loaded_trace_run_id") == run_id
        else None
    )
    if trace:
        view = present_run_trace(trace)
        for warning in view["warnings"]:
            st.warning(warning)
        summary = view["summary"]
        state = view["state"]
        state_message = (
            f"账本派生状态：{state} · API Delivery 汇总：{summary['delivery_status']} · "
            f"完整性：{summary['integrity_status']}"
        )
        if state == "COMPLETE":
            st.success(state_message)
        elif state in {"PENDING", "LEGACY_UNAVAILABLE"}:
            st.info(state_message)
        else:
            st.warning(state_message)
        summary_cols = st.columns(4)
        summary_cols[0].metric("契约", summary["contract_version"])
        summary_cols[1].metric("调用数", summary["total"])
        summary_cols[2].metric("完整性失败", summary["invalid_count"])
        summary_cols[3].metric(
            "投递完整", "是" if summary["delivery_complete"] is True else "否/未知"
        )

        with st.expander("六态判定口径"):
            st.markdown(
                "- **EMPTY**：v2 Run 当前没有已提交调用记录；不证明从未发生调用。\n"
                "- **MISSING**：有效调用记录缺少同事务 Outbox。\n"
                "- **INVALID**：载荷、哈希、投影或绑定校验失败。\n"
                "- **PENDING**：Outbox 待投递或租约处理中。\n"
                "- **COMPLETE**：当前持久化调用均有效且 Outbox 已投递。\n"
                "- **DEGRADED**：死信或其他非完整投递状态。"
            )

        st.markdown("**Telemetry Delivery**")
        st.dataframe(view["count_rows"], use_container_width=True, hide_index=True)
        st.markdown("**MODEL / TOOL Invocation 终态**")
        if view["invocation_rows"]:
            st.dataframe(view["invocation_rows"], use_container_width=True, hide_index=True)
        else:
            st.info("该 Run 没有可展示的 invocation_v2 记录。")

        st.markdown("**Run 状态事件（与调用账本是两类不同事实）**")
        st.dataframe(
            view["event_rows"],
            use_container_width=True,
            height=560,
            hide_index=True,
        )

# ---------------------------------------------------------------- ⑥ Fail-Closed
with tab6:
    st.subheader("两个合成故障注入：真实执行如何 fail closed")
    st.caption(
        "这里展示的是可复现的合成故障，不是生产事故。非法 Artifact 会经过真实 "
        "Supervisor → EvidenceAuditor → 数据库持久化；验收要求 Run=HUMAN_REVIEW、"
        "Claim 被打回、Artifact/RunEvent 可核验且 ReportVersion=0。"
    )
    failure_report_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "reports"
        / "local"
        / "v16_fail_closed_system.json"
    )
    if not failure_report_path.is_file():
        st.info(
            "尚无本机系统实证。运行 scripts/run_fail_closed_cases.py "
            "--execute-system 后刷新；一键启动会自动执行。"
        )
    else:
        try:
            failure_report = FailureCaseReport.model_validate_json(failure_report_path.read_bytes())
        except (OSError, ValidationError):
            st.error("Fail-closed 报告不可读或契约校验失败，拒绝展示为有效证据。")
        else:
            st.caption(
                f"生成时间：{failure_report.generated_at.isoformat()} · "
                f"Execution ID：{failure_report.execution_id}"
            )
            if not failure_report.system_execution_performed:
                st.warning("当前文件只是 Contract Validator 预检，不是系统执行证明。")
            elif not failure_report.all_passed:
                st.error("至少一个系统门禁断言失败；不得将该报告作为完成证据。")
            else:
                st.success("真实状态机与数据库门禁全部通过；证明范围不包含 HTTP 端点或生产事故。")
            for result in failure_report.results:
                evidence = result.system_evidence
                title = f"{result.case_id} · {result.injection_type.value}"
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(result.safe_display.public_message)
                    st.caption(
                        f"动作 {result.safe_display.action} · 工作流 "
                        f"{result.safe_display.workflow_status} · "
                        f"违规码 {', '.join(result.observed_violation_codes) or '—'}"
                    )
                    if evidence is None:
                        st.caption("无 Supervisor/数据库系统证据；仅可视为合同预检。")
                        continue
                    cols = st.columns(4)
                    cols[0].metric("Run 状态", evidence.persisted_run_status)
                    cols[1].metric("Claim 状态", "/".join(evidence.claim_review_statuses))
                    cols[2].metric(
                        "Artifact / Event",
                        f"{evidence.artifact_count} / {evidence.run_event_count}",
                    )
                    cols[3].metric("报告版本数", evidence.report_count)
                    st.caption("状态路径：" + " → ".join(evidence.state_transitions))
                    st.caption(
                        f"Artifact 哈希复核：{evidence.artifact_hashes_verified} · "
                        f"列/Claim 投影复核：{evidence.artifact_claim_projection_verified} · "
                        f"Audit 事件：{evidence.audit_completed} · HTTP 调用："
                        f"{evidence.http_endpoint_called}"
                    )
