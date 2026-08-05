"""稳定错误码（文档 §14.6）。不向客户端暴露堆栈、SQL 或内部路径。"""


class CreditLensError(Exception):
    error_code: str = "INTERNAL_ERROR"
    retryable: bool = False

    def __init__(self, message: str = "", details: dict | None = None):
        super().__init__(message or self.error_code)
        self.message = message or self.error_code
        self.details = details or {}


class AclDeniedError(CreditLensError):
    error_code = "ACL_DENIED"


class CaseNotFoundError(CreditLensError):
    error_code = "CASE_NOT_FOUND"


class DocumentParseFailedError(CreditLensError):
    error_code = "DOCUMENT_PARSE_FAILED"


class DataQualityBlockedError(CreditLensError):
    error_code = "DATA_QUALITY_BLOCKED"


class UploadIntegrityMismatchError(CreditLensError):
    error_code = "UPLOAD_INTEGRITY_MISMATCH"


class InvalidStateTransitionError(CreditLensError):
    error_code = "INVALID_STATE_TRANSITION"


class IdempotencyConflictError(CreditLensError):
    error_code = "IDEMPOTENCY_CONFLICT"


class ConcurrentReviewConflictError(CreditLensError):
    """WP3：并发审批冲突（expected_state_version 不匹配），API 映射 409。"""

    error_code = "REVIEW_CONFLICT"
    retryable = True


class ActionNotAuthorizedError(CreditLensError):
    """WP3：复核动作未授权（仅 REVIEWER/OWNER 可审批）。"""

    error_code = "ACTION_NOT_AUTHORIZED"


class InsufficientEvidenceError(CreditLensError):
    error_code = "INSUFFICIENT_EVIDENCE"
