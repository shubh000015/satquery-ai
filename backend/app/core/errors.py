from app.schemas.imagery import ValidationReport


class AgentError(Exception):
    """Base class for failures the API should surface verbatim to the client."""

    code = "agent-error"
    status_code = 400

    def __init__(self, detail: str, issues: list[str] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.issues = issues or []


class AssetNotFoundError(AgentError):
    code = "asset-not-found"
    status_code = 404


class InputValidationError(AgentError):
    """Raised when the validator rejects the input set (diagram: 'Show error')."""

    code = "input-invalid"
    status_code = 422

    def __init__(self, report: ValidationReport) -> None:
        messages = [issue.message for issue in report.issues if issue.severity == "error"]
        super().__init__(
            detail=messages[0] if messages else "Input validation failed.",
            issues=messages,
        )
        self.report = report


class UnsupportedFormatError(AgentError):
    code = "unsupported-format"
    status_code = 415
