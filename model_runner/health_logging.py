from __future__ import annotations
import logging

class HealthcheckAccessFilter(logging.Filter):
    """Log only health state transitions, not every successful probe.

    First /health result is logged. Repeated results with the same healthy/unhealthy
    state are suppressed. A failure is logged once, and the first successful recovery
    is logged again. Non-health HTTP access logs are untouched.
    """
    def __init__(self) -> None:
        super().__init__()
        self._state: str | None = None

    @staticmethod
    def _health_status(record: logging.LogRecord) -> tuple[bool, int | None]:
        args = record.args if isinstance(record.args, tuple) else ()
        if len(args) >= 5:
            try:
                method, path, status = str(args[1]), str(args[2]), int(args[4])
                return method.upper() == "GET" and path.split("?", 1)[0] == "/health", status
            except Exception:
                pass
        msg = record.getMessage()
        is_health = '"GET /health' in msg or 'GET /health HTTP/' in msg
        if not is_health:
            return False, None
        status = None
        for token in reversed(msg.replace('"',' ').split()):
            if token.isdigit() and len(token) == 3:
                status = int(token); break
        return True, status

    def filter(self, record: logging.LogRecord) -> bool:
        is_health, status = self._health_status(record)
        if not is_health:
            return True
        new_state = "healthy" if status is not None and 200 <= status < 400 else "unhealthy"
        if new_state == self._state:
            return False
        self._state = new_state
        return True

_FILTER: HealthcheckAccessFilter | None = None

def install_healthcheck_access_filter() -> None:
    global _FILTER
    if _FILTER is None:
        _FILTER = HealthcheckAccessFilter()
        logging.getLogger("uvicorn.access").addFilter(_FILTER)
