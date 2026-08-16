import logging
from mammography_agent.health_logging import HealthcheckAccessFilter


def access_record(path="/health", status=200):
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:12345", "GET", path, "1.1", status),
        exc_info=None,
    )


def test_health_filter_logs_only_transitions():
    f=HealthcheckAccessFilter()
    assert f.filter(access_record(status=200)) is True
    assert f.filter(access_record(status=200)) is False
    assert f.filter(access_record(status=503)) is True
    assert f.filter(access_record(status=503)) is False
    assert f.filter(access_record(status=200)) is True


def test_health_filter_keeps_non_health_access_logs():
    f=HealthcheckAccessFilter()
    assert f.filter(access_record(path="/datasets", status=200)) is True
