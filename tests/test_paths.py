from pathlib import Path
import os
from mammography_agent.workspace import safe_workspace_path

def test_workspace_rejects_escape():
    try:
        safe_workspace_path("../escape")
        assert False
    except ValueError:
        assert True
