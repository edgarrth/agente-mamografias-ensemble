"""Small compatibility helpers for Streamlit presentation APIs.

These helpers are deliberately isolated from the inference pipeline. They only
select presentation arguments supported by the Streamlit version installed in
the Web container.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def image_fill_width_kwargs(image_callable: Callable[..., Any]) -> dict[str, Any]:
    """Return a full-width argument supported by the installed ``st.image``.

    Streamlit 1.61 removed ``use_column_width``. Intermediate 1.x releases
    support ``use_container_width`` and current releases support
    ``width='stretch'``. The project intentionally keeps a broad Streamlit 1.x
    range, so the UI chooses the compatible spelling at runtime.
    """
    try:
        parameters = inspect.signature(image_callable).parameters
    except (TypeError, ValueError):
        # Current API is the safest fallback when introspection is unavailable.
        return {"width": "stretch"}

    # Current Streamlit: deprecated column argument is gone and width accepts
    # semantic values such as "stretch".
    if "use_column_width" not in parameters and "width" in parameters:
        return {"width": "stretch"}

    # Transitional Streamlit versions (e.g. 1.45/1.50).
    if "use_container_width" in parameters:
        return {"use_container_width": True}

    # Older supported Streamlit versions (e.g. 1.38).
    if "use_column_width" in parameters:
        return {"use_column_width": True}

    return {"width": "stretch"}
