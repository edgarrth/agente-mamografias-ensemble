from mammography_agent.streamlit_compat import image_fill_width_kwargs


def test_image_width_kwargs_for_streamlit_138_style_signature():
    def image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto"):
        return None

    assert image_fill_width_kwargs(image) == {"use_column_width": True}


def test_image_width_kwargs_for_transitional_streamlit_style_signature():
    def image(
        image,
        caption=None,
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
        *,
        use_container_width=False,
    ):
        return None

    assert image_fill_width_kwargs(image) == {"use_container_width": True}


def test_image_width_kwargs_for_current_streamlit_style_signature():
    def image(
        image,
        caption=None,
        width="content",
        clamp=False,
        channels="RGB",
        output_format="auto",
        *,
        use_container_width=None,
    ):
        return None

    assert image_fill_width_kwargs(image) == {"width": "stretch"}


def test_streamlit_app_does_not_call_removed_column_width_directly():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py").read_text()
    assert "use_column_width=True" not in source
    assert "image_fill_width_kwargs(st.image)" in source
