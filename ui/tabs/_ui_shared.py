# ui/tabs/_ui_shared.py
import gradio as gr
from typing import Dict, List

def build_path_dropdown_selector(
    label: str,
    path_label: str = "선택된 경로",
):
    """
    공통 Dropdown + State + Path Textbox selector

    반환:
        dropdown
        map_state
        path_output
        refresh_fn(path_map, default_key)
    """

    dropdown = gr.Dropdown(label=label, choices=[], value=None)
    map_state = gr.State({})
    path_output = gr.Textbox(
        label=path_label,
        interactive=False,
        lines=3,
        elem_id="log_box"
    )

    def refresh_selector(
        path_map: Dict[str, str],
        sorted_keys: List[str],
        default_key: str | None,
    ):
        return (
            gr.update(choices=sorted_keys, value=default_key),
            path_map,
            path_map.get(default_key, "") if default_key else "",
        )

    def on_change(selected_key, path_map):
        if selected_key and selected_key in path_map:
            return path_map[selected_key]
        return ""

    dropdown.change(
        fn=on_change,
        inputs=[dropdown, map_state],
        outputs=[path_output],
    )

    return {
        "dropdown": dropdown,
        "map_state": map_state,
        "path_output": path_output,
        "refresh_fn": refresh_selector,
    }


def build_log_textbox(
    label: str,
    value: str = "",
    lines: int = 3,
    elem_id: str = "log_box",
):
    """
    공통 로그용 Textbox
    - 경로 표시
    - 상태 로그
    - 결과 경로 출력
    """
    return gr.Textbox(
        label=label,
        value=value,
        lines=lines,
        interactive=False,
        elem_id=elem_id,
    )

def build_markdown_log_box(
    title: str,
    value: str = "",
):
    """
    공통 Markdown 로그 박스
    - 경로 설정 상태
    - 안내 메시지
    - 설명용 로그
    """
    with gr.Group():
        gr.Markdown(f"### [{title}]")
        log_md = gr.Markdown(value=value)

    return log_md