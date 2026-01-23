# ui/tabs/tab3_1_left2_train_compare.py
import os
import gradio as gr

from core.config import PROJECT_ROOT
from core.utils_csv import _file_to_path
from core.utils_jeeeun import (
    scan_run_dirs,
    build_artifact_path_map,
    sort_map_by_mtime_desc,
    pick_default_key,
)
from ui.tabs._ui_shared import build_path_dropdown_selector


def build_train_compare(
    *,
    task,
    results_csv_path,
    compare_csv_state,
    compare_enabled_state,
):
    with gr.Tab(label="다른 training과 비교하기"):
        gr.Markdown("### 1. 최근 트레이닝 모니터링 소스 (Primary)")
        gr.Markdown(
            "트레이닝 후 트레이닝 경로가 자동으로 선택됩니다.<br>"
            "비워두면 최신 트레이닝 경로를 자동으로 선택합니다."
        )

        with gr.Accordion(label="다른 csv 파일 불러오기", open=False):
            results_csv_file = gr.File(
                label="탐색기에서 results.csv 선택",
                file_types=[".csv"],
                file_count="single",
            )

        gr.Markdown("### 2. 이전 runs 선택 (Compare)")
        gr.Markdown(
            "원하는 모델이 검색되지 않으면, "
            "[새로운 training 시작] 탭에서 YOLO Task를 확인해주세요."
        )

        runs_selector = build_path_dropdown_selector(
            label="비교 대상 run 선택"
        )
        runs_dropdown = runs_selector["dropdown"]
        runs_map_state = runs_selector["map_state"]
        prev_results_csv_path = runs_selector["path_output"]

        btn_overlay_plots = gr.Button("데이터 불러오기")

    # ===== 이벤트 =====

    results_csv_file.change(
        fn=lambda f: _file_to_path(f),
        inputs=[results_csv_file],
        outputs=[results_csv_path],
    )

    def refresh_runs_by_task(task_name: str):
        base_dir = os.path.join(PROJECT_ROOT, "runs", task_name)

        run_dir_map = scan_run_dirs(base_dir)
        path_map = build_artifact_path_map(
            run_dir_map,
            artifact="results_csv",
        )

        sorted_keys = sort_map_by_mtime_desc(path_map)
        default_key = pick_default_key(sorted_keys)

        return runs_selector["refresh_fn"](
            path_map,
            sorted_keys,
            default_key,
        )

    task.change(
        fn=refresh_runs_by_task,
        inputs=[task],
        outputs=[
            runs_dropdown,
            runs_map_state,
            prev_results_csv_path,
        ],
    )

    def on_overlay_plots(primary_csv, compare_csv):
        return compare_csv, True

    btn_overlay_plots.click(
        fn=on_overlay_plots,
        inputs=[
            results_csv_path,
            prev_results_csv_path,
        ],
        outputs=[
            compare_csv_state,
            compare_enabled_state,
        ],
    )

    return {
        "compare_selector": {
            "dropdown": runs_dropdown,
            "map_state": runs_map_state,
            "path": prev_results_csv_path,
        },
        "refresh": {
            "runs": refresh_runs_by_task,
        },
    }
