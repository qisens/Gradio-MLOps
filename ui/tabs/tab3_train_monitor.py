# ui/tabs/tab3_train_monitor.py
import os
import gradio as gr

from core.config import (
    PROJECT_ROOT, RUNS_DIR, METRIC_COLUMNS, LOSS_COLUMNS,
    UPLOAD_DATA_DIR, UPLOAD_MODEL_DIR
)
from core.utils_csv import _build_runs_map, _on_run_change, _file_to_path
from core.yolo_train import run_epoch_eval_manual
from core.utilities import build_folder_picker, save_uploaded_file
from core.train_monitor_service import (
    refresh_6plots_compare, prev_page, next_page,
    build_epoch_conf_monitor_ui
)

def build_tab3_train_monitor(trainer):
    with gr.Tab("3. Train Monitor"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 학습 실행")
                task = gr.Radio(["detect", "segment"], value="segment", label="YOLO Task")

                data_yaml_file = gr.File(label="data.yaml 업로드", file_types=[".yaml", ".yml"], file_count="single")
                data_yaml_path = gr.Textbox(label="서버 저장 경로 (data.yaml)", interactive=False)

                model_pt_file = gr.File(label="모델(.pt) 업로드", file_types=[".pt"], file_count="single")
                model_pt_path = gr.Textbox(label="서버 저장 경로 (model.pt)", interactive=False)

                data_yaml_file.change(fn=lambda f: save_uploaded_file(f, UPLOAD_DATA_DIR), inputs=[data_yaml_file], outputs=[data_yaml_path])
                model_pt_file.change(fn=lambda f: save_uploaded_file(f, UPLOAD_MODEL_DIR), inputs=[model_pt_file], outputs=[model_pt_path])

                with gr.Row():
                    monitor_imgsz = gr.Slider(label="imgsz", minimum=256, maximum=2048, step=64, value=640)
                    monitor_epochs = gr.Slider(label="epochs", minimum=1, maximum=500, step=1, value=100)
                with gr.Row():
                    monitor_batch = gr.Slider(label="batch", minimum=1, maximum=128, step=1, value=16)
                    monitor_lr0 = gr.Number(label="lr0", value=0.001)

                with gr.Row():
                    btn_start_train = gr.Button("학습 시작 (CLI)", variant="primary")
                    btn_stop_train = gr.Button("학습 강제 종료", variant="stop")
                train_status = gr.Textbox(label="상태", interactive=False)

                btn_stop_train.click(fn=lambda: trainer.stop_train(), inputs=[], outputs=[train_status])

                gr.Markdown("### Epoch별 모델 평가 (수동 실행)")
                with gr.Row():
                    with gr.Column(scale=1):
                        weights_path_tb, _, _, _ = build_folder_picker(
                            label="weights 폴더", root_dir=PROJECT_ROOT, default_path=os.path.join(PROJECT_ROOT, "runs")
                        )
                    with gr.Column(scale=1):
                        eval_img_path_tb, _, _, _ = build_folder_picker(
                            label="평가 이미지 폴더", root_dir=PROJECT_ROOT, default_path=os.path.join(PROJECT_ROOT, "datasets")
                        )

                with gr.Row():
                    eval_imgsz = gr.Slider(label="imgsz", minimum=256, maximum=2048, step=64, value=640)
                    eval_conf = gr.Number(label="conf_thres", value=0.25)
                    eval_iou = gr.Number(label="iou_thres", value=0.5)
                    eval_device = gr.Textbox(label="device", value="0")

                btn_eval = gr.Button("선택 경로로 Epoch 평가 실행", variant="primary")
                eval_log = gr.Textbox(label="평가 로그", lines=12, interactive=False)
                btn_eval.click(fn=run_epoch_eval_manual, inputs=[weights_path_tb, eval_img_path_tb, eval_imgsz, eval_conf, eval_iou, eval_device], outputs=[eval_log])

                gr.Markdown("### 모니터링 소스 (Primary)")
                results_csv_path = gr.Textbox(label="results.csv 경로 (비우면 최신 자동)", value="")
                results_csv_file = gr.File(label="탐색기에서 results.csv 선택", file_types=[".csv"], file_count="single")
                refresh_sec = gr.Slider(label="갱신 주기(초)", minimum=1, maximum=10, step=1, value=2)

                gr.Markdown("### 이전 runs 선택 (Compare)")
                compare_enabled = gr.Checkbox(value=True, label="이전 run과 비교(오버레이)")
                btn_refresh_runs = gr.Button("runs 목록 갱신")
                runs_dropdown = gr.File(label="비교 대상 run 선택", file_types=[".csv"], file_count="single")
                runs_map_state = gr.State(value={})
                prev_results_csv_path = gr.Textbox(label="(Compare) 선택된 results.csv 경로", interactive=False, value="")

            with gr.Column(scale=3):
                gr.Markdown("### 실시간 지표 (6개씩 묶어서)")
                view_mode = gr.Radio(["metrics", "loss"], value="metrics", label="표출 그룹")

                with gr.Row():
                    btn_prev = gr.Button("◀ 이전")
                    page_state = gr.State(1)
                    page_view = gr.Markdown("페이지: 1")
                    btn_next = gr.Button("다음 ▶")

                plot6 = []
                for _ in range(2):
                    with gr.Row():
                        for _ in range(3):
                            plot6.append(gr.Plot(label=""))

                last_update = gr.Markdown("마지막 갱신: -")
                with gr.Accordion("Epoch/Best/Last Conf 추세(스캔)", open=False):
                    build_epoch_conf_monitor_ui(default_weights_dir="")

        timer = gr.Timer(value=2.0)

        results_csv_file.change(fn=lambda f: _file_to_path(f), inputs=[results_csv_file], outputs=[results_csv_path])

        btn_refresh_runs.click(fn=_build_runs_map, inputs=[task], outputs=[runs_dropdown, runs_map_state, prev_results_csv_path])
        runs_dropdown.change(fn=_on_run_change, inputs=[runs_dropdown, runs_map_state], outputs=[prev_results_csv_path])

        btn_prev.click(fn=prev_page, inputs=[page_state], outputs=[page_state])
        btn_next.click(
            fn=lambda p, m: next_page(p, METRIC_COLUMNS if m == "metrics" else LOSS_COLUMNS),
            inputs=[page_state, view_mode],
            outputs=[page_state],
        )
        page_state.change(fn=lambda p: f"페이지: {int(p)}", inputs=[page_state], outputs=[page_view])

        timer.tick(
            fn=lambda primary_csv, rs, p, m, comp_csv, comp_on: refresh_6plots_compare(
                primary_csv, rs, int(p), m, RUNS_DIR, METRIC_COLUMNS, LOSS_COLUMNS,
                compare_csv_path=comp_csv, compare_enabled=comp_on
            ),
            inputs=[results_csv_path, refresh_sec, page_state, view_mode, prev_results_csv_path, compare_enabled],
            outputs=[*plot6, last_update, timer, page_state],
        )

        btn_start_train.click(
            fn=lambda t, dy, mp, isz, ep, ba, lr0: trainer.start_train(t, dy, mp, isz, ep, ba, lr0),
            inputs=[task, data_yaml_path, model_pt_path, monitor_imgsz, monitor_epochs, monitor_batch, monitor_lr0],
            outputs=[train_status],
        )
