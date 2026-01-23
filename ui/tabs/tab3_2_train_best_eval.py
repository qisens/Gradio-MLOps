# ui/tabs/tab3_2_train_best_eval.py
import os
import gradio as gr
from core.config import PROJECT_ROOT
from core.yolo_train import run_epoch_eval_manual
from core.train_monitor_service import EpochConfMonitor
from core.utilities import build_folder_picker
from ui.tabs._ui_shared import build_path_dropdown_selector, build_markdown_log_box, build_log_textbox
from core.utils_jeeeun import scan_run_dirs, build_artifact_path_map, sort_map_by_mtime_desc, pick_default_key

epoch_conf_monitor = EpochConfMonitor()

def build_best_eval(task):
    with gr.Tab(label="best 모델 평가"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Epoch별 모델 평가 (수동 실행)")

                with gr.Column():
                    gr.Markdown("모델 선택하기")
                    weight_selector = build_path_dropdown_selector(label="weights 폴더 선택")
                    weight_dropdown = weight_selector["dropdown"]
                    weight_map_state = weight_selector["map_state"]
                    weights_path_tb = weight_selector["path_output"]

                with gr.Column():
                    gr.Markdown("평가 대상 이미지 폴더 선택하기")
                    eval_img_path_tb, _, _, _ = build_folder_picker(
                        label="평가 이미지 폴더",
                        root_dir=os.path.join(PROJECT_ROOT, "test_img"),
                        default_path=os.path.join(PROJECT_ROOT, "test_img"),
                    )

                with gr.Row():
                    gr.Markdown("model inference parameter")
                    eval_imgsz = gr.Slider(256, 2048, step=64, value=640)
                    eval_conf = gr.Number(value=0.25)
                    eval_iou = gr.Number(value=0.5)
                    eval_device = gr.Textbox(value="0")

                run_title = gr.Textbox(label="그래프 타이틀(옵션)", value="", placeholder="예) demo_exp22 / segment 비교 등")
                btn_eval = gr.Button("Epoch 평가 및 그래프 그리기", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("epoch 별 confidence plot")
                eval_log = build_log_textbox(label="평가 로그", lines=7)
                plot_img = gr.Plot(label="Weights vs Conf")
                table = gr.Dataframe(interactive=False)
                csv_file = gr.File(label="CSV 다운로드(요약)")
                save_csv_status = build_markdown_log_box(
                    title="csv 파일 생성 상태",
                    value="왼쪽에서 경로들을 확인하고 버튼을 클릭해주세요.<br>confidence 평가 결과를 csv로 저장 할 수 있습니다.",
                )

        def refresh_weights_by_task(task_name):
            base_dir = os.path.join(PROJECT_ROOT, "runs", task_name)
            run_dir_map = scan_run_dirs(base_dir)
            path_map = build_artifact_path_map(run_dir_map, artifact="weights_dir")
            sorted_keys = sort_map_by_mtime_desc(path_map)
            default_key = pick_default_key(sorted_keys)
            return weight_selector["refresh_fn"](
                path_map, sorted_keys, default_key
            )

        task.change(
            fn=refresh_weights_by_task,
            inputs=[task],
            outputs=[weight_dropdown, weight_map_state, weights_path_tb],
        )

        def on_eval_and_plot(
            weights_dir, eval_img_path, imgsz, conf, iou, device, run_title
        ):
            eval_log_msg = run_epoch_eval_manual(
                weights_dir, eval_img_path, imgsz, conf, iou, device
            )
            fig, df, csv_path, status = epoch_conf_monitor.update_epoch_conf_view(
                weights_dir, run_title
            )
            return (
                f"{eval_log_msg}\n{status}",
                fig,
                df,
                csv_path,
                status,
            )

        btn_eval.click(
            fn=on_eval_and_plot,
            inputs=[
                weights_path_tb,
                eval_img_path_tb,
                eval_imgsz,
                eval_conf,
                eval_iou,
                eval_device,
                run_title,
            ],
            outputs=[
                eval_log,
                plot_img,
                table,
                csv_file,
                save_csv_status,
            ],
        )

    return {
        "weights_selector": {
            "dropdown": weight_dropdown,
            "map_state": weight_map_state,
            "path": weights_path_tb,
        },
        "refresh": {
            "weights": refresh_weights_by_task,
        },
    }
