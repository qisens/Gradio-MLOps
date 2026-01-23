# ui/tabs/tab3_1_left1_train_new.py
import gradio as gr
from core.config import UPLOAD_DATA_DIR, UPLOAD_MODEL_DIR
from core.utilities import save_uploaded_file
from ui.tabs._ui_shared import build_log_textbox, build_markdown_log_box


def build_train_new(
    *,
    trainer,
    results_csv_path,
    epoch_tick,
):
    with gr.Tab(label="새로운 training 시작"):
        task = gr.Radio(["segment", "detect"], value="segment", label="YOLO Task")

        gr.Markdown("### 학습 실행")

        data_yaml_file = gr.File(
            label="data.yaml 업로드",
            file_types=[".yaml", ".yml"],
            file_count="single",
        )
        data_yaml_path = build_log_textbox(label="서버 저장 경로 (data.yaml)")

        with gr.Accordion(label="모델 선택해서 training", open=False):
            model_pt_file = gr.File(
                label="모델(.pt) 업로드",
                file_types=[".pt"],
                file_count="single",
            )
            model_pt_path = gr.Textbox(
                label="서버 저장 경로 (model.pt)",
                interactive=False,
            )

        data_yaml_file.change(
            fn=lambda f: save_uploaded_file(f, UPLOAD_DATA_DIR),
            inputs=[data_yaml_file],
            outputs=[data_yaml_path],
        )
        model_pt_file.change(
            fn=lambda f: save_uploaded_file(f, UPLOAD_MODEL_DIR),
            inputs=[model_pt_file],
            outputs=[model_pt_path],
        )

        with gr.Row():
            monitor_imgsz = gr.Slider(256, 2048, 640, step=64, label="imgsz")
            monitor_epochs = gr.Slider(1, 500, 11, step=1, label="epochs")

        with gr.Row():
            monitor_batch = gr.Slider(1, 128, 16, step=1, label="batch")
            monitor_lr0 = gr.Number(0.001, label="lr0")

        with gr.Row():
            btn_start_train = gr.Button("학습 시작 (CLI)", variant="primary")
            btn_stop_train = gr.Button("학습 강제 종료", variant="stop")

        log_box = build_log_textbox(label="학습 로그", lines=20)

        # ===== 이벤트 =====

        btn_start_train.click(
            fn=lambda: 0,
            outputs=[epoch_tick],
            queue=False,
        )

        btn_start_train.click(
            fn=trainer.start_train_stream,
            inputs=[
                task,
                data_yaml_path,
                model_pt_path,
                monitor_imgsz,
                monitor_epochs,
                monitor_batch,
                monitor_lr0,
            ],
            outputs=[log_box, epoch_tick, results_csv_path],
        )

        btn_stop_train.click(
            fn=lambda: trainer.stop_train(),
            outputs=[log_box],
        )

    return {
        "task": task,
    }
