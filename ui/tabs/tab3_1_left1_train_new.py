# ui/tabs/tab3_1_left1_train_new.py
import gradio as gr
from core.config import UPLOAD_TRAINING_INFO_DIR
from core.utilities import save_uploaded_file
from ui.tabs._ui_shared import build_log_textbox, build_markdown_log_box
import os
import shutil
import yaml

def build_train_new(
    *,
    trainer,
    results_csv_path,
    epoch_tick,
    train_log_message_state,
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
        btn_check_conf = gr.Button("데이터 label에 conf 포함여부 체크")

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
            fn=lambda f: save_uploaded_file(f, UPLOAD_TRAINING_INFO_DIR),
            inputs=[data_yaml_file],
            outputs=[data_yaml_path],
        )
        model_pt_file.change(
            fn=lambda f: save_uploaded_file(f, UPLOAD_TRAINING_INFO_DIR),
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

        # log_box = build_log_textbox(label="학습 로그", lines=20)

        # ===== 이벤트 =====
        btn_check_conf.click(
            fn=sanitize_segmentation_labels,
            inputs=[data_yaml_path]
        )

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
            outputs=[train_log_message_state, epoch_tick, results_csv_path],
        )

        btn_stop_train.click(
            fn=lambda: trainer.stop_train(),
            outputs=[train_log_message_state],
        )

    return {
        "task": task,
    }

def has_conf(parts):
    if len(parts) < 3:
        return False
    try:
        conf = float(parts[1])
        return 0.0 <= conf <= 1.0
    except:
        return False


def sanitize_segmentation_labels(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_path = data['path']

    def get_label_dir(img_rel):
        return os.path.join(base_path, img_rel.replace('images', 'labels'))

    label_dirs = [get_label_dir(data['train'])]
    if 'val' in data:
        label_dirs.append(get_label_dir(data['val']))

    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            continue

        backup_dir = label_dir + "_with_conf"

        # 1️⃣ 백업 (복사)
        if not os.path.exists(backup_dir):
            print(f"[BACKUP] {label_dir} → {backup_dir}")
            shutil.copytree(label_dir, backup_dir)
        else:
            print(f"[SKIP BACKUP] already exists: {backup_dir}")

        changed_files = []

        # 2️⃣ 파일 단위 처리
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(label_dir, fname)

            new_lines = []
            file_changed = False

            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if has_conf(parts):
                        # 🔥 conf 제거 (두 번째 값)
                        parts = [parts[0]] + parts[2:]
                        file_changed = True

                    new_lines.append(" ".join(parts))

            # 3️⃣ 변경된 파일만 overwrite
            if file_changed:
                with open(fpath, "w") as f:
                    f.write("\n".join(new_lines))

                changed_files.append(fname)

        # 4️⃣ 로그 출력
        if changed_files:
            print(f"[MODIFIED] {label_dir}")
            for f in changed_files:
                print(f"  - {f}")
        else:
            print(f"[CLEAN] {label_dir} (수정 필요 없음)")

    print("✅ sanitize 완료")