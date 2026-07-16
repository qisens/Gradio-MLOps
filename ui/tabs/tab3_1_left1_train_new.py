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

        with gr.Accordion(label="모델 선택해서 training(새로 트레이닝하는 경우 필요없는과정)", open=False):
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
            # 초보자가 파라미터의 의미와 권장 수치를 알 수 있도록 info 속성을 추가하여 툴팁([?])을 제공합니다.
            monitor_imgsz = gr.Slider(label="imgsz", minimum=256, maximum=2048, step=64, value=640,
                                      info="기본값 : 640 (객체 크기에 따라 조정)")
            monitor_epochs = gr.Slider(label="epochs", minimum=1, maximum=500, step=1, value=100,
                                       info="성능 수렴 시 Early Stopping 자동 적용")
        with gr.Row():
            # Batch 사이즈 설정 시 OOM(Out of Memory) 에러를 방지하기 위해 권장 설정을 안내합니다.
            monitor_batch = gr.Slider(label="batch", minimum=1, maximum=128, step=1, value=8,
                                      info="VRAM OOM 에러 방지를 위해 하단의 'Batch 최적화' 스캔 권장")
            monitor_lr0 = gr.Number(label="lr0", value=0.001,
                                    info="권장: 0.001 (AdamW 옵티마이저 기준)")


        with gr.Row():
            # 시스템 VRAM을 감지하여 안전한 Batch 사이즈를 자동 세팅하기 위한 신규 버튼입니다.
            btn_auto_batch = gr.Button("🔍 시스템 VRAM 스캔 및 Batch 최적화", variant="secondary")

        with gr.Row():
            btn_start_train = gr.Button("학습 시작 (CLI)", variant="primary")
            btn_stop_train = gr.Button("학습 강제 종료", variant="stop")


        # ===== 이벤트 =====
        btn_auto_batch.click(
            fn=auto_recommend_batch,
            inputs=[monitor_imgsz],  # 현재 설정된 이미지 사이즈를 입력으로 받음
            outputs=[monitor_batch]  # 계산된 최적의 Batch 사이즈를 슬라이더에 반영
        )

        start_evt = btn_start_train.click(
            fn=lambda: 0,
            outputs=[epoch_tick],
            queue=False,
        )
        start_evt = start_evt.then(
            fn=sanitize_segmentation_labels,
            inputs=[data_yaml_path],
        )
        start_evt.then(
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


def auto_recommend_batch(imgsz):
    """
    현재 시스템의 GPU 여유 메모리(VRAM)를 감지하여,
    설정된 이미지 사이즈(imgsz)에 맞는 안전한 권장 Batch 사이즈를 계산하고 반환합니다.
    """
    import torch
    try:
        # 1. GPU 사용 가능 여부 확인
        if not torch.cuda.is_available():
            gr.Warning("⚠️ GPU를 찾을 수 없습니다. 기본값 8로 설정합니다.")
            return 8

        # 2. 시스템 여유 메모리(free)와 전체 메모리(total)를 획득 (단위: 바이트)
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_mem_gb = free_mem / (1024 ** 3)  # GB 단위로 변환

        # 3. 이미지 크기(imgsz)에 따른 1배치당 VRAM 소모량 추정
        # (YOLO Segmentation 모델 기준, imgsz 640일 때 배치 1당 대략 0.3GB 소모한다고 가정)
        mem_per_sample = (imgsz / 640) ** 2 * 0.3

        # 4. 시스템 안정성 및 백그라운드 프로세스를 고려하여 잔여 메모리의 85%만 가용 영역으로 산정 (OOM 보호)
        safe_mem = free_mem_gb * 0.85

        # 5. 안전 가용 메모리를 샘플당 소모량으로 나누어 최대 허용 배치 사이즈 계산
        rec_batch = int(safe_mem / mem_per_sample)

        # 6. 하드웨어 효율과 YOLO 권장 설정에 맞게 2의 제곱수로 하향 정렬 (2, 4, 8, 16, 32, 64, 128)
        pow_2 = [2, 4, 8, 16, 32, 64, 128]
        # 계산된 rec_batch보다 작거나 같은 2의 제곱수 중 가장 큰 값을 선택, 최소값은 2로 보장
        best_batch = max([b for b in pow_2 if b <= rec_batch], default=2)

        # 7. 사용자에게 성공적으로 계산되었음을 알리는 팝업 안내
        gr.Info(f"🔍 시스템 여유 VRAM: {free_mem_gb:.1f}GB. OOM 방지를 위해 Batch를 {best_batch}(으)로 자동 최적화했습니다.")
        return best_batch

    except Exception as e:
        # 에러 발생 시 시스템 다운을 막고 기본값 반환
        gr.Warning(f"⚠️ VRAM 감지 실패: {str(e)} (안전 기본값 8을 유지합니다.)")
        return 8

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