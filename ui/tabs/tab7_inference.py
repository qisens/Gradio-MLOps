# ui/tabs/tab7_inference.py
import gradio as gr
import os
from core.file_browser import IMAGE_EXTS
from ui.tabs._ui_shared import build_path_dropdown_selector, build_markdown_log_box, build_log_textbox
from core.utilities import build_folder_picker
from core.config import PROJECT_ROOT
from core.inf_conf import _predict_one
import cv2
import shutil
from pathlib import Path
from ui.shared.js_assets import save_polygons_for_editor_from_seg_txt #json 만들기 위함

from datetime import datetime
def get_today_ymd():
    return datetime.now().strftime("%y%m%d")

def build_inference_tab(
    default_img_dir: str,
    default_model_dir: str,
):
    """
    7번째 탭 UI 구성
    """
    with gr.Row():
        # ======================
        # Left: 원본 이미지 선택 + 실행 버튼
        # ======================
        with gr.Column(scale=2):

            with gr.Column():
                gr.Markdown("### 이미지 폴더 선택하기")
                eval_img_path_tb, _, _, _ = build_folder_picker(
                    label="평가 이미지 폴더",
                    root_dir=os.path.join(PROJECT_ROOT, "test_img"),
                    default_path=os.path.join(PROJECT_ROOT, "test_img"),
                )

            with gr.Column():
                gr.Markdown("### 모델 경로 선택하기 (best.pt 모델 사용)")
                weight_selector = build_path_dropdown_selector(label="weights 폴더 선택")
                weights_dropdown = weight_selector["dropdown"]
                weights_map_state = weight_selector["map_state"]
                weights_dir_tb = weight_selector["path_output"]

            with gr.Accordion(label="추론 파라미터 선택", open=False):
                eval_imgsz_slider = gr.Slider(256, 2048, step=64, value=640, label="imgsz")
                eval_conf_tb = gr.Number(value=0.25, label="conf")
                eval_iou_tb = gr.Number(value=0.5, label="iou")
                eval_device_tb = gr.Textbox(value="0", label="device")

            btn_infer = gr.Button("추론 시작", variant="primary")
            progress_md = gr.Markdown("⏳ inference 대기중...")
            infer_log_tb = build_log_textbox(label="추론 상태 로그", lines=15)

        with gr.Column(scale=3):
            gr.Markdown("### inference 결과")
            viewer_orig_name = build_log_textbox(label="현재 파일명", lines=1)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 원본 이미지")
                    viewer_orig_img = gr.Image(type="numpy", label="원본 이미지")

                with gr.Column(scale=1):
                    gr.Markdown("### 추론 결과")
                    viewer_infer_img = gr.Image(type="numpy", label="추론 결과")

            with gr.Row():
                btn_prev_img = gr.Button("이전")
                btn_next_img = gr.Button("다음")

            gr.Markdown("### 복사할 이미지 체크 - 이미지와 txt가 저장됩니다.")
            with gr.Row():
                btn_mark_bad = gr.Button("복사할 이미지로 체크")
                btn_unmark_bad = gr.Button("체크 해제")
                btn_save_bad = gr.Button("선택한 이미지 저장")
            with gr.Row():
                bad_list_md = build_log_textbox(label="선택된 이미지 리스트", lines=10)


    ''' state '''
    server_img_dir_state = gr.State()  # 원본 이미지 폴더
    server_infer_dir_state = gr.State()  # inference 결과 폴더

    viewer_state = gr.State()  # tab1의 SourceState

    ''' event '''
    def infer_folder(
            img_dir: str,
            weights_dir: str,
            imgsz: int,
            conf: float,
            iou: float,
            device: str,
    ):
        logs = []

        if not os.path.isdir(img_dir):
            yield "❌ 이미지 폴더가 유효하지 않습니다.", "", "", "❌"
            return

        model_path = os.path.join(weights_dir, "best.pt")
        if not os.path.exists(model_path):
            yield "❌ best.pt가 존재하지 않습니다.", "", "", "❌"
            return

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])

        total = len(img_files)
        if total == 0:
            yield "❌ 이미지 없음", "", "", "❌"
            return

        img_save_dir, txt_save_dir = build_inf_save_dir(
            PROJECT_ROOT,
            img_folder=img_dir,
            model_path=model_path,
        )

        logs.append(f"📁 이미지 저장 위치: {img_save_dir}")
        yield "\n".join(logs), img_dir, img_save_dir, f"inference 시작 - 0 / {total}"

        for i, fname in enumerate(img_files, 1):
            img_path = os.path.join(img_dir, fname)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                logs.append(f"{fname}: ❌ 로드 실패")
                yield "\n".join(logs), img_dir, img_save_dir, f"inference 에러 - {i} / {total}"
                continue

            vis, summary, res = _predict_one(
                model_path=model_path,
                img_bgr=img_bgr,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
            )

            # 1. overlay 이미지 저장
            img_out_path = os.path.join(img_save_dir, fname)
            cv2.imwrite(img_out_path, vis[:, :, ::-1])

            # 2. txt 저장
            txt_name = os.path.splitext(fname)[0] + ".txt"
            txt_out_path = os.path.join(txt_save_dir, txt_name)
            save_yolo_txt_from_res(res, txt_out_path)

            logs.append(f"{fname}: ✅ inference 완료")
            yield "\n".join(logs), img_dir, img_save_dir, f"inference 진행중 - {i} / {total}"

        logs.append("🎉 inference 완료")
        yield "\n".join(logs), img_dir, img_save_dir, f"inference 완료 - {total} / {total}"

    def build_inf_save_dir(project_root: str, img_folder: str, model_path: str):
        # dataset name
        dataset_name = os.path.basename(os.path.normpath(img_folder))

        # model info
        # .../demo_exp20_epoch200/weights/best.pt
        model_name = os.path.splitext(os.path.basename(model_path))[0]  # best
        train_name = os.path.basename(
            os.path.dirname(os.path.dirname(model_path))
        )  # demo_exp20_epoch200

        dir_name = f"{dataset_name}__{train_name}__{model_name}"

        save_root = os.path.join(project_root, "7_inference", "inf_results", dir_name)
        img_save_dir = os.path.join(save_root, "images")
        txt_save_dir = os.path.join(save_root, "labels")

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(txt_save_dir, exist_ok=True)

        return img_save_dir, txt_save_dir

    def init_infer_view(orig_dir: str, infer_dir: str):
        imgs = sorted([
            f for f in os.listdir(infer_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])

        if not imgs:
            return None, None, "", "❌ 결과 이미지 없음", {
                "orig_dir": orig_dir,
                "infer_dir": infer_dir,
                "images": [],
                "idx": 0,
                "bad_images": [],
            }

        first = imgs[0]

        orig_img = cv2.imread(os.path.join(orig_dir, first))
        infer_img = cv2.imread(os.path.join(infer_dir, first))

        return (
            orig_img[:, :, ::-1] if orig_img is not None else None,
            infer_img[:, :, ::-1] if infer_img is not None else None,
            first,
            {
                "orig_dir": orig_dir,
                "infer_dir": infer_dir,
                "images": imgs,
                "idx": 0,
                "bad_images": [],
            }
        )

    # yolo txt 저장
    def save_yolo_txt_from_res(res, txt_path: str):
        """
        Ultralytics YOLO segmentation 결과(res) → YOLO seg txt 저장
        포맷: class conf x1 y1 x2 y2 ...
        """
        if res.masks is None or res.boxes is None:
            open(txt_path, "w").close()
            return

        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        polygons = res.masks.xyn  # normalized polygon

        lines = []
        for cls, conf, poly in zip(cls_ids, confs, polygons):
            if poly is None or len(poly) < 3:
                continue

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lines.append(f"{cls} {conf:.6f} {coords}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

    # json 생성
    def generate_json_from_img_txt(
            img_path: str,
            txt_path: str,
            json_out_path: str,
            conf_threshold: float = 0.25,
    ):
        """
        inference img + txt → polygon json 생성
        """
        if not os.path.exists(img_path):
            return False, f"[JSON] 이미지 없음: {os.path.basename(img_path)}"

        if not os.path.exists(txt_path):
            return False, f"[JSON] txt 없음: {os.path.basename(txt_path)}"

        json_path, _ = save_polygons_for_editor_from_seg_txt(
            image_path=img_path,
            txt_path=txt_path,
            classes_txt_path=None,  # 필요하면 나중에 추가
            json_path=json_out_path,
            conf_threshold=conf_threshold,
            assume_normalized="auto",
        )

        return True, json_path

    # 이미지 이전 / 다음
    def viewer_move(step: int, state: dict):
        if not state or not state.get("images"):
            return state, None, None, "", ""

        images = state["images"]
        idx = state["idx"]

        new_idx = max(0, min(idx + step, len(images) - 1))
        fname = images[new_idx]

        orig_path = os.path.join(state["orig_dir"], fname)
        infer_path = os.path.join(state["infer_dir"], fname)

        orig_img = cv2.imread(orig_path)
        infer_img = cv2.imread(infer_path)

        state["idx"] = new_idx

        return (
            state,
            orig_img[:, :, ::-1] if orig_img is not None else None,
            infer_img[:, :, ::-1] if infer_img is not None else None,
            fname,  # orig name
        )

    def on_prev(state: dict):
        return viewer_move(-1, state)

    def on_next(state: dict):
        return viewer_move(1, state)

    # bad img 선택
    def mark_bad(state: dict):
        if not state or not state.get("images"):
            return state, "선택된 이미지 없음"

        state.setdefault("bad_images", [])

        fname = state["images"][state["idx"]]
        if fname not in state["bad_images"]:
            state["bad_images"].append(fname)

        return state, render_bad_list(state)

    def unmark_bad(state: dict):
        if not state or not state.get("images"):
            return state, "선택된 이미지 없음"

        state.setdefault("bad_images", [])

        fname = state["images"][state["idx"]]
        if fname in state["bad_images"]:
            state["bad_images"].remove(fname)

        return state, render_bad_list(state)

    def save_bad_images(state: dict):
        if not state or not state.get("bad_images"):
            return "⚠️ 저장할 이미지가 없습니다."

        orig_dir = state["orig_dir"]
        infer_dir = Path(state["infer_dir"])
        labels_dir = infer_dir.parent / "labels"
        labels_dir = str(labels_dir)

        dataset_name = os.path.basename(os.path.normpath(orig_dir))
        date_tag = datetime.now().strftime("%y%m%d")
        dataset_name_with_date = f"{dataset_name}_{date_tag}"

        save_root = os.path.join(PROJECT_ROOT, "7_inference", "bad_cases")
        save_img_dir = os.path.join(save_root, dataset_name_with_date, "images")
        save_txt_dir = os.path.join(save_root, dataset_name_with_date, "labels")
        save_json_dir = os.path.join(save_root, dataset_name_with_date, "json")
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_txt_dir, exist_ok=True)
        os.makedirs(save_json_dir, exist_ok=True)

        logs = []
        count = 0

        for fname in state["bad_images"]:
            src_img = os.path.join(orig_dir, fname)
            dst_img = os.path.join(save_img_dir, fname)

            txt_fname = fname.replace(".jpg", ".txt")
            src_txt = os.path.join(labels_dir, txt_fname)
            dst_txt = os.path.join(save_txt_dir, txt_fname)

            json_fname = fname.replace(".jpg", ".json")
            dst_json = os.path.join(save_json_dir, json_fname)

            if not os.path.exists(src_img):
                logs.append(f"[MISSING IMAGE] {fname}")
                continue

            if not os.path.exists(src_txt):
                logs.append(f"[MISSING TXT] {txt_fname}")
                continue

            # 1. copy image
            shutil.copy2(src_img, dst_img)

            # 2. copy txt
            shutil.copy2(src_txt, dst_txt)

            # 3. generate json
            ok, msg = generate_json_from_img_txt(
                img_path=dst_img,
                txt_path=dst_txt,
                json_out_path=dst_json,
                conf_threshold=0.25,
            )

            if not ok:
                logs.append(msg)
                continue

            count += 1

        summary = f"✅ 이미지 {count}개 복사 완료\n📁 {save_img_dir}"

        if logs:
            summary += "\n\n⚠️ 로그:\n" + "\n".join(logs)

        return summary

    def render_bad_list(state: dict):
        if not state or not state.get("bad_images"):
            return "선택된 이미지 없음"

        lines = "\n".join([f"- {f}" for f in state["bad_images"]])
        return f"**총 {len(state['bad_images'])}개 선택됨**\n\n{lines}"

    ''' event 등록 '''
    btn_infer.click(
        fn=infer_folder,
        inputs=[
            eval_img_path_tb,
            weights_dir_tb,
            eval_imgsz_slider,
            eval_conf_tb,
            eval_iou_tb,
            eval_device_tb,
        ],
        outputs=[
            infer_log_tb,
            server_img_dir_state,
            server_infer_dir_state,
            progress_md,
        ],
    ).then(
        fn=init_infer_view,
        inputs=[
            server_img_dir_state,
            server_infer_dir_state,
        ],
        outputs=[
            viewer_orig_img,
            viewer_infer_img,
            viewer_orig_name,
            viewer_state,
        ]
    )

    # 이미지 이전 / 다음
    btn_prev_img.click(
        fn=on_prev,
        inputs=[viewer_state],
        outputs=[
            viewer_state,
            viewer_orig_img,
            viewer_infer_img,
            viewer_orig_name,
        ],
    )

    btn_next_img.click(
        fn=on_next,
        inputs=[viewer_state],
        outputs=[
            viewer_state,
            viewer_orig_img,
            viewer_infer_img,
            viewer_orig_name,
        ],
    )

    # bad img 선택
    btn_mark_bad.click(
        fn=mark_bad,
        inputs=[viewer_state],
        outputs=[viewer_state, bad_list_md],
    )

    btn_unmark_bad.click(
        fn=unmark_bad,
        inputs=[viewer_state],
        outputs=[viewer_state, bad_list_md],
    )

    btn_save_bad.click(
        fn=save_bad_images,
        inputs=[viewer_state],
        outputs=[infer_log_tb],  # 로그창 재활용
    )

    return {
        "weights_selector": {
            "dropdown": weight_selector["dropdown"],
            "map_state": weight_selector["map_state"],
            "path": weight_selector["path_output"],  # ⭐ alias 맞추기
        }
    }


def build_tab7_inference():
    with gr.Tab("7. 모델 inference"):
        tab7 = build_inference_tab(
            # default_img_dir="/home/gpuadmin/seongje_maixcam/yolo11_seg_dataset/images/val",
            # default_model_dir="/home/gpuadmin/seongje_gradio2/test_yolo_project/runs/segment",
            default_img_dir="/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/test_img",
            default_model_dir="/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/runs/segment",
        )
        return tab7