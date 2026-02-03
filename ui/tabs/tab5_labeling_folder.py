# ui/tabs/tab5_labeling.py
import time, json, base64
import os
import cv2
import tempfile
from pathlib import Path
from io import BytesIO
from PIL import Image
import gradio as gr
import colorsys
from ui.shared.js_assets import save_polygons_for_editor_from_seg_txt
from ui.tabs._ui_shared import build_markdown_log_box, build_log_textbox
from core.config import PROJECT_ROOT
from core.utilities import build_folder_picker


def load_classes_txt(classes_file):
    """
    classes.txt → (class_names, class_colors)
    """
    if not classes_file:
        return [], []

    # gr.File 이 list로 들어오는 경우 방어
    if isinstance(classes_file, list):
        classes_file = classes_file[0] if classes_file else None
    if classes_file is None:
        return [], []

    text = Path(classes_file.name).read_text(encoding="utf-8")
    class_names = [l.strip() for l in text.splitlines() if l.strip()]

    class_colors = []
    for i in range(len(class_names)):
        h = i / max(1, len(class_names))
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 1.0)
        class_colors.append(
            "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
        )

    return class_names, class_colors

def pil_to_b64(img):
    try:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def load_polygon_for_edit(selected_file, image_file, classes_file):
    """
    업로드된 JSON + 이미지로 JS editor에 폴리곤 로드하기
    """

    if selected_file is None or image_file is None:
        return {"status": "error", "message": "JSON + 이미지 필요"}

    class_names, class_colors = load_classes_txt(classes_file) if classes_file else ([], [])

    poly_raw = Path(selected_file.name).read_text(encoding="utf-8")
    img = Image.open(image_file.name).convert("RGB")
    b64img = pil_to_b64(img)

    return {
        "polygon_json": poly_raw,
        "image_b64": b64img,
        "meta": {
            "names": class_names,
            "colors": class_colors
        },
        "_nonce": time.time(),
        "loaded_file_path": selected_file.name
    }

def load_case_folder_and_first(case_folder, classes_file):
    """
    bad_cases/xxx 선택 →
    images, json, labels 경로 자동 설정 →
    첫 번째 케이스 로드
    """
    if not case_folder:
        raise gr.Error("bad_case 폴더를 선택해주세요.")

    base = Path(case_folder)
    img_dir = base / "images"
    json_dir = base / "json"
    lbl_dir = base / "labels"

    if not img_dir.exists() or not json_dir.exists():
        raise gr.Error("images / json 폴더 구조가 올바르지 않습니다.")

    # 이미지 기준 파일 리스트
    images = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]]
    )
    if not images:
        raise gr.Error("images 폴더에 이미지가 없습니다.")

    # 상태용 paths
    paths = {
        "base": str(base),
        "images": str(img_dir),
        "json": str(json_dir),
        "labels": str(lbl_dir),
    }

    # 첫 번째 로드
    idx = 0
    img_path = images[idx]
    json_path = json_dir / f"{img_path.stem}.json"
    cur_text = f"{idx + 1} / {len(images)} : {img_path.name}"

    if not json_path.exists():
        raise gr.Error(f"JSON 없음: {json_path.name}")

    class_names, class_colors = load_classes_txt(classes_file)

    poly_raw = json_path.read_text(encoding="utf-8")
    img = Image.open(img_path).convert("RGB")
    b64img = pil_to_b64(img)

    payload = {
        "polygon_json": poly_raw,
        "image_b64": b64img,
        "meta": {
            "names": class_names,
            "colors": class_colors,
        },
        "_nonce": time.time(),
        "loaded_file_path": str(json_path),
    }

    return paths, [p.name for p in images], idx, payload, cur_text

def load_by_index(cur_index, file_list, case_paths, classes_file):
    if not file_list or not case_paths:
        raise gr.Error("파일 목록이 없습니다.")

    idx = int(cur_index)
    idx = max(0, min(idx, len(file_list) - 1))

    img_name = file_list[idx]
    img_path = Path(case_paths["images"]) / img_name
    json_path = Path(case_paths["json"]) / f"{Path(img_name).stem}.json"

    class_names, class_colors = load_classes_txt(classes_file)

    poly_raw = json_path.read_text(encoding="utf-8")
    img = Image.open(img_path).convert("RGB")
    b64img = pil_to_b64(img)

    payload = {
        "polygon_json": poly_raw,
        "image_b64": b64img,
        "meta": {
            "names": class_names,
            "colors": class_colors,
        },
        "_nonce": time.time(),
        "loaded_file_path": str(json_path),
    }

    # 🔥 여기 추가
    cur_text = f"{idx+1} / {len(file_list)} : {img_name}"

    return idx, payload, cur_text


def save_current_overwrite(cur_index, file_list, case_paths, current_json):
    if not file_list or not case_paths:
        return "❌ 저장할 데이터가 없습니다."

    if current_json is None:
        return "❌ 현재 JSON이 없습니다."

    # 🔥 핵심 1: string → dict 변환
    if isinstance(current_json, str):
        try:
            current_json = json.loads(current_json)
        except Exception as e:
            return f"❌ JSON 파싱 실패: {e}"

    # 🔥 핵심 2: annotations dict 구조 확인
    if not isinstance(current_json, dict) or "annotations" not in current_json:
        return "❌ JSON 구조가 올바르지 않습니다."

    idx = int(cur_index)
    img_name = file_list[idx]
    stem = Path(img_name).stem

    json_path = Path(case_paths["json"]) / f"{stem}.json"
    txt_path  = Path(case_paths["labels"]) / f"{stem}.txt"
    img_path  = Path(case_paths["images"]) / img_name

    # ✅ JSON 저장 (사람이 읽을 수 있는 형태)
    json_path.write_text(
        json.dumps(current_json, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ✅ YOLO TXT 생성
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    lines = []
    for ann in current_json.get("annotations", []):
        cid = int(ann.get("class_id", 0))
        seg = ann.get("segmentation", [])
        if not seg:
            continue

        poly = seg[0]
        coords = []
        for i in range(0, len(poly), 2):
            x = max(0.0, min(1.0, poly[i] / w))
            y = max(0.0, min(1.0, poly[i + 1] / h))
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")

        if coords:
            lines.append(f"{cid} " + " ".join(coords))

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return f"✅ Saved: {json_path.name}, {txt_path.name}"


def build_tab5_labeling_folder(current_tab: gr.State):
    ''' UI 컴포넌트 관련 '''
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### base 경로 선택하기")
            case_folder, _, _, _ = build_folder_picker(
                label="Select base folder",
                root_dir=os.path.join(PROJECT_ROOT, "7_inference/bad_cases"),
                default_path=os.path.join(PROJECT_ROOT, "7_inference/bad_cases"),
            )

            gr.Markdown("### classes.txt 파일 불러오기")
            classes_txt_file = gr.File(
                label="Select classes.txt",
                file_types=[".txt"]
            )

        with gr.Column(scale=1):
            js_log_box = build_log_textbox(label="JS Log", lines=18)

    with gr.Row():
        load_btn = gr.Button(
            "Load Folder case",
            size="lg"
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div>
                <h3>[단축키 & 조작 안내]</h3>
                <ul>
                    <li><b>Click + Drag</b> : 포인트 이동</li>
                    <li><b>Ctrl + Click</b> : 포인트 삭제</li>
                    <li><b>Shift + Click</b> : 포인트 추가</li>
                    <li><b>Ctrl + Z / Y</b> : Undo / Redo</li>
                </ul>
            </div>
            """)

            # NEW polygon controls : Class ID + 버튼 영역
            gr.HTML("새로운 폴리곤 추가 버튼을 눌러 레이블링을 추가하고, 레이블링 완료 후 finish 버튼을 눌러주세요.")
            with gr.Row():
                with gr.Column(scale=2):
                    new_class = gr.Dropdown(
                        choices=[],
                        label="Class ID",
                        value=None
                    )
                    # 아래 칸용 dummy dropdown (높이 맞추기용)
                    gr.Dropdown(
                        choices=[],
                        label="Class ID",
                        visible=False
                    )
                with gr.Column(scale=1):
                    add_mode_btn = gr.Button("➕ New Polygon")
                    finish_poly_btn = gr.Button("✔ Finish Polygon")

            gr.HTML("레이블링 수정 완료 후 저장하기 버튼을 눌러주세요")
            save_btn = gr.Button(
                "💾 Save (JSON + TXT overwrite)",
                size="lg"
            )

    gr.Markdown("### 파일 이동하기")
    with gr.Row():
        prev_btn = gr.Button("⬅ Prev")
        cur_file_text = build_log_textbox(label="Current file", lines=1)

        next_btn = gr.Button("Next ➡")


    ''' 상태 관련 '''
    load_out = gr.JSON(visible=False)
    case_paths = gr.State({})
    file_list = gr.State([])
    cur_index = gr.State(0)


    ''' 버튼 클릭 리스너 및 ui 관련 함수 '''

    def build_class_dropdown(classes_file):
        names, colors = load_classes_txt(classes_file)
        choices = [
            (f"{i:02d} — {names[i]} [{colors[i]}]", i)
            for i in range(len(names))
        ]
        return gr.update(choices=choices, value=0 if choices else None)

    classes_txt_file.change(
        fn=build_class_dropdown,
        inputs=[classes_txt_file],
        outputs=[new_class]
    )

    def update_current_file_text(cur_index, file_list):
        if not file_list:
            return ""
        idx = int(cur_index)
        total = len(file_list)
        name = file_list[idx]
        return f"{idx + 1} / {total} : {name}"

    load_btn.click(
        fn=load_case_folder_and_first,
        inputs=[case_folder, classes_txt_file],
        outputs=[case_paths, file_list, cur_index, load_out, cur_file_text]
    ).then(
        fn=None,
        inputs=load_out,
        outputs=None,
        js="(p) => window.js_editor(p)"
    ).then(
        fn=lambda idx, fl: f"{idx+1} / {len(fl)} : {fl[idx]}",
        inputs=[cur_index, file_list],
        outputs=[cur_file_text]
    )

    #  폴리곤 추가 / 종료
    add_mode_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => add_mode_on()"
    )
    finish_poly_btn.click(
        fn=None,
        inputs=new_class,
        outputs=None,
        js="(cls_value) => finish_poly(cls_value)"
    )

    # 이미지 이전/다음 버튼 클릭
    prev_btn.click(
        fn=lambda i: i - 1,
        inputs=[cur_index],
        outputs=[cur_index]
    ).then(
        fn=load_by_index,
        inputs=[cur_index, file_list, case_paths, classes_txt_file],
        outputs=[cur_index, load_out, cur_file_text]
    ).then(
        fn=None,
        inputs=load_out,
        outputs=None,
        js="(p) => window.js_editor(p)"
    ).then(
        fn=update_current_file_text,
        inputs=[cur_index, file_list],
        outputs=[cur_file_text]
    )

    next_btn.click(
        fn=lambda i: i + 1,
        inputs=[cur_index],
        outputs=[cur_index]
    ).then(
        fn=load_by_index,
        inputs=[cur_index, file_list, case_paths, classes_txt_file],
        outputs=[cur_index, load_out, cur_file_text]
    ).then(
        fn=None,
        inputs=load_out,
        outputs=None,
        js="(p) => window.js_editor(p)"
    ).then(
        fn=update_current_file_text,
        inputs=[cur_index, file_list],
        outputs=[cur_file_text]
    )

    save_btn.click(
        fn=None,
        inputs=None,
        outputs=load_out,  # 🔥 핵심
        js="""
        () => {
            const cur = window.get_current_json();
            return cur ? JSON.parse(JSON.stringify(cur)) : null;
        }
        """
    )

    save_btn.click(
        fn=save_current_overwrite,
        inputs=[cur_index, file_list, case_paths, load_out],
        outputs=[js_log_box]
    )

    def reset_folder_state(tab_name):
        if tab_name == "folder":
            return {}, [], 0, None
        else:
            # 아무 변화 없게 현재 상태 그대로 반환
            return gr.update(), gr.update(), gr.update(), gr.update()

    current_tab.change(
        fn=reset_folder_state,
        inputs=current_tab,
        outputs=[case_paths, file_list, cur_index, load_out],
    )
