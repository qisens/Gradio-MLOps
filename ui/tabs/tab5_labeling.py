# ui/tabs/tab5_labeling.py
import time, json, base64
from io import BytesIO
from PIL import Image
import gradio as gr

from ui.shared.js_assets import save_polygons_for_editor_from_seg_txt

COCO_CLASSES = ["oil", "water", "bubble"]
COCO_COLORS = ["#ff0000", "#00aaff", "#ffaa00"]

def pil_to_b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def load_polygon_for_edit(selected_file, image_file):
    if selected_file is None:
        return json.dumps({"status": "error", "message": "⚠ JSON 파일이 선택되지 않았습니다."})
    if image_file is None:
        return json.dumps({"status": "error", "message": "⚠ 이미지 파일도 함께 업로드해야 합니다."})

    file_path = selected_file.name
    with open(file_path, "r") as f:
        poly_raw = f.read()

    img_path = image_file.name
    img = Image.open(img_path)
    b64img = pil_to_b64(img)

    return {
        "polygon_json": poly_raw,
        "image_b64": b64img,
        "meta": {"colors": COCO_COLORS, "names": COCO_CLASSES},
        "_nonce": time.time(),
        "loaded_file_path": file_path
    }

def gen_json_return_file_and_json(img_f, txt_f, cls_f):
    json_path, out = save_polygons_for_editor_from_seg_txt(
        image_path=img_f.name,
        txt_path=txt_f.name,
        classes_txt_path=cls_f.name if cls_f else None,
        json_path="temp_generated.json",
        conf_threshold=0.25,
        assume_normalized="auto"
    )
    return json_path, out

def build_tab5_labeling():
    from ui.shared.js_assets import load_all_js
    ALL_JS = load_all_js("./json")

    with gr.Tab("5. Labeling"):
        gr.HTML(f"<script>{ALL_JS}</script>")

        with gr.Row():
            load_json_file = gr.File(label="Select polygon JSON file(select only)", file_types=[".json"])
            image_file = gr.File(label="Select image file (.jpg or .png)", file_types=["image"])
            load_btn = gr.Button("Load Polygon + Image + rendering")

            infer_txt_file = gr.File(label="Select inference txt (YOLO seg)", file_types=[".txt"])
            classes_txt_file = gr.File(label="Select classes.txt", file_types=[".txt"])
            gen_btn = gr.Button("Generate JSON from Image+TXT")
            gen_out = gr.JSON(visible=False)
            gen_file = gr.File(label="Download JSON", visible=True)

            gen_btn.click(
                fn=gen_json_return_file_and_json,
                inputs=[image_file, infer_txt_file, classes_txt_file],
                outputs=[gen_file, gen_out]
            ).then(
                fn=None,
                inputs=[gen_out],
                outputs=None,
                js=r"""
                (p) => { window.js_editor(p); return null; }
                """
            )
            # gen_btn.click(
            #     fn=lambda img_f, txt_f, cls_f: save_polygons_for_editor_from_seg_txt(
            #         image_path=img_f.name,
            #         txt_path=txt_f.name,
            #         classes_txt_path=cls_f.name if cls_f else None,
            #         json_path="temp_generated.json",
            #         conf_threshold=0.25,
            #         assume_normalized="auto"
            #     )[1],
            #     inputs=[image_file, infer_txt_file, classes_txt_file],
            #     outputs=gen_out
            # ).then(
            #     fn=None,
            #     inputs=[gen_out],
            #     outputs=None,
            #     js="""
            #     async (p) => {
            #         window.js_editor(p);
            #         const jsonText = (typeof p === "string") ? p : JSON.stringify(p, null, 2);
            #         await window.save_json_via_filepicker("temp_generated.json", jsonText);
            #         return null;
            #     }
            #     """
            # )

        with gr.Row():
            js_log_box = gr.Textbox(label="JS Log", elem_id="js_log_box", lines=8, max_lines=12, interactive=False)

        gr.HTML("""
        <div style="display:flex; justify-content:center; width:100%; margin-top:10px;">
            <canvas id="edit_canvas" width="800" height="600" style="border:1px solid white;"></canvas>
        </div>
        """)

        # ---------------------------------------
        # Load JSON + Attach JS Editor
        # ---------------------------------------
        load_out = gr.JSON(visible=False)
        ev = load_btn.click(fn=load_polygon_for_edit, inputs=[load_json_file, image_file], outputs=load_out)
        ev.then(fn=None, inputs=load_out, outputs=None, js="(p) => window.js_editor(p)")

        # ---------------------------------------
        # save json
        # ---------------------------------------
        with gr.Row():
            save_name = gr.Textbox(label="Save as (filename.json)", value="edited.json")
            save_btn = gr.Button("Save(in json)")

        # ---------------------------------------
        # NEW: polygon 추가 기능
        # ---------------------------------------
        with gr.Row():
            drop_choices = [(f"{i:02d} — {COCO_CLASSES[i]}  [{COCO_COLORS[i]}]", i) for i in range(len(COCO_CLASSES))]
            new_class = gr.Dropdown(choices=drop_choices, label="Class ID for New Polygon", value=0)
            add_mode_btn = gr.Button("➕ New Polygon", scale=1)
            finish_poly_btn = gr.Button("✔ Finish Polygon", scale=1)

            add_mode_btn.click(fn=None, inputs=None, outputs=None, js="() => add_mode_on()")
            finish_poly_btn.click(fn=None, inputs=new_class, outputs=None, js="(cls_value) => finish_poly(cls_value)")

        edited_json_state = gr.State()

        save_btn.click(
            fn=None, inputs=[save_name], outputs=edited_json_state,
            js="""
            (filename) => {
                const cur = window.get_current_json();
                return (typeof cur === "string") ? cur : JSON.stringify(cur, null, 2);
            }
            """
        )
        save_btn.click(
            fn=None, inputs=[save_name, edited_json_state], outputs=None,
            js="""
            (filename, jsonText) => {
                return window.save_json_via_filepicker(filename || "edited.json", jsonText);
            }
            """
        )
