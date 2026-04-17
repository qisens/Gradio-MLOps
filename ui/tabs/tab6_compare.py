# ui/tabs/tab6_compare.py
import gradio as gr
import os
from core.file_browser import list_dir, join_path, parent_dir, filter_files, IMAGE_EXTS, MODEL_EXTS
from core.inf_conf import compare_infer_two_models


def build_compare_tab(
    default_img_dir: str,
    default_model_dir: str,
):
    """
    6번째 탭 UI 구성
    """
    with gr.Row():
        # ======================
        # Left: 원본 이미지 선택 + 실행 버튼
        # ======================
        with gr.Column(scale=2):
            gr.Markdown("### 원본 이미지 선택")

            img_cur = gr.Textbox(label="현재 경로", value=default_img_dir)
            with gr.Row():
                img_btn_up = gr.Button("상위 폴더")
                img_btn_refresh = gr.Button("새로고침")

            img_dirs = gr.Dropdown(label="폴더", choices=[], interactive=True)
            img_files = gr.Dropdown(label="이미지 파일", choices=[], interactive=True)

            img_selected = gr.Textbox(label="선택된 이미지 경로", interactive=False)

            btn_compare = gr.Button("추론결과 비교 버튼", variant="primary")
            status = gr.Textbox(label="상태", interactive=False)

            gr.Markdown("### 추론 파라미터")
            with gr.Row():
                imgsz = gr.Slider(label="imgsz", minimum=256, maximum=2048, step=64, value=640)
                conf_thres = gr.Slider(label="conf", minimum=0.0, maximum=1.0, step=0.01, value=0.25)
                iou_thres = gr.Slider(label="iou", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
            device = gr.Textbox(label="device", value="0")

        # ======================
        # Middle: 기존 모델 + 결과
        # ======================
        with gr.Column(scale=3):
            gr.Markdown("### 기존 모델 경로 선택")

            old_cur = gr.Textbox(label="현재 경로", value=default_model_dir)
            with gr.Row():
                old_btn_up = gr.Button("상위 폴더", key="old_up")
                old_btn_refresh = gr.Button("새로고침", key="old_refresh")

            old_dirs = gr.Dropdown(label="폴더", choices=[], interactive=True)
            old_files = gr.Dropdown(label="모델 파일(.pt)", choices=[], interactive=True)

            old_model_path = gr.Textbox(label="선택된 기존 모델 경로", interactive=False)

            old_vis = gr.Image(label="기존 모델의 추론결과 이미지", type="numpy")
            old_conf_table = gr.Dataframe(
                headers=["cls", "count", "conf_mean", "conf_min", "conf_max"],
                datatype=["number", "number", "number", "number", "number"],
                label="객체별 conf 요약(기존)",
                interactive=False
            )

        # ======================
        # Right: 최신 모델 + 결과
        # ======================
        with gr.Column(scale=3):
            gr.Markdown("### 최신 모델 경로 선택")

            new_cur = gr.Textbox(label="현재 경로", value=default_model_dir)
            with gr.Row():
                new_btn_up = gr.Button("상위 폴더", key="new_up")
                new_btn_refresh = gr.Button("새로고침", key="new_refresh")

            new_dirs = gr.Dropdown(label="폴더", choices=[], interactive=True)
            new_files = gr.Dropdown(label="모델 파일(.pt)", choices=[], interactive=True)

            new_model_path = gr.Textbox(label="선택된 최신 모델 경로", interactive=False)

            new_vis = gr.Image(label="최신 모델의 추론결과 이미지", type="numpy")
            new_conf_table = gr.Dataframe(
                headers=["cls", "count", "conf_mean", "conf_min", "conf_max"],
                datatype=["number", "number", "number", "number", "number"],
                label="객체별 conf 요약(최신)",
                interactive=False
            )

    # -----------------------
    # 공통: 브라우저 갱신 함수
    # -----------------------
    def _refresh_browser(cur_path: str, mode: str):
        cur, dirs, files = list_dir(cur_path)
        if mode == "image":
            files = filter_files(files, IMAGE_EXTS)
        elif mode == "model":
            files = filter_files(files, MODEL_EXTS)
        return (
            cur,
            gr.update(choices=dirs, value=None),
            gr.update(choices=files, value=None),
        )

    def _enter_dir(cur_path: str, dir_name: str, mode: str):
        if not dir_name:
            return _refresh_browser(cur_path, mode)
        nxt = join_path(cur_path, dir_name)
        return _refresh_browser(nxt, mode)

    def _go_up(cur_path: str, mode: str):
        return _refresh_browser(parent_dir(cur_path), mode)

    def _pick_file(cur_path: str, file_name: str):
        if not cur_path or not file_name:
            return ""
        return os.path.join(cur_path, file_name)

    # -----------------------
    # 이미지 탐색 이벤트
    # -----------------------
    img_btn_refresh.click(
        fn=lambda p: _refresh_browser(p, "image"),
        inputs=[img_cur],
        outputs=[img_cur, img_dirs, img_files],
    )
    img_btn_up.click(
        fn=lambda p: _go_up(p, "image"),
        inputs=[img_cur],
        outputs=[img_cur, img_dirs, img_files],
    )
    img_dirs.change(
        fn=lambda p, d: _enter_dir(p, d, "image"),
        inputs=[img_cur, img_dirs],
        outputs=[img_cur, img_dirs, img_files],
    )
    img_files.change(
        fn=_pick_file,
        inputs=[img_cur, img_files],
        outputs=[img_selected],
    )

    # -----------------------
    # 기존 모델 탐색 이벤트
    # -----------------------
    old_btn_refresh.click(
        fn=lambda p: _refresh_browser(p, "model"),
        inputs=[old_cur],
        outputs=[old_cur, old_dirs, old_files],
    )
    old_btn_up.click(
        fn=lambda p: _go_up(p, "model"),
        inputs=[old_cur],
        outputs=[old_cur, old_dirs, old_files],
    )
    old_dirs.change(
        fn=lambda p, d: _enter_dir(p, d, "model"),
        inputs=[old_cur, old_dirs],
        outputs=[old_cur, old_dirs, old_files],
    )
    old_files.change(
        fn=_pick_file,
        inputs=[old_cur, old_files],
        outputs=[old_model_path],
    )

    # -----------------------
    # 최신 모델 탐색 이벤트
    # -----------------------
    new_btn_refresh.click(
        fn=lambda p: _refresh_browser(p, "model"),
        inputs=[new_cur],
        outputs=[new_cur, new_dirs, new_files],
    )
    new_btn_up.click(
        fn=lambda p: _go_up(p, "model"),
        inputs=[new_cur],
        outputs=[new_cur, new_dirs, new_files],
    )
    new_dirs.change(
        fn=lambda p, d: _enter_dir(p, d, "model"),
        inputs=[new_cur, new_dirs],
        outputs=[new_cur, new_dirs, new_files],
    )
    new_files.change(
        fn=_pick_file,
        inputs=[new_cur, new_files],
        outputs=[new_model_path],
    )

    # -----------------------
    # 비교 버튼: 두 모델 추론
    # -----------------------
    btn_compare.click(
        fn=lambda ip, om, nm, s, c, i, d: compare_infer_two_models(
            img_path=ip,
            old_model_path=om,
            new_model_path=nm,
            imgsz=int(s),
            conf_thres=float(c),
            iou_thres=float(i),
            device=str(d).strip() or "0",
        ),
        inputs=[img_selected, old_model_path, new_model_path, imgsz, conf_thres, iou_thres, device],
        outputs=[old_vis, new_vis, old_conf_table, new_conf_table, status],
    )

    # 초기 1회 갱신(사용자가 버튼 누르기 전 기본 목록이 보이게)
    img_btn_refresh.click(fn=lambda p: _refresh_browser(p, "image"), inputs=[img_cur], outputs=[img_cur, img_dirs, img_files])
    old_btn_refresh.click(fn=lambda p: _refresh_browser(p, "model"), inputs=[old_cur], outputs=[old_cur, old_dirs, old_files])
    new_btn_refresh.click(fn=lambda p: _refresh_browser(p, "model"), inputs=[new_cur], outputs=[new_cur, new_dirs, new_files])


def build_tab6_compare(img_dir, model_dir):
    with gr.Tab("6. 모델 추론결과 비교"):
        build_compare_tab(
            default_img_dir=img_dir,
            default_model_dir=model_dir,
        )