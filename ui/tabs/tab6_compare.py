# ui/tabs/tab6_compare.py
import gradio as gr
import os
import pandas as pd
from core.file_browser import list_dir, join_path, parent_dir, filter_files, IMAGE_EXTS, MODEL_EXTS
from core.inf_conf import compare_infer_two_models, summarize_conf_for_dir


def build_single_img_compare_tab(
    default_img_dir: str,
    default_model_dir: str,
):
    """
    6-1 탭 UI 구성
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
            device=str(d).strip() or "cuda:0"
        ),
        inputs=[img_selected, old_model_path, new_model_path, imgsz, conf_thres, iou_thres, device],
        outputs=[old_vis, new_vis, old_conf_table, new_conf_table, status],
    )

    # 초기 1회 갱신(사용자가 버튼 누르기 전 기본 목록이 보이게)
    img_btn_refresh.click(fn=lambda p: _refresh_browser(p, "image"), inputs=[img_cur], outputs=[img_cur, img_dirs, img_files])
    old_btn_refresh.click(fn=lambda p: _refresh_browser(p, "model"), inputs=[old_cur], outputs=[old_cur, old_dirs, old_files])
    new_btn_refresh.click(fn=lambda p: _refresh_browser(p, "model"), inputs=[new_cur], outputs=[new_cur, new_dirs, new_files])

#새로 작성한 부분
# ============================================================
# 6-2) Directory Summary Compare (새로 추가)
# ============================================================
def build_dir_compare_tab(
    default_img_dir: str,
    default_model_dir: str,
):
    """
    6-2 탭 UI 구성 (권장 안정 버전)
    - 폴더 이동은 자유
    - [이 폴더를 비교 대상으로 설정] 버튼으로 선택 확정
    - 6-1의 '파일 선택'과 동일한 확정 구조
    """

    with gr.Row():
        # ======================
        # Left: 이미지 폴더 탐색 + 선택 확정
        # ======================
        with gr.Column(scale=2):
            gr.Markdown("### 비교 이미지 폴더 선택")

            img_cur = gr.Textbox(label="현재 경로", value=default_img_dir)

            with gr.Row():
                img_btn_up = gr.Button("상위 폴더")
                img_btn_refresh = gr.Button("새로고침")

            img_dirs = gr.Dropdown(label="폴더", choices=[], interactive=True)

            btn_set_dir = gr.Button("이 폴더를 비교 대상으로 설정")
            img_selected_dir = gr.Textbox(
                label="선택된 이미지 폴더 경로",
                interactive=False,
            )

            gr.Markdown("### 추론 파라미터")
            with gr.Row():
                imgsz = gr.Slider(256, 2048, step=64, value=640, label="imgsz")
                conf_thres = gr.Slider(0.0, 1.0, step=0.01, value=0.25, label="conf")
                iou_thres = gr.Slider(0.0, 1.0, step=0.01, value=0.5, label="iou")

            device = gr.Textbox(label="device", value="0")

            btn_compare = gr.Button("폴더 전체 conf 요약 비교", variant="primary")
            status = gr.Textbox(label="상태", interactive=False)

        # ======================
        # Middle: 기존 모델
        # ======================
        with gr.Column(scale=3):
            gr.Markdown("### 기존 모델 경로 선택")

            old_cur = gr.Textbox(label="현재 경로", value=default_model_dir)
            with gr.Row():
                old_btn_up = gr.Button("상위 폴더", key="old_up_dir")
                old_btn_refresh = gr.Button("새로고침", key="old_refresh_dir")

            old_dirs = gr.Dropdown(label="폴더", choices=[], interactive=True)
            old_files = gr.Dropdown(label="모델 파일(.pt)", choices=[], interactive=True)

            old_model_path = gr.Textbox(label="선택된 기존 모델 경로", interactive=False)

            old_conf_table = gr.Dataframe(
                headers=["cls", "count", "conf_mean", "conf_min", "conf_max"],
                datatype=["number"] * 5,
                label="폴더 전체 conf 요약(기존)",
                interactive=False,
            )

        # ======================
        # Right: 최신 모델
        # ======================
        with gr.Column(scale=3):
            gr.Markdown("### 최신 모델 경로 선택")

            new_cur = gr.Textbox(label="현재 경로", value=default_model_dir)
            with gr.Row():
                new_btn_up = gr.Button("상위 폴더", key="new_up_dir")
                new_btn_refresh = gr.Button("새로고침", key="new_refresh_dir")

            new_dirs = gr.Dropdown(label="폴더", choices=[], interactive=True)
            new_files = gr.Dropdown(label="모델 파일(.pt)", choices=[], interactive=True)

            new_model_path = gr.Textbox(label="선택된 최신 모델 경로", interactive=False)

            new_conf_table = gr.Dataframe(
                headers=["cls", "count", "conf_mean", "conf_min", "conf_max"],
                datatype=["number"] * 5,
                label="폴더 전체 conf 요약(최신)",
                interactive=False,
            )

    # ✅ 전체 평균(ALL) 표시 UI는 Row 밖에 따로 하나 더 만들어도 되고,
    #   위 Row 안에 넣어도 되는데, 보통 보기 좋게 아래에 한 줄 추가함.
    with gr.Row():
        old_overall = gr.Number(
            label="OLD · Overall conf (ALL)",
            interactive=False,
            precision=4,
        )
        new_overall = gr.Number(
            label="NEW · Overall conf (ALL)",
            interactive=False,
            precision=4,
        )
        delta_overall = gr.Number(
            label="Δ (NEW - OLD)",
            interactive=False,
            precision=4,
        )

    # =====================================================
    # 공통 유틸
    # =====================================================
    def _refresh_img(cur_path: str):
        cur, dirs, _ = list_dir(cur_path)
        return cur, gr.update(choices=dirs, value=None)

    def _enter_img_dir(cur_path: str, dir_name: str):
        if not dir_name:
            return _refresh_img(cur_path)
        nxt = join_path(cur_path, dir_name)
        return _refresh_img(nxt)

    def _refresh_model(cur_path: str):
        cur, dirs, files = list_dir(cur_path)
        files = filter_files(files, MODEL_EXTS)
        return cur, gr.update(choices=dirs, value=None), gr.update(choices=files, value=None)

    def _enter_model_dir(cur_path: str, dir_name: str):
        if not dir_name:
            return _refresh_model(cur_path)
        nxt = join_path(cur_path, dir_name)
        return _refresh_model(nxt)

    def _pick_file(cur_path: str, file_name: str):
        if not cur_path or not file_name:
            return ""
        return os.path.join(cur_path, file_name)

    def _extract_overall_mean(df: pd.DataFrame):
        """
        cls = -1 (ALL) 행의 conf_mean 추출
        """
        if df is None or df.empty:
            return None
        row = df[df["cls"] == -1]
        if row.empty:
            return None
        return float(row.iloc[0]["conf_mean"])

    # =====================================================
    # 이미지 폴더 탐색 이벤트
    # =====================================================
    img_btn_refresh.click(
        fn=_refresh_img,
        inputs=[img_cur],
        outputs=[img_cur, img_dirs],
    )

    img_btn_up.click(
        fn=lambda p: _refresh_img(parent_dir(p)),
        inputs=[img_cur],
        outputs=[img_cur, img_dirs],
    )

    img_dirs.change(
        fn=_enter_img_dir,
        inputs=[img_cur, img_dirs],
        outputs=[img_cur, img_dirs],
    )

    # ✅ 선택 확정 (핵심)
    btn_set_dir.click(
        fn=lambda cur: cur,
        inputs=[img_cur],
        outputs=[img_selected_dir],
    )

    # =====================================================
    # 기존 모델 탐색
    # =====================================================
    old_btn_refresh.click(
        fn=_refresh_model,
        inputs=[old_cur],
        outputs=[old_cur, old_dirs, old_files],
    )

    old_btn_up.click(
        fn=lambda p: _refresh_model(parent_dir(p)),
        inputs=[old_cur],
        outputs=[old_cur, old_dirs, old_files],
    )

    old_dirs.change(
        fn=_enter_model_dir,
        inputs=[old_cur, old_dirs],
        outputs=[old_cur, old_dirs, old_files],
    )

    old_files.change(
        fn=_pick_file,
        inputs=[old_cur, old_files],
        outputs=[old_model_path],
    )

    # =====================================================
    # 최신 모델 탐색
    # =====================================================
    new_btn_refresh.click(
        fn=_refresh_model,
        inputs=[new_cur],
        outputs=[new_cur, new_dirs, new_files],
    )

    new_btn_up.click(
        fn=lambda p: _refresh_model(parent_dir(p)),
        inputs=[new_cur],
        outputs=[new_cur, new_dirs, new_files],
    )

    new_dirs.change(
        fn=_enter_model_dir,
        inputs=[new_cur, new_dirs],
        outputs=[new_cur, new_dirs, new_files],
    )

    new_files.change(
        fn=_pick_file,
        inputs=[new_cur, new_files],
        outputs=[new_model_path],
    )

    # =====================================================
    # 비교 실행
    # =====================================================
    def _run_dir_compare(sel_dir, old_m, new_m, s, c, i, d):
        if not sel_dir or not os.path.isdir(sel_dir):
            return [], [], None, None, None, "[ERROR] 비교할 이미지 폴더를 먼저 선택해줘"

        if not old_m or not new_m:
            return [], [], None, None, None, "[ERROR] 두 모델 경로를 모두 선택해줘"

        d = str(d).strip()
        dev = f"cuda:{d}" if d.isdigit() else (d or "cuda:0")

        # --- raw summary (ALL 포함) ---
        old_df_all = summarize_conf_for_dir(
            img_dir=sel_dir,
            model_path=old_m,
            imgsz=int(s),
            conf_thres=float(c),
            iou_thres=float(i),
            device=dev,
        )
        new_df_all = summarize_conf_for_dir(
            img_dir=sel_dir,
            model_path=new_m,
            imgsz=int(s),
            conf_thres=float(c),
            iou_thres=float(i),
            device=dev,
        )

        # --- ALL 값 추출 ---
        old_all = _extract_overall_mean(old_df_all)
        new_all = _extract_overall_mean(new_df_all)

        delta = None
        if old_all is not None and new_all is not None:
            delta = round(new_all - old_all, 4)

        # --- ✅ 테이블용: ALL(-1) 행 제거 ---
        old_df = old_df_all[old_df_all["cls"] != -1].reset_index(drop=True)
        new_df = new_df_all[new_df_all["cls"] != -1].reset_index(drop=True)

        return (
            old_df,
            new_df,
            old_all,
            new_all,
            delta,
            "[OK] 폴더 전체 conf 요약 완료",
        )

    btn_compare.click(
        fn=_run_dir_compare,
        inputs=[
            img_selected_dir,
            old_model_path,
            new_model_path,
            imgsz,
            conf_thres,
            iou_thres,
            device,
        ],
        outputs=[
            old_conf_table,
            new_conf_table,
            old_overall,
            new_overall,
            delta_overall,
            status,
        ],
    )

    # ✅ 여기 3줄이 "마지막 refresh 버튼들" 맞음 (자동 1회 실행)
    img_btn_refresh.click(fn=_refresh_img, inputs=[img_cur], outputs=[img_cur, img_dirs])
    old_btn_refresh.click(fn=_refresh_model, inputs=[old_cur], outputs=[old_cur, old_dirs, old_files])
    new_btn_refresh.click(fn=_refresh_model, inputs=[new_cur], outputs=[new_cur, new_dirs, new_files])

def build_tab6_compare():
    with gr.Tab("6. 모델 추론결과 비교"):
        with gr.Tabs():

            # ---------------------------
            # 내부 탭 1: 단일 이미지 비교
            # ---------------------------
            with gr.TabItem("Single Image Compare"):
                build_single_img_compare_tab(
                    default_img_dir="/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/test_img",
                    default_model_dir="/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/runs/segment",
                )

            # ---------------------------
            # 내부 탭 2: 디렉토리 전체 비교
            # (지금은 자리만 만들어두기)
            # ---------------------------
            with gr.TabItem("Directory Summary Compare"):
                build_dir_compare_tab(
                    default_img_dir="/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/test_img",
                    default_model_dir="/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/runs/segment",
                )
