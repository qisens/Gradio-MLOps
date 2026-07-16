import gradio as gr
import pandas as pd
import random

from core.config import PROJECT_ROOT
from core.dataset_service import (
    build_existing_dataset_stats_df,
    list_new_images_for_checkbox_onelevel,
    ensure_out_dataset_root,
    split_new_dataset_by_selection_onelevel,
    copy_existing_dataset_into_final,
    upload_files_to_labeling_dataset
)
from ui.tabs._ui_shared import build_markdown_log_box

def build_tab2_dataset():
    with gr.Tab("2. Dataset 설정"):
        final_out_root_state = gr.State("")  # 실제 저장 루트(out_root/dataset_name)

        with gr.Accordion(label="로컬 업로드", open=False):
            gr.Markdown("### ⬆️ 로컬 업로드 → test_yolo_project/datasets_for_labeling/<폴더명>/images,labels 저장")

            labeling_dataset_name = gr.Textbox(
                label="업로드 저장 폴더명 (datasets_for_labeling 하위 생성)",
                placeholder="예) labeling_20251231_v1",
            )

            local_images = gr.File(
                label="로컬 이미지 업로드(여러개)",
                file_count="multiple",
                file_types=["image"],
            )
            local_txts = gr.File(
                label="로컬 txt 업로드(여러개)",
                file_count="multiple",
                file_types=[".txt"],
            )

            upload_btn = gr.Button("⬆️ 서버로 업로드 저장")
            upload_log = gr.Textbox(label="업로드 로그", lines=8)
            upload_path_view = gr.Textbox(label="생성된 dataset_root", interactive=False)

            def _upload_to_labeling_root(name, imgs, txts):
                log, info = upload_files_to_labeling_dataset(
                    dataset_name=name,
                    img_files=imgs,
                    txt_files=txts,
                    overwrite=True,
                )
                return log, (info.get("dataset_root", "") if info else "")

            upload_btn.click(
                fn=_upload_to_labeling_root,
                inputs=[labeling_dataset_name, local_images, local_txts],
                outputs=[upload_log, upload_path_view],
            )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Column():
                    with gr.Column():
                        use_existing = gr.Radio(
                            ["Yes", "No"],
                            value="Yes",
                            label="기존 데이터셋 활용 여부 체크",
                        )

                        existing_hint = gr.Markdown(
                            "기존 데이터셋을 사용하려면 기존데이터셋 활용여부에서 `Yes`를 선택하세요.",
                            visible=False
                        )

                        existing_dataset_dir = gr.FileExplorer(
                            label="기존 데이터셋 경로 선택 (폴더)",
                            root_dir=PROJECT_ROOT,
                            file_count="single",
                            visible=True
                        )

                new_dataset_dir = gr.FileExplorer(
                    label="이번 데이터셋 경로 선택 (폴더) - (images/, labels/ 한 레벨)",
                    root_dir=PROJECT_ROOT,
                    file_count="single",
                )

                out_dataset_dir = gr.FileExplorer(
                    label="신규 데이터셋 저장 상위 경로 선택 (폴더)",
                    root_dir=PROJECT_ROOT,
                    file_count="single",
                )

                dataset_name = gr.Textbox(
                    label="신규 데이터셋 폴더명 (out 상위경로 하위에 생성)",
                    placeholder="예) dataset_20251231_v1",
                )

                create_out_btn = gr.Button("📁 신규 데이터셋 저장 폴더 생성/확인")



            with gr.Column(scale=3):
                log_box = gr.Textbox(
                    label="검증 / 실행 로그",
                    lines=10,
                    interactive=False,
                    elem_id="log_box"
                )

                existing_stats_df = gr.Dataframe(label="기존 데이터셋 통계", interactive=False)

                gr.Markdown("⭐ 신규 이미지 목록 ⭐")

                new_choices_state = gr.State([])
                with gr.Row():
                    select_all_btn = gr.Button("전체 선택")
                    deselect_all_btn = gr.Button("전체 해제")

                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 🎲 데이터 자동 랜덤 분할 (Smart Split)")
                        # 버튼 하나로 체크박스를 자동 조작
                        random_split_btn = gr.Button("🪄 랜덤 분할 적용", variant="secondary")
                    # Train 데이터의 비율을 정하는 슬라이더 (0.0 ~ 1.0)
                    train_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                        label="Train 비율 설정 (나머지는 Val)"
                    )

                new_list_box = gr.CheckboxGroup(
                    label="체크 = Train dataset / 미체크 = Val dataset",
                    choices=[],
                    value=[],
                )

                selection_stats = gr.Markdown(
                    "**Train:** 0 | **Val:** 0 | **Total:** 0"
                )

                out_root_view= build_markdown_log_box(
                    title="최종 저장 경로",
                    value="왼쪽에서 저장 경로를 설정해 주세요.",
                )

                split_btn = gr.Button("✅ 체크 기준으로 train/val 분할 복사 실행")

                split_result_df = gr.Dataframe(label="분할/복사 결과", interactive=False)

        # 1) 기존 데이터셋 사용 여부에 따라 existing 경로 활성/비활성
        def _toggle_existing(v):
            if v == "Yes":
                return (
                    gr.update(visible=False),  # hint 숨김
                    gr.update(visible=True),  # explorer 표시
                )
            else:
                return (
                    gr.update(visible=True),  # hint 표시
                    gr.update(visible=False),  # explorer 숨김
                )

        use_existing.change(
            _toggle_existing,
            inputs=use_existing,
            outputs=[existing_hint, existing_dataset_dir]
        )

        # 2) 기존 데이터셋 경로 설정되면 통계표 로드
        def _load_existing_stats(use, ex_root):
            if use != "Yes":
                df = pd.DataFrame([{"split": "train", "count": 0}, {"split": "val", "count": 0}])
                return df, "[INFO] 기존 데이터셋 미사용"
            df, msg = build_existing_dataset_stats_df(ex_root)  # images/train|val 기준
            return df, msg

        existing_dataset_dir.change(
            _load_existing_stats,
            inputs=[use_existing, existing_dataset_dir],
            outputs=[existing_stats_df, log_box]
        )

        # 3) 신규 데이터셋 선택되면 (images/ 한 레벨) 이미지 목록 로드

        # 4) out_dataset_dir + dataset_name 으로 최종 저장 루트 생성
        def _create_out(use, ex_root, out_root, name):
            # 1) 최종 저장 루트 생성 + 기본 구조 생성
            msg, final_root = ensure_out_dataset_root(out_root, name)
            if not final_root:
                return msg, "", ""

            # 2) 기존 데이터셋 활용이면: existing_root -> final_root로 복사 (cache 제외)
            if use == "Yes":
                copy_msg = copy_existing_dataset_into_final(
                    existing_root=ex_root,
                    final_root=final_root,
                    overwrite=True,
                    exclude_names={"cache", "__cache__", ".cache", "raw"}  # 필요시 추가
                )
                msg = msg + "\n" + copy_msg

            return msg, final_root, final_root

        create_out_btn.click(
            _create_out,
            inputs=[use_existing, existing_dataset_dir, out_dataset_dir, dataset_name],
            outputs=[log_box, final_out_root_state, out_root_view]
        )

        def _make_stats_text(train_cnt, total_cnt):
            val_cnt = total_cnt - train_cnt
            if total_cnt == 0:
                return "**Train:** 0 | **Val:** 0 | **Total:** 0"
            return (
                f"**Train:** {train_cnt} ({train_cnt/total_cnt*100:.1f}%) | "
                f"**Val:** {val_cnt} ({val_cnt/total_cnt*100:.1f}%) | "
                f"**Total:** {total_cnt}"
            )

        def _apply_random_split(new_root, ratio):
            # 1. 현재 선택된 경로에서 이미지 목록을 새로 가져옴
            choices, msg = list_new_images_for_checkbox_onelevel(new_root)

            # [추가된 부분] 데이터가 없을 때 경고 팝업 띄우기
            if not choices:
                gr.Warning("⚠️ 분할할 이미지가 없습니다! 왼쪽에서 '이번 데이터셋 경로'를 먼저 선택해주세요.")
                return gr.update(), "⚠️ 분할할 이미지가 없습니다. 경로를 먼저 선택하세요.", _make_stats_text(0, 0)

            # 2. 리스트를 복사하여 무작위로 섞음
            # random.seed(42): 매번 같은 결과를 얻기 위한 고정값 (재현성 확보)
            random.seed(42)
            shuffled_list = list(choices)
            random.shuffle(shuffled_list)

            # 3. 비율(ratio)에 따라 자를 위치 계산
            split_point = int(len(shuffled_list) * ratio)

            # 4. 앞부분은 Train(체크 대상), 뒷부분은 Val(미체크 대상)
            train_selected = shuffled_list[:split_point]

            # UI 업데이트: 체크박스 값 변경, 로그 출력, 하단 통계 갱신
            log_msg = f"[INFO] 전체 {len(shuffled_list)}개 데이터 중 {len(train_selected)}개를 Train으로 자동 배정했습니다. (비율: {ratio * 100}%)"

            # [추가된 부분] 성공했을 때 기분 좋은 안내 팝업 띄우기
            gr.Info(f" 성공! {len(train_selected)}개의 이미지가 Train으로 배정되었습니다.")

            return (
                gr.update(value=train_selected),
                log_msg,
                _make_stats_text(len(train_selected), len(shuffled_list))
            )

        # 랜덤 분할 버튼 이벤트 연결
        random_split_btn.click(
            _apply_random_split,
            inputs=[new_dataset_dir, train_ratio],
            outputs=[new_list_box, log_box, selection_stats]
        )

        # 5) 전체 선택 / 전체 해제
        def _load_new_list(new_root):
            choices, msg = list_new_images_for_checkbox_onelevel(new_root)
            return gr.update(choices=choices, value=[]), msg, choices, _make_stats_text(0, len(choices))

        new_dataset_dir.change(
            _load_new_list,
            inputs=[new_dataset_dir],
            outputs=[new_list_box, log_box, new_choices_state, selection_stats]
        )

        def _select_all(choices):
            return choices

        select_all_btn.click(
            fn=_select_all,
            inputs=[new_choices_state],
            outputs=[new_list_box],
        )

        deselect_all_btn.click(
            fn=lambda: [],
            inputs=[],
            outputs=[new_list_box],
        )

        # 6) 체크 기준 분할 복사 실행
        def _split(selected_train, new_root, final_out_root):
            msg, df = split_new_dataset_by_selection_onelevel(
                new_root=new_root,
                final_out_root=final_out_root,
                selected_train_filenames=selected_train,
                overwrite=True,
            )
            return msg, df

        split_btn.click(
            _split,
            inputs=[new_list_box, new_dataset_dir, final_out_root_state],
            outputs=[log_box, split_result_df]
        )


