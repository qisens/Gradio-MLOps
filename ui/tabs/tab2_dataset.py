import gradio as gr
import pandas as pd

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

                new_list_box = gr.CheckboxGroup(
                    label="신규 이미지 목록(체크=Train / 미체크=Val)",
                    choices=[],
                    value=[],
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
        def _load_new_list(new_root):
            choices, msg = list_new_images_for_checkbox_onelevel(new_root)
            return gr.update(choices=choices, value=[]), msg

        new_dataset_dir.change(
            _load_new_list,
            inputs=[new_dataset_dir],
            outputs=[new_list_box, log_box]
        )

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

        # 5) 체크 기준 분할 복사 실행
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


