# ui/tabs/tab3_train_monitor.py
import os
import gradio as gr
from core.config import (
    PROJECT_ROOT, RUNS_DIR, METRIC_COLUMNS, LOSS_COLUMNS,
    UPLOAD_DATA_DIR, UPLOAD_MODEL_DIR
)
from core.utils_csv import _file_to_path
from core.yolo_train import run_epoch_eval_manual
from core.utilities import build_folder_picker, save_uploaded_file
from core.train_monitor_service import (
    TrainResultsPlotter,
    # build_epoch_conf_monitor_ui
)
from core.utils_jeeeun import (
    scan_run_dirs,
    build_artifact_path_map,
    sort_map_by_mtime_desc,
    pick_default_key,
)
from ui.tabs._ui_shared import (
    build_path_dropdown_selector,
    build_markdown_log_box,
    build_log_textbox
)
from core.train_monitor_service import EpochConfMonitor
epoch_conf_monitor = EpochConfMonitor()

mode = "all"
# view_mode = "matrics"
# view_mode = "loss"

PLOT_ROWS = 2
PLOT_COLS = 2
PLOT_PAGE_SIZE = PLOT_ROWS * PLOT_COLS

def build_tab3_train_monitor(trainer):
    with gr.Tab("3. Train Monitor"):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab(label="Training"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Tab(label="새로운 training 시작"):
                                gr.Markdown("### 학습 실행")
                                task = gr.Radio(["segment", "detect"], value="segment", label="YOLO Task")

                                data_yaml_file = gr.File(label="data.yaml 업로드", file_types=[".yaml", ".yml"], file_count="single")
                                data_yaml_path = build_log_textbox(label="서버 저장 경로 (data.yaml)")

                                with gr.Accordion(label="모델 선택해서 training", open=False):
                                    model_pt_file = gr.File(label="모델(.pt) 업로드", file_types=[".pt"], file_count="single")
                                    model_pt_path = gr.Textbox(label="서버 저장 경로 (model.pt)", interactive=False)

                                data_yaml_file.change(fn=lambda f: save_uploaded_file(f, UPLOAD_DATA_DIR), inputs=[data_yaml_file], outputs=[data_yaml_path])
                                model_pt_file.change(fn=lambda f: save_uploaded_file(f, UPLOAD_MODEL_DIR), inputs=[model_pt_file], outputs=[model_pt_path])

                                with gr.Row():
                                    monitor_imgsz = gr.Slider(label="imgsz", minimum=256, maximum=2048, step=64, value=640)
                                    monitor_epochs = gr.Slider(label="epochs", minimum=1, maximum=500, step=1, value=11)
                                with gr.Row():
                                    monitor_batch = gr.Slider(label="batch", minimum=1, maximum=128, step=1, value=16)
                                    monitor_lr0 = gr.Number(label="lr0", value=0.001)

                                with gr.Row():
                                    btn_start_train = gr.Button("학습 시작 (CLI)", variant="primary")
                                    btn_stop_train = gr.Button("학습 강제 종료", variant="stop")

                            with gr.Tab(label="다른 training과 비교하기"):
                                gr.Markdown("### 1. 최근 트레이닝 모니터링 소스 (Primary)")
                                gr.Markdown("트레이닝 후 트레이닝 경로가 자동으로 선택됩니다.<br>비워두면 최신 트레이닝 경로를 자동으로 선택합니다.")
                                results_csv_path = build_log_textbox(label="results.csv 경로")
                                with gr.Accordion(label="다른 csv 파일 불러오기", open=False):
                                    results_csv_file = gr.File(label="탐색기에서 results.csv 선택", file_types=[".csv"], file_count="single")

                                gr.Markdown("### 2. 이전 runs 선택 (Compare)")
                                gr.Markdown("원하는 모델이 검색되지 않으면, [새로운 training 시작] 탭에서 YOLO Task를 확인해주세요.")

                                runs_selector = build_path_dropdown_selector(
                                    label="비교 대상 run 선택"
                                )
                                runs_dropdown = runs_selector["dropdown"]
                                runs_map_state = runs_selector["map_state"]
                                prev_results_csv_path = runs_selector["path_output"]

                                btn_overlay_plots = gr.Button("데이터 불러오기")

                        with gr.Column(scale=3):
                            with gr.Row():
                                btn_refresh_plot = gr.Button("🔄 새로고침")
                                btn_prev = gr.Button("◀ 이전")
                                page_state = gr.State(1)
                                page_view = gr.Markdown("페이지: 1")
                                btn_next = gr.Button("다음 ▶")

                            plot_list = []

                            for _ in range(PLOT_ROWS):
                                with gr.Row():
                                    for _ in range(PLOT_COLS):
                                        plot_list.append(gr.Plot(label=""))

                            last_update = gr.Markdown("[plot 갱신]")
                            log_box = build_log_textbox(label="학습 로그", lines=20)

                with gr.Tab(label="best 모델 평가"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### Epoch별 모델 평가 (수동 실행)")

                            gr.Markdown("모델 선택하기")
                            with gr.Column(scale=1):
                                weight_selector = build_path_dropdown_selector(
                                    label="weights 폴더 선택"
                                )
                                weight_dropdown = weight_selector["dropdown"]
                                weight_map_state = weight_selector["map_state"]
                                weights_path_tb = weight_selector["path_output"]

                            gr.Markdown("평가 대상 이미지 폴더 선택하기")
                            with gr.Column(scale=1):
                                eval_img_path_tb, _, _, _ = build_folder_picker(
                                    label="평가 이미지 폴더", root_dir=os.path.join(PROJECT_ROOT, "test_img"), default_path=os.path.join(PROJECT_ROOT, "test_img")
                                )

                            gr.Markdown("model inference parameter")
                            with gr.Row():
                                eval_imgsz = gr.Slider(label="imgsz", minimum=256, maximum=2048, step=64, value=640)
                                eval_conf = gr.Number(label="conf_thres", value=0.25)
                                eval_iou = gr.Number(label="iou_thres", value=0.5)
                                eval_device = gr.Textbox(label="device", value="0")

                            run_title = gr.Textbox(
                                label="그래프 타이틀(옵션)",
                                value="",
                                placeholder="예) demo_exp22 / segment 비교 등",
                            )

                            # btn_eval = gr.Button("선택 경로로 Epoch 평가 실행", variant="primary")
                            btn_eval = gr.Button("Epoch 평가 및 그래프 그리기", variant="primary")

                        with gr.Column(scale=3):
                            gr.Markdown("### Weights별(conf txt) 추세 모니터링")

                            # btn_refresh = gr.Button("스캔 & 그래프 업데이트", variant="primary")
                            eval_log = build_log_textbox(label="평가 로그", lines=7)

                            plot_img = gr.Plot(label="Weights vs Conf")
                            table = gr.Dataframe(label="파일 목록", interactive=False)
                            csv_file = gr.File(label="CSV 다운로드(요약)")
                            save_csv_status = build_markdown_log_box(
                                title="csv 파일 생성 상태",
                                value="왼쪽 경로들을 확인하고 버튼을 클릭하면 confidence 평가 결과를 csv로 저장 할 수 있습니다.",
                            )

                            # btn_refresh.click(
                            #     fn=update_epoch_conf_view,
                            #     inputs=[weights_path_tb, run_title],
                            #     outputs=[plot_img, table, csv_file, status],
                            # )


        epoch_tick = gr.State(0)
        compare_csv_state = gr.State("")
        compare_enabled_state = gr.State(False)

        # [새로운 training 시작] 탭 관련
        # 버튼 클릭 리스너 두개 등록 가능. train 새로 시작할때 state 초기화 위함
        btn_start_train.click(
            fn=lambda: 0,
            inputs=[],
            outputs=[epoch_tick],
            queue=False,
        )
        btn_start_train.click(
            fn=trainer.start_train_stream,
            inputs=[
                task, data_yaml_path, model_pt_path,
                monitor_imgsz, monitor_epochs,
                monitor_batch, monitor_lr0
            ],
            outputs=[log_box, epoch_tick, results_csv_path],
        )

        def stop_train_and_timer():
            msg = trainer.stop_train()
            return msg

        btn_stop_train.click(
            fn=stop_train_and_timer,
            inputs=[],
            outputs=[log_box],
        )

        # plot 관련
        plotter = TrainResultsPlotter(
            RUNS_DIR,
            METRIC_COLUMNS,
            LOSS_COLUMNS,
            page_size=PLOT_PAGE_SIZE,
        )

        # plot 페이지 관련
        def on_epoch_tick(_, primary_csv, page, compare_csv, compare_enabled):
            return plotter.refresh_plots(
                csv_path=primary_csv,
                page_now=page,
                mode=mode,
                compare_csv_path=compare_csv,
                compare_enabled=compare_enabled,
            )

        epoch_tick.change(
            fn=on_epoch_tick,
            inputs=[
                epoch_tick,
                results_csv_path,
                page_state,
                compare_csv_state,
                compare_enabled_state,
            ],
            outputs=[page_state, *plot_list, last_update],
        )

        def on_prev_page(primary_csv, page, compare_csv, compare_enabled):
            return plotter.refresh_plots(
                csv_path=primary_csv,
                page_now=int(page) - 1,
                mode=mode,
                compare_csv_path=compare_csv,
                compare_enabled=compare_enabled,
            )

        def on_next_page(primary_csv, page, compare_csv, compare_enabled):
            return plotter.refresh_plots(
                csv_path=primary_csv,
                page_now=int(page) + 1,
                mode=mode,
                compare_csv_path=compare_csv,
                compare_enabled=compare_enabled,
            )

        btn_prev.click(
            fn=on_prev_page,
            inputs=[
                results_csv_path,
                page_state,
                compare_csv_state,
                compare_enabled_state,
            ],
            outputs=[page_state, *plot_list, last_update],
        )
        btn_next.click(
            fn=on_next_page,
            inputs=[
                results_csv_path,
                page_state,
                compare_csv_state,
                compare_enabled_state,
            ],
            outputs=[page_state, *plot_list, last_update],
        )

        page_state.change(
            fn=lambda p: f"페이지: {int(p)}",
            inputs=[page_state],
            outputs=[page_view],
        )

        # plot 새로고침
        def on_refresh_plots(primary_csv, page, compare_csv, compare_enabled):
            return plotter.refresh_plots(
                csv_path=primary_csv,
                page_now=page,
                mode=mode,
                compare_csv_path=compare_csv,
                compare_enabled=compare_enabled,
            )

        btn_refresh_plot.click(
            fn=on_refresh_plots,
            inputs=[results_csv_path, page_state],
            outputs=[page_state, *plot_list, last_update],
        )

        # [다른 training과 비교하기 tab]
        # 1. 최근 트레이닝 모니터링 소스 (Primary)

        results_csv_file.change(fn=lambda f: _file_to_path(f), inputs=[results_csv_file], outputs=[results_csv_path])

        # 2. 이전 runs 선택 (Compare)
        def refresh_runs_by_task(task_name: str):
            base_dir = os.path.join(PROJECT_ROOT, "runs", task_name)

            run_dir_map = scan_run_dirs(base_dir)
            path_map = build_artifact_path_map(
                run_dir_map,
                artifact="results_csv",
            )

            sorted_keys = sort_map_by_mtime_desc(path_map)
            default_key = pick_default_key(sorted_keys)

            return runs_selector["refresh_fn"](
                path_map,
                sorted_keys,
                default_key,
            )

        task.change(
            fn=refresh_runs_by_task,
            inputs=[task],
            outputs=[
                runs_dropdown,
                runs_map_state,
                prev_results_csv_path,
            ],
        )

        def on_overlay_plots(primary_csv, compare_csv):
            return (
                compare_csv,  # compare_csv_state
                True,  # compare_enabled_state
                *plotter.refresh_plots(
                    csv_path=primary_csv,
                    page_now=1,
                    mode=mode,
                    compare_csv_path=compare_csv,
                    compare_enabled=True,
                )
            )

        btn_overlay_plots.click(
            fn=on_overlay_plots,
            inputs=[
                results_csv_path,
                prev_results_csv_path,
            ],
            outputs=[
                compare_csv_state,
                compare_enabled_state,
                page_state,
                *plot_list,
                last_update,
            ],
        )

        #[best 모델 평가] 탭 관련
        def refresh_weights_by_task(task_name: str):
            base_dir = os.path.join(PROJECT_ROOT, "runs", task_name)
            run_dir_map = scan_run_dirs(base_dir)
            path_map = build_artifact_path_map(
                run_dir_map,
                artifact="weights_dir",
            )

            sorted_keys = sort_map_by_mtime_desc(path_map)
            default_key = pick_default_key(sorted_keys)

            return weight_selector["refresh_fn"](
                path_map,
                sorted_keys,
                default_key,
            )

        task.change(
            fn=refresh_weights_by_task,
            inputs=[task],
            outputs=[
                weight_dropdown,
                weight_map_state,
                weights_path_tb,
            ],
        )

        # 버튼 클릭 리스너. epoch conf 평가 및 그래프 그리는 두가지 동작 한번에
        def on_eval_and_plot(
                weights_dir,
                eval_img_path,
                imgsz,
                conf,
                iou,
                device,
                run_title,
        ):
            # 1) epoch 평가 실행
            eval_log_msg = run_epoch_eval_manual(
                weights_dir,
                eval_img_path,
                imgsz,
                conf,
                iou,
                device,
            )

            # 2) conf txt 스캔 + plot
            fig, df, csv_path, status = epoch_conf_monitor.update_epoch_conf_view(
                weights_dir,
                run_title,
            )

            # 로그 합치기
            merged_status = f"{eval_log_msg}\n{status}"

            return (
                merged_status,  # eval_log
                fig,
                df,
                csv_path,
                status,
            )

        btn_eval.click(
            fn=on_eval_and_plot,
            inputs=[
                weights_path_tb,
                eval_img_path_tb,
                eval_imgsz,
                eval_conf,
                eval_iou,
                eval_device,
                run_title,
            ],
            outputs=[
                eval_log,
                plot_img,
                table,
                csv_file,
                save_csv_status,
            ],
        )

    return {
        "task": task,

        "compare_selector": {
            "dropdown": runs_dropdown,
            "path": prev_results_csv_path,
            "map_state": runs_map_state,
        },

        "weights_selector": {
            "dropdown": weight_dropdown,
            "path": weights_path_tb,
            "map_state": weight_map_state,
        },

        "refresh": {
            "runs": refresh_runs_by_task,
            "weights": refresh_weights_by_task,
        },
    }