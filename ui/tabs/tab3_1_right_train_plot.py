# ui/tabs/tab3_1_right_train_plot.py
import gradio as gr
from core.train_monitor_service import TrainResultsPlotter
from core.config import RUNS_DIR, METRIC_COLUMNS, LOSS_COLUMNS
from ui.tabs._ui_shared import build_log_textbox, build_markdown_log_box

PLOT_ROWS = 2
PLOT_COLS = 2
PLOT_PAGE_SIZE = PLOT_ROWS * PLOT_COLS
mode = "all"


def build_train_plot(
    *,
    results_csv_path,
    epoch_tick,
    compare_csv_state,
    compare_enabled_state,
    train_log_message_state,
):
    plotter = TrainResultsPlotter(
        RUNS_DIR,
        METRIC_COLUMNS,
        LOSS_COLUMNS,
        page_size=PLOT_PAGE_SIZE,
    )

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

        last_update = build_markdown_log_box(title="plot 갱신", value="",)
        log_box = build_log_textbox(label="학습 로그", lines=20)


    # ===== 이벤트 =====

    def refresh(csv_path, page, compare_csv, compare_enabled, log_message):
        return (
            *plotter.refresh_plots(
                csv_path=csv_path,
                page_now=page,
                mode=mode,
                compare_csv_path=compare_csv,
                compare_enabled=compare_enabled,
            ),
            log_message,
        )

    def register_refresh(trigger):
        trigger(
            fn=refresh,
            inputs=[
                results_csv_path,
                page_state,
                compare_csv_state,
                compare_enabled_state,
                train_log_message_state,
            ],
            outputs=[
                page_state,
                *plot_list,
                last_update,
                log_box,
            ],
        )
    register_refresh(epoch_tick.change)
    register_refresh(btn_refresh_plot.click)
    # 🔥 compare 관련 트리거 추가
    register_refresh(compare_csv_state.change)
    register_refresh(compare_enabled_state.change)

    def move_page(p, delta):
        return max(1, int(p) + delta)

    btn_prev.click(
        fn=lambda p: move_page(p, -1),
        inputs=[page_state],
        outputs=[page_state],
    )

    btn_next.click(
        fn=lambda p: move_page(p, +1),
        inputs=[page_state],
        outputs=[page_state],
    )

    page_state.change(
        fn=refresh,
        inputs=[
            results_csv_path,
            page_state,
            compare_csv_state,
            compare_enabled_state,
            train_log_message_state,  # 로그 state 쓰는 경우
        ],
        outputs=[
            page_state,
            *plot_list,
            last_update,
            log_box,
        ],
    )

    page_state.change(
        fn=lambda p: f"페이지: {int(p)}",
        inputs=[page_state],
        outputs=[page_view],
    )

    return {
        "page_state": page_state,
        "plots": plot_list,
        "last_update": last_update,
    }
