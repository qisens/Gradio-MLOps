# ui/tabs/tab3_train_monitor.py
import gradio as gr

from ui.tabs.tab3_1_left1_train_new import build_train_new
from ui.tabs.tab3_1_left2_train_compare import build_train_compare
from ui.tabs.tab3_1_right_train_plot import build_train_plot
from ui.tabs.tab3_2_train_best_eval import build_best_eval


def build_tab3_train_monitor(trainer):
    with gr.Tab("3. Train Monitor"):
        # ===== shared state =====
        results_csv_path = gr.State("")
        epoch_tick = gr.State(0)
        compare_csv_state = gr.State("")
        compare_enabled_state = gr.State(False)

        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column(scale=2):
                    tab1 = build_train_new(
                        trainer=trainer,
                        results_csv_path=results_csv_path,
                        epoch_tick=epoch_tick,
                    )
                    task = tab1["task"]

                    compare_tab = build_train_compare(
                        task=task,
                        results_csv_path=results_csv_path,
                        compare_csv_state=compare_csv_state,
                        compare_enabled_state=compare_enabled_state,
                    )

                # 🔥 plot은 1,2번 탭이 **공동 사용**
                plot_panel = build_train_plot(
                    results_csv_path=results_csv_path,
                    epoch_tick=epoch_tick,
                    compare_csv_state=compare_csv_state,
                    compare_enabled_state=compare_enabled_state,
                )

        with gr.Tab("best 모델 평가"):
            best_eval_tab = build_best_eval(task=task)

    return {
        "task": task,
        "compare_selector": compare_tab["compare_selector"],
        "weights_selector": best_eval_tab["weights_selector"],
        "refresh": {
            "runs": compare_tab["refresh"]["runs"],
            "weights": best_eval_tab["refresh"]["weights"],
        },
    }
