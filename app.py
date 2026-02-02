# app.py
import gradio as gr

from core.config import PROJECT_ROOT, YOLO_CLI
from core.yolo_train import YoloTrainer

from ui.shared.js_assets import load_all_js
from ui.tabs.tab1_viewer import build_tab1_viewer
from ui.tabs.tab2_dataset import build_tab2_dataset
from ui.tabs.tab3_train_monitor import build_tab3_train_monitor
from ui.tabs.tab4_perf_monitor import build_tab4_perf_monitor
from ui.tabs.tab5_main import build_tab5
from ui.tabs.tab6_compare import build_tab6_compare
from ui.tabs.tab7_inference import build_tab7_inference
from core.yolo_train import YoloTrainer
from core.utils_csv import _build_runs_map

def create_demo():
    all_js = load_all_js("./json")

    trainer = YoloTrainer(yolo_cli=YOLO_CLI, project_root=PROJECT_ROOT)

    with gr.Blocks(css="""
    /* gradio textbox 로그 스타일 */
    #log_box textarea {
        background-color: #f3f4f6;
        color: #111827;
        font-family: monospace;
        font-size: 13px;
    }
    
    /* html textbox 로그 스타일*/
    #logbox {
        height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 13px;
        background-color: #f3f4f6;
        color: #111827;
        padding: 8px;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
    }
    """) as demo:
        with gr.Tabs():
            build_tab1_viewer()
            build_tab2_dataset()
            tab3 = build_tab3_train_monitor(trainer=trainer)
            build_tab4_perf_monitor()
            build_tab5()
            build_tab6_compare()
            tab7 = build_tab7_inference()

        demo.load(
            fn=lambda task_name: (
                *tab3["refresh"]["runs"](task_name),
                *tab3["refresh"]["weights"](task_name),
                *tab3["refresh"]["weights"](task_name),  # ← tab7도 같은 weights 사용

            ),
            inputs=[tab3["task"]],
            outputs=[
                # compare runs selector
                tab3["compare_selector"]["dropdown"],
                tab3["compare_selector"]["map_state"],
                tab3["compare_selector"]["path"],

                # weights selector
                tab3["weights_selector"]["dropdown"],
                tab3["weights_selector"]["map_state"],
                tab3["weights_selector"]["path"],

                # tab7 weights selector
                tab7["weights_selector"]["dropdown"],
                tab7["weights_selector"]["map_state"],
                tab7["weights_selector"]["path"],
            ],
        )

    return demo, all_js
    #return demo

if __name__ == "__main__":
    demo, ALL_JS = create_demo()
    #demo = create_demo()
    # demo.launch(js=ALL_JS, share=True)
    demo.launch(js=ALL_JS, server_port=7861)
