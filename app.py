# app.py
import gradio as gr

from core.config import PROJECT_ROOT, YOLO_CLI
from core.yolo_train import YoloTrainer

from ui.shared.js_assets import load_all_js
from ui.tabs.tab1_viewer import build_tab1_viewer
from ui.tabs.tab2_dataset import build_tab2_dataset
from ui.tabs.tab3_train_monitor import build_tab3_train_monitor
from ui.tabs.tab4_perf_monitor import build_tab4_perf_monitor
from ui.tabs.tab5_labeling import build_tab5_labeling
from ui.tabs.tab6_compare import build_tab6_compare
from core.yolo_train import YoloTrainer

def create_demo():
    all_js = load_all_js("./json")

    trainer = YoloTrainer(yolo_cli=YOLO_CLI, project_root=PROJECT_ROOT)

    with gr.Blocks() as demo:
        with gr.Tabs():
            build_tab1_viewer()
            build_tab2_dataset()
            build_tab3_train_monitor(trainer=trainer)
            build_tab4_perf_monitor()
            build_tab5_labeling()
            build_tab6_compare()

    return demo, all_js
    #return demo

if __name__ == "__main__":
    demo, ALL_JS = create_demo()
    #demo = create_demo()
    # demo.launch(js=ALL_JS, share=True)
    demo.launch(js=ALL_JS, server_port=7861)
    #demo.launch(js=ALL_JS)
    #demo.launch(share=True)
    #demo.launch()
