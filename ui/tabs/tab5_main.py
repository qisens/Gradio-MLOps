# ui/tabs/tab5_main.py
import gradio as gr

from ui.tabs.tab5_labeling import build_tab5_labeling
from ui.tabs.tab5_labeling_folder import build_tab5_labeling_folder
from ui.tabs.tab5_labeling_canvas import build_tab5_labeling_canvas
from ui.shared.js_assets import load_all_js

def build_tab5():
    with gr.Tab("5. Labeling"):
        ALL_JS = load_all_js("./json")
        gr.HTML(f"<script>{ALL_JS}</script>")

        with gr.Tabs() as tabs:
            with gr.Tab("Folder", id="folder"):
                build_tab5_labeling_folder()

            with gr.Tab("Single File", id="single"):
                build_tab5_labeling()

        # 공통 캔버스 부분
        build_tab5_labeling_canvas()

        tabs.select(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { window.reset_editor && window.reset_editor(); }"
        )

