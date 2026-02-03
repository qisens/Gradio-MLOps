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

        current_tab = gr.State("folder")

        with gr.Tabs() as tabs:
            with gr.Tab("Folder", id="folder") as tab_folder:
                build_tab5_labeling_folder(current_tab)

            with gr.Tab("Single File", id="single") as tab_single:
                build_tab5_labeling(current_tab)

        # 공통 캔버스 부분
        build_tab5_labeling_canvas()

        bind_tab_reset(tab_folder, "folder", current_tab)
        bind_tab_reset(tab_single, "single", current_tab)

def bind_tab_reset(tab, tab_name, current_tab):
    tab.select(
        fn=lambda: tab_name,
        inputs=None,
        outputs=[current_tab],
        js=f"""() => {{
            window.currentTab = "{tab_name}";
            window.reset_editor && window.reset_editor();
        }}"""
    )