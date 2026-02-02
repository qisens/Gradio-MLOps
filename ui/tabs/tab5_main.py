import gradio as gr

from ui.tabs.tab5_labeling import build_tab5_labeling
from ui.tabs.tab5_labeling_folder import build_tab5_labeling_folder
from ui.shared.js_assets import load_all_js

def build_tab5():
    with gr.Tab("5. Labeling"):
        gr.HTML("""
            <style>
            /* Accordion 전체 높이 제한 */
            #optional-accordion {
                max-height: 290px;      /* 왼쪽 영역 높이에 맞게 조절 */
                overflow: hidden;
            }

            /* Accordion 내부만 스크롤 */
            #optional-accordion .wrap {
                max-height: 280px;
                overflow-y: auto;
                padding-right: 6px;
            }
            </style>
            """)

        ALL_JS = load_all_js("./json")
        gr.HTML(f"<script>{ALL_JS}</script>")

        with gr.Tabs():
            with gr.Tab("Folder"):
                build_tab5_labeling_folder()

            with gr.Tab("Single File"):
                build_tab5_labeling()

