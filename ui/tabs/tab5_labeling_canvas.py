# ui/tabs/tab5_labeling_canvas.py
# ui/tabs/tab5_labeling_canvas.py
import gradio as gr

def build_tab5_labeling_canvas():
    """
    Tab5 전체에서 공통으로 사용하는 단일 Canvas
    JS editor는 항상 이 canvas(#edit_canvas)에만 그림
    """

    gr.HTML("""
    <div style="
        display:flex;
        justify-content:center;
        width:100%;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <canvas
            id="edit_canvas"
            width="800"
            height="600"
            style="border:1px solid white;"
        ></canvas>
    </div>
    """)
