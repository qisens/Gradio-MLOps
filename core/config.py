import os

#DEFAULT_INPUT_DIR = "./input_images"
# EXPLORER_ROOT = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/test_yolo_project/datasets/"
EXPLORER_ROOT = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/1_image_viewer/"
INF_RESULTS_ROOT = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/4_model_score"

UPLOAD_ROOT = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo"
UPLOAD_NEWDATASET_ROOT = f"{UPLOAD_ROOT}/datasets_for_labeling"
# UPLOAD_DATA_DIR = f"{UPLOAD_ROOT}/configs"
UPLOAD_DATA_DIR = f"{UPLOAD_ROOT}/3_Train_monitoring/configs"
# UPLOAD_MODEL_DIR = f"{UPLOAD_ROOT}/base_model"
UPLOAD_MODEL_DIR = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/3_Train_monitoring/"

TRAIN_ENV_PY = "/home/qisens/anaconda3/envs/gradio/bin/python"
YOLO_CLI = "/home/qisens/anaconda3/envs/gradio/bin/yolo"

# PROJECT_ROOT = os.path.abspath("./test_yolo_project")
PROJECT_ROOT = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/"
# RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
RUNS_DIR = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/runs"

METRIC_COLUMNS = [
    "metrics/mAP50-95(B)", "metrics/mAP50(B)",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50-95(M)", "metrics/mAP50(M)",
    "metrics/precision(M)", "metrics/recall(M)",
]

LOSS_COLUMNS = [
    "train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss",
]

# LABELING_DEST_ROOT = os.path.join(
#         os.path.abspath("./test_yolo_project"),
#         "datasets_for_labeling"
#     )
LABELING_DEST_ROOT = "/home/qisens/jeeeun/workspace/gradio/mlops_q2/1113_demo/4_model_score/datasets_for_labeling"