import os

#DEFAULT_INPUT_DIR = "./input_images"
EXPLORER_ROOT = "/home/gpuadmin"
INF_RESULTS_ROOT = "/home/gpuadmin/seongje_gradio2/inf_results"

UPLOAD_ROOT = "/home/gpuadmin/seongje_gradio2/test_yolo_project"
UPLOAD_NEWDATASET_ROOT = f"{UPLOAD_ROOT}/datasets_for_labeling"
UPLOAD_DATA_DIR = f"{UPLOAD_ROOT}/configs"
UPLOAD_MODEL_DIR = f"{UPLOAD_ROOT}/base_model"

TRAIN_ENV_PY = "/home/gpuadmin/anaconda3/envs/gr_ultra/bin/python"
YOLO_CLI = "/home/gpuadmin/anaconda3/envs/gr_ultra/bin/yolo"

PROJECT_ROOT = os.path.abspath("./test_yolo_project")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

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

LABELING_DEST_ROOT = os.path.join(
        os.path.abspath("./test_yolo_project"),
        "datasets_for_labeling"
    )