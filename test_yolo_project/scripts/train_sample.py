import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolov8n-seg.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name", default="demo_exp")
    p.add_argument("--device", default="0")  # GPU면 "0", CPU면 "cpu"
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        task="segment",
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        device=args.device,
    )

if __name__ == "__main__":
    main()

