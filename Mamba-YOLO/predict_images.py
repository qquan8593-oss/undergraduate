"""Run FP32 image detection and save annotated images (Unicode paths supported)."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MAMBA_SCAN_BACKEND', 'torch')
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Image path or image directory')
    parser.add_argument('--weights', default=str(ROOT / 'weights/visdrone_last.pt'))
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'output_dir/predict_last')
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        parser.error('Weights not found: ' + args.weights)
    if not Path(args.source).exists():
        parser.error('Source not found: ' + args.source)
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error('conf and iou must be between 0 and 1')
    torch.set_num_threads(4)
    model = YOLO(args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, device=args.device, half=False,
                            save=False, stream=True, max_det=300)
    for result in results:
        if result.boxes is not None and not torch.isfinite(result.boxes.data).all():
            raise RuntimeError('Non-finite prediction boxes')
        target = args.output / (Path(result.path).stem + '_pred.jpg')
        ok, encoded = cv2.imencode('.jpg', result.plot())
        if not ok:
            raise RuntimeError('Could not encode output image')
        encoded.tofile(str(target))
        boxes = result.boxes
        print(f'Saved: {target}; boxes: {len(boxes) if boxes is not None else 0}')
        if boxes is not None and len(boxes):
            print('Maximum confidence:', float(boxes.conf.max()))
        else:
            print('No boxes passed the threshold. Try --conf 0.05 for diagnosis; low-confidence boxes may be incorrect.')


if __name__ == '__main__':
    main()
