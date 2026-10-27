from ultralytics import YOLO
import argparse
import os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default=ROOT + '/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers; 0 is portable on Windows')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
    parser.add_argument('--name', default='mambayolo', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    args = {
        "data": os.path.join(ROOT, opt.data),
        "imgsz": opt.imgsz,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": os.path.join(ROOT, opt.project),
        "name": opt.name,
    }
    model_conf = os.path.join(ROOT, opt.config)
    model = YOLO(model_conf)
    if task == 'train':
        model.train(**args)
    else:
        model.val(data=args['data'], imgsz=opt.imgsz, batch=opt.batch_size,
                  device=opt.device, workers=opt.workers, project=args['project'],
                  name=opt.name, split='test' if task == 'test' else 'val')
