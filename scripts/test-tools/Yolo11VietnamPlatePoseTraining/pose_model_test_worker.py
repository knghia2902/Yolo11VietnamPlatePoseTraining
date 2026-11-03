"""One bounded inference process; never writes annotations or training data."""
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import time


def main():
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    weights, source, output = map(Path, sys.argv[1:])
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model = YOLO(str(weights))
        if model.task != 'pose' or model.model.yaml['kpt_shape'] != [4, 3]:
            raise ValueError('Expected four-corner pose model')
        result = model.predict(str(source), imgsz=960, conf=0.25, device='cpu', verbose=False)[0]
    image = np.array(Image.open(source).convert('RGB'))
    detections = []
    for i, box in enumerate(result.boxes):
        points = result.keypoints.xy[i].cpu().numpy().astype(np.float32)
        width = max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3]))
        height = max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1]))
        w, h = max(16, min(1000, round(float(width)))), max(16, min(1000, round(float(height))))
        target = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (w,h))
        buffer = io.BytesIO(); Image.fromarray(crop).save(buffer, format='JPEG', quality=95)
        detections.append(dict(confidence=float(box.conf[0]), box=box.xyxy[0].cpu().tolist(),
            points=points.tolist(), crop=base64.b64encode(buffer.getvalue()).decode('ascii')))
    Path(output).write_text(json.dumps(dict(width=image.shape[1], height=image.shape[0], detections=detections,
        elapsed_seconds=round(time.perf_counter()-start, 2))), encoding='utf-8')


if __name__ == '__main__':
    main()
