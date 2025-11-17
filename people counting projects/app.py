# people_counter_app.py
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import time
import os
import uuid

app = Flask(__name__)

# ---------------------------
# CONFIG
VIDEO_SOURCE = 0  # unused for upload flow; kept for reference
SIDEBAR_WIDTH = 300
FRAME_WIDTH = 960   # resize for perf
FRAME_HEIGHT = 540
CONF_THRESHOLD = 0.4
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# ---------------------------

# Try ultralytics YOLOv8 first (preferred), otherwise fallback to yolov5 via torch.hub
use_ultralytics = False
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # auto-downloads if not present
    use_ultralytics = True
    print("Using ultralytics YOLOv8")
except Exception as e:
    print("Ultralytics not available, trying yolov5 (torch.hub)...")
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("Using yolov5 via torch.hub")

# Simple centroid tracker (keeps IDs for detected bounding boxes across frames)
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        # next object ID to assign
        self.next_object_id = 0
        # dict of object_id -> centroid
        self.objects = dict()
        # how many consecutive frames an object has been missing
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        # total unique IDs assigned (used as total people count)
        self.total_count = 0

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        self.total_count += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        rects: list of bounding boxes [(x1,y1,x2,y2), ...]
        returns dict of object_id -> centroid
        """
        # if no detections
        if len(rects) == 0:
            # mark all existing as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        # compute centroids for input rects
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # if no existing tracked objects, register all
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # build object IDs and centroids arrays
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute distance matrix between existing objects and new input centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

            # find smallest value pairs (greedy)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                # ignore if distance too large
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # compute unused rows and cols
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # if more object centroids than input centroids -> some objects disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # register new objects
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


def draw_sidebar(frame, total_count, fps=None):
    h, w = frame.shape[:2]
    sidebar = np.zeros((h, SIDEBAR_WIDTH, 3), dtype=np.uint8) + 30
    # Text positions
    cv2.putText(sidebar, "TOTAL PEOPLE", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    cv2.putText(sidebar, str(total_count), (20,140), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,255,0), 6, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(sidebar, f"FPS: {fps:.1f}", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(sidebar, ts, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
    return sidebar


def process_video(input_path, output_path):
    """
    Process the uploaded video, detect people per frame, update tracker to count unique people,
    and write a processed video with overlays. Returns total unique people count.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    tracker = CentroidTracker(max_disappeared=40, max_distance=60)

    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rects = []

        # run detection
        if use_ultralytics:
            results = model(frame)[0]
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    if conf < CONF_THRESHOLD:
                        continue
                    # person class is 0
                    if cls == 0:
                        xyxy = box.xyxy.cpu().numpy().astype(int).reshape(-1)
                        x1, y1, x2, y2 = xyxy
                        rects.append((x1, y1, x2, y2))
        else:
            results = model(frame)
            df = results.pandas().xyxy[0]
            persons = df[df['name'] == 'person']
            for _, r in persons.iterrows():
                if r['confidence'] < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = int(r['xmin']), int(r['ymin']), int(r['xmax']), int(r['ymax'])
                rects.append((x1, y1, x2, y2))

        objects = tracker.update(rects)

        # draw boxes and IDs
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        for object_id, centroid in objects.items():
            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # sidebar with total unique count
        elapsed = time.time() - start if frame_count > 0 else 0.0
        fps_proc = frame_count / elapsed if elapsed > 0 else fps
        sidebar = draw_sidebar(frame, tracker.total_count, fps=fps_proc)
        combined = np.hstack((frame, sidebar))

        # initialize writer
        if out is None:
            h, w = combined.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out.write(combined)
        frame_count += 1

    cap.release()
    if out is not None:
        out.release()

    total = tracker.total_count
    return total


# Templates (rendered inline so we don't need separate template files)
INDEX_HTML = '''
<!doctype html>
<html>
  <head>
    <title>People Counter - Upload Video</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      body { font-family: Arial, sans-serif; background:#111; color:#eee; display:flex; justify-content:center; align-items:flex-start; min-height:100vh; }
      .container { margin-top:40px; width:90vw; max-width:900px; }
      .card { background:#1a1a1a; padding:20px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.6); }
      input[type=file] { color:#fff; }
      button { padding:10px 16px; border-radius:6px; border:none; background:#0b84ff; color:#fff; cursor:pointer; }
      .note { color:#bbb; font-size:0.9rem; margin-top:8px }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h2>Upload a video to count people</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
          <input type="file" name="video" accept="video/*" required />
          <div style="height:12px"></div>
          <button type="submit">Upload & Process</button>
        </form>
        <p class="note">Processed video will be saved and shown with a sidebar that displays the total unique people counted.</p>
      </div>
    </div>
  </body>
</html>
'''

RESULT_HTML = '''
<!doctype html>
<html>
  <head>
    <title>People Count Result</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      body { font-family: Arial, sans-serif; background:#111; color:#eee; display:flex; justify-content:center; align-items:flex-start; min-height:100vh; }
      .container { margin-top:20px; width:95vw; max-width:1200px; }
      .card { background:#1a1a1a; padding:20px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.6); }
      video { width:100%; height:auto; border-radius:6px; }
      a.btn { display:inline-block; margin-top:8px; padding:8px 12px; background:#0b84ff; color:#fff; border-radius:6px; text-decoration:none; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h2>Processed Video</h2>
        <p>Total unique people counted: <strong>{{ total }}</strong></p>
        <video controls>
          <source src="{{ url }}" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <div style="margin-top:10px">
          <a href="/" class="btn">Process another</a>
        </div>
      </div>
    </div>
  </body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No file part", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400

    # save uploaded file
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # processed output filename
    out_filename = f"processed_{filename}.mp4"
    output_path = os.path.join(PROCESSED_FOLDER, out_filename)

    # process video (this runs synchronously)
    total = process_video(input_path, output_path)

    # after processing redirect to result page
    return redirect(url_for('result', filename=out_filename, total=total))


from werkzeug.utils import secure_filename

@app.route('/result')
def result():
    filename = request.args.get('filename')
    total = request.args.get('total', '0')
    if not filename:
        return redirect(url_for('index'))
    url = url_for('processed_file', filename=filename)
    return render_template_string(RESULT_HTML, url=url, total=total)


@app.route('/processed/<path:filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
