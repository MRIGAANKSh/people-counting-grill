# people_counter_app.py
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, Response
import cv2
import numpy as np
import time
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------------------
# CONFIG
VIDEO_SOURCE = 0  # webcam index used by /live
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
model = None
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # auto-downloads if not present
    use_ultralytics = True
    print("Using ultralytics YOLOv8")
except Exception as e:
    print("Ultralytics not available or failed to import. Trying yolov5 (torch.hub)...")
    try:
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        use_ultralytics = False
        print("Using yolov5 via torch.hub")
    except Exception as e2:
        print("Failed to load any YOLO model. Object detection will not run.")
        model = None

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
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
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


def detect_persons_on_frame(frame):
    """
    Run the loaded model on the frame and return list of rects for person detections: [(x1,y1,x2,y2), ...]
    If model is not loaded, returns empty list.
    """
    rects = []
    if model is None:
        return rects

    if use_ultralytics:
        try:
            results = model(frame)[0]
            if getattr(results, "boxes", None) is not None:
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
        except Exception:
            return rects
    else:
        try:
            results = model(frame)
            df = results.pandas().xyxy[0]
            persons = df[df['name'] == 'person']
            for _, r in persons.iterrows():
                if r['confidence'] < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = int(r['xmin']), int(r['ymin']), int(r['xmax']), int(r['ymax'])
                rects.append((x1, y1, x2, y2))
        except Exception:
            return rects

    return rects


def process_and_save(input_path, output_path):
    """
    Process the uploaded video, detect people per frame, update tracker to count unique people,
    and write a processed video with overlays. Returns total unique people count.
    (This is synchronous - it will block until finished.)
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
        rects = detect_persons_on_frame(frame)
        objects = tracker.update(rects)

        # draw boxes and IDs
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        for object_id, centroid in objects.items():
            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

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

    return tracker.total_count


# Generator for live webcam feed (used by /video_feed)
def generate_live_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    tracker = CentroidTracker(max_disappeared=40, max_distance=60)
    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rects = detect_persons_on_frame(frame)
        objects = tracker.update(rects)

        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        for object_id, centroid in objects.items():
            cv2.putText(frame, f"ID {object_id}", (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

        elapsed = time.time() - start if frame_count > 0 else 0.0
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        sidebar = draw_sidebar(frame, tracker.total_count, fps=fps)
        combined = np.hstack((frame, sidebar))

        ret, buffer = cv2.imencode('.jpg', combined)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        frame_count += 1

    cap.release()


# Generator to stream frames from a processed video file (MJPEG)
def generate_processed_frames(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # ensure sidebar exists (the processed video already has sidebar, but we can stream as-is)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


# ---------------------------
# HTML Templates (inline)
# ---------------------------

INDEX_HTML = '''
<!doctype html>
<html>
  <head>
    <title>People Counter - Upload or Live</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      body { font-family: Arial, sans-serif; background:#111; color:#eee; display:flex; justify-content:center; align-items:flex-start; min-height:100vh; }
      .container { margin-top:40px; width:90vw; max-width:900px; }
      .card { background:#1a1a1a; padding:20px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.6); }
      input[type=file] { color:#fff; }
      button { padding:10px 16px; border-radius:6px; border:none; background:#0b84ff; color:#fff; cursor:pointer; }
      .note { color:#bbb; font-size:0.9rem; margin-top:8px }
      .links { margin-top:14px; }
      a { color:#0b84ff; text-decoration:none; margin-right:12px; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h2>People Counter — Upload a video or open Live Stream</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
          <input type="file" name="video" accept="video/*" required />
          <div style="height:12px"></div>
          <button type="submit">Upload & Process</button>
        </form>

        <div class="links">
          <a href="{{ url_for('live') }}">Open Live Stream (webcam)</a>
          <a href="{{ url_for('show_processed_list') }}">View Processed Videos</a>
        </div>

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
          <source src="{{ processed_url }}" type="video/mp4">
          Your browser does not support the video tag.
        </video>

        <div style="margin-top:10px">
          <a href="{{ stream_page }}" class="btn">Stream as MJPEG</a>
          <a href="{{ download_url }}" class="btn" style="background:#22bb66">Download MP4</a>
          <a href="/" class="btn" style="background:#555">Process another</a>
        </div>
      </div>
    </div>
  </body>
</html>
'''

LIVE_HTML = '''
<!doctype html>
<html>
  <head>
    <title>Live People Counter</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      body { font-family: Arial, sans-serif; background:#111; color:#eee; display:flex; justify-content:center; align-items:flex-start; min-height:100vh; }
      .container { margin-top:20px; width:95vw; max-width:1200px; }
      .card { background:#1a1a1a; padding:20px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.6); text-align:center; }
      img { width:100%; max-width:1100px; height:auto; border-radius:6px; }
      a.btn { display:inline-block; margin-top:8px; padding:8px 12px; background:#0b84ff; color:#fff; border-radius:6px; text-decoration:none; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h2>Live Webcam — YOLO + Tracking</h2>
        <p>Streaming processed webcam frames (press Stop in your browser to stop).</p>
        <img src="{{ url_for('video_feed') }}" alt="Live feed" />
        <div style="margin-top:10px">
          <a href="/" class="btn">Back</a>
        </div>
      </div>
    </div>
  </body>
</html>
'''

PROCESSED_LIST_HTML = '''
<!doctype html>
<html>
  <head>
    <title>Processed Videos</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      body { font-family: Arial, sans-serif; background:#111; color:#eee; display:flex; justify-content:center; align-items:flex-start; min-height:100vh; }
      .container { margin-top:20px; width:95vw; max-width:1200px; }
      .card { background:#1a1a1a; padding:20px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.6); }
      ul { list-style:none; padding:0; }
      li { padding:8px 0; }
      a { color:#0b84ff; text-decoration:none; margin-right:8px; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h2>Saved Processed Videos</h2>
        {% if files %}
        <ul>
          {% for f in files %}
            <li>
              {{ f }} —
              <a href="{{ url_for('result', filename=f, total=0) }}">Open page</a>
              <a href="{{ url_for('stream_page', filename=f) }}">Stream</a>
              <a href="{{ url_for('processed_file', filename=f) }}">Download</a>
            </li>
          {% endfor %}
        </ul>
        {% else %}
          <p>No processed videos found.</p>
        {% endif %}
        <div style="margin-top:10px"><a href="/" class="btn" style="background:#0b84ff; padding:8px 12px; color:#fff; border-radius:6px; text-decoration:none">Back</a></div>
      </div>
    </div>
  </body>
</html>
'''

STREAM_PAGE_HTML = '''
<!doctype html>
<html>
  <head>
    <title>Stream Processed Video</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      body { font-family: Arial, sans-serif; background:#111; color:#eee; display:flex; justify-content:center; align-items:flex-start; min-height:100vh; }
      .container { margin-top:20px; width:95vw; max-width:1200px; }
      .card { background:#1a1a1a; padding:20px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.6); text-align:center; }
      img { width:100%; max-width:1100px; height:auto; border-radius:6px; }
      a.btn { display:inline-block; margin-top:8px; padding:8px 12px; background:#0b84ff; color:#fff; border-radius:6px; text-decoration:none; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h2>Streaming: {{ filename }}</h2>
        <img src="{{ stream_url }}" alt="Streamed processed video" />
        <div style="margin-top:10px">
          <a href="/" class="btn">Back</a>
        </div>
      </div>
    </div>
  </body>
</html>
'''

# ---------------------------
# Routes
# ---------------------------

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

    # process video synchronously and save
    total = process_and_save(input_path, output_path)

    # redirect to result page
    return redirect(url_for('result', filename=out_filename, total=total))


@app.route('/result')
def result():
    filename = request.args.get('filename')
    total = request.args.get('total', '0')
    if not filename:
        return redirect(url_for('index'))
    processed_url = url_for('processed_file', filename=filename)
    stream_page = url_for('stream_page', filename=filename)
    download_url = processed_url
    return render_template_string(RESULT_HTML, processed_url=processed_url, total=total, stream_page=stream_page, download_url=download_url)


@app.route('/processed/<path:filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=False)


@app.route('/view_stream/<path:filename>')
def stream_page(filename):
    stream_url = url_for('stream_processed', filename=filename)
    return render_template_string(STREAM_PAGE_HTML, stream_url=stream_url, filename=filename)


@app.route('/stream_processed/<path:filename>')
def stream_processed(filename):
    return Response(generate_processed_frames(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/processed_list')
def show_processed_list():
    files = sorted(os.listdir(PROCESSED_FOLDER))
    return render_template_string(PROCESSED_LIST_HTML, files=files)


# LIVE streaming routes
@app.route('/live')
def live():
    return render_template_string(LIVE_HTML)


@app.route('/video_feed')
def video_feed():
    return Response(generate_live_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------
# Run app
# ---------------------------
if __name__ == '__main__':
    # run with threaded=True so MJPEG streaming endpoints can work while other requests happen
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
