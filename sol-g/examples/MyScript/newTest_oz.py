from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
    

def tracking(frame):
    model = YOLO("yolov8n.pt").to('cuda')

    tracker = DeepSort(max_age=30)
    
    results = model(frame)
    
    id1name = model.names

    detections = []
    class_name = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            #conf = float(box.conf[0].cpu())
            cls = int(box.cls[0].cpu())
            class_name.append(id1name[cls])
            detections.append(([x1, y1, x2 - x1, y2 - y1], cls))


    track_id_to_label = {}
    tracks = tracker.update_tracks(detections, frame=frame)

    

    for i,track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        #if 
        min_dist = float('inf')
        matched_class = "unknown"

        track_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        for det in detections:
            
            dx, dy, dw, dh = det[0]
            dist_x = (dx + dw * 0.5) - track_center[0]
            dist_y = (dy + dh * 0.5) - track_center[1]
            dist = dist_x * dist_x + dist_y * dist_y
            if dist < min_dist:
                min_dist = dist
                matched_class = id1name[det[1]] if det[1] < len(id1name) else "unknown"
            #print("print",matched_class)


        if not matched_class == "person":
            track_id_to_label[track_id] = matched_class
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id} ,{track_id_to_label[track_id]}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)            
    
    return frame