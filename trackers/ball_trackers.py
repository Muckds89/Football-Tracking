import cv2


from ultralytics import YOLO
import logging
from utils.utils import Utils
from collections import deque

logging.basicConfig(level=logging.INFO)

@staticmethod
def _point_inside_xyxy(point, bbox_xyxy, margin=0):
    x, y = point
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)

@staticmethod
def _shrink_xyxy_bbox(bbox_xyxy, shrink_x=0.15, shrink_y=0.10):
    x1, y1, x2, y2 = bbox_xyxy

    w = x2 - x1
    h = y2 - y1

    dx = int(w * shrink_x)
    dy = int(h * shrink_y)

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = x2 - dx
    ny2 = y2 - dy

    if nx2 <= nx1 or ny2 <= ny1:
        return bbox_xyxy

    return (nx1, ny1, nx2, ny2)
@staticmethod
def _shrink_xyxy_bbox(bbox_xyxy, shrink_x=0.25, shrink_y=0.20):
    x1, y1, x2, y2 = bbox_xyxy

    w = x2 - x1
    h = y2 - y1

    dx = int(w * shrink_x)
    dy = int(h * shrink_y)

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = x2 - dx
    ny2 = y2 - dy

    if nx2 <= nx1 or ny2 <= ny1:
        return bbox_xyxy

    return (nx1, ny1, nx2, ny2)

@staticmethod
def _iou_xywh_vs_xyxy(ball_bbox_xywh, obj_bbox_xyxy):
    bx, by, bw, bh = ball_bbox_xywh
    bx2 = bx + bw
    by2 = by + bh

    ox1, oy1, ox2, oy2 = obj_bbox_xyxy

    ix1 = max(bx, ox1)
    iy1 = max(by, oy1)
    ix2 = min(bx2, ox2)
    iy2 = min(by2, oy2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    ball_area = max(1, bw * bh)
    obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))
    union = ball_area + obj_area - inter

    return inter / union if union > 0 else 0.0



def _bbox_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def _bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return int(x), int(y), int(x + w), int(y + h)


def _center_of_xywh(bbox):
    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)


def _distance(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


class BallTracker:
    def __init__(self, model_path="yolov8n.pt", yolo_conf=0.20):
        self.model_path = model_path
        self.model = YOLO(model_path) if model_path else None
        self.yolo_conf = yolo_conf

        self.ball_class_names = {"sports ball", "ball"}

    def _create_csrt(self):
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerCSRT_create()

    def sanitize_bbox(self,bbox, frame_shape):
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))

        # Reject non-positive sizes
        if w <= 0 or h <= 0:
            return None

        # Clamp origin
        x = max(0, x)
        y = max(0, y)

        # Clamp size so bbox stays inside frame
        if x >= frame_w or y >= frame_h:
            return None

        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return None

        return (x, y, w, h)

    def _is_reasonable_ball_box(self, bbox, min_size=4, max_size=80):
        x, y, w, h = bbox
        if w < min_size or h < min_size:
            return False
        if w > max_size or h > max_size:
            return False

        aspect = w / h
        if aspect < 0.4 or aspect > 2.5:
            return False

        return True

    def _find_ball_candidate_nearby(
        self,
        prev_frame,
        curr_frame,
        last_bbox,
        search_radius=120,
        min_area=8,
        max_area=250
    ):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        cx, cy = _center_of_xywh(last_bbox)

        h, w = curr_gray.shape[:2]
        sx1 = max(0, cx - search_radius)
        sy1 = max(0, cy - search_radius)
        sx2 = min(w, cx + search_radius)
        sy2 = min(h, cy + search_radius)

        roi = thresh[sy1:sy2, sx1:sx2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = float("inf")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            fx = sx1 + x
            fy = sy1 + y

            cand_bbox = (fx, fy, cw, ch)
            if not self._is_reasonable_ball_box(cand_bbox):
                continue


            cand_center = _center_of_xywh(cand_bbox)
            score = _distance((cx, cy), cand_center)

            if score < best_score:
                best_score = score
                best_bbox = cand_bbox

        return best_bbox

    def _detect_ball_with_yolo_nearby(
        self,
        frame,
        last_bbox=None,
        search_radius=180,
        imgsz=960
    ):
        """
        Returns bbox in xywh format or None.
        Uses YOLO either on whole frame or on a search ROI around last known position.
        """
        if self.model is None:
            return None

        h, w = frame.shape[:2]

        if last_bbox is None:
            sx1, sy1, sx2, sy2 = 0, 0, w, h
        else:
            cx, cy = _center_of_xywh(last_bbox)
            sx1 = max(0, cx - search_radius)
            sy1 = max(0, cy - search_radius)
            sx2 = min(w, cx + search_radius)
            sy2 = min(h, cy + search_radius)

        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            return None

        results = self.model(roi, conf=self.yolo_conf, imgsz=imgsz, verbose=False)
        if not results or results[0].boxes is None:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        best_bbox = None
        best_score = float("inf")

        target_center = None
        if last_bbox is not None:
            target_center = _center_of_xywh(last_bbox)

        for box, cls_id, conf in zip(boxes, cls_ids, confs):
            class_name = self.model.names[int(cls_id)]

            if class_name not in self.ball_class_names:
                continue

            x1, y1, x2, y2 = map(int, box)
            fx1, fy1 = sx1 + x1, sy1 + y1
            fx2, fy2 = sx1 + x2, sy1 + y2

            cand_bbox = _bbox_xyxy_to_xywh((fx1, fy1, fx2, fy2))
            if not self._is_reasonable_ball_box(cand_bbox):
                continue

            cand_center = _center_of_xywh(cand_bbox)

            if target_center is None:
                # Prefer higher confidence when we don't have history yet
                score = -float(conf)
            else:
                # Prefer nearest plausible YOLO ball to previous location
                score = _distance(target_center, cand_center) - 20.0 * float(conf)

            if score < best_score:
                best_score = score
                best_bbox = cand_bbox

        return best_bbox
    def _center_of_xyxy(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


    
    def _predict_next_center(self, last_bbox, prev_bbox):
        if prev_bbox is None:
            return _center_of_xywh(last_bbox)

        c1 = _center_of_xywh(prev_bbox)
        c2 = _center_of_xywh(last_bbox)

        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]

        return c2[0] + dx, c2[1] + dy 

    def _get_player_bboxes_for_frame(self, player_tracks, frame_idx):
        if player_tracks is None:
            return []

        players = player_tracks.get("players", [])
        if frame_idx >= len(players):
            return []

        players_this_frame = players[frame_idx]
        if isinstance(players_this_frame, dict):
            player_iter = players_this_frame.values()
        elif isinstance(players_this_frame, list):
            player_iter = players_this_frame
        else:
            return []

        out = []
        for player in player_iter:
            if not isinstance(player, dict):
                continue
            bbox = player.get("bbox")
            if bbox is None:
                continue
            out.append(tuple(map(int, bbox)))
        return out
    
    def _reject_if_on_player(
        self,
        cand_bbox_xywh,
        player_bboxes_xyxy,
        inside_margin=0,
        max_iou=0.25,
        shrink_x=0.25,
        shrink_y=0.20
    ):
        cand_center = _center_of_xywh(cand_bbox_xywh)

        for pb in player_bboxes_xyxy:
            # Use only the inner/core player area as rejection zone
            inner_pb = _shrink_xyxy_bbox(pb, shrink_x=shrink_x, shrink_y=shrink_y)

            # Reject if candidate center is deep inside player body
            if _point_inside_xyxy(cand_center, inner_pb, margin=inside_margin):
                return True

            # Also reject if overlap with inner core is too high
            iou = _iou_xywh_vs_xyxy(cand_bbox_xywh, inner_pb)
            if iou > max_iou:
                return True          
            

        return False

    def _history_span(self, centers):
        if len(centers) < 2:
            return 999.0

        xs = [p[0] for p in centers]
        ys = [p[1] for p in centers]

        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        return max(span_x, span_y)
    
    def track_ball_hybrid(
        self,
        video_frames,
        player_tracks=None,
        init_frame_idx=0,
        init_bbox=None,
        max_frames=None,
        yolo_every=5,
        motion_every=1,
        search_radius=70,
        warmup_frames=10,
        max_jump=40
    ):
        if max_frames is not None:
            video_frames = video_frames[:max_frames]

        ball_tracks = [[] for _ in range(len(video_frames))]

        if init_frame_idx >= len(video_frames):
            raise ValueError("init_frame_idx is outside video_frames length")

        if init_bbox is None:
            return {"ball": ball_tracks}

        tracker = self._create_csrt()

        x, y, w, h = map(int, init_bbox)
        init_bbox = (x, y, w, h)

        init_frame = video_frames[init_frame_idx]
        safe_bbox = self.sanitize_bbox(init_bbox, init_frame.shape)
        if safe_bbox is None:
            logging.error(f"Invalid init_bbox: {init_bbox}, frame size: {init_frame.shape[:2]}")
            return {"ball": ball_tracks}

        center_history = deque(maxlen=10)
        center_history.append(_center_of_xywh(safe_bbox))
        still_count = 0

        ok = tracker.init(init_frame, safe_bbox)
        logging.info(f"tracker.init ok = {ok}")

        debug_frame = init_frame.copy()
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.imwrite("debug_init_frame_with_bbox.jpg", debug_frame)

        if not ok:
            logging.error("Failed to initialize hybrid ball tracker")
            return {"ball": ball_tracks}

        last_good_bbox = safe_bbox
        prev_good_bbox = None

        ball_tracks[init_frame_idx].append({
            "bbox": _bbox_xywh_to_xyxy(safe_bbox),
            "source": "init"
        })

        for frame_idx in range(init_frame_idx + 1, len(video_frames)):
            frame = video_frames[frame_idx]
            accepted_bbox = None
            source = None

            player_bboxes = self._get_player_bboxes_for_frame(player_tracks, frame_idx)
            predicted_center = self._predict_next_center(last_good_bbox, prev_good_bbox)
            frames_since_init = frame_idx - init_frame_idx

            # 1. CSRT first
            ok, bbox = tracker.update(frame)
            if ok:
                csrt_bbox = tuple(map(int, bbox))
                csrt_bbox = self.sanitize_bbox(csrt_bbox, frame.shape)

                if csrt_bbox is not None and self._is_reasonable_ball_box(csrt_bbox):
                    new_center = _center_of_xywh(csrt_bbox)
                    jump = _distance(_center_of_xywh(last_good_bbox), new_center)
                    pred_dist = _distance(predicted_center, new_center)

                    if jump < max_jump and pred_dist < max_jump * 1.5:
                        if not self._reject_if_on_player(csrt_bbox, player_bboxes):
                            accepted_bbox = csrt_bbox
                            source = "csrt"

            # 2. Warmup: trust only CSRT
            if frames_since_init < warmup_frames:
                if accepted_bbox is not None:
                    new_center = _center_of_xywh(accepted_bbox)
                    last_center = _center_of_xywh(last_good_bbox)
                    move_px = _distance(last_center, new_center)

                    # frame-to-frame stillness
                    if move_px <= 3:
                        still_count += 1
                    else:
                        still_count = 0

                    # history-based stillness
                    tmp_history = list(center_history) + [new_center]
                    xs = [p[0] for p in tmp_history]
                    ys = [p[1] for p in tmp_history]
                    span = max(max(xs) - min(xs), max(ys) - min(ys)) if len(tmp_history) >= 2 else 999

                    # reject likely static false lock (cone, shoe patch, etc.)
                    if still_count >= 8 and span < 6:
                        logging.info(
                            f"Rejected static candidate at frame {frame_idx} "
                            f"(source={source}, move={move_px:.1f}, span={span:.1f})"
                        )
                        accepted_bbox = None
                        source = None
                    else:
                        center_history.append(new_center)
                        prev_good_bbox = last_good_bbox
                        last_good_bbox = accepted_bbox
                        ball_tracks[frame_idx].append({
                            "bbox": _bbox_xywh_to_xyxy(accepted_bbox),
                            "source": source
                        })

                if accepted_bbox is None:
                    logging.info(f"Ball lost at frame {frame_idx}")
                continue

            # 3. YOLO correction
            if frame_idx % yolo_every == 0:
                yolo_bbox = self._detect_ball_with_yolo_nearby(
                    frame=frame,
                    last_bbox=last_good_bbox,
                    search_radius=search_radius,
                    imgsz=960
                )

                if yolo_bbox is not None:
                    yolo_bbox = self.sanitize_bbox(yolo_bbox, frame.shape)
                    if yolo_bbox is not None and self._is_reasonable_ball_box(yolo_bbox):
                        yolo_center = _center_of_xywh(yolo_bbox)
                        jump = _distance(_center_of_xywh(last_good_bbox), yolo_center)
                        pred_dist = _distance(predicted_center, yolo_center)

                        if jump < max_jump and pred_dist < max_jump * 1.5:
                            if not self._reject_if_on_player(yolo_bbox, player_bboxes):
                                accepted_bbox = yolo_bbox
                                source = "yolo"
                                tracker = self._create_csrt()
                                tracker.init(frame, accepted_bbox)

            # 4. Motion fallback
            if accepted_bbox is None and frame_idx > 0 and frame_idx % motion_every == 0:
                prev_frame = video_frames[frame_idx - 1]
                motion_bbox = self._find_ball_candidate_nearby(
                    prev_frame=prev_frame,
                    curr_frame=frame,
                    last_bbox=last_good_bbox,
                    search_radius=search_radius,
                    min_area=6,
                    max_area=220
                )

                if motion_bbox is not None:
                    motion_bbox = self.sanitize_bbox(motion_bbox, frame.shape)
                    if motion_bbox is not None and self._is_reasonable_ball_box(motion_bbox):
                        motion_center = _center_of_xywh(motion_bbox)
                        jump = _distance(_center_of_xywh(last_good_bbox), motion_center)
                        pred_dist = _distance(predicted_center, motion_center)

                        if jump < max_jump and pred_dist < max_jump * 1.5:
                            if not self._reject_if_on_player(motion_bbox, player_bboxes):
                                accepted_bbox = motion_bbox
                                source = "motion"
                                tracker = self._create_csrt()
                                tracker.init(frame, accepted_bbox)

            if accepted_bbox is not None:
                prev_good_bbox = last_good_bbox
                last_good_bbox = accepted_bbox
                ball_tracks[frame_idx].append({
                    "bbox": _bbox_xywh_to_xyxy(accepted_bbox),
                    "source": source
                })
            else:
                logging.info(f"Ball lost at frame {frame_idx}")

        return {"ball": ball_tracks}    
          
    def draw_tracks_on_video(self, video_frames, player_tracks, ball_tracks, output_path):
        if len(video_frames) == 0:
            raise ValueError("video_frames is empty")

        height, width = video_frames[0].shape[:2]
        logging.info(f"Drawing tracks on video with resolution: {width}x{height}")

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (width, height)
        )

        # for debug just trim the frame from 0 to 60

        # video_frames = video_frames[:60]

        for frame_idx, frame in enumerate(video_frames):
            annotated = frame.copy()

            current_players = player_tracks.get("players", [])
            current_ball = ball_tracks.get("ball", [])
            players_this_frame = current_players[frame_idx] if frame_idx < len(current_players) else {}
            logging.info(f"Frame {frame_idx}: {len(players_this_frame)} players, {len(current_ball[frame_idx]) if frame_idx < len(current_ball) else 0} balls")

            # normalize to iterable of player dicts
            if isinstance(players_this_frame, dict):
                player_iter = players_this_frame.values()
            elif isinstance(players_this_frame, list):
                player_iter = players_this_frame
            else:
                player_iter = []

            for player in player_iter:
                if not isinstance(player, dict):
                    continue

                bbox = player.get("bbox")
                if bbox is None:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    "player",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            ball_this_frame = current_ball[frame_idx] if frame_idx < len(current_ball) else []
            logging.info(f"Frame {frame_idx}: {len(ball_this_frame)} balls detected")

            # draw ball
            for ball in ball_this_frame:
                bbox = ball.get("bbox")
                if not bbox:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                source = ball.get("source", "unknown")
                cv2.putText(
                    annotated,
                    f"ball:{source}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2
                )

            out.write(annotated)

        out.release()
        logging.info(f"Saved final output to {output_path}")