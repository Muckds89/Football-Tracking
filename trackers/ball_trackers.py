import cv2
from ultralytics import YOLO
import sys
sys.path.append('../')
import logging
logging.basicConfig(level=logging.INFO)

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
    def __init__(self, model_path=None):
        self.model_path = model_path

    def _create_csrt(self):
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerCSRT_create()

    def _find_ball_candidate_nearby(
        self,
        prev_frame,
        curr_frame,
        last_bbox,
        search_radius=120,
        min_area=8,
        max_area=250
    ):
        """
        Motion-based fallback detector near the previous ball position.
        Returns bbox in xywh format or None.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        lx, ly, lw, lh = last_bbox
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

            # map ROI contour bbox back to full-frame coords
            fx = sx1 + x
            fy = sy1 + y

            # reject weird elongated shapes
            if cw <= 0 or ch <= 0:
                continue
            aspect = cw / ch
            if aspect < 0.4 or aspect > 2.5:
                continue

            cand_bbox = (fx, fy, cw, ch)
            cand_center = _center_of_xywh(cand_bbox)

            # prefer candidates near previous center
            score = _distance((cx, cy), cand_center)

            if score < best_score:
                best_score = score
                best_bbox = cand_bbox

        return best_bbox

    def track_ball_with_csrt(self, video_frames, init_frame_idx, init_bbox, reinit_every=5):
        tracker = self._create_csrt()
        logging.info("Initialized CSRT tracker for ball tracking")

        ball_tracks = [[] for _ in range(len(video_frames))]
        logging.info(f"Tracking ball in {len(video_frames)} frames")

        if init_frame_idx >= len(video_frames):
            raise ValueError("init_frame_idx is outside video_frames length")

        init_frame = video_frames[init_frame_idx]
        x, y, w, h = map(int, init_bbox)
        init_bbox = (x, y, w, h)

        logging.info(f"Tracking ball in frame {init_frame_idx}")
        logging.info(f"init_frame shape: {init_frame.shape}, dtype: {init_frame.dtype}")

        ok = tracker.init(init_frame, init_bbox)
        logging.info(f"tracker.init ok = {ok}")

        debug_frame = init_frame.copy()
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.imwrite("debug_init_frame_with_bbox.jpg", debug_frame)

        if not ok:
            logging.error("Failed to initialize CSRT ball tracker")
            return {"ball": ball_tracks}

        last_good_bbox = init_bbox

        ball_tracks[init_frame_idx].append({
            "bbox": _bbox_xywh_to_xyxy(init_bbox)
        })

        for frame_idx in range(init_frame_idx + 1, len(video_frames)):
            frame = video_frames[frame_idx]
            ok, bbox = tracker.update(frame)

            accepted_bbox = None

            if ok:
                bx, by, bw, bh = map(int, bbox)

                # basic sanity checks to reduce drift
                if 4 <= bw <= 80 and 4 <= bh <= 80:
                    accepted_bbox = (bx, by, bw, bh)

            # periodic re-detection or recovery if tracking failed
            should_reinit = (frame_idx - init_frame_idx) % reinit_every == 0 or accepted_bbox is None

            if should_reinit and frame_idx > 0:
                prev_frame = video_frames[frame_idx - 1]
                candidate = self._find_ball_candidate_nearby(
                    prev_frame=prev_frame,
                    curr_frame=frame,
                    last_bbox=last_good_bbox,
                    search_radius=120,
                    min_area=8,
                    max_area=250
                )

                if candidate is not None:
                    accepted_bbox = candidate
                    tracker = self._create_csrt()
                    tracker.init(frame, accepted_bbox)
                    logging.info(f"Reinitialized ball tracker at frame {frame_idx} with bbox {accepted_bbox}")

            if accepted_bbox is not None:
                last_good_bbox = accepted_bbox
                ball_tracks[frame_idx].append({
                    "bbox": _bbox_xywh_to_xyxy(accepted_bbox)
                })

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

        video_frames = video_frames[:60]

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
                cv2.putText(
                    annotated,
                    "ball",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2
                )

            out.write(annotated)

        out.release()
        logging.info(f"Saved final output to {output_path}")