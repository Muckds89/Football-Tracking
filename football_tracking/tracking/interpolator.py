class Interpolator:
    def __init__(self):
        pass
    def interpolate_ball_tracks(ball_tracks, max_gap=5):
        filled = ball_tracks[:]
        n = len(filled)

        def center_of(track):
            if track is None:
                return None
            if track.get("center") is not None:
                return track["center"]
            if track.get("bbox") is not None:
                x1, y1, x2, y2 = track["bbox"]
                return [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            return None

        i = 0
        while i < n:
            if filled[i] is not None:
                i += 1
                continue

            start_gap = i - 1
            j = i
            while j < n and filled[j] is None:
                j += 1
            end_gap = j

            gap_len = end_gap - i

            if (
                start_gap >= 0 and
                end_gap < n and
                gap_len <= max_gap and
                filled[start_gap] is not None and
                filled[end_gap] is not None
            ):
                c1 = center_of(filled[start_gap])
                c2 = center_of(filled[end_gap])

                if c1 is not None and c2 is not None:
                    for k in range(1, gap_len + 1):
                        alpha = k / (gap_len + 1)
                        cx = int(c1[0] * (1 - alpha) + c2[0] * alpha)
                        cy = int(c1[1] * (1 - alpha) + c2[1] * alpha)

                        filled[i + k - 1] = {
                            "frame": i + k - 1,
                            "center": [cx, cy],
                            "bbox": None,
                            "conf": 0.0,
                            "interpolated": True
                        }

            i = end_gap

        return filled