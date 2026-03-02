# gesture_classifier.py

class GestureClassifier:
    def get_finger_status(self, hand_landmarks):
        """Return a 5-element list indicating whether each finger is open (1) or closed (0).

        Accepts either a MediaPipe-like `hand_landmarks` object (with `.landmark` list
        where each element has `.x`, `.y`, `.z`) or a sequence of 21 `(x,y,z)` tuples.
        Uses wrist-relative distance heuristics to be robust to flips and rotations.
        """
        fingers = []

        # helper to read a landmark
        def lm(i):
            try:
                # Prefer .landmark attribute (MediaPipe NormalizedLandmarkList)
                lm_list = getattr(hand_landmarks, 'landmark', None)
                if lm_list is not None:
                    l = lm_list[i]
                    return (float(getattr(l, 'x', 0.0)), float(getattr(l, 'y', 0.0)), float(getattr(l, 'z', 0.0)))

                # If passed a sequence of landmarks or tuples
                if isinstance(hand_landmarks, (list, tuple)) and len(hand_landmarks) > i:
                    item = hand_landmarks[i]
                    if hasattr(item, 'x'):
                        return (float(getattr(item, 'x', 0.0)), float(getattr(item, 'y', 0.0)), float(getattr(item, 'z', 0.0)))
                    if isinstance(item, (list, tuple)):
                        x = float(item[0]) if len(item) > 0 else 0.0
                        y = float(item[1]) if len(item) > 1 else 0.0
                        z = float(item[2]) if len(item) > 2 else 0.0
                        return (x, y, z)
            except Exception:
                pass
            return (0.0, 0.0, 0.0)

        # wrist base
        wx, wy, wz = lm(0)

        import math
        # compute hand scale (max distance from wrist to other landmarks)
        dists = []
        for i in range(1, 21):
            x, y, z = lm(i)
            d = math.sqrt((x - wx) ** 2 + (y - wy) ** 2 + (z - wz) ** 2)
            dists.append(d)
        hand_size = max(dists) if dists else 1.0
        if hand_size == 0:
            hand_size = 1.0

        TH = 0.12  # threshold fraction of hand size

        # Thumb: compare tip(4) vs ip(3) distance from wrist
        try:
            tx, ty, tz = lm(4)
            ipx, ipy, ipz = lm(3)
            dist_tip = math.sqrt((tx - wx) ** 2 + (ty - wy) ** 2 + (tz - wz) ** 2)
            dist_ip = math.sqrt((ipx - wx) ** 2 + (ipy - wy) ** 2 + (ipz - wz) ** 2)
            fingers.append(1 if (dist_tip - dist_ip) > TH * hand_size else 0)
        except Exception:
            fingers.append(0)

        # other fingers: tip indices [8,12,16,20], pip indices [6,10,14,18]
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for t, p in zip(tips, pips):
            try:
                tx, ty, tz = lm(t)
                px, py, pz = lm(p)
                dist_tip = math.sqrt((tx - wx) ** 2 + (ty - wy) ** 2 + (tz - wz) ** 2)
                dist_pip = math.sqrt((px - wx) ** 2 + (py - wy) ** 2 + (pz - wz) ** 2)
                fingers.append(1 if (dist_tip - dist_pip) > TH * hand_size else 0)
            except Exception:
                fingers.append(0)

        return fingers

    def classify(self, fingers):
        """Classify fingers into a single digit string '0'..'9'.

        Supported inputs:
        - a single 5-element list `fingers` -> returns 0..5 by raised finger count
        - a list of two 5-element lists `[left_fingers, right_fingers]` ->
          two-hand mode: if left hand shows all five fingers (modifier), add 5 to
          right-hand count to obtain 5..9. Otherwise returns right-hand count.

        Returns 'UNKNOWN' when the combination doesn't map to 0..9.
        """
        # two-hand mode when list of two finger-lists passed
        try:
            # detect if this is a pair of hands
            if isinstance(fingers, list) and len(fingers) == 2 and all(isinstance(f, list) and len(f) == 5 for f in fingers):
                    left, right = fingers
                    left_count = left.count(1)
                    right_count = right.count(1)
                    # combined value: sum of raised fingers across both hands
                    value = left_count + right_count
                    # clamp to 0..10 and return as string
                    if value < 0:
                        return "UNKNOWN"
                    value = min(value, 10)
                    return str(value)

            # single hand mode
            if isinstance(fingers, list) and len(fingers) == 5:
                total = fingers.count(1)
                if 0 <= total <= 5:
                    return str(total)
                return "UNKNOWN"

        except Exception:
            pass

        return "UNKNOWN"