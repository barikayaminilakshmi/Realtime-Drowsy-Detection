import numpy as np

def euclidean(p1, p2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))

def eye_aspect_ratio(pts):
    # pts = 6 points around the eye
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def mouth_aspect_ratio(mouth_pts):
    # mouth_pts = [left, right, top, bottom]
    left, right, top, bottom = mouth_pts
    horiz = euclidean(left, right)
    vert = euclidean(top, bottom)
    return (vert / horiz) if horiz != 0 else 0.0