import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# DOTS and MESH (magenta)
drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 255))
mesh_spec    = mp_draw.DrawingSpec(thickness=1, color=(255, 0, 255))

# GREEN style ONLY for eye contours
eye_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))

# RED style ONLY for lips
lips_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))

# CYAN style ONLY for nose contour
nose_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=1, color=(255, 255, 0))

# Iris indices
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

def draw_iris_circle(img, face_landmarks, indices, color=(0, 255, 255)):
    h, w, _ = img.shape
    cx = int(face_landmarks.landmark[indices[0]].x * w)
    cy = int(face_landmarks.landmark[indices[0]].y * h)
    center = (cx, cy)

    dists = []
    for idx in indices[1:]:
        px = int(face_landmarks.landmark[idx].x * w)
        py = int(face_landmarks.landmark[idx].y * h)
        d = math.dist([cx, cy], [px, py])
        dists.append(d)

    if dists:
        radius = int(sum(dists) / len(dists))
        cv2.circle(img, center, radius, color, 2)

def draw_nostril_circle(img, face_landmarks, outer_idx, inner_idx, color=(0, 165, 255)):
    """Draw a small circle per nostril using outer+inner as diameter."""
    h, w, _ = img.shape

    ox = face_landmarks.landmark[outer_idx].x * w
    oy = face_landmarks.landmark[outer_idx].y * h
    ix = face_landmarks.landmark[inner_idx].x * w
    iy = face_landmarks.landmark[inner_idx].y * h

    cx = int((ox + ix) / 2.0)
    cy = int((oy + iy) / 2.0)
    radius = int(math.dist([ox, oy], [ix, iy]) / 2.0)

    if radius > 0:
        cv2.circle(img, (cx, cy), radius, color, 2)

# Open webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # --- TESSELATION (magenta, unchanged) ---
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # ONE 'L'
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mesh_spec
                )

                # --- EYE CONTOURS (green) ---
                if hasattr(mp_face_mesh, "FACEMESH_LEFT_EYE"):
                    mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eye_spec
                    )
                if hasattr(mp_face_mesh, "FACEMESH_RIGHT_EYE"):
                    mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=eye_spec
                    )

                # --- LIP CONTOUR (red) ---
                if hasattr(mp_face_mesh, "FACEMESH_LIPS"):
                    mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=lips_spec
                    )

                # --- NOSE CONTOUR (cyan) ---
                if hasattr(mp_face_mesh, "FACEMESH_NOSE"):
                    mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_NOSE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=nose_spec
                    )

                # --- NOSTRIL CIRCLES (orange) ---
                # left nostril: outer=98, inner=97
                draw_nostril_circle(frame, face_landmarks, 98, 97)
                # right nostril: outer=327, inner=326
                draw_nostril_circle(frame, face_landmarks, 327, 326)

                # --- IRIS CIRCLES (yellow) ---
                draw_iris_circle(frame, face_landmarks, LEFT_IRIS)
                draw_iris_circle(frame, face_landmarks, RIGHT_IRIS)

        cv2.imshow("Face Mesh + Regions (Eyes/Lips/Nose/Nostrils/Iris)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()