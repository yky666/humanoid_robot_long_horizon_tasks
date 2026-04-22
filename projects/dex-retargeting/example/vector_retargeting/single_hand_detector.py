import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type
        self.fallback_detected_hand_type = inverse_hand_dict[hand_type]
        # Temporal consistency: prevent wrist frame flips (palm ↔ dorsal)
        self._prev_wrist_rot = None
        # Wrist rotation EMA smoothing (reduces frame jitter → mapping bias)
        self._smooth_wrist_rot = None
        self._wrist_rot_alpha = 0.45  # EMA weight for new frame (lower = smoother)
        # Jitter-based adaptive smoothing (dorsal / occluded scenarios)
        self._raw_jp_history = []      # last 3 raw joint_pos (before smoothing)
        self._smooth_joint_pos = None  # EMA-smoothed output
        self._jitter_ema = 0.0         # running jitter score (0 = stable, 1 = max jitter)
        _FINGER_TIP_IDS = [4, 8, 12, 16, 20]
        self._tip_ids = _FINGER_TIP_IDS

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"
    ):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1
                )

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )

        return image

    def detect(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None

        num_box = len(results.multi_hand_landmarks)
        if num_box == 1:
            desired_hand_num = 0
        else:
            desired_hand_num = -1
        fallback_hand_num = -1
        if desired_hand_num < 0:
            for i in range(num_box):
                label = results.multi_handedness[i].ListFields()[0][1][0].label
                if label == self.detected_hand_type:
                    desired_hand_num = i
                    break
                if label == self.fallback_detected_hand_type and fallback_hand_num < 0:
                    fallback_hand_num = i
            if desired_hand_num < 0:
                desired_hand_num = fallback_hand_num
        if desired_hand_num < 0:
            return 0, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]

        # Parse 3d keypoint from MediaPipe hand detector
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)

        # --- Temporal consistency: prevent sudden frame flips (palm ↔ dorsal) ---
        if self._prev_wrist_rot is not None:
            R_diff = self._prev_wrist_rot.T @ mediapipe_wrist_rot
            cos_angle = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle > np.pi / 2:  # > 90 deg → likely flipped
                # Try flipping normal (col-1) and z (col-2) of the frame
                flipped_rot = mediapipe_wrist_rot.copy()
                flipped_rot[:, 1] *= -1
                flipped_rot[:, 2] *= -1
                R_diff_f = self._prev_wrist_rot.T @ flipped_rot
                cos_f = np.clip((np.trace(R_diff_f) - 1.0) / 2.0, -1.0, 1.0)
                if np.arccos(cos_f) < angle:
                    mediapipe_wrist_rot = flipped_rot
        self._prev_wrist_rot = mediapipe_wrist_rot.copy()

        # --- Wrist rotation EMA smoothing ---
        # Reduce frame-to-frame orientation noise that causes mapping bias
        if self._smooth_wrist_rot is not None:
            # Adaptive alpha: smooth more when jitter is high
            alpha_r = self._wrist_rot_alpha
            if self._jitter_ema > 0.30:
                alpha_r = 0.15  # very smooth for dorsal/occluded
            # Linear interpolation of rotation matrices + re-orthogonalize
            blended = alpha_r * mediapipe_wrist_rot + (1 - alpha_r) * self._smooth_wrist_rot
            U, _, Vt = np.linalg.svd(blended)
            mediapipe_wrist_rot = U @ Vt  # nearest orthogonal matrix
            # Ensure proper rotation (det = +1)
            if np.linalg.det(mediapipe_wrist_rot) < 0:
                U[:, -1] *= -1
                mediapipe_wrist_rot = U @ Vt
        self._smooth_wrist_rot = mediapipe_wrist_rot.copy()

        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        # --- Jitter-based adaptive smoothing ---
        # Keep a short history of raw (unsmoothed) joint_pos for jitter detection
        self._raw_jp_history.append(joint_pos.copy())
        if len(self._raw_jp_history) > 3:
            self._raw_jp_history.pop(0)

        alpha = 1.0  # default: no smoothing
        if len(self._raw_jp_history) >= 3:
            # Detect direction reversals at finger tips between consecutive frames
            d_curr = self._raw_jp_history[-1] - self._raw_jp_history[-2]
            d_prev = self._raw_jp_history[-2] - self._raw_jp_history[-3]
            dots = np.sum(d_curr * d_prev, axis=1)  # [21]
            tip_dots = dots[self._tip_ids]
            # fraction of finger tips that reversed direction
            jitter_ratio = float(np.mean(tip_dots < 0))
            # EMA of jitter ratio for temporal stability
            self._jitter_ema = 0.4 * jitter_ratio + 0.6 * self._jitter_ema

            if self._jitter_ema > 0.55:
                alpha = 0.12  # very heavy smoothing — dorsal fist jitter
            elif self._jitter_ema > 0.30:
                t = (self._jitter_ema - 0.30) / 0.25
                alpha = 1.0 - t * 0.88  # 1.0 → 0.12 linearly
            # else alpha stays 1.0

        if self._smooth_joint_pos is not None:
            joint_pos = alpha * joint_pos + (1 - alpha) * self._smooth_joint_pos
        self._smooth_joint_pos = joint_pos.copy()

        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    ) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        # Use 5 palm-plane points for robust SVD normal estimation
        # [wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp]
        palm_points = keypoint_3d_array[[0, 5, 9, 13, 17], :]

        # Compute vector from wrist to middle_mcp
        x_vector = palm_points[0] - palm_points[2]

        # Direction from pinky_mcp to index_mcp (for z-axis check)
        pinky_to_index = palm_points[1] - palm_points[4]  # index_mcp - pinky_mcp

        # Normal fitting with SVD on 5 points (much more stable than 3)
        palm_centered = palm_points - np.mean(palm_points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(palm_centered)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / (np.linalg.norm(x) + 1e-8)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * pinky_to_index) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
