"""Speaker tracking: largest face = active speaker, with transition guards."""

from config import SPEAKER_SIZE_RATIO, SPEAKER_SWITCH_FRAMES


class SpeakerTracker:
    def __init__(self):
        self.current_speaker = None       # (person_id, person_name)
        self._candidate = None            # who we might switch to
        self._candidate_count = 0         # consecutive frames the candidate is largest

    def update(self, faces_with_identity):
        """Given a list of (face, person_id, person_name, confidence), return the active speaker.

        Args:
            faces_with_identity: list of tuples (face_obj, person_id, person_name, confidence)
                where face_obj has .bbox attribute.

        Returns:
            (person_id, person_name, confidence) of the active speaker, or None if in transition.
        """
        if not faces_with_identity:
            return None

        # Sort by face area descending (largest = closest to camera)
        sorted_faces = sorted(
            faces_with_identity,
            key=lambda x: _face_area(x[0]),
            reverse=True,
        )

        largest = sorted_faces[0]
        largest_face, largest_pid, largest_name, largest_conf = largest

        # ── Guard 1: Size ratio ──────────────────────────────────
        # If there's a second face and it's nearly as large, we're in transition
        if len(sorted_faces) >= 2:
            second = sorted_faces[1]
            largest_area = _face_area(largest_face)
            second_area = _face_area(second[0])

            if second_area > 0:
                ratio = largest_area / second_area
                if ratio < SPEAKER_SIZE_RATIO:
                    # Transition happening, skip this frame
                    return None

        # ── Guard 2: Stability buffer ────────────────────────────
        speaker_key = (largest_pid, largest_name)

        if self.current_speaker is None:
            # First detection ever — accept immediately
            self.current_speaker = speaker_key
            self._candidate = None
            self._candidate_count = 0
            return (largest_pid, largest_name, largest_conf)

        if speaker_key == self.current_speaker:
            # Same speaker, reset any pending candidate
            self._candidate = None
            self._candidate_count = 0
            return (largest_pid, largest_name, largest_conf)

        # Different person is now largest — track as candidate
        if speaker_key == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = speaker_key
            self._candidate_count = 1

        if self._candidate_count >= SPEAKER_SWITCH_FRAMES:
            # Stable enough — switch speaker
            self.current_speaker = speaker_key
            self._candidate = None
            self._candidate_count = 0
            return (largest_pid, largest_name, largest_conf)

        # Still in transition — return current speaker
        current_pid, current_name = self.current_speaker
        # Find confidence for current speaker in this frame
        for face, pid, name, conf in faces_with_identity:
            if (pid, name) == self.current_speaker:
                return (current_pid, current_name, conf)

        # Current speaker not in frame at all — still return them during buffer
        return (current_pid, current_name, 0.0)

    def reset(self):
        """Reset tracker state (e.g. between chunks)."""
        self.current_speaker = None
        self._candidate = None
        self._candidate_count = 0


def _face_area(face):
    """Calculate bounding box area for a face."""
    bbox = face.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height
