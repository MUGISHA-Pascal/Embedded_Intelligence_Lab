from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np


class FaceLocker:
    """Manages face locking state for a specific identity."""

    def __init__(
        self,
        target_identity: str,
        lock_threshold: float = 0.45,
        unlock_timeout: float = 2.0,
    ) -> None:
        """
        Initialize face locker.
        
        Args:
            target_identity: Name of the identity to lock onto
            lock_threshold: Minimum similarity score to establish/maintain lock
            unlock_timeout: Seconds without detection before releasing lock
        """
        self.target_identity = target_identity
        self.lock_threshold = lock_threshold
        self.unlock_timeout = unlock_timeout
        
        self.is_locked = False
        self.locked_box: Optional[Tuple[int, int, int, int]] = None
        self.last_seen_time: Optional[float] = None
        self.lock_start_time: Optional[float] = None
        self.frame_count_since_lock = 0

    def update(
        self,
        detected_faces: list[Tuple[Tuple[int, int, int, int], str, float]],
        current_time: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Update lock state based on detected faces.
        
        Args:
            detected_faces: List of (box, name, score) tuples from recognition
            current_time: Current timestamp
            
        Returns:
            Locked face box if locked, None otherwise
        """
        target_face = None
        best_score = -1.0
        
        # Find the target identity in detected faces
        for box, name, score in detected_faces:
            if name == self.target_identity and score >= self.lock_threshold:
                if score > best_score:
                    best_score = score
                    target_face = box
        
        # If target found
        if target_face is not None:
            self.last_seen_time = current_time
            
            if not self.is_locked:
                # Establish lock
                self.is_locked = True
                self.locked_box = target_face
                self.lock_start_time = current_time
                self.frame_count_since_lock = 0
            else:
                # Update locked box position
                self.locked_box = target_face
                self.frame_count_since_lock += 1
            
            return self.locked_box
        
        # Target not found - check if we should maintain or release lock
        if self.is_locked:
            if self.last_seen_time is not None:
                time_since_seen = current_time - self.last_seen_time
                if time_since_seen > self.unlock_timeout:
                    # Release lock
                    self.release_lock()
                else:
                    # Maintain lock with last known position
                    return self.locked_box
        
        return None

    def release_lock(self) -> None:
        """Release the current lock."""
        self.is_locked = False
        self.locked_box = None
        self.last_seen_time = None
        self.lock_start_time = None
        self.frame_count_since_lock = 0

    def get_lock_duration(self, current_time: float) -> float:
        """Get duration of current lock in seconds."""
        if self.is_locked and self.lock_start_time is not None:
            return current_time - self.lock_start_time
        return 0.0
