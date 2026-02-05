from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from .utils import ensure_dir


class ActionLogger:
    """Records face action history to timestamped files."""

    def __init__(self, output_dir: str = "action_history") -> None:
        """
        Initialize action logger.
        
        Args:
            output_dir: Directory to store action history files
        """
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.current_file: Optional[str] = None
        self.file_handle: Optional[any] = None

    def start_logging(self, identity_name: str) -> str:
        """
        Start logging for a specific identity.
        
        Args:
            identity_name: Name of the locked identity
            
        Returns:
            Path to the created log file
        """
        # Close any existing file
        self.stop_logging()
        
        # Create filename: <face>_history_<timestamp>.txt
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{identity_name.lower()}_history_{timestamp}.txt"
        self.current_file = os.path.join(self.output_dir, filename)
        
        # Open file for writing
        self.file_handle = open(self.current_file, "w", encoding="utf-8")
        
        # Write header
        self.file_handle.write(f"Face Locking Action History\n")
        self.file_handle.write(f"Identity: {identity_name}\n")
        self.file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 60 + "\n\n")
        self.file_handle.flush()
        
        return self.current_file

    def log_action(self, action: str, description: str = "", value: str = "") -> None:
        """
        Log a single action to the current file.
        
        Args:
            action: Action type (e.g., "moved_left", "blink", "smile")
            description: Optional description of the action
            value: Optional value or measurement
        """
        if self.file_handle is None:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        log_entry = f"[{timestamp}] {action}"
        if description:
            log_entry += f" - {description}"
        if value:
            log_entry += f" (value: {value})"
        log_entry += "\n"
        
        self.file_handle.write(log_entry)
        self.file_handle.flush()

    def log_lock_event(self, event_type: str, details: str = "") -> None:
        """
        Log lock-related events (lock established, lock released).
        
        Args:
            event_type: Type of lock event
            details: Additional details
        """
        if self.file_handle is None:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"\n[{timestamp}] *** {event_type.upper()} ***"
        if details:
            log_entry += f" - {details}"
        log_entry += "\n\n"
        
        self.file_handle.write(log_entry)
        self.file_handle.flush()

    def log_multiple_actions(self, actions: List[str]) -> None:
        """
        Log multiple actions detected in the same frame.
        
        Args:
            actions: List of action strings
        """
        for action in actions:
            description = self._get_action_description(action)
            self.log_action(action, description)

    def _get_action_description(self, action: str) -> str:
        """Get human-readable description for an action."""
        descriptions = {
            "moved_left": "Face moved to the left",
            "moved_right": "Face moved to the right",
            "blink": "Eye blink detected",
            "smile": "Smile or laugh detected",
        }
        return descriptions.get(action, "Action detected")

    def stop_logging(self) -> None:
        """Stop logging and close the current file."""
        if self.file_handle is not None:
            self.file_handle.write("\n" + "=" * 60 + "\n")
            self.file_handle.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.close()
            self.file_handle = None
            self.current_file = None

    def get_current_file(self) -> Optional[str]:
        """Get the path to the current log file."""
        return self.current_file

    def __del__(self) -> None:
        """Ensure file is closed on deletion."""
        self.stop_logging()
