#!/usr/bin/env python3
"""
Face Locking System - Main Entry Point
Simple usage: python3 main.py
"""

import os
import sys


def get_python_command():
    """Get the correct Python command for this system."""
    # Use the same Python that's running this script
    return sys.executable


def list_enrolled_identities():
    """List all enrolled identities."""
    identities_dir = "data/identities"
    if not os.path.exists(identities_dir):
        return []
    
    identities = [d for d in os.listdir(identities_dir) 
                  if os.path.isdir(os.path.join(identities_dir, d)) and d != ".gitkeep"]
    return identities


def main():
    """Main entry point."""
    python_cmd = get_python_command()
    
    print("\n" + "=" * 60)
    print("FACE LOCKING SYSTEM")
    print("=" * 60)
    
    # Check for enrolled identities
    identities = list_enrolled_identities()
    
    if not identities:
        print("\nNo enrolled faces found.")
        print("\nLet's enroll your face first!")
        print("=" * 60)
        os.system(f"{python_cmd} live_enroll.py")
        
        # Check again after enrollment
        identities = list_enrolled_identities()
        if not identities:
            print("\n✗ Enrollment failed or cancelled.")
            return
    
    print(f"\nFound {len(identities)} enrolled face(s):")
    for i, name in enumerate(identities, 1):
        print(f"  {i}. {name}")
    
    print("\nWhat would you like to do?")
    print("  1. Lock onto existing face")
    print("  2. Enroll a new face")
    print("  3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 60)
        print("Starting Face Locking System...")
        print("=" * 60 + "\n")
        os.system(f"{python_cmd} -m src.run_pipeline")
    
    elif choice == "2":
        print("\n" + "=" * 60)
        print("Starting Live Enrollment...")
        print("=" * 60 + "\n")
        os.system(f"{python_cmd} live_enroll.py")
        
        # After enrollment, ask if they want to test
        print("\n" + "=" * 60)
        test = input("Start face locking now? (yes/no): ").strip().lower()
        if test == "yes":
            os.system(f"{python_cmd} -m src.run_pipeline")
    
    elif choice == "3":
        print("\n✓ Goodbye!")
        print("=" * 60 + "\n")
    
    else:
        print("\n✗ Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        print("=" * 60 + "\n")
        sys.exit(0)
