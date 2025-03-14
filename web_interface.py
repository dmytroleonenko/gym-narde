#!/usr/bin/env python3

"""
Narde Web Interface

This script launches a web interface that allows playing against a trained
Deep Q-Network (DQN) model in the Narde backgammon variant game.
"""

import os
import sys
import atexit
import logging
from datetime import datetime
from web.app import app

def display_feedback_on_exit():
    """Display user feedback and logs when the server shuts down"""
    try:
        # Check if user feedback exists
        if os.path.exists("user_feedback.txt"):
            with open("user_feedback.txt", "r") as f:
                feedback_entries = f.read().strip()
            
            if feedback_entries:
                print("\n" + "="*80)
                print("USER FEEDBACK HISTORY:")
                print("-"*80)
                print(feedback_entries)
                print("="*80 + "\n")
        
        # Display path to log file
        if os.path.exists("debug.log"):
            print(f"Debug logs have been saved to: {os.path.abspath('debug.log')}")
            
            # Get the last 20 lines of the log file
            with open("debug.log", "r") as f:
                log_lines = f.readlines()
                last_lines = log_lines[-20:]
            
            print("\nLast 20 log entries:")
            print("-"*80)
            for line in last_lines:
                print(line.strip())
            print("-"*80)
            
    except Exception as e:
        print(f"Error displaying feedback: {e}")

def clear_logs():
    """Clear log files to start fresh with each server run"""
    # Clear debug log but keep a backup
    if os.path.exists("debug.log"):
        try:
            # Rename the existing log to include a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"debug_backup_{timestamp}.log"
            os.rename("debug.log", backup_name)
            print(f"Previous logs backed up to {backup_name}")
        except Exception as e:
            print(f"Error backing up log file: {e}")
    
    # Create empty log file
    with open("debug.log", "w") as f:
        f.write(f"=== New Log Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    # Also backup the feedback file but don't clear it
    if os.path.exists("user_feedback.txt"):
        try:
            with open("user_feedback.txt", "r") as f:
                feedback_content = f.read()
            
            # Create a backup with timestamp
            with open(f"user_feedback_backup_{timestamp}.txt", "w") as f:
                f.write(feedback_content)
            
            print(f"Previous feedback backed up to user_feedback_backup_{timestamp}.txt")
        except Exception as e:
            print(f"Error backing up feedback file: {e}")

if __name__ == "__main__":
    # Register the exit handler
    atexit.register(display_feedback_on_exit)
    
    # Clear logs at startup
    clear_logs()
    
    print("Starting Narde Web Interface...")
    print("Open your browser and navigate to http://localhost:5858")
    app.run(host='0.0.0.0', port=5858, debug=True)
