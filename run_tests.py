#!/usr/bin/env python3
"""Run all tests for the Narde environment and training utilities."""

import os
import sys
import importlib
import unittest
import pytest
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    print("=" * 80)

def run_test_module(module_name):
    """Run a test module and return success status."""
    print_header(f"Running {module_name}")
    
    try:
        # Import the module dynamically
        test_module = importlib.import_module(module_name)
        
        # Find and run all test functions in the module
        test_functions = []
        for name in dir(test_module):
            if name.startswith('test_'):
                test_func = getattr(test_module, name)
                if callable(test_func):
                    test_functions.append(test_func)
        
        # Run each test function
        for test_func in test_functions:
            print(f"\n{Fore.YELLOW}Running {test_func.__name__}...{Style.RESET_ALL}")
            try:
                test_func()
                print(f"{Fore.GREEN}✓ {test_func.__name__} passed{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✗ {test_func.__name__} failed: {str(e)}{Style.RESET_ALL}")
                return False
        
        return True
    
    except Exception as e:
        print(f"{Fore.RED}Error running {module_name}: {str(e)}{Style.RESET_ALL}")
        return False

def main():
    """Run all tests."""
    # Define the test modules to run
    test_modules = [
        "tests.test_gym_interface",
        "tests.test_narde_env", 
        "tests.test_game_mechanics",
        "tests.test_worker_functions"
    ]
    
    # Track overall success
    all_passed = True
    
    # Run each test module
    for module in test_modules:
        if not run_test_module(module):
            all_passed = False
    
    # Print summary
    print_header("Test Results")
    if all_passed:
        print(f"{Fore.GREEN}All tests passed successfully!{Style.RESET_ALL}")
        return 0
    else:
        print(f"{Fore.RED}Some tests failed. See output above for details.{Style.RESET_ALL}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 