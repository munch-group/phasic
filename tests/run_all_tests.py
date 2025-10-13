#!/usr/bin/env python3
"""
Master test runner for PtDAlgorithms Python API test suite.

Runs all test files in the correct order and reports results.
"""

import subprocess
import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Change to tests directory
TESTS_DIR = Path(__file__).parent
os.chdir(TESTS_DIR.parent)

class TestRunner:
    def __init__(self):
        self.results = []
        self.total_passed = 0
        self.total_failed = 0

    def run_test(self, name, command, description, required_deps=None):
        """Run a single test command and record results."""
        print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
        print(f"{BOLD}{BLUE}Running: {name}{RESET}")
        print(f"{description}")
        if required_deps:
            print(f"Required: {', '.join(required_deps)}")
        print(f"{BLUE}{'='*80}{RESET}\n")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=False,
                text=True,
                timeout=300  # 5 minute timeout per test
            )

            if result.returncode == 0:
                print(f"\n{GREEN}✓ {name} PASSED{RESET}")
                self.results.append((name, True, None))
                self.total_passed += 1
                return True
            else:
                print(f"\n{RED}✗ {name} FAILED (exit code: {result.returncode}){RESET}")
                self.results.append((name, False, f"Exit code: {result.returncode}"))
                self.total_failed += 1
                return False

        except subprocess.TimeoutExpired:
            print(f"\n{RED}✗ {name} TIMEOUT{RESET}")
            self.results.append((name, False, "Timeout"))
            self.total_failed += 1
            return False
        except FileNotFoundError as e:
            print(f"\n{YELLOW}⊘ {name} SKIPPED - {e}{RESET}")
            self.results.append((name, None, f"Skipped: {e}"))
            return None
        except Exception as e:
            print(f"\n{RED}✗ {name} ERROR: {e}{RESET}")
            self.results.append((name, False, str(e)))
            self.total_failed += 1
            return False

    def print_summary(self):
        """Print final test summary."""
        print(f"\n\n{BOLD}{BLUE}{'='*80}{RESET}")
        print(f"{BOLD}{BLUE}TEST SUMMARY{RESET}")
        print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

        for name, passed, error in self.results:
            if passed is True:
                print(f"{GREEN}✓ {name}{RESET}")
            elif passed is False:
                print(f"{RED}✗ {name}{RESET}")
                if error:
                    print(f"  {RED}  → {error}{RESET}")
            else:
                print(f"{YELLOW}⊘ {name} - SKIPPED{RESET}")
                if error:
                    print(f"  {YELLOW}  → {error}{RESET}")

        print(f"\n{BOLD}Total: {self.total_passed + self.total_failed} tests{RESET}")
        print(f"{GREEN}Passed: {self.total_passed}{RESET}")
        print(f"{RED}Failed: {self.total_failed}{RESET}")

        if self.total_failed == 0:
            print(f"\n{BOLD}{GREEN}ALL TESTS PASSED! 🎉{RESET}")
            return 0
        else:
            print(f"\n{BOLD}{RED}SOME TESTS FAILED{RESET}")
            return 1

def main():
    runner = TestRunner()

    # Test 1: Standalone comprehensive API test (no pytest needed)
    runner.run_test(
        name="Comprehensive API (Standalone)",
        command="python tests/test_api_comprehensive.py",
        description="Standalone comprehensive test suite (no pytest required) - 31 tests",
        required_deps=["ptdalgorithms"]
    )

    # Test 2: Comprehensive API with pytest
    runner.run_test(
        name="Comprehensive API (pytest)",
        command="pytest tests/test_comprehensive_api.py -v",
        description="pytest-based comprehensive test suite - 80+ test methods, 500+ assertions",
        required_deps=["pytest", "ptdalgorithms"]
    )

    # Test 3: JAX integration tests
    runner.run_test(
        name="JAX Integration",
        command="pytest tests/test_jax_integration.py -v",
        description="JAX-specific functionality tests - 60+ tests",
        required_deps=["pytest", "JAX", "ptdalgorithms"]
    )

    # Test 4: Symbolic DAG tests
    runner.run_test(
        name="Symbolic DAG",
        command="pytest tests/test_symbolic_dag.py -v",
        description="Symbolic DAG and parameterized edges - 40+ tests",
        required_deps=["pytest", "JAX", "ptdalgorithms"]
    )

    # Test 5: Utilities and integration tests
    runner.run_test(
        name="Utilities & Integration",
        command="pytest tests/test_utilities_integration.py -v",
        description="Utilities and integration features - 50+ tests",
        required_deps=["pytest", "ptdalgorithms"]
    )

    # Print final summary
    exit_code = runner.print_summary()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
