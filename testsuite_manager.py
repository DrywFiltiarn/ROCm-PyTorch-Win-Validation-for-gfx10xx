import os
import sys
import time
import importlib
import torch
import contextlib
from datetime import datetime
import ctypes

def setup_windows_console():
    if os.name == 'nt':
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            mode.value |= 4 
            kernel32.SetConsoleMode(handle, mode)
        except:
            pass

class Colors:
    HEADER, OKBLUE, OKCYAN, OKGREEN, WARNING, FAIL, ENDC, BOLD = (
        '\033[95m', '\033[94m', '\033[96m', '\033[92m', 
        '\033[93m', '\033[91m', '\033[0m', '\033[1m'
    )

class ROCmModularSuite:
    def __init__(self):
        setup_windows_console()
        self.test_dir = "tests"
        self.log_dir = "logs"
        self.tests = {}
        self.results = {}
        self.milestone = "2.8.0"
        self._current_msg = "Ready."
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.refresh_tests()

    def get_mem_info(self):
        if not torch.cuda.is_available(): return "N/A", "N/A"
        f, t = torch.cuda.mem_get_info()
        u_gb, t_gb = (t-f)/(1024**3), t/(1024**3)
        a, r = torch.cuda.memory_allocated()/(1024**3), torch.cuda.memory_reserved()/(1024**3)
        return f"{u_gb:.2f}/{t_gb:.2f} GB", f"Alloc: {a:.2f}GB | Res: {r:.2f}GB"

    def refresh_tests(self):
        test_files = [f[:-3] for f in os.listdir(self.test_dir) if f.endswith(".py") and f != "__init__.py"]
        self.tests = {str(i): name for i, name in enumerate(sorted(test_files), 1)}
        self.results = {idx: "Pending" for idx in self.tests}
        self._current_msg = "Suite reloaded. Statuses reset."

    def display_menu(self, unattended=False):
        os.system('cls')
        vram, mem = self.get_mem_info()
        
        out = []
        out.append(f"{Colors.HEADER}{Colors.BOLD}=== ROCm MODULAR VALIDATION SUITE ==={Colors.ENDC}")
        out.append(f"PyTorch: {Colors.OKCYAN}{torch.__version__}{Colors.ENDC} | HIP: {Colors.OKCYAN}{torch.version.hip}{Colors.ENDC}")
        out.append(f"Milestone: {Colors.WARNING}{self.milestone}{Colors.ENDC} | GPU: {torch.cuda.get_device_name(0)}")
        out.append(f"VRAM: {vram:<15} | {mem}")
        
        msg_color = Colors.OKGREEN if "Complete" in self._current_msg else Colors.OKBLUE
        out.append(f"Status: {msg_color}{self._current_msg:<60}{Colors.ENDC}")
        out.append("-" * 85)
        
        for idx in sorted(self.tests.keys(), key=int):
            name = self.tests[idx]
            res = self.results[idx]
            color = Colors.OKGREEN if "Passed" in res else Colors.FAIL if "Failed" in res else Colors.OKBLUE if "Running" in res else Colors.WARNING
            out.append(f" [{idx:>2}] {name:<30} {color}{res:<25}{Colors.ENDC}")
        
        out.append("-" * 85)
        if not unattended:
            out.append(f"{Colors.BOLD}[A] Run All  [S] Save Summary  [R] Reload/Reset  [L] Clear Logs  [Q] Quit{Colors.ENDC}")
            out.append("Selection: ")
        else:
            out.append("")
        
        sys.stdout.write("\n".join(out))
        sys.stdout.flush()

    def execute_test(self, idx, silent=False, unattended=False):
        test_name = self.tests[idx]
        module_name = f"{self.test_dir}.{test_name}"
        log_path = os.path.join(self.log_dir, f"{test_name}.log")
        
        self.results[idx] = "Running..."
        self.display_menu(unattended)
        
        start_time = time.time()
        success = False
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
            with open(log_path, "w", encoding="utf-8") as f:
                with contextlib.redirect_stdout(f):
                    success = module.run()
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\nCRITICAL CRASH: {str(e)}\n")
            success = False
        
        duration = time.time() - start_time
        self.results[idx] = f"{'Passed' if success else 'Failed'} ({duration:.2f}s)"

        if not silent:
            self.display_menu(unattended)
            print(f"\n{Colors.BOLD}--- LOG: {test_name}.log ---{Colors.ENDC}")
            try:
                with open(log_path, "r", encoding="utf-8") as f: print(f.read())
            except: print("Error reading log.")
            input(f"\n{Colors.WARNING}Press [Enter] to return to menu...{Colors.ENDC}")

    def save_summary(self):
        now = datetime.now()
        ts_f = now.strftime("%Y%m%d_%H%M%S")
        ts_h = now.strftime("%Y-%m-%d %H:%M:%S")
        path = os.path.join(self.log_dir, f"validation_summary_{ts_f}.txt")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== ROCm VALIDATION SUMMARY | {self.milestone} ===\n")
            f.write(f"Date: {ts_h}\n")
            f.write(f"PyTorch: {torch.__version__} | HIP: {torch.version.hip}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write("-" * 55 + "\n")
            for idx in sorted(self.tests.keys(), key=int):
                f.write(f"[{self.results[idx]:<20}] {self.tests[idx]}\n")
            f.write("-" * 55 + "\n")
        self._current_msg = "Summary saved to logs/"

    def run(self, unattended=False):
        if unattended:
            self._current_msg = "Running unattended batch validation..."
            for idx in sorted(self.tests.keys(), key=int):
                self.execute_test(idx, silent=True, unattended=unattended)
            self.save_summary(); self.display_menu(unattended)
            
            has_failures = any("Failed" in res for res in self.results.values())
            sys.exit(1 if has_failures else 0)

        while True:
            self.display_menu(unattended)
            choice = input().upper()
            if choice == 'Q': break
            elif choice == 'R': self.refresh_tests()
            elif choice == 'S': self.save_summary(); self.display_menu(unattended)
            elif choice == 'L':
                for f in os.listdir(self.log_dir): os.remove(os.path.join(self.log_dir, f))
                self._current_msg = "Logs cleared."
            elif choice == 'A':
                self._current_msg = "Running batch validation..."
                for idx in sorted(self.tests.keys(), key=int):
                    self.execute_test(idx, silent=True, unattended=unattended)
                self.save_summary()
                self._current_msg = "Batch Complete. Reports saved."
                self.display_menu(unattended)
            elif choice in self.tests:
                self.execute_test(choice, silent=False, unattended=unattended)

if __name__ == "__main__":
    if os.name == 'nt': os.system('chcp 65001 > nul')
    suite = ROCmModularSuite()
    is_unattended = "--unattended" in sys.argv
    suite.run(unattended=is_unattended)