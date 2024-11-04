import subprocess
import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed and accessible"""
    dependencies = {
        'Tesseract': {
            'command': ['tesseract', '--version'],
            'error': 'Tesseract is not installed or not in PATH'
        },
        'Poppler': {
            'command': ['pdfinfo', '-v'],
            'error': 'Poppler is not installed or not in PATH'
        },
        'Ghostscript': {
            'command': ['gswin64c', '--version'],
            'error': 'Ghostscript is not installed or not in PATH'
        }
    }

    print("Checking dependencies...")
    print("="*50)
    
    all_good = True
    for name, info in dependencies.items():
        try:
            result = subprocess.run(
                info['command'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()
                print(f"✓ {name}: {version}")
            else:
                print(f"✗ {name}: {info['error']}")
                all_good = False
        except FileNotFoundError:
            print(f"✗ {name}: {info['error']}")
            all_good = False
            
    print("\nEnvironment Variables:")
    print(f"PATH: {os.environ['PATH']}")
    
    return all_good

if __name__ == "__main__":
    if check_dependencies():
        print("\nAll dependencies are installed correctly!")
    else:
        print("\nSome dependencies are missing. Please install them before running the PDF processor.")
        sys.exit(1)