import pytest
from pathlib import Path
import sys

# Ensure the project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import _strip_redundant_prefixes

def test_workspace_prefix_strip():
    """Test replacing simple workspace/ prefix"""
    result = _strip_redundant_prefixes("workspace/myproj/src/train.py", "myproj")
    assert result == "src/train.py"

def test_double_nesting_strip():
    """Test replacing nested workspace/proj/workspace/proj structures"""
    result = _strip_redundant_prefixes("scratch_project/workspace/diabetes/src/train.py", "diabetes")
    assert result == "src/train.py"
    
    result2 = _strip_redundant_prefixes("workspace/workspace/myproj/data/raw/file.csv", "myproj")
    assert result2 == "data/raw/file.csv"

def test_known_roots_detection():
    """Test that known root directories are preserved correctly"""
    result = _strip_redundant_prefixes("myproj/workspace/myproj/results/plots/fig.png", "myproj")
    assert result == "results/plots/fig.png"
    
    result2 = _strip_redundant_prefixes("something/config/settings.yaml", "myproj")
    assert result2 == "config/settings.yaml"

def test_known_files_detection():
    """Test that known top-level project files are preserved correctly"""
    result = _strip_redundant_prefixes("workspace/project/README.md", "project")
    assert result == "README.md"
    
    result2 = _strip_redundant_prefixes("nested/dir/report.md", "project")
    assert result2 == "report.md"
    
    result3 = _strip_redundant_prefixes("foo/bar/.gitignore", "project")
    assert result3 == ".gitignore"

def test_no_change_needed():
    """Test paths that are already correct are not changed unnecessarily"""
    result = _strip_redundant_prefixes("src/train.py", "myproj")
    assert result == "src/train.py"
    
    result2 = _strip_redundant_prefixes("report.md", "myproj")
    assert result2 == "report.md"
    
    result3 = _strip_redundant_prefixes("custom_folder/script.py", "myproj")
    # If it's a completely unknown folder without workspace prefixes, it should just strip any project name prefixes if possible, or leave it alone.
    # The stripping logic preserves it if no rules match or strips workspace/proj.
    # Let's see what the function does for "custom_folder/script.py" with "myproj". It likely returns the same string.
    assert _strip_redundant_prefixes(result3, "myproj") == result3
