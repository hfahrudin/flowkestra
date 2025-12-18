
import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import os

from flowkestra.worker import Worker

@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory structure for the test."""
    temp_dir = tempfile.mkdtemp()
    
    # Create origin and target directories
    origin_dir = Path(temp_dir) / "origin"
    origin_dir.mkdir()
    target_dir = Path(temp_dir) / "target"
    target_dir.mkdir()

    # Create a dummy script
    output_file = target_dir / "success.txt"
    script_content = f"""
import sys
with open("{output_file}", "w") as f:
    f.write("Success!")
"""
    script_path = origin_dir / "dummy_script.py"
    script_path.write_text(script_content)

    # Create a dummy requirements file
    (origin_dir / "requirements.txt").write_text("numpy")
    
    yield {
        "temp_dir": temp_dir,
        "origin_dir": origin_dir,
        "target_dir": target_dir,
        "script_path": "dummy_script.py",
        "requirements_path": "requirements.txt",
        "output_file": output_file
    }
    
    shutil.rmtree(temp_dir)

def test_local_worker_e2e(temp_project_dir):
    """
    End-to-end test for a local worker.
    This test will:
    1. Set up a temporary project structure.
    2. Create and run a Worker instance.
    3. Verify that the worker's script ran successfully.
    """
    worker_id = "e2e_worker"
    main_states = {worker_id: {}}

    worker = Worker(
        worker_id=worker_id,
        workdir=temp_project_dir["target_dir"],
        origin_dir=temp_project_dir["origin_dir"],
        main_states=main_states,
        requirements=temp_project_dir["requirements_path"],
        pipelines={"step1": {"script": temp_project_dir["script_path"]}},
        suppress_output=False,
        clean_workdir_after_run=False # Keep workdir for inspection
    )

    worker.run()

    # Verify that the script ran successfully
    assert temp_project_dir["output_file"].exists()
    assert temp_project_dir["output_file"].read_text() == "Success!"

    worker.close()
