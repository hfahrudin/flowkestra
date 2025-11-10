import argparse
import sys
from orkeslora.manager import TrainingManager

def main():
    parser = argparse.ArgumentParser(
        description="Orkeslora: Run local or remote ML training from a YAML config"
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    manager = TrainingManager(args.file)
    manager.setup()

    script_path = manager.config.get("script_path")
    if not script_path:
        print("Error: 'script_path' must be defined in the YAML file")
        sys.exit(1)

    deployed_script = manager.deploy_script(script_path)
    additional_env = manager.config.get("env_vars", {})

    manager.run_training(deployed_script, additional_env)

    # Download artifacts if any
    artifacts = manager.config.get("artifacts", [])
    for artifact in artifacts:
        manager.download_artifact(artifact["source"], artifact["target"])

    manager.close()
    print("Training finished successfully!")
