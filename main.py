from orkeslora.tracking import MLflowTracking
import os

def main():
    # Example of local tracking
    print("Running local MLflow experiment...")
    local_tracker = MLflowTracking(experiment_name="Local Experiment")
    local_tracker.start_run("Local Run 1")
    local_tracker.log_param("learning_rate", 0.01)
    local_tracker.log_metric("accuracy", 0.95)
    
    # Create a dummy artifact file
    with open("artifact.txt", "w") as f:
        f.write("This is a dummy artifact.")
    
    local_tracker.log_artifact("artifact.txt")
    local_tracker.set_tag("version", "1.0")
    local_tracker.end_run()
    os.remove("artifact.txt")
    print("Local experiment completed.")

    # Example of remote tracking (assuming a remote server is running)
    # To run this, you need to have a remote MLflow server.
    # For example: mlflow server --host 127.0.0.1 --port 5000
    remote_tracking_uri = "http://127.0.0.1:5000"
    print(f"\nRunning remote MLflow experiment on {remote_tracking_uri}...")
    try:
        remote_tracker = MLflowTracking(tracking_uri=remote_tracking_uri, experiment_name="Remote Experiment")
        remote_tracker.start_run("Remote Run 1")
        remote_tracker.log_param("batch_size", 64)
        remote_tracker.log_metric("loss", 0.12)
        remote_tracker.set_tag("status", "release")
        remote_tracker.end_run()
        print("Remote experiment completed.")
    except Exception as e:
        print(f"Could not connect to remote MLflow server: {e}")
        print("Please ensure the MLflow server is running to test remote tracking.")

if __name__ == "__main__":
    main()
