import os
import time
import subprocess
import socket
import mlflow
from pathlib import Path
from contextlib import asynccontextmanager

ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "false").lower() in ("1", "true", "yes")

def configure_monitoring():
    if not ENABLE_MLFLOW:
        print("[monitoring] ENABLE_MLFLOW is false → skipping MLflow configuration.")
        return
    print("[monitoring] ENABLE_MLFLOW is true → MLflow server will start at runtime.")

def wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def get_lifespan():
    if not ENABLE_MLFLOW:
        @asynccontextmanager
        async def noop_lifespan(app):
            yield
        return noop_lifespan

    @asynccontextmanager
    async def mlflow_lifespan(app):
        artifact_dir = os.getenv("MLFLOW_ARTIFACT_ROOT", "mlruns")
        abs_artifact_path = Path(artifact_dir).absolute()
        abs_artifact_path.mkdir(parents=True, exist_ok=True)

        raw_backend = os.getenv("MLFLOW_BACKEND_STORE_URI", "")
        if raw_backend:
            print(f"[monitoring] MLflow server will use backend store: {raw_backend}")
        else:
            print("[monitoring] WARNING: MLFLOW_BACKEND_STORE_URI not set → server will default to file store.")

        host = os.getenv("MLFLOW_SERVER_HOST", "127.0.0.1")
        port_str = os.getenv("MLFLOW_SERVER_PORT", "5000")
        try:
            port = int(port_str)
        except ValueError:
            print(f"[monitoring] ERROR: Invalid MLFLOW_SERVER_PORT='{port_str}'. Using default 5000.")
            port = 5000

        raw_tracking = os.getenv("MLFLOW_TRACKING_URI", "")
        if raw_tracking.startswith("http://") or raw_tracking.startswith("https://"):
            tracking_uri = raw_tracking
        elif raw_tracking:
            tracking_uri = "http://" + raw_tracking
            print(f"[monitoring] WARNING: Prepended 'http://' to tracking URI → now using {tracking_uri}")
        else:
            tracking_uri = f"http://{host}:{port}"

        max_retries = 3
        attempt = 0
        mlflow_proc = None

        while attempt < max_retries:
            attempt += 1
            print(f"[monitoring] Attempt {attempt}/{max_retries} to launch MLflow server…")

            cmd = [
                "mlflow", "server",
                "--backend-store-uri", raw_backend,
                "--default-artifact-root", str(abs_artifact_path),
                "--host", host,
                "--port", str(port),
            ]

            mlflow_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if wait_for_port(host, port, timeout=20):
                print(f"[monitoring] MLflow server is now listening at {host}:{port}")
                break  # success
            else:
                print(f"[monitoring] ERROR: MLflow server did not start within timeout on {host}:{port}")
                mlflow_proc.terminate()
                try:
                    mlflow_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    mlflow_proc.kill()
                    mlflow_proc.wait()

                if attempt == max_retries:
                    raise RuntimeError("MLflow server failed to start after multiple attempts.")

        mlflow.set_tracking_uri(tracking_uri)
        print(f"[monitoring] MLflow client connected (tracking URI = {tracking_uri}).\n")

        try:
            yield
        finally:
            print("[monitoring] Shutting down MLflow server subprocess…")
            if mlflow_proc and mlflow_proc.poll() is None:
                mlflow_proc.terminate()
                try:
                    mlflow_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[monitoring] MLflow server did not terminate gracefully; killing.")
                    mlflow_proc.kill()
                    mlflow_proc.wait()
            print("[monitoring] MLflow server fully shut down.")

    return mlflow_lifespan
