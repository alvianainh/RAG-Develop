import os
import time
from functools import wraps

ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "false").lower() in ("1", "true", "yes")

if ENABLE_MLFLOW:
    import mlflow

def monitor_rag_component(component_name):
    if not ENABLE_MLFLOW:
        # No-op decorator if MLflow is disabled
        def noop_decorator(func):
            return func
        return noop_decorator

    # Actual decorator when MLflow is enabled
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                mlflow.set_experiment(component_name)
                run = mlflow.start_run(run_name=component_name)
                run_id = run.info.run_id
                print(f"[DEBUG][{component_name}] Started run ID: {run_id}")

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"[DEBUG][{component_name}] Exception: {e!r}")
                    mlflow.log_metric("duration_seconds", duration)
                    mlflow.set_tag("component", component_name)
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("error", type(e).__name__)
                    raise
                else:
                    duration = time.time() - start_time
                    mlflow.log_metric("duration_seconds", duration)
                    mlflow.set_tag("component", component_name)
                    mlflow.set_tag("status", "SUCCESS")
                    print(f"[DEBUG][{component_name}] Finished run ID: {run_id}")
                    return result
                finally:
                    mlflow.end_run()
            except Exception as logging_error:
                print(f"[monitoring] Decorator skipping measure RAG performance\nError: {logging_error}")
                return func(*args, **kwargs)

        return wrapper
    return decorator
