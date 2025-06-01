# import mlflow
# import time
# from functools import wraps
# from monitoring.setup import ENABLE_MLFLOW

# def monitor_rag_component(component_name):
#     """
#     Decorator that, for each call to the wrapped function:
#     1) Sets the MLflow experiment name to component_name (creating it if necessary).
#     2) Starts a top‐level MLflow run named component_name.
#     3) Logs duration, status, and tags (no file artifacts).
#     4) Ends the run as soon as the function returns or raises.
#     """
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             if not ENABLE_MLFLOW:
#                 print("Decorator skipping measure RAG performance")
#                 return func(*args, **kwargs)

#             # 1) Ensure the experiment is named after this component
#             mlflow.set_experiment(component_name)

#             # 2) Start a new run for this component
#             run = mlflow.start_run(run_name=component_name)
#             run_id = run.info.run_id
#             print(f"[DEBUG][{component_name}] Started run ID: {run_id}")

#             start_time = time.time()
#             try:
#                 result = func(*args, **kwargs)
#             except Exception as e:
#                 duration = time.time() - start_time
#                 print(f"[DEBUG][{component_name}] Exception: {e!r}")

#                 mlflow.log_metric("duration_seconds", duration)
#                 mlflow.set_tag("component", component_name)
#                 mlflow.set_tag("status", "FAILED")
#                 mlflow.set_tag("error", type(e).__name__)

#                 mlflow.end_run()
#                 raise
#             else:
#                 duration = time.time() - start_time
#                 mlflow.log_metric("duration_seconds", duration)
#                 mlflow.set_tag("component", component_name)
#                 mlflow.set_tag("status", "SUCCESS")

#                 mlflow.end_run()
#                 print(f"[DEBUG][{component_name}] Finished run ID: {run_id}")
#                 return result

#         return wrapper
#     return decorator


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
