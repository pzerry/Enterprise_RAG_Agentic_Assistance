"""Local DeepEval evaluations of the application's actual components."""
import os

# This suite writes local reports; no Confident AI account is required.
os.environ.setdefault('DEEPEVAL_TELEMETRY_OPT_OUT', 'YES')

