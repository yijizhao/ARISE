"""Training-view selection: ARISE and the Random baseline."""

from .arise import select as select_arise
from .common import SamplingError, covered_steps
from .random_sampling import select_random

METHODS = {"ARISE": select_arise, "Random": select_random}

# What --budget means for each method.
BUDGET_UNIT = {
    "ARISE": "raw-data budget beta (C = floor(beta * T))",
    "Random": "training-sample ratio q (n = floor(q * N))",
}
