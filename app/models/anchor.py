"""
app/models/anchor.py
=======================
Shared shape for the "where does this forecast cascade start from" state,
so every model class and the rolling-forecast orchestrator agree on one
contract instead of each model reaching into its own frozen artifacts.

predict()/explain() on every model class accept an OPTIONAL `anchor`. When
omitted, the model falls back to the frozen last_row/last_scfi it loaded
from model_artifacts/ at startup (today's behavior, unchanged). When
supplied, the model cascades forward from this live state instead - built
fresh from Supabase by app/forecasting/feature_engineering.py.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass
class ForecastAnchor:
    last_row: dict
    last_scfi: float
    last_date: date
    # (seq_len, n_features), SCALED - only populated/consumed by LstmModel.
    x_seq_tail: "np.ndarray | None" = None
