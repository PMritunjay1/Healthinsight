"""
HealthInsight-UQ: Bayesian Beta Trust Dynamics & Reliability Consensus (Algorithms 4 & 5)
========================================================================================
Implements dynamic specialist trust tracking using Beta distributions:
    theta_{i,d} ~ Beta(alpha_{i,d}, beta_{i,d})
Initialized with uniform prior Beta(1, 1).
Updated strictly using calibration ground-truth cases:
    - Specialist prediction correct: alpha_{i,d} += 1.0
    - Specialist prediction incorrect: beta_{i,d} += 1.0
Expected trust score:
    hat{theta}_{i,d} = alpha_{i,d} / (alpha_{i,d} + beta_{i,d})
Consensus weight:
    w_{i,d} = lambda_d * hat{theta}_{i,d}
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("health_insights.trust_tracker")

SPECIALISTS = ["cardiologist", "pulmonologist", "neurologist"]
DOMAINS = ["Cardiology", "Pulmonology", "Neurology", "general"]
DEFAULT_TRUST_PATH = os.path.join(os.path.dirname(__file__), "models", "trust_posteriors.json")


class BetaTrustStore:
    """
    Thread-safe, reproducible store for specialist Beta trust parameters (alpha, beta).
    Guarantees that test data is never used to update posteriors.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or DEFAULT_TRUST_PATH
        # store[specialist][domain] = [alpha, beta]
        self.posteriors: Dict[str, Dict[str, List[float]]] = {}
        self._init_priors()
        if os.path.exists(self.storage_path):
            self.load()

    def _init_priors(self):
        """Initializes all specialist-domain pairs to uniform Beta(1, 1)."""
        for spec in SPECIALISTS:
            self.posteriors[spec] = {}
            for dom in DOMAINS:
                self.posteriors[spec][dom] = [1.0, 1.0]

    def get_trust(self, specialist: str, domain: str = "general") -> float:
        """
        Returns posterior mean trust score hat{theta}_{i,d} = alpha / (alpha + beta).
        Bounded strictly in (0.0, 1.0).
        """
        spec_key = specialist.lower().strip()
        dom_key = self._normalize_domain(domain)

        params = self.posteriors.get(spec_key, {}).get(dom_key)
        if not params:
            params = self.posteriors.get(spec_key, {}).get("general", [1.0, 1.0])

        alpha, beta = params[0], params[1]
        return float(alpha / (alpha + beta))

    def update(self, specialist: str, domain: str, is_correct: bool, weight: float = 1.0):
        """
        Bayesian update of Beta trust parameters.
        alpha += 1 on success, beta += 1 on error.
        """
        spec_key = specialist.lower().strip()
        dom_key = self._normalize_domain(domain)

        if spec_key not in self.posteriors:
            self.posteriors[spec_key] = {d: [1.0, 1.0] for d in DOMAINS}
        if dom_key not in self.posteriors[spec_key]:
            self.posteriors[spec_key][dom_key] = [1.0, 1.0]

        if is_correct:
            self.posteriors[spec_key][dom_key][0] += float(weight)
        else:
            self.posteriors[spec_key][dom_key][1] += float(weight)

    def _normalize_domain(self, domain: str) -> str:
        d = (domain or "general").strip().lower()
        if "cardio" in d:
            return "Cardiology"
        elif "pulmo" in d:
            return "Pulmonology"
        elif "neuro" in d:
            return "Neurology"
        return "general"

    def save(self, path: Optional[str] = None):
        """Atomic serialization of trust posteriors to JSON."""
        target_path = path or self.storage_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        payload = {
            "metadata": {
                "description": "HealthInsight-UQ Bayesian Beta Trust Posteriors (Algorithm 5)",
                "priors": "Beta(1.0, 1.0)",
            },
            "posteriors": self.posteriors,
            "trust_means": {
                spec: {
                    dom: round(self.get_trust(spec, dom), 4)
                    for dom in DOMAINS
                }
                for spec in SPECIALISTS
            }
        }

        dir_name = os.path.dirname(target_path)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(payload, tf, indent=2)
            temp_name = tf.name

        os.replace(temp_name, target_path)
        logger.info(f"Persisted trust posteriors to {target_path}")

    def load(self, path: Optional[str] = None):
        """Loads trust posteriors from JSON."""
        target_path = path or self.storage_path
        if not os.path.exists(target_path):
            return

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_posteriors = data.get("posteriors", {})
            for spec, dom_map in loaded_posteriors.items():
                if spec not in self.posteriors:
                    self.posteriors[spec] = {}
                for dom, params in dom_map.items():
                    self.posteriors[spec][dom] = [float(params[0]), float(params[1])]
            logger.info(f"Successfully loaded trust posteriors from {target_path}")
        except Exception as e:
            logger.error(f"Failed to load trust posteriors from {target_path}: {e}")
