import logging
import math
from typing import Dict, Any, List
import numpy as np
from backend.models import RACOutput, AgentResponse

logger = logging.getLogger("RAC_Module")


class RACModule:
    """
    Robust Agreement & Consensus (RAC) Module.
    Evaluates cross-agent diagnostic alignment, computes conflict penalties,
    and scales agent confidence scores mathematically.
    """
    def __init__(self):
        import os, pickle, warnings
        self.calibrator = None
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        clean_path = os.path.join(models_dir, 'calibrator_clean.pkl')
        legacy_path = os.path.join(models_dir, 'calibrator.pkl')
        calibrator_path = clean_path if os.path.exists(clean_path) else legacy_path
        if os.path.exists(calibrator_path):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(calibrator_path, 'rb') as f:
                        self.calibrator = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load calibrator: {e}")

    def calibrate_confidence(self, c_base: float, u_cons: float = 0.0, h_agr: float = 0.0) -> float:
        """Applies loaded post-hoc calibrator to confidence score."""
        calibrated_confidence = c_base
        if self.calibrator is not None:
            try:
                if hasattr(self.calibrator, "predict_proba"):
                    if hasattr(self.calibrator, "temperature"):
                        calibrated_confidence = float(self.calibrator.predict_proba(np.array([c_base]))[0])
                    elif hasattr(self.calibrator, "coef_"):
                        n_feat = self.calibrator.coef_.shape[1]
                        if n_feat == 3:
                            feat = np.array([[c_base, u_cons, h_agr]])
                            calibrated_confidence = float(self.calibrator.predict_proba(feat)[0][1])
                        else:
                            calibrated_confidence = float(self.calibrator.predict_proba(np.array([[c_base]]))[0][1])
                elif hasattr(self.calibrator, "predict"):
                    calibrated_confidence = float(self.calibrator.predict([c_base])[0])
            except Exception as e:
                logger.error(f"Failed to apply calibrator: {e}")
        return max(0.0, min(1.0, float(calibrated_confidence)))

    def _extract_primary_diagnoses(self, parsed_results: Dict[str, dict]) -> List[str]:
        """Extract primary diagnoses from all diagnostic and specialist agents."""
        diagnoses = []
        for key in ["cardiologist", "pulmonologist", "neurologist", "diagnosis"]:
            out = parsed_results.get(key)
            if out and isinstance(out, dict):
                p_diag = out.get("primary_diagnosis", "")
                if p_diag and p_diag != "Unknown":
                    diagnoses.append(p_diag.lower())
        return diagnoses

    def _calculate_agreement(self, parsed_results: Dict[str, dict]) -> float:
        if not parsed_results:
            return 1.0

        primary_diagnoses = self._extract_primary_diagnoses(parsed_results)
        if not primary_diagnoses:
            return 1.0

        # Collect keywords from all primary diagnoses
        diag_keywords = set()
        for diag in primary_diagnoses:
            cleaned = diag.replace(",", " ").replace("-", " ").replace("_", " ")
            for w in cleaned.split():
                if len(w) > 3:
                    diag_keywords.add(w)

        total_agents = len(parsed_results)
        agreeing_agents = 0

        # Pre-process diagnostic/specialist agents and their keywords
        diag_agent_keys = ["cardiologist", "pulmonologist", "neurologist", "diagnosis"]
        active_diag_agents = {}
        for agent_name in diag_agent_keys:
            out = parsed_results.get(agent_name)
            if out and isinstance(out, dict):
                p_diag = (out.get("primary_diagnosis") or "").strip().lower()
                conf = float(out.get("confidence", 0.0) or 0.0)
                has_active = bool(out.get("active_diagnoses"))
                if p_diag and p_diag != "unknown":
                    cleaned = p_diag.replace(",", " ").replace("-", " ").replace("_", " ")
                    words = {w for w in cleaned.split() if len(w) > 3}
                    active_diag_agents[agent_name] = (p_diag, words, conf, has_active)

        for agent_name, output_dict in parsed_results.items():
            if not output_dict:
                continue

            # Diagnostic / specialist agents
            if agent_name in diag_agent_keys:
                if agent_name not in active_diag_agents:
                    continue
                # If only 1 diagnostic agent is active, there is no specialist disagreement
                if len(active_diag_agents) <= 1:
                    agreeing_agents += 1
                else:
                    # Multiple diagnostic agents: verify alignment with at least one other diagnostic agent
                    my_diag, my_words, my_conf, my_active = active_diag_agents[agent_name]
                    agrees = False
                    for other_name, (other_diag, other_words, other_conf, other_active) in active_diag_agents.items():
                        if other_name == agent_name:
                            continue
                        if my_diag == other_diag or (my_words and (my_words & other_words)):
                            agrees = True
                            break
                    if agrees:
                        agreeing_agents += 1
                continue

            # For risk and treatment agents, check keyword presence or compatibility
            output_str = str(output_dict).lower()
            overlap = any(kw in output_str for kw in diag_keywords)
            if overlap:
                agreeing_agents += 1
            else:
                # If risk agent is high/critical and case has urgent specialist finding, count as agreement
                risk_lvl = output_dict.get("risk_level", "").lower()
                if risk_lvl in ("high", "critical", "medium"):
                    agreeing_agents += 1

        agreement = agreeing_agents / max(1, total_agents)
        return min(1.0, max(0.0, agreement))

    def evaluate(self, agent_results: Dict[str, AgentResponse]) -> RACOutput:
        """
        Mathematically evaluates agreement across general and specialist agents,
        computes conflict penalties, and returns scaled confidence scores.
        """
        parsed_results = {}
        for name, resp in agent_results.items():
            if resp.output:
                if hasattr(resp.output, "model_dump"):
                    parsed_results[name] = resp.output.model_dump()
                elif isinstance(resp.output, dict):
                    parsed_results[name] = resp.output

        # 1. Agreement Score
        agreement_score = self._calculate_agreement(parsed_results)

        # 2. Conflict Penalty
        conflict_penalty = round(1.0 - agreement_score, 4)

        # 3. Adjusted Confidence per agent
        adjusted_conf_map = {}
        for name, resp in agent_results.items():
            if not resp.output:
                continue

            base_conf = 0.8
            if hasattr(resp.output, "confidence"):
                base_conf = getattr(resp.output, "confidence", 0.8)
            elif isinstance(resp.output, dict) and "confidence" in resp.output:
                base_conf = resp.output["confidence"]

            # Mathematical calibration adjustment with bounded conflict penalty
            adjusted_conf = base_conf * (0.80 + 0.20 * agreement_score) * (1.0 - (conflict_penalty * 0.15))

            # Penalty for generic/unspecified diagnoses
            output_str = str(parsed_results.get(name, "")).lower()
            generic_terms = ["unspecified", "syndrome", "unknown", "idiopathic", "other_"]
            if any(gt in output_str for gt in generic_terms):
                adjusted_conf -= 0.10

            # Clamp between 0.60 and 0.98
            adjusted_conf = max(0.60, min(0.98, adjusted_conf))
            adjusted_conf_map[name] = round(adjusted_conf, 4)

        logger.info(f"RAC evaluated: agreement={agreement_score:.2f}, penalty={conflict_penalty:.2f}, map={adjusted_conf_map}")

        # Full 18-disease vocabulary pooled agreement entropy H_agr
        from scripts.utils.label_extractor import ALL_DISEASE_CLASSES
        pooled_probs = np.zeros(len(ALL_DISEASE_CLASSES), dtype=np.float32)
        m_specs = 0
        c_base_list = []
        u_cons_list = []

        for name, resp in agent_results.items():
            if name in ["cardiologist", "pulmonologist", "neurologist"] and resp.output:
                out = resp.output
                c_val = getattr(out, "confidence", 0.0) if hasattr(out, "confidence") else (out.get("confidence", 0.0) if isinstance(out, dict) else 0.0)
                c_base_list.append(float(c_val) if c_val is not None else 0.0)

                u_val = getattr(out, "u_cons", None)
                if u_val is None:
                    u_val = getattr(out, "u_instability", 0.0)
                u_cons_list.append(float(u_val) if u_val is not None else 0.0)

                # Extract disease probabilities into pooled 18-disease vector
                d_probs = getattr(out, "disease_probabilities", {}) if hasattr(out, "disease_probabilities") else (out.get("disease_probabilities", {}) if isinstance(out, dict) else {})
                if d_probs:
                    m_specs += 1
                    for idx, dis in enumerate(ALL_DISEASE_CLASSES):
                        if dis in d_probs:
                            pooled_probs[idx] += float(d_probs[dis])

        c_base = max(c_base_list) if c_base_list else 0.0
        u_cons = max(u_cons_list) if u_cons_list else 0.0

        # Calculate normalized Shannon agreement entropy across 18-disease vocabulary
        h_agr = 0.0
        sum_p = np.sum(pooled_probs)
        if m_specs > 0 and sum_p > 0:
            import math
            p_bar = pooled_probs / sum_p
            shannon_h = -sum(p * math.log(p) for p in p_bar if p > 0)
            max_h = math.log(len(ALL_DISEASE_CLASSES))
            h_agr = shannon_h / max_h if max_h > 0 else 0.0

        # Calibrated Confidence Inference (Clean Calibrator)
        calibrated_confidence = self.calibrate_confidence(c_base, u_cons, h_agr)

        return RACOutput(
            agreement_score=round(agreement_score, 4),
            conflict_penalty=conflict_penalty,
            adjusted_confidence_map=adjusted_conf_map,
            e_domain=round(h_agr, 4),
            c_base=round(c_base, 4),
            u_instability=round(u_cons, 4),
            u_cons=round(u_cons, 4),
            calibrated_confidence=round(calibrated_confidence, 4)
        )

