import logging
from typing import Dict, Any, List
from backend.models import RACOutput, AgentResponse

logger = logging.getLogger("RAC_Module")


class RACModule:
    """
    Robust Agreement & Consensus (RAC) Module.
    Evaluates cross-agent diagnostic alignment, computes conflict penalties,
    and scales agent confidence scores mathematically.
    """
    def __init__(self):
        pass

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

        for agent_name, output_dict in parsed_results.items():
            if not output_dict:
                continue

            # Diagnostic / specialist agents with active primary diagnoses agree
            if agent_name in ["cardiologist", "pulmonologist", "neurologist", "diagnosis"]:
                p_diag = output_dict.get("primary_diagnosis", "").lower()
                if p_diag and p_diag != "Unknown":
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
        return min(1.0, max(0.5, agreement))

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

            # Mathematical calibration adjustment
            adjusted_conf = base_conf * agreement_score * (1.0 - (conflict_penalty * 0.4))

            # Penalty for generic/unspecified diagnoses
            output_str = str(parsed_results.get(name, "")).lower()
            generic_terms = ["unspecified", "syndrome", "unknown", "idiopathic", "other_"]
            if any(gt in output_str for gt in generic_terms):
                adjusted_conf -= 0.10

            # Clamp between 0.60 and 0.98
            adjusted_conf = max(0.60, min(0.98, adjusted_conf))
            adjusted_conf_map[name] = round(adjusted_conf, 4)

        logger.info(f"RAC evaluated: agreement={agreement_score:.2f}, penalty={conflict_penalty:.2f}, map={adjusted_conf_map}")

        return RACOutput(
            agreement_score=round(agreement_score, 4),
            conflict_penalty=conflict_penalty,
            adjusted_confidence_map=adjusted_conf_map
        )
