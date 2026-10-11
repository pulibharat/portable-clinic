"""
Matcher for Portable Clinic.
Performs cosine similarity lookups against enrolled face embeddings.
"""

import numpy as np
from .config import SIMILARITY_THRESHOLD, MATCH_MARGIN


class Matcher:
    """Normalized embedding matrix lookup with best-per-person similarity scoring."""

    def __init__(self, db: dict[str, list[np.ndarray]]) -> None:
        self.names: list[str] = []
        rows: list[np.ndarray] = []
        for name, embs in db.items():
            for e in embs:
                self.names.append(name)
                rows.append(e)

        if rows:
            mat = np.array(rows, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = mat / norms
        else:
            self.matrix = None

    def match(self, emb: np.ndarray) -> tuple[str, float]:
        """
        Matches emb against known embeddings.
        Returns: (person_name, similarity_score).
        Returns ("Unknown", score) if below SIMILARITY_THRESHOLD or MATCH_MARGIN.
        """
        if self.matrix is None or len(self.names) == 0:
            return "Unknown", 0.0

        e = emb.astype(np.float32)
        norm_e = np.linalg.norm(e)
        if norm_e == 0:
            return "Unknown", 0.0
        e = e / (norm_e + 1e-8)

        sims = self.matrix @ e

        # Best similarity per person (MAX across all enrolled poses of that person)
        best_per_person: dict[str, float] = {}
        for name, s in zip(self.names, sims):
            val = float(s)
            if name not in best_per_person or val > best_per_person[name]:
                best_per_person[name] = val

        ranked = sorted(best_per_person.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0

        if best_sim < SIMILARITY_THRESHOLD:
            return "Unknown", best_sim

        if (best_sim - second_sim) < MATCH_MARGIN:
            return "Unknown", best_sim

        return best_name, best_sim
