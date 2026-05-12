# phase5_recommender.py
"""Phase 5 recommender: multi‑objective optimization using NSGA‑III.
It extends the Phase 4 recommender, adds Google rating and review count as
additional objectives, and ensures reproducibility via fixed random seeds.
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from pymoo.algorithms.moo.nsga3 import NSGA3
# pymoo 0.6 provides reference directions via util module
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# Import Phase4Recommender
from .phase4_recommender import Phase4Recommender

class Phase5Recommender(Phase4Recommender):
    """Phase 5 recommendation engine.

    Adds two additional objectives:
        * mean_norm_rating (to be maximized → negative for minimization)
        * mean_norm_review_count (to be maximized → negative for minimization)
    """

    def __init__(self, start_poi_name: str = None, config_path: str = None):
        # Initialise Phase4 (no arguments)
        super().__init__()
        self.start_poi_name = start_poi_name
        # Load configuration if provided
        if config_path is not None:
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                self.seed = cfg.get("seed", 42)
                self.population_size = cfg.get("population_size", 200)
                self.generations = cfg.get("generations", 50)
                self.weights = cfg.get("weights", {"distance": 0.1, "rating": 0.2, "review_count": 0.2, "landscape": 0.5})
            except Exception as e:
                raise RuntimeError(f"Failed to load config {config_path}: {e}")
        else:
            self.seed = 42
            self.population_size = 200
            self.generations = 50
            self.weights = {"distance": 0.2, "rating": 0.2, "review_count": 0.2, "landscape": 0.4}
        # Reproducible seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Load additional Google info and prepare normalized features
        self._load_google_info()
        self._prepare_normalized_features()
        # Run Phase4 recommendation to obtain candidates and target POI (top_n unused)
        _, self.target_poi, self.candidates = super().recommend(self.start_poi_name, top_n=0)

    def _load_google_info(self):
        project_root = Path(__file__).parents[3]
        csv_path = project_root / "data" / "processed" / "poi" / "poi_google_info.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Google info CSV not found at {csv_path}")
        self.google_df = pd.read_csv(csv_path)

    def _prepare_normalized_features(self):
        if "poi_id" not in self.poi_df.columns:
            if "id" in self.poi_df.columns:
                self.poi_df = self.poi_df.rename(columns={"id": "poi_id"})
            else:
                raise KeyError("poi_id column not found in Phase4 POI dataframe")
        merged = pd.merge(self.poi_df, self.google_df, how="left", on="poi_id")
        merged["rating"] = merged["rating"].fillna(0)
        merged["review_count"] = merged["review_count"].fillna(0)
        self.poi_df = merged
        scaler = MinMaxScaler()
        self.poi_df[["norm_rating", "norm_review_count"]] = scaler.fit_transform(
            self.poi_df[["rating", "review_count"]]
        )

    def _objective_function(self, selected_values, top_n):
        selected_indices = np.argsort(selected_values)[-top_n:]
        subset = self.candidates.iloc[selected_indices]
        total_distance = subset["distance_m"].sum()
        mean_rating = subset["norm_rating"].mean()
        mean_review = subset["norm_review_count"].mean()
        mean_landscape = subset["norm_ls_density"].mean()
        return np.array([total_distance, -mean_rating, -mean_review, -mean_landscape])

    class MultiObjProblem(Problem):
        def __init__(self, parent, top_n):
            self.parent = parent
            self.top_n = top_n
            n_var = len(parent.candidates)
            super().__init__(n_var=n_var, n_obj=4, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            F = []
            for ind in X:
                F.append(self.parent._objective_function(ind, self.top_n))
            out["F"] = np.column_stack(F).T

    def recommend(self, target_poi_name: str = None, top_n: int = 10):
        """Run NSGA‑III and return the best solution according to user weights.
        If target_poi_name is given, recompute Phase4 candidates.
        """
        if target_poi_name:
            _, self.target_poi, self.candidates = super().recommend(target_poi_name, top_n=0)
        if len(self.candidates) < top_n:
            raise ValueError("Not enough candidates after Phase4 filtering")
            
        # Calculate Landscape Score (Euclidean KDE) for candidates
        target_ls_cluster = self.target_poi.get('ls_cluster')
        if target_ls_cluster is not None and len(self.candidates) > 0:
            ls_points = self.ls_coords_m[self.ls_valid_df['cluster'] == target_ls_cluster]
            bandwidth = 500.0
            lat_to_m = 111000
            lng_to_m = 82000
            
            densities = []
            for _, row in self.candidates.iterrows():
                poi_coord = np.array([float(row['lat']) * lat_to_m, float(row['lng']) * lng_to_m])
                dists = np.linalg.norm(ls_points - poi_coord, axis=1)
                k_vals = np.where(dists <= bandwidth, (1 - (dists/bandwidth)**2)**2, 0)
                densities.append(np.sum(k_vals))
                
            # Need to avoid SettingWithCopyWarning by explicitly doing assignment or using loc
            self.candidates.loc[:, 'ls_density'] = densities
            scaler = MinMaxScaler()
            self.candidates.loc[:, 'norm_ls_density'] = scaler.fit_transform(self.candidates[['ls_density']])
        else:
            self.candidates.loc[:, 'norm_ls_density'] = 0.0

        problem = self.MultiObjProblem(self, top_n)
        ref_dirs = get_reference_directions("uniform", n_partitions=7, n_dim=4)
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=self.population_size, seed=self.seed)
        res = minimize(problem, algorithm, termination=('n_gen', self.generations), verbose=False)
        pareto_vectors = res.X
        pareto_objs = res.F
        w = np.array([
            self.weights.get("distance", 0.4),
            self.weights.get("rating", 0.2),
            self.weights.get("review_count", 0.2),
            self.weights.get("landscape", 0.2)
        ])
        scores = np.dot(pareto_objs, w)
        best_idx = np.argmin(scores)
        best_vector = pareto_vectors[best_idx]
        selected_indices = np.argsort(best_vector)[-top_n:]
        selected = self.candidates.iloc[selected_indices]
        return {
            "selected_pois": selected.to_dict(orient="records"),
            "objectives": {
                "distance": float(pareto_objs[best_idx, 0]),
                "mean_rating": -float(pareto_objs[best_idx, 1]),
                "mean_review_count": -float(pareto_objs[best_idx, 2]),
                "mean_landscape": -float(pareto_objs[best_idx, 3])
            },
            "pareto_front": pareto_objs.tolist()
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase5 multi‑objective recommendation")
    parser.add_argument("start_poi", help="Name of the starting POI")
    parser.add_argument("--config", default=None, help="Path to phase5.yaml config file")
    parser.add_argument("--top_n", type=int, default=10, help="Number of POIs per candidate solution")
    args = parser.parse_args()
    recommender = Phase5Recommender(start_poi_name=args.start_poi, config_path=args.config)
    result = recommender.recommend(top_n=args.top_n)
    print("Selected POIs:")
    for poi in result["selected_pois"]:
        print(poi)
    print("Objectives:", result["objectives"])
