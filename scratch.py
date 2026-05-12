from src.urban_analysis.prototype.phase5_recommender import Phase5Recommender
recommender = Phase5Recommender(start_poi_name="函館朝市")
res = recommender.recommend(top_n=10)
print(res["objectives"])
