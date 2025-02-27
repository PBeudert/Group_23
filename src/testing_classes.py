from MovieDataProcessor import MovieDataProcessor
processor = MovieDataProcessor()
print("hello world")
#print(processor.movie_metadata.head())
#print(processor.movie_metadata.describe())
print(processor.__movie_type__(363))
print(processor.__actor_count__())
#print(processor.__actor_distributions__())