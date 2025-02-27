from MovieDataProcessor import MovieDataProcessor
processor = MovieDataProcessor()

#print(processor.movie_metadata.head())
#print(processor.movie_metadata.describe())
print(processor.__movie_type__(1))
print(processor.__actor_count__())
#print(processor.__actor_distributions__())