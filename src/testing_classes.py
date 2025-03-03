from MovieDataProcessor import MovieDataProcessor
processor = MovieDataProcessor()

#print(processor.movie_metadata.head())
#print(processor.movie_metadata.describe())
#print(processor.__movie_type__(363))
print(processor.__actor_count__())
#print(processor.__actor_distributions__())



processor = MovieDataProcessor()



processor.__actor_distributions__("F",2.1,1.3,True)

processor.debug_column_4()

print("hello world")
