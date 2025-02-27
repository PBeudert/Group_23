from MovieDataProcessor import MovieDataProcessor
processor = MovieDataProcessor()
print("hello world")
#print(processor.movie_metadata.head())
#print(processor.movie_metadata.describe())
print(processor.__movie_type__(363))
print(processor.__actor_count__())
#print(processor.__actor_distributions__())




from MovieDataProcessor import MovieDataProcessor

processor = MovieDataProcessor()

# Call the corrected method
df_result = processor.__actor_distributions__(
    gender="All",
    max_height=2.0,  # 2 meters (reasonable limit)
    min_height=1.5,  # 1.5 meters (reasonable limit)
    plot=True
)

print(df_result)






