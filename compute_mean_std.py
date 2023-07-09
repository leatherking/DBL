import statistics
seed1 = [71.5, 63.4, 58.5, 55.05, 50.44, 43.55, 41.17, 39.39, 33.5, 27.86]
seed2 = [74.2, 65.6, 57.57, 52.18, 48.5, 44.73, 36.17, 35.33, 30.08, 24.97]
seed3 = [74.3, 66.45, 60.43, 56.08, 53.04, 49.23, 43.77, 41.29, 35.63, 31.88]

mean1 = sum(seed1)/len(seed1)
mean2 = sum(seed2)/len(seed2)
mean3 = sum(seed3)/len(seed3)

data = [mean1, mean2, mean3]

mean = statistics.mean(data)
std = statistics.stdev(data)
print("===> mean: ",mean)
print("===> std: ",std)


