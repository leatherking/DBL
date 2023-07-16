import statistics
seed1 = [78.6, 69.85, 65.9, 61.68, 55.46, 52.85, 51.46, 49.3, 46.09, 45.25]
seed2 = [80.2, 72.7, 63.93, 59.22, 55.22, 50.87, 47.86, 46.44, 45.69, 44.18]
seed3 = [76.4, 70.2, 65.5, 61.25, 56.94, 54.05, 51.26, 47.69, 46.34, 43.66]

mean1 = sum(seed1)/len(seed1)
mean2 = sum(seed2)/len(seed2)
mean3 = sum(seed3)/len(seed3)

data = [mean1, mean2, mean3]

mean = statistics.mean(data)
std = statistics.stdev(data)
print("===> mean: ",mean)
print("===> std: ",std)


