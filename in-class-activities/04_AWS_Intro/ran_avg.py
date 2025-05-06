import random

n = 100
ran_sum = 0
for i in range(n):
    ran_sum += random.random()

print("Random Average", ran_sum / n)
