import random
s=open("train.txt","r")
t=open("mask_to_bad.txt", 'w')
m=s.readlines()
rand_list = random.sample(range(0,15000), 700)
for item in rand_list:
    print(m[item], file=t)
print(rand_list)