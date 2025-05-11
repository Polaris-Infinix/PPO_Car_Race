from collections import deque
rewards=[2,5,8,5,4]
adv=deque(maxlen=len(rewards))
# print(adv[0])
adv.appendleft(2)
print(adv[0])
# adv.appendleft(2)

for i in adv:
    print(True)
    break

print(False)