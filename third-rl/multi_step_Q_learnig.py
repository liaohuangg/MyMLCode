import numpy as np
import random

'''
The multi-step decision-making problem of robots
'''
q = np.zeros((7,7)) #init matrix
q = np.matrix(q)

# give the choice that robot could do
r = np.array([[-1,-1,-1,0,-1,-1,-1],
              [-1,-1,0,-1,-1,-1,-1],
              [-1,0,-1,0,-1,0,-1],
              [0,-1,0,-1,0,-1,-1],
              [-1,-1,-1,0,-1,0,100],
              [-1,-1,0,-1,0,-1,100],
              [-1,-1,-1,-1,0,0,100]])
r = np.matrix(r)

# train
# define the greedy parameter
gamma = 0.8
for i in range(10):
    # for each train, choose one state randomly
    state = random.randint(0,6)
    while state != 6:
        # choose non-neg action
        r_pos_action = []
        for action in range(7):
            if(r[state, action] >= 0):
                r_pos_action.append(action)
        # the next state choose valid action randomly
        next_state = r_pos_action[random.randint(0, len(r_pos_action) - 1)]
        
        # multi-step Q-learning - 2 step
        if next_state != 6:
            r_2nd_pos_action = []
            for action in range(7):
                if(r[next_state, action] >= 0):
                    r_2nd_pos_action.append(action)
            # the next state choose valid action randomly
            next_next_state = r_2nd_pos_action[random.randint(0, len(r_2nd_pos_action) - 1)]
            print()
            q[state, next_state] = r[state, next_state] +  r[next_state, next_next_state] + gamma * gamma * q[next_next_state].max()
            state = next_state
        else :
             # update q matrix by choosing the action that may yield the greatest benefit next.
            q[state, next_state] = r[state, next_state] + gamma * q[next_state].max()
            state = next_state
print("\nprint Q\n")
print(q)

# use Q matrix to teach us how to move to the target
state = random.randint(0,6)
print('\nrobot at {}'.format(state))
count = 0
while state != 6:
    if count > 20: # if we have tried 20 time, failed
        print('\nfail\n')
        break
    
    # choose q_max
    q_max = q[state].max()
    q_max_action = []
    for action in range(7):
        if q[state, action] == q_max:
            q_max_action.append(action)
    # choose a valid action randomly
    randomIndex = random.randint(0, len(q_max_action) - 1)
    next_state = q_max_action[randomIndex]
    print("\nrobot goes to " + str(next_state) + "\n")
    state = next_state