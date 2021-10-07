import numpy as np
import cvxpy as cp
import json
import os
import sys

arr = [0.5, 1, 2]
team = 66
y = arr[team%3]
# y=1
cost = -10/y

all_actions = [
##   0      1       2        3       4        5       6      7         8        9
    'UP', 'DOWN', 'RIGHT', 'LEFT', 'STAY', 'SHOOT', 'HIT', 'CRAFT', 'GATHER', 'NONE'
]

class Position:
    def __init__(self, letter, actions, probs): 
        self.letter = letter
        self.actions = {}
        for ind, act in enumerate(actions):     ## actions has indices of all actions dict, for actions from this state
            self.actions[act] = probs[ind]      ## storing actions like action:(probability of success) for eg- 'UP':0.85

pos = {}
pos['C'] = Position('C', ['UP','DOWN','RIGHT', 'LEFT', 'STAY', 'SHOOT', 'HIT', 'NONE'],[0.85,0.85,0.85,0.85,0.85,0.5,0.1,1.0])
pos['E'] = Position('E', ['LEFT','STAY','SHOOT','HIT', 'NONE'],[1.0,1.0,0.9,0.2,1.0])
pos['W'] = Position('W', ['RIGHT','STAY','SHOOT', 'NONE'],[1.0,1.0,0.25,1.0])
#                                                                     1    2     3 arrows
pos['N'] = Position('N', ['DOWN','STAY','CRAFT', 'NONE'],[0.85,0.85,[0.5, 0.35, 0.15],1.0])
pos['S'] = Position('S', ['UP','STAY','GATHER', 'NONE'],[0.85,0.85,0.75,1.0])

class State:
    def __init__(self, material, arrows, mm_state, mm_health, letter):
        self.material = material
        self.arrows = arrows
        self.mm_state = mm_state
        self.mm_health = mm_health
        self.letter = letter        

TASK = 1
ARROWS_RANGE=4
MM_HEALTH_RANGE=5
MM_STATE_RANGE=2    # D=0, R=1
MATERIAL_RANGE=3
POSITION_RANGE=5    # C=0, E=1, W=2, N=3, S=4  
TOT_STATES = POSITION_RANGE*MATERIAL_RANGE*ARROWS_RANGE*MM_STATE_RANGE*MM_HEALTH_RANGE


# index in utilities array
material_ind = 1
arrow_ind = 2
state_ind = 3
health_ind = 4
position_ind = 0

pos_map = ['C','E','W','N','S']
pos_inv_map = {'C':0,'E':1,'W':2,'N':3,'S':4}
mm_state_map = ['D','R']
mm_state_inv_map = {'D':0,'R':1}

# computing hash value for a particular state
def state_to_hash(state):
    return (pos_inv_map[state.letter] * (MATERIAL_RANGE* ARROWS_RANGE* MM_STATE_RANGE * MM_HEALTH_RANGE) +
            state.material * (ARROWS_RANGE* MM_STATE_RANGE* MM_HEALTH_RANGE) +
            state.arrows * (MM_STATE_RANGE* MM_HEALTH_RANGE) +
            mm_state_inv_map[state.mm_state] *( MM_HEALTH_RANGE) +
            state.mm_health)

# Computing the state using hash value
def hash_to_state(hash_num):
    if type(hash_num) != int:
        raise ValueError

    if not (0 <= hash_num < TOT_STATES):
        raise ValueError

    pos = pos_map[hash_num // (MATERIAL_RANGE* ARROWS_RANGE* MM_STATE_RANGE * MM_HEALTH_RANGE)]
    hash_num = hash_num % (MATERIAL_RANGE*ARROWS_RANGE* MM_STATE_RANGE * MM_HEALTH_RANGE)
    mat = hash_num//(ARROWS_RANGE* MM_STATE_RANGE * MM_HEALTH_RANGE)
    hash_num = hash_num % (ARROWS_RANGE* MM_STATE_RANGE * MM_HEALTH_RANGE)
    arrow = hash_num // (MM_STATE_RANGE * MM_HEALTH_RANGE)
    hash_num = hash_num % (MM_STATE_RANGE * MM_HEALTH_RANGE)
    mm_st = mm_state_map[hash_num // (MM_HEALTH_RANGE)]
    hash_num = hash_num % MM_HEALTH_RANGE
    mm_helth = hash_num
    
    return State(mat, arrow, mm_st, mm_helth, pos)

def state_to_tuple(state):
    return ((state.letter, state.material, state.arrows, state.mm_state, state.mm_health))
    
success_move = {
    'C':{
        'UP':'N', 'DOWN':'S','RIGHT':'E', 'LEFT':'W', 'STAY':'C', 'SHOOT':'C', 'HIT':'C','NONE':'C'
    },
    'E':{
        'LEFT':'C', 'STAY':'E', 'SHOOT':'E', 'HIT':'E','NONE':'E'
    },
    'W':{
        'RIGHT':'C', 'STAY':'W', 'SHOOT':'W', 'NONE':'W'
    },
    'N':{
        'DOWN':'C', 'STAY':'N', 'CRAFT':'N', 'NONE':'N'
    },
    'S':{
        'UP':'C', 'STAY':'S', 'GATHER':'S', 'NONE':'S'
    }
}
unsuccess_move = {
    'C':{
        'UP':'E', 'DOWN':'E','RIGHT':'E', 'LEFT':'E', 'STAY':'E', 'SHOOT':'C', 'HIT':'C','NONE':'C'
    },
    'E':{
        'LEFT':'C', 'STAY':'E', 'SHOOT':'E', 'HIT':'E','NONE':'E'
    },
    'W':{
        'RIGHT':'C', 'STAY':'W', 'SHOOT':'W', 'NONE':'W'
    },
    'N':{
        'DOWN':'E', 'STAY':'E', 'CRAFT':'N', 'NONE':'N'
    },
    'S':{
        'UP':'E', 'STAY':'E', 'GATHER':'S', 'NONE':'S'
    }
}

mm_state_prob={
    'D': { 'D':0.8, 'R':0.2},
    'R': {'D':0.5, 'R':0.5}
}

def movement(curr, action, success):
    if(success):
        return success_move[curr][action]
    else:
        return unsuccess_move[curr][action]
    
def get_actions(state):
    actions=[]
    for action in pos[state.letter].actions:   # loop over all the actions for the current state
        if state.arrows==0 and action=='SHOOT':
            continue
        if state.material==0 and action=='CRAFT':
            continue
        if state.mm_health==0 and not action=='NONE':
            continue
        if state.mm_health>0 and action=='NONE':
            continue
        actions.append(action)

    return actions

a_cols = 0
att = 0
for i in range(TOT_STATES):
    t = hash_to_state(i)
    a_cols = a_cols + len(get_actions(t))

def get_d_next_states(state, act_type):     # this function is called when MM is dormant or MM is ready and IJ is not at C or E
    ways = []
    if(act_type=='UP'):
        ways = []
        ways.append((pos[state.letter].actions['UP']*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['UP']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['UP'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['UP'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='DOWN'):
        ways = []
        ways.append((pos[state.letter].actions['DOWN']*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['DOWN']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['DOWN'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['DOWN'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
        
    elif(act_type=='RIGHT'):
        ways = []
        ways.append((pos[state.letter].actions['RIGHT']*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['RIGHT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['RIGHT'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['RIGHT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
    
    elif(act_type=='LEFT'):
        ways = []
        ways.append((pos[state.letter].actions['LEFT']*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['LEFT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['LEFT'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['LEFT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='STAY'):
        ways = []
        ways.append((pos[state.letter].actions['STAY']*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['STAY']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['STAY'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['STAY'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
        
    elif(act_type=='SHOOT'):
        ways = []
        ways.append((pos[state.letter].actions['SHOOT']*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows-1, 'D', state.mm_health-1, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['SHOOT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows-1, 'R', state.mm_health-1, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['SHOOT'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows-1, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['SHOOT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows-1, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='HIT'):
        ways = []
        ways.append((pos[state.letter].actions['HIT']*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', max(0,state.mm_health-2), movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['HIT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', max(0,state.mm_health-2), movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['HIT'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['HIT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='GATHER'):
        ways = []
        # print(state.letter)
        # print(pos[state.letter].actions)
        ways.append((pos[state.letter].actions['GATHER']*mm_state_prob[state.mm_state]['D'], State(min(state.material+1, MATERIAL_RANGE-1), state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['GATHER']*mm_state_prob[state.mm_state]['R'], State(min(state.material+1, MATERIAL_RANGE-1), state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['GATHER'])*mm_state_prob[state.mm_state]['D'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append(((1-pos[state.letter].actions['GATHER'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='CRAFT'):
        ways = []
        if state.material > 0:
            ways.append((pos[state.letter].actions['CRAFT'][0]*mm_state_prob[state.mm_state]['D'], State(state.material-1, min(state.arrows+1,3), 'D', state.mm_health, movement(state.letter, act_type, 1))))
            ways.append((pos[state.letter].actions['CRAFT'][0]*mm_state_prob[state.mm_state]['R'], State(state.material-1, min(state.arrows+1,3), 'R', state.mm_health, movement(state.letter, act_type, 1))))
            ways.append((pos[state.letter].actions['CRAFT'][1]*mm_state_prob[state.mm_state]['D'], State(state.material-1, min(state.arrows+2,3), 'D', state.mm_health, movement(state.letter, act_type, 0))))
            ways.append((pos[state.letter].actions['CRAFT'][1]*mm_state_prob[state.mm_state]['R'], State(state.material-1, min(state.arrows+2,3), 'R', state.mm_health, movement(state.letter, act_type, 0))))
            ways.append((pos[state.letter].actions['CRAFT'][2]*mm_state_prob[state.mm_state]['D'], State(state.material-1, min(state.arrows+3,3), 'D', state.mm_health, movement(state.letter, act_type, 1))))
            ways.append((pos[state.letter].actions['CRAFT'][2]*mm_state_prob[state.mm_state]['R'], State(state.material-1, min(state.arrows+3,3), 'R', state.mm_health, movement(state.letter, act_type, 1))))

    elif(act_type=='NONE'):
        return []
    return ways

def get_r_next_states(state, act_type):     # this function is called only when mm_state in the current position is R and the current position is C or E
    ways = []
    if not (state.mm_state=='R' and (state.letter=='C' or state.letter=='E')):
        print("ERROR")
    if(act_type=='UP'):
        ways = []
        ways.append((mm_state_prob[state.mm_state]['D'], State(state.material, 0, 'D', min(MM_HEALTH_RANGE-1, state.mm_health+1), state.letter)))
        ways.append((pos[state.letter].actions['UP']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['UP'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='DOWN'):
        ways = []
        ways.append((mm_state_prob[state.mm_state]['D'], State(state.material, 0, 'D', min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter)))
        ways.append((pos[state.letter].actions['DOWN']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['DOWN'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
        
    elif(act_type=='RIGHT'):
        ways = []
        ways.append((mm_state_prob[state.mm_state]['D'], State(state.material, 0, 'D', min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter)))
        ways.append((pos[state.letter].actions['RIGHT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['RIGHT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
    
    elif(act_type=='LEFT'):
        ways = []
        ways.append((mm_state_prob[state.mm_state]['D'], State(state.material, 0, 'D', min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter)))
        ways.append((pos[state.letter].actions['LEFT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['LEFT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='STAY'):
        ways = []
        ways.append((mm_state_prob[state.mm_state]['D'], State(state.material, 0, 'D', min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter)))
        ways.append((pos[state.letter].actions['STAY']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['STAY'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
        
    elif(act_type=='SHOOT'):
        ways = []
        ways.append((mm_state_prob[state.mm_state]['D'], State(state.material, 0, 'D', min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter)))
        ways.append((pos[state.letter].actions['SHOOT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows-1, 'R', state.mm_health-1, movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['SHOOT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows-1, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='HIT'):
        ways = []
        ways.append((mm_state_prob[state.mm_state]['D'], State(state.material, 0, 'D', min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter)))
        ways.append((pos[state.letter].actions['HIT']*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', max(0,state.mm_health-2), movement(state.letter, act_type, 1))))
        ways.append(((1-pos[state.letter].actions['HIT'])*mm_state_prob[state.mm_state]['R'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='NONE'):
        return []
    return ways
        
def get_a_matrix():
    a = np.zeros((TOT_STATES, a_cols), dtype=np.float64)

    idx = 0
    for i in range(TOT_STATES):
        st = hash_to_state(i)
        actions = get_actions(st)

        for action in actions:
            a[i][idx] += 1.0
            next_states = []
            if(st.mm_state=='D'):
                next_states = get_d_next_states(st,action)
            else:
                if(st.letter=='C' or st.letter=='E'):
                    next_states = get_r_next_states(st,action)
                else:
                    next_states = get_d_next_states(st,action)
            # if(i==1 and idx==5):
            #     print(action)
            # if action=='NONE':
            #     a[i][idx]+=1
            for next_state in next_states:
                # a[i][idx] += next_state[0]
                # if st.material == next_state[1].material and st.arrows==next_state[1].arrows and st.mm_state==next_state[1].mm_state and st.mm_health==next_state[1].mm_health and st.letter==next_state[1].letter:
                # # # # if st.mm_state==next_state[1].mm_state:
                #     continue
                a[state_to_hash(next_state[1])][idx] -= next_state[0]
                # a[i][idx]+=next_state[0]

            idx += 1

    return a

def get_r_matrix():
        
    r = np.full((1, a_cols), cost)

    idx = 0
    for i in range(TOT_STATES):
        state = hash_to_state(i)
        actions = get_actions(state)
        for action in actions:
            if ((state.letter=='C' or state.letter=='E') and (state.mm_state=='R')):
                r[0][idx] = mm_state_prob[state.mm_state]['D']*(cost-40) + mm_state_prob[state.mm_state]['R']*(pos[state.letter].actions[action])*(cost) + mm_state_prob[state.mm_state]['R']*(1-pos[state.letter].actions[action])*(cost)
                # r[0][idx] = -40
            if action == 'NONE':
                r[0][idx] = 0
            idx += 1
    return r

def get_alpha_matrix():
    alpha = np.zeros((TOT_STATES, 1))
    start = State(2,3,'R',4,'C')
    start_hash = state_to_hash(start)
    alpha[start_hash][0] = 1.0
    return alpha

def solver(alpha, a,r):
    x = cp.Variable(shape=(a_cols, 1), name="x")
    
    constr = [ cp.matmul(a, x) == alpha, x >= 0 ]

    obj = cp.Maximize(cp.matmul(r, x))
    problem = cp.Problem(obj,constr)

    sol = problem.solve()
    obj = sol
    list_x = list(x.value)
    fin_x = [ float(val) for val in list_x]
    return obj, fin_x

def get_policy(x):
    idx = 0
    policy=[]
    for i in range(TOT_STATES):
        state = hash_to_state(i)
        actions = get_actions(state)
        #### changing this can change the policy
        act_idx = np.argmax(x[idx : idx+len(actions)])
        ####
        idx += len(actions)
        best_action = actions[act_idx]
        temp = []
        temp.append(tuple((state.letter, state.material, state.arrows, state.mm_state, 25*state.mm_health)))
        temp.append(best_action)
        policy.append(temp)
        
    return policy

def get_answer_dict(alpha,a,r,objective,x, policy):
    answer_dict = {}
    a=np.around(a,3)
    x_= [ round(val,3) for val in x]
    answer_dict["a"] = a.tolist()
    r = [float(val) for val in np.transpose(r)]
    answer_dict["r"] = r
    alp = [float(val) for val in alpha]
    answer_dict["alpha"] = alp
    answer_dict["x"] = x_
    answer_dict["policy"] = policy
    answer_dict["objective"] = round(float(objective),3)

    return answer_dict

a = get_a_matrix()
r = get_r_matrix()
alpha = get_alpha_matrix()
temp = solver(alpha, a,r)
objective = temp[0]
x = temp[1]
policy = get_policy(x)
results = get_answer_dict(alpha,a,r,objective,x,policy)
# print("objective="+str(results['objective']))


os.makedirs('outputs', exist_ok=True)
json_object = json.dumps(results)
with open("outputs/part_3_output.json", 'w') as f:
    f.write(json_object)
    
