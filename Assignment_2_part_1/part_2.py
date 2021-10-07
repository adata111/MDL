import numpy as np
from copy import deepcopy
import os
import sys

arr = [0.5, 1, 2]
x = 66
y = arr[x%3]
cost = -10/y

gamma = 0.999
delta = 0.001

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

class State:
    def __init__(self, material, arrows, mm_state, mm_health, letter):
        self.material = material
        self.arrows = arrows
        self.mm_state = mm_state
        self.mm_health = mm_health
        self.letter = letter

pos = {}
pos['C'] = Position('C', ['UP','DOWN','RIGHT', 'LEFT', 'STAY', 'SHOOT', 'HIT'],[0.85,0.85,0.85,0.85,0.85,0.5,0.1])
pos['E'] = Position('E', ['LEFT','STAY','SHOOT','HIT'],[1.0,1.0,0.9,0.2])
pos['W'] = Position('W', ['RIGHT','STAY','SHOOT'],[1.0,1.0,0.25])
#                                                             1    2     3 arrows
pos['N'] = Position('N', ['DOWN','STAY','CRAFT'],[0.85,0.85,[0.5, 0.35, 0.15]])
pos['S'] = Position('S', ['UP','STAY','GATHER'],[0.85,0.85,0.75])

TASK = 1
ARROWS_RANGE=4
MM_HEALTH_RANGE=5
MM_STATE_RANGE=2    # D=0, R=1
MATERIAL_RANGE=3
POSITION_RANGE=5    # C=0, E=1, W=2, N=3, S=4  

# index in utilities array
material_ind = 1
arrow_ind = 2
state_ind = 3
health_ind = 4
position_ind = 0

pos_map = ['C','E','W','N','S']
mm_state_map = ['D','R']

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

def get_utility_hist_index(state,it):
    return (state.letter, state.material, state.arrows, state.mm_state, state.mm_health, it)

utilities_history = {}
utilities = np.zeros((POSITION_RANGE, MATERIAL_RANGE, ARROWS_RANGE, MM_STATE_RANGE, MM_HEALTH_RANGE))
for state, _ in np.ndenumerate(utilities):
    utilities_history[(pos_map[state[0]],state[1],state[2],mm_state_map[state[3]],state[4],0)] = [0, None]
    # print(state)

def movement(curr, action, success):
    if(success):
        return success_move[curr][action]
    else:
        return unsuccess_move[curr][action]

def get_action_utility(act_type, state, it, next_mm_state = 'B', ready_prob=0):
    # state = State(*state)
    reward=0
    ways = []
    if(act_type=='UP'):
        ways = []
        # print(pos)
        # print(state.letter)
        ways.append((pos[state.letter].actions['UP'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['UP'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['UP']), State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append((1-(pos[state.letter].actions['UP']), State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='DOWN'):
        ways = []
        ways.append((pos[state.letter].actions['DOWN'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['DOWN'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['DOWN']), State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append((1-(pos[state.letter].actions['DOWN']), State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
        
    elif(act_type=='RIGHT'):
        ways = []
        ways.append((pos[state.letter].actions['RIGHT'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['RIGHT'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['RIGHT']), State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append((1-(pos[state.letter].actions['RIGHT']), State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
    
    elif(act_type=='LEFT'):
        ways = []
        ways.append((pos[state.letter].actions['LEFT'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['LEFT'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['LEFT']), State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append((1-(pos[state.letter].actions['LEFT']), State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='STAY'):
        ways = []
        ways.append((pos[state.letter].actions['STAY'], State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['STAY'], State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['STAY']), State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append((1-(pos[state.letter].actions['STAY']), State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))
        
    elif(act_type=='SHOOT' and state.arrows>0):
        ways = []
        ways.append((pos[state.letter].actions['SHOOT'], State(state.material, state.arrows-1, 'D', state.mm_health-1, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['SHOOT'], State(state.material, state.arrows-1, 'R', state.mm_health-1, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['SHOOT']), State(state.material, state.arrows-1, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['SHOOT']), State(state.material, state.arrows-1, 'R', state.mm_health, movement(state.letter, act_type, 1))))

    elif(act_type=='HIT'):
        ways = []
        ways.append((pos[state.letter].actions['HIT'], State(state.material, state.arrows, 'D', max(0,state.mm_health-2), movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['HIT'], State(state.material, state.arrows, 'R', max(0,state.mm_health-2), movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['HIT']), State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['HIT']), State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))

    elif(act_type=='GATHER'):
        ways = []
        # print(state.letter)
        # print(pos[state.letter].actions)
        ways.append((pos[state.letter].actions['GATHER'], State(min(state.material+1, MATERIAL_RANGE-1), state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((pos[state.letter].actions['GATHER'], State(min(state.material+1, MATERIAL_RANGE-1), state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 1))))
        ways.append((1-(pos[state.letter].actions['GATHER']), State(state.material, state.arrows, 'D', state.mm_health, movement(state.letter, act_type, 0))))
        ways.append((1-(pos[state.letter].actions['GATHER']), State(state.material, state.arrows, 'R', state.mm_health, movement(state.letter, act_type, 0))))

    elif(act_type=='CRAFT'):
        ways = []
        if state.material > 0:
            ways.append((pos[state.letter].actions['CRAFT'][0], State(state.material-1, min(state.arrows+1,3), 'D', state.mm_health, movement(state.letter, act_type, 1))))
            ways.append((pos[state.letter].actions['CRAFT'][0], State(state.material-1, min(state.arrows+1,3), 'R', state.mm_health, movement(state.letter, act_type, 1))))
            ways.append((pos[state.letter].actions['CRAFT'][1], State(state.material-1, min(state.arrows+2,3), 'D', state.mm_health, movement(state.letter, act_type, 0))))
            ways.append((pos[state.letter].actions['CRAFT'][1], State(state.material-1, min(state.arrows+2,3), 'R', state.mm_health, movement(state.letter, act_type, 0))))
            ways.append((pos[state.letter].actions['CRAFT'][2], State(state.material-1, min(state.arrows+3,3), 'D', state.mm_health, movement(state.letter, act_type, 1))))
            ways.append((pos[state.letter].actions['CRAFT'][2], State(state.material-1, min(state.arrows+3,3), 'R', state.mm_health, movement(state.letter, act_type, 1))))

    elif(act_type=='NONE'):
        return utilities_history[get_utility_hist_index(state, it)][0]

    my_util=0
    for way in ways:    # way[0]= probability of action, way[1] = new state,
        if ((act_type=='SHOOT' or act_type=='HIT') and (state.mm_health>way[1].mm_health) and (way[1].mm_health==0)):
            reward=50
        else: 
            reward=0
        c=cost
        if TASK == 2.2 and act_type=='STAY':
            c=0
        if(next_mm_state=='B'):
            if way[1].mm_state == 'D':
                my_util += way[0]*0.8*(c+reward+gamma*utilities_history[get_utility_hist_index(way[1],it)][0])
                
            if way[1].mm_state == 'R':
                my_util += way[0]*0.2*(c+reward+gamma*utilities_history[get_utility_hist_index(way[1],it)][0])
        else:
            if way[1].mm_state==next_mm_state:
                my_util += way[0]*ready_prob*(c+reward+gamma*utilities_history[get_utility_hist_index(way[1],it)][0])

    return my_util

def ready_action_util(act_type, state, it):
    ways = []
    util = 0
    if state.letter == 'C':
        # if act_type=='SHOOT':
        #     ways.append((0.5*0.5, State(state.material,0,'D',min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter), -40))
        # elif act_type=='HIT':
        #     ways.append((0.5*0.9, State(state.material,0,'D',min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter), -40))
        # else:
        ways.append((0.5, State(state.material,0,'D',min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter), -40))
    elif state.letter == 'E':
        # if act_type=='SHOOT':
        #     ways.append((0.5*0.1, State(state.material,0,'D',min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter), -40))
        # elif act_type=='HIT':
        #     ways.append((0.5*0.8, State(state.material,0,'D',min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter), -40))
        # else:
        ways.append((0.5, State(state.material,0,'D',min(MM_HEALTH_RANGE-1,state.mm_health+1), state.letter), -40))
    else:
        ## IJ will complete the step and MM will go to D state
        util+=get_action_utility(act_type,state,it, 'D', 0.5)
    ## for MM not attacking
    ## IJ will complete step, 0.5 prob multiplied and MM will remain in R state
    util+=get_action_utility(act_type,state,it, 'R', 0.5)
    for way in ways:
        c=cost
        if TASK == 2.2 and act_type=='STAY':
            c=0
        util += way[0]*(c+way[2]+gamma*utilities_history[get_utility_hist_index(way[1],it)][0])
    return util

def print_all(path):
    with open(path, 'w') as f:
        prev=sys.stdout
        sys.stdout=f
        it = 0
        for data in utilities_history:

            if data[5]==0:
                continue
            if not data[5]==it:
                it=data[5]
                print()
                print("iteration="+str(it-1))
            tem = data
            tem = tem[:5]
            temp = list(tem)
            temp[4] = 25*temp[4]
            tem = "("+str(temp[0])+","+str(temp[1])+","+str(temp[2])+","+str(temp[3])+","+str(temp[4])+")"
            print('{}:{}=[{:.3f}]'.format(tem, utilities_history[data][1], utilities_history[data][0]))
            # print(state, file=f)
        sys.stdout = prev

def value_iteration(utilities, TASK):
    policies = np.full((POSITION_RANGE, MATERIAL_RANGE, ARROWS_RANGE, MM_STATE_RANGE, MM_HEALTH_RANGE), -1, dtype='int')

    done = False
    iteration = 0
    diff = np.NINF
    while not done: 
        temp = np.zeros(utilities.shape)
        diff = -np.inf
        __count=0
        for stat, util in np.ndenumerate(utilities):
            state=State(stat[material_ind],stat[arrow_ind],mm_state_map[stat[state_ind]],stat[health_ind],pos_map[stat[position_ind]])
            # print(stat[4])
            new_util = -np.inf
            best_util = -np.inf
            best_action = None
            if state.mm_health==0:
                best_util = get_action_utility('NONE',state,iteration)
                best_action = 'NONE'
            else:
                for action in pos[pos_map[stat[position_ind]]].actions:   # loop over all the actions for the current state
                    if state.arrows==0 and action=='SHOOT':
                        continue
                    if state.material==0 and action=='CRAFT':
                        continue
                    if(state.mm_state=='D'):
                        new_util = get_action_utility(action, state,iteration) #passing prev iter num, for edge case, index=0, all values are already set so 0 is previous itr
                    else:
                        new_util = ready_action_util(action, state,iteration)
                    
                    if new_util>best_util:
                        best_util = new_util
                        best_action = action

            temp[stat] = best_util

            utilities_history[get_utility_hist_index(state,iteration+1)] = [best_util, best_action]

            diff = max(diff, abs(util - best_util))

        utilities = deepcopy(temp)

        if diff < delta:
            done = True
        iteration+=1

os.makedirs('outputs', exist_ok=True)
# task 1
TASK = 1
#print(TASK)
value_iteration(utilities, TASK)
print_all('outputs/part_2_trace.txt')                    
#print()                
#print()

# task 2.1
TASK = 2.1
#print(TASK)
success_move['E']['LEFT']='W'
unsuccess_move['E']['LEFT']='W'

utilities_history = {}
utilities = np.zeros((POSITION_RANGE, MATERIAL_RANGE, ARROWS_RANGE, MM_STATE_RANGE, MM_HEALTH_RANGE))
for state, _ in np.ndenumerate(utilities):
    utilities_history[(pos_map[state[0]],state[1],state[2],mm_state_map[state[3]],state[4],0)] = [0, None]

value_iteration(utilities, TASK)
print_all('outputs/part_2_task_2.1_trace.txt')   

success_move['E']['LEFT']='C'
unsuccess_move['E']['LEFT']='C'                
#print()                
#print()

# task 2.2
TASK = 2.2
#print(TASK)
utilities_history = {}
utilities = np.zeros((POSITION_RANGE, MATERIAL_RANGE, ARROWS_RANGE, MM_STATE_RANGE, MM_HEALTH_RANGE))
for state, _ in np.ndenumerate(utilities):
    utilities_history[(pos_map[state[0]],state[1],state[2],mm_state_map[state[3]],state[4],0)] = [0, None]
value_iteration(utilities, TASK)
print_all('outputs/part_2_task_2.2_trace.txt')                  
#print()                
#print()

# task 2.3
TASK = 2.3
#print(TASK)
gamma = 0.25
utilities_history = {}
utilities = np.zeros((POSITION_RANGE, MATERIAL_RANGE, ARROWS_RANGE, MM_STATE_RANGE, MM_HEALTH_RANGE))
for state, _ in np.ndenumerate(utilities):
    utilities_history[(pos_map[state[0]],state[1],state[2],mm_state_map[state[3]],state[4],0)] = [0, None]
value_iteration(utilities, TASK)
print_all('outputs/part_2_task_2.3_trace.txt')    

