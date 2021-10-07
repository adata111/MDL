import numpy as np
import random
import copy
import client
from client import get_errors, submit

import sys
mystdout = open("atl30.txt","a")
mystdout_less = open("ats30.txt","a")
diagram = open("fig30.txt", "a")
old_stdout = sys.stdout
sys.stdout = mystdout

initial_weights = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
pop_size = 16       # must always keep even because the way new population is created, we will get index out of bounds error
features = len(initial_weights)
gen = 6
n = 4
SECRET_KEY = '8OeK1o9obxFbkN15KFJbAfiT13UtlKKVLaAPf3KHTIMrzBDY7l'
gen_fit_func = np.zeros(gen+1)

def mutate(vector,mut_prob, range_val):
# Generating a random number from uniform distribution between given range
    temp = np.copy(vector)
# num = np.random.uniform(low=-range_val,high=range_val)
    sys.stdout = diagram
    print("original vector: ",vector)
    for i in range(features):
        # Mutating temp[i] based on the probability
        ran = np.random.random()
        if ran < mut_prob:
            if(temp[i] < 1e-30 and temp[i]>-1e-30):
                temp[i] += np.random.choice([-1e-20, 1e-20])
            # f = temp[i]/range_val
            f = range_val
            num = np.random.uniform(low=-f, high=f)
            temp[i] = temp[i] * (1 + num)
        # If vector[i] is going out of bounds the value is adjusted accordingly
        if(temp[i]>10) :
            temp[i]=10
        elif(temp[i]<-10):
            temp[i]=-10

    print("mutated vector: ", temp)
    sys.stdout = mystdout
    return temp

def russian_roulette(fit_func):   

    fit_func = -fit_func
    if(np.ptp(fit_func) == 0):
    	if(np.min(fit_func)==0):
    		normalized_fit_func = fit_func+1
    	else:
        	normalized_fit_func = fit_func/np.min(fit_func)
    else:
        normalized_fit_func = (fit_func - np.min(fit_func)) / (np.ptp(fit_func)) # to get fit_func in [0,1]
    
    thresholds = []
    prev_prob = 0.0
    fit_func_sum = np.sum(normalized_fit_func)
    for val in normalized_fit_func:
        prev_prob = prev_prob + (val/fit_func_sum)
        thresholds.append(prev_prob)

    return thresholds

def get_parent_indices(thresholds, k):
# num = np.random.random() # in [0, 1)

    num = np.random.random((2,1))
    sys.stdout =old_stdout
    print(num[0]," ",num[1])
    sys.stdout = mystdout
    ret = []
    flag = 0
    for j in range(2):
        flag = 0
        for i in range(pop_size):
            if num[j] < thresholds[i]:
                ret.append(i)
                flag=1
                break
        if flag==0:
            ret.append(0) 
        
    return ret

def fitness(errs, fit_change):
    # if(fit_change>10):
    #     return errs[0]+errs[1]+2*abs(errs[0]-errs[1])
    # elif(fit_change>5):
    #     return errs[0]+errs[1]+abs(errs[0]-errs[1])
    # else:
    return errs[0]+errs[1]+abs(errs[0]-errs[1])

def simulated_binary_crossover(parent1, parent2):
    child1 = np.empty(features)
    child2 = np.empty(features)
    u = random.random() 
    eta_c = 3       #distribution index
    beta = 0 
    if (u < 0.5):
        beta = (2 * u)**((eta_c + 1)**-1)
    else:
        beta = ((2*(1-u))**-1)**((eta_c + 1)**-1)


    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    child1 = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)

    return child1, child2


def cross_over(parent1, parent2):
    # setting up pivot
    pivot = np.random.randint(features)
    # Copying each parent in a child vector
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    # Swapping the elements in the children vectors which come after the pivot
    child1[pivot:] = parent2[pivot:]
    child2[pivot:] = parent1[pivot:]
    # Returning the generated children
    return child1, child2

def init_pop(mut_prob, range_val):
    population = np.zeros((pop_size, features))
    for i in range(pop_size):
        arr = np.copy(initial_weights)
        if(i==0):
            population[i] = np.copy(mutate(arr, 0.4, range_val))
        else:
            population[i] = np.copy(mutate(arr,mut_prob,range_val))
    sys.stdout = mystdout_less
    print("Initial Population: ")
    print(population)
    sys.stdout = mystdout
    print("Initial Population: ")
    print(population)
    return population

def get_pop_errors(pop, fit_change):

    fit_func1 = np.zeros(pop_size)
    train_err1 = np.zeros(pop_size)
    val_err1 = np.zeros(pop_size)

    # find population errors
    for j in range(pop_size):
        arr = pop[j].tolist()
        error = get_errors(SECRET_KEY, arr)

        # store value of fitness function
        fit_func1[j] = fitness(error, fit_change)
        train_err1[j] = error[0]
        val_err1[j] = error[1]

    return sort_arrays(fit_func1, train_err1, val_err1, pop)

def sort_arrays(fit_func1, train_err1, val_err1, pop):
# sort population according to fitness function

    s = len(pop)
    fit_func = np.zeros(s)
    train_err = np.zeros(s)
    val_err = np.zeros(s)

    sys.stdout = diagram
    print("\n before sorting")
    for j in range(s):
        print(j, " ", pop[j])
    sys.stdout = mystdout

    print("\n before sorting")
    for j in range(s):
        print(j, " ", pop[j])

    sorted_idx = fit_func1.argsort()
    pop1 = copy.deepcopy(pop)


    sys.stdout = diagram
    print("\n after sorting")
    for j in range(s):
        fit_func[j] = fit_func1[sorted_idx[j]]
        train_err[j] = train_err1[sorted_idx[j]]
        val_err[j] = val_err1[sorted_idx[j]]
        pop[j] = pop1[sorted_idx[j]]
        print(sorted_idx[j], " ", pop[j],fit_func[j])

    print()
    sys.stdout = mystdout

    print("\n after sorting")
    for j in range(s):
        fit_func[j] = fit_func1[sorted_idx[j]]
        train_err[j] = train_err1[sorted_idx[j]]
        val_err[j] = val_err1[sorted_idx[j]]
        pop[j] = pop1[sorted_idx[j]]
        print(sorted_idx[j], " ", pop[j],fit_func[j])

    print()
    return fit_func, train_err, val_err, pop 


def main():
    range_val = 0.1
    mut_prob = 0.5
    fit = 0
    den = 6000
    population = np.zeros((pop_size, features))
    # generate initial population
    population = init_pop( 0.9, range_val)

    par_fit_func1 = np.zeros(pop_size)
    par_train_err1 = np.zeros(pop_size)
    par_val_err1 = np.zeros(pop_size)

    par_fit_func = np.zeros(pop_size)
    par_train_err = np.zeros(pop_size)
    par_val_err = np.zeros(pop_size)

    best_vector = [2.11883826e-20,-1.30947698e-12,-1.26981469e-13,4.35985059e-11,-1.69882674e-10,-1.60736677e-15,6.56637029e-16,3.36778076e-05,-2.05256036e-06,-2.21911133e-08,1.01591767e-09]
    best_fit_func = -1
    best_train_err = -1
    best_val_err = -1

    # find error for population and sort
    sys.stdout = diagram
    print("parent\n")
    sys.stdout = mystdout
    fit+=1
    par_fit_func, par_train_err, par_val_err, population = get_pop_errors(population,fit)

    # sort population according to fitness func value
    # par_fit_func, par_train_err, par_val_err, population = sort_arrays(par_fit_func1, par_train_err1, par_val_err1, population)

    if(best_fit_func == -1 or best_fit_func >= par_fit_func[0]):
        best_vector = np.copy(population[0])
        best_fit_func = np.copy(par_fit_func[0])
        best_train_err = np.copy(par_train_err[0])
        best_val_err = np.copy(par_val_err[0])

    else:
        sys.stdout = mystdout_less
        print("No improvements")
        sys.stdout = mystdout
        print("No improvements")

    sys.stdout = mystdout_less
    print("Best vector = ", population[0], "\n")
    print("Fit function value = ", par_fit_func[0], "\n")
    print("Train error value = ", par_train_err[0], "\n")
    print("Validation error value = ", par_val_err[0], "\n")

    sys.stdout = mystdout
    print("Best vector = ", population[0], "\n")
    print("Fit function value = ", par_fit_func[0], "\n")
    print("Train error value = ", par_train_err[0], "\n")
    print("Validation error value = ", par_val_err[0], "\n")
    gen_fit_func[0] = par_fit_func[0] 


    ##### Start for loop for generations from here
    for i in range(gen):

        sys.stdout = diagram    
        print("Generation: ", i+1)
        sys.stdout = mystdout_less
        print("Generation: ", i+1)
        sys.stdout = mystdout
        print("Generation: ", i+1)

        # if(i>0 and gen_fit_func[i-1]<=gen_fit_func[i]): # check if lowest error for previous iteration's generation and the new generation created(current iteration generation) is same
        #     # if it is same then mutate the population
        #     print("No improvements in last generation")
        #     for pop_i in range(pop_size):
        #         population[pop_i] = mutate(population[pop_i],0.9,0.15)
            
        #     print(population)
        #     if(i==gen-1):
        #         break
        #     # get errors and sort this population
        #     par_fit_func, par_train_err, par_val_err, population = get_pop_errors(population)
        
        if(i==gen-1):
            sys.stdout = mystdout_less
            print(population)
            sys.stdout = mystdout
            print(population)
            break
            
        # array containing originally generated children
        child_org = np.zeros((pop_size, features))
        # array containing children population
        child_pop = np.zeros((pop_size, features))

        new_pop_size = 0
        k = 6
        thresholds = russian_roulette(par_fit_func)
        sys.stdout = diagram
        print("thresholds\n",thresholds)
        sys.stdout = mystdout

        while(new_pop_size < pop_size):

            # find k best parents and use 2 out of k parents for crossover
            # select_idx = random.sample(range(k), 2)
            select_idx = get_parent_indices(thresholds,k)

            par_cross = simulated_binary_crossover(population[select_idx[0]], population[select_idx[1]])

            # mutate generated children
            sys.stdout = old_stdout
            mut1 = mutate(par_cross[0], mut_prob, range_val)
            # print("crossed: ", par_cross[0])
            # print("mutated: ",mut1)
            mut2 = mutate(par_cross[1], mut_prob, range_val)
            # print("crossed: ", par_cross[1])
            # print("mutated: ",mut2)

            # check if generated children are not same as parents. If they are same, then continue (abort this iteration)
            if mut1.tolist() == population[select_idx[0]].tolist() or mut1.tolist() == population[select_idx[1]].tolist():
                print("Discarded")
                continue
            elif mut2.tolist() == population[select_idx[0]].tolist() or mut2.tolist() == population[select_idx[1]].tolist():
                print("Discarded")
                continue
            else:

                sys.stdout = diagram
                print("\nselected indices")
                print(select_idx[0], " ", population[select_idx[0]])
                print(select_idx[1], " ", population[select_idx[1]])

                print("\nafter crossover")
                print(select_idx[0], " ", par_cross[0])
                print(select_idx[1], " ", par_cross[1])

                print("\nafter mutation")   
                print(select_idx[0], " ", mut1)
                print(select_idx[1], " ", mut2)


                sys.stdout = mystdout
                print("\nselected indices")
                print(select_idx[0], " ", population[select_idx[0]])
                print(select_idx[1], " ", population[select_idx[1]])

                print("\nafter crossover")
                print(select_idx[0], " ", par_cross[0])
                print(select_idx[1], " ", par_cross[1])

                print("\nafter mutation")   
                print(select_idx[0], " ", mut1)
                print(select_idx[1], " ", mut2)

                child_org[new_pop_size] = np.copy(par_cross[0])
                child_pop[new_pop_size] = np.copy(mut1)
                new_pop_size += 1

                child_pop[new_pop_size] = np.copy(mut2)
                new_pop_size +=1

        child_fit_func = np.zeros(pop_size)
        child_train_err = np.zeros(pop_size)
        child_val_err = np.zeros(pop_size)

        # Find error for child population
        sys.stdout = diagram
        print("child_pop\n")
        sys.stdout = mystdout
        print("\nChild population")
        fit+=1
        child_fit_func, child_train_err, child_val_err, child_pop = get_pop_errors(child_pop,fit)

        # Temporary storage
        arr_pop = np.zeros((pop_size, features))
        arr_fit_func = np.zeros(pop_size)
        arr_train_err = np.zeros(pop_size)
        arr_val_err = np.zeros(pop_size)

        # select n best parents and n best children
        for j in range(n):
            arr_pop[j] = np.copy(population[j])
            arr_fit_func[j] = np.copy(par_fit_func[j])
            arr_train_err[j] = np.copy(par_train_err[j])
            arr_val_err[j] = np.copy(par_val_err[j])

            arr_pop[j+n] = np.copy(child_pop[j])
            arr_fit_func[j+n] = np.copy(child_fit_func[j])
            arr_train_err[j+n] = np.copy(child_train_err[j])
            arr_val_err[j+n] = np.copy(child_val_err[j])

        # club all the remaining parents and children

        club_pop = np.zeros((pop_size, features))
        club_fit_func = np.zeros(pop_size)
        club_train_err = np.zeros(pop_size)
        club_val_err = np.zeros(pop_size)

        club_pop = np.copy(np.concatenate([population[n:], child_pop[n:]]))
        # print(club_pop)
        club_fit_func1 = np.copy(np.concatenate([par_fit_func[n:], child_fit_func[n:]]))
        club_train_err1 = np.copy(np.concatenate([par_train_err[n:], child_train_err[n:]]))
        club_val_err1 = np.copy(np.concatenate([par_val_err[n:], child_val_err[n:]]))

        # Sorting the clubbed elements
        sys.stdout = diagram
        print("clubbed\n")
        sys.stdout = mystdout
        print("\nClubbed remaining parents and children")
        club_fit_func, club_train_err, club_val_err, club_pop = sort_arrays(club_fit_func1, club_train_err1, club_val_err1, club_pop)
        
        # select (pop_size - 2*n) elements from the combined array to generate new generation
        select = 0
        
        while(select + 2*n < pop_size):
            arr_pop[select + 2*n] = np.copy(club_pop[select])
            arr_fit_func[select + 2*n] = np.copy(club_fit_func[select])
            arr_train_err[select + 2*n] = np.copy(club_train_err[select])
            arr_val_err[select + 2*n] = np.copy(club_val_err[select])
            select += 1

        # Storing the temporary array in new generation population andd Sorting the new generation
        sys.stdout = mystdout
        print("New generation: ")
        par_fit_func, par_train_err, par_val_err, population = sort_arrays(arr_fit_func, arr_train_err, arr_val_err, arr_pop)

        sys.stdout = mystdout_less
        print("New generation: ")
        print(population) 
        sys.stdout = mystdout
        print("New generation: ")
        print(population) 

        # if the error generated is less than the previous generation then update the error
        # After sorting the 0th element will have the best elekeepdimsment
        if(best_fit_func == -1 or best_fit_func > par_fit_func[0]):
            best_vector = np.copy(population[0])
            best_fit_func = np.copy(par_fit_func[0])
            best_train_err = np.copy(par_train_err[0])
            best_val_err = np.copy(par_val_err[0])

        else:
            sys.stdout = mystdout_less
            print("No improvements")
            sys.stdout = mystdout
            print("No improvements")

        gen_fit_func[i+1] = par_fit_func[0]
        if(i==gen-1):
            sys.stdout = mystdout_less
            print("total err",par_fit_func)
            print("train err" , par_train_err)
            print("validation err",par_val_err)
            sys.stdout = mystdout
            print("total err",par_fit_func)
            print("train err" , par_train_err)
            print("validation err",par_val_err)

        sys.stdout = mystdout_less
        print("Avg fit func = ",(np.mean(par_fit_func)), "\n")
        print("Best vector = ", best_vector, "\n")
        print("Fit function value = ", best_fit_func, "\n")
        print("Train error value = ", best_train_err, "\n")
        print("Validation error value = ", best_val_err, "\n")

        sys.stdout = mystdout
        print("Best vector = ", best_vector, "\n")
        print("Fit function value = ", best_fit_func, "\n")
        print("Train error value = ", best_train_err, "\n")
        print("Validation error value = ", best_val_err, "\n")

    return best_vector

print("************************************************************************\n\n")
best_vector = main()

sys.stdout = old_stdout
submit(SECRET_KEY, best_vector.tolist())
