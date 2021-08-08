# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>

#EXERCISE 1
import pulp
def convert_to_ilp(nodes, edges):
    prob = pulp.LpProblem("Graph_model_prob",pulp.LpMinimize)
    for i in range(len(nodes)):
        for j in range(len(nodes[i].costs)):
            #create variable unary costs
            locals()['unary_'+str(i)+'_label_'+str(j)]=pulp.LpVariable('unary_'+str(i)+'_label_'+str(j),lowBound=0, upBound = 1,cat=pulp.LpInteger)
            
            try:
                objective+=nodes[i].costs[j]*locals()['unary_'+str(i)+'_label_'+str(j)]
            except UnboundLocalError:
                objective=nodes[i].costs[j]*locals()['unary_'+str(i)+'_label_'+str(j)]
            
            try :
                locals()['constraint_unary_'+str(i)]+=locals()['unary_'+str(i)+'_label_'+str(j)]
            except KeyError:
                locals()['constraint_unary_'+str(i)]=locals()['unary_'+str(i)+'_label_'+str(j)]
    for ed in edges:
        for k in range(len(nodes[ed.right].costs)):
            for j in range(len(nodes[ed.left].costs)):
                #create variable edges
                locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]=pulp.LpVariable('pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k),lowBound=0, upBound = 1,cat=pulp.LpInteger)
                
                objective+=ed.costs[j,k]*locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                
                try :
                    locals()['constraint_pairwise_'+str(ed.left)+'_'+str(ed.right)]+=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                except KeyError:
                    locals()['constraint_pairwise_'+str(ed.left)+'_'+str(ed.right)]=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                    
                try :
                    locals()['constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k)]+=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                except KeyError:
                    locals()['constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k)]=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                
                try :
                    locals()['constraint_pairwise_edgeL_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)]+=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                except KeyError:
                    locals()['constraint_pairwise_edgeL_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)]=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                 
                    
                
        # The objective function is added to 'prob' first
    prob+=objective
        
    for i in range(len(nodes)):
        prob.addConstraint(locals()['constraint_unary_'+str(i)]==1,'constraint_unary_'+str(i))
    for ed in edges:
        prob.addConstraint(locals()['constraint_pairwise_'+str(ed.left)+'_'+str(ed.right)]==1,'constraint_pairwise_'+str(ed.left)+'_'+str(ed.right))
        for k in range(len(nodes[ed.right].costs)):
            prob.addConstraint(locals()['constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k)]==locals()['unary_'+str(ed.right)+'_label_'+str(k)],'constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k))
        for j in range(len(nodes[ed.left].costs)):
            prob.addConstraint(locals()['constraint_pairwise_edgeL_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)]==locals()['unary_'+str(ed.left)+'_label_'+str(j)],'constraint_pairwise_edgeL_'+str(ed.right)+'_'+str(ed.left)+'_label_'+str(j))

    return prob
        
def ilp_to_labeling(nodes, edges, prob):
    assignment=[0]*len(nodes)
    assigned=0
    for v in reversed(prob.variables()):
        if v.value()==1 and v.name[0]!='p':
            assignment[int(v.name[6])]=int(v.name[-1])
            assigned=assigned+1
        if assigned==len(nodes):
            break
    return assignment
    

#%%
#EXERCISE 2

def convert_to_lp(nodes, edges):
    prob = pulp.LpProblem("Graph_model_prob",pulp.LpMinimize)
    for i in range(len(nodes)):
        for j in range(len(nodes[i].costs)):
            #create variable unarycosts
            locals()['unary_'+str(i)+'_label_'+str(j)]=pulp.LpVariable('unary_'+str(i)+'_label_'+str(j),lowBound=0, upBound = 1)
            
            try:
                objective+=nodes[i].costs[j]*locals()['unary_'+str(i)+'_label_'+str(j)]
            except UnboundLocalError:
                objective=nodes[i].costs[j]*locals()['unary_'+str(i)+'_label_'+str(j)]
            
            try :
                locals()['constraint_unary_'+str(i)]+=locals()['unary_'+str(i)+'_label_'+str(j)]
            except KeyError:
                locals()['constraint_unary_'+str(i)]=locals()['unary_'+str(i)+'_label_'+str(j)]
    for ed in edges:
        for k in range(len(nodes[ed.right].costs)):
            for j in range(len(nodes[ed.left].costs)):
                #create variable edges
                locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]=pulp.LpVariable('pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k),lowBound=0, upBound = 1)
                
                
                objective+=ed.costs[j,k]*locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                
                try :
                    locals()['constraint_pairwise_'+str(ed.left)+'_'+str(ed.right)]+=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                except KeyError:
                    locals()['constraint_pairwise_'+str(ed.left)+'_'+str(ed.right)]=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                      
                try :
                    locals()['constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k)]+=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                except KeyError:
                    locals()['constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k)]=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                
                try :
                    locals()['constraint_pairwise_edgeL_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)]+=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
                except KeyError:
                    locals()['constraint_pairwise_edgeL_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)]=locals()['pairwise_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)+'_'+str(k)]
#                    try :
#                        locals()['constraint_pairwise_edge_'+str(ed.right)+'_'+str(ed.left)+'_label_'+str(j)]+=locals()['pairwise_'+str(ed.right)+'_'+str(ed.left)+'_label_'+str(k)+'_'+str(j)]
#                    except KeyError:
#                        locals()['constraint_pairwise_edge_'+str(ed.right)+'_'+str(ed.left)+'_label_'+str(j)]=locals()['pairwise_'+str(ed.right)+'_'+str(ed.left)+'_label_'+str(k)+'_'+str(j)]
                    
                    
    prob.setObjective(objective)
        
    for i in range(len(nodes)):
        prob.addConstraint(locals()['constraint_unary_'+str(i)]==1,'constraint_unary_'+str(i))
    for ed in edges:
        prob.addConstraint(locals()['constraint_pairwise_'+str(ed.left)+'_'+str(ed.right)]==1,'constraint_pairwise_'+str(ed.left)+'_'+str(ed.right))
        for k in range(len(nodes[ed.right].costs)):
            prob.addConstraint(locals()['constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k)]==locals()['unary_'+str(ed.right)+'_label_'+str(k)],'constraint_pairwise_edgeR_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(k))
        for j in range(len(nodes[ed.left].costs)):
            prob.addConstraint(locals()['constraint_pairwise_edgeL_'+str(ed.left)+'_'+str(ed.right)+'_label_'+str(j)]==locals()['unary_'+str(ed.left)+'_label_'+str(j)],'constraint_pairwise_edgeL_'+str(ed.right)+'_'+str(ed.left)+'_label_'+str(j))

    return prob
def Naive_rounding(assignment,prob):
    for v in reversed(prob.variables()):
        if v.name[0]=='p':
            check_label1=assignment[int(v.name[9])]==int(v.name[-3])
            check_label2=assignment[int(v.name[11])]==int(v.name[-1])
            if check_label1 and check_label2:
                v.value=1
            else:
                v.value=0
        elif v.name[0]=='u':
            if assignment[int(v.name[6])]==int(v.name[-1]):
                v.value=1
            else:
                v.value=0
                
    
def lp_to_labeling(nodes, edges, prob):

    varsdict = {}
    for v in reversed(prob.variables()):
        if v.name[0]=='p':
            break
        varsdict[v.name] = v.varValue
            
    assignment=[0]*len(nodes)
    for i in range(len(nodes)):
        maxi=0
        for j in range(len(nodes[i].costs)):
            if varsdict['unary_'+str(i)+'_label_'+str(j)]>maxi:
                assignment[i]=j
                maxi=varsdict['unary_'+str(i)+'_label_'+str(j)]
               
            

    return assignment
    
#for v in lp.variables():
#    print(v.name,'=', v.value())
    
#%%

## 2.2 Naive roinding is performed on relaxed LP solutions to round the fractional varianles so as to obtain an approximate solution of a non-relaxed problem. So, ultimately we obtain rounded values of solution from LP problems with the hope to obtain similar solution to the ILP problem.
## 2.3 Rounding does not give expected outputs.
####Exercise 2.4

from model_2_4 import *
count=0
for acyclic in ACYCLIC_MODELS:
    count=1+count
    print ('\n ACYCLIC GRAPH ', count)
    lp = convert_to_lp(acyclic[0],acyclic[1])
    assert(lp.solve())
    print ('LP_problem:')
#    for var in lp.variables():
#        print('{} -> {}'.format(var.name, var.value()))
    print('Optimal value = ', lp.objective.value())
    
    
    ilp = convert_to_ilp(acyclic[0],acyclic[1])
    assert(ilp.solve())
    print ('ILP_problem:')
#    for var in ilp.variables():
#        print('{} -> {}'.format(var.name, var.value()))
    print('Optimal value = ', ilp.objective.value())
    
count=0
for cyclic in CYCLIC_MODELS:
    count=1+count
    print ('\n CYCLIC GRAPH ', count)
    lp = convert_to_lp(cyclic[0],cyclic[1])
    assert(lp.solve())
    print ('LP_problem:')
#    for var in lp.variables():
#        print('{} -> {}'.format(var.name, var.value()))
    print('Optimal value = ', lp.objective.value())
    
    
    ilp = convert_to_ilp(cyclic[0],cyclic[1])
    assert(ilp.solve())
    print ('ILP_problem:')
#    for var in ilp.variables():
#        print('{} -> {}'.format(var.name, var.value()))
    print('Optimal value = ', ilp.objective.value())

## Acyclic graphs always give integral solution while cyclic don't.

#%%
####Exercise 2.5
    
from model_2_5 import *
from tsukuba_visualize import *
import matplotlib.pyplot as plt
models=all_models()
count=0
for model in models:
    print('MODEL DOWN',2**(5-count))

    if count==2:
#        lp=convert_to_lp(model[0],model[1])
#        lp.solve()
#        print('objective LP =', lp.objective.value())
#        
        
        ilp=convert_to_ilp(model[0],model[1])
        ilp.solve()
        print('objective ILP =', ilp.objective.value())
        
        assignment=ilp_to_labeling(model[0],model[1],ilp)
        img=to_image(assignment,(48,36))
        plt.imshow(img)
        break
    count=count+1
## In terms of variables = O(L^2*N)
## In terms of constraints = O(L*N)
