def convert_to_lp(nodes, edges):
    prob = pulp.LpProblem("Graph_model_prob",pulp.LpMinimize)
    i in range(len(nodes))
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
