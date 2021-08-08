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
            assignment[int(v.name[6])]=int(v.name[-1])       #v.name[6]- the 6th character for unary_1_label_2 and similarly for v.name[-1].
            assigned=assigned+1
        if assigned==len(nodes):
            break
    return assignment

##2
import pulp
def convert_to_lp(nodes, edges):
    prob = pulp.LpProblem("Graph_model_prob",pulp.LpMinimize)
    for i in range(len(nodes)):
        for j in range(len(nodes[i].costs)):
            #create variable unary costs
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
        
def lp_to_labeling(nodes, edges, prob):
    prob.roundSolution()
    assignment=[0]*len(nodes)
    assigned=0
    for v in reversed(prob.variables()):
        if v.value()==1 and v.name[0]!='p':
            assignment[int(v.name[6])]=int(v.name[-1])       #v.name[6]- the 6th character for unary_1_label_2 and similarly for v.name[-1].
            assigned=assigned+1
        if assigned==len(nodes):
            break
    return assignment
    
