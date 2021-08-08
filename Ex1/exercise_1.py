# This is the file where should insert your own code.
#
# Author1: Enric Fita Sanmartin <enfisan@alumni.uv.es>
# Author2: Shivali Dubey <dshivali145@gmail.com>

import math
import copy
import numpy as np
#%%
# For exercise 1.2
def evaluate_energy(nodes, edges, assignment):

    energy = 0
    for i in range(len(nodes)):
        energy = nodes[i].costs[assignment[i]] + energy
        
    for j in range(len(edges)):
        energy = edges[j].costs[assignment[edges[j].left],assignment[edges[j].right]] + energy
            
        
    return energy


#%%
# For exercise 1.3
    
def bruteforce(nodes, edges):
    assignment = [0] * len(nodes)
    min_energy=evaluate_energy(nodes, edges,assignment)
    min_assignment=assignment.copy()
    labels=[1]
    for node in reversed(range(len(nodes))):
        labels.append(labels[-1]*len(nodes[node].costs))
        
    for it in range(1,labels[-1]):
        for node in range(0,len(nodes)):
            if node==0:
                assignment[node]=((math.ceil((it+1)/labels[-2]))-1)%len(nodes[node].costs)
                modul=it+1
            else:
                modul=modul%labels[-1-node]
                assignment[node]=((math.ceil((modul)/labels[-2-node]))-1)%len(nodes[node].costs)
        modul=modul%labels[-1-node]
        assignment[-1]=(math.ceil((modul)/labels[-2-len(nodes)+1])-1)%len(nodes[-1].costs)
        
#        print(it+1)
#        print(labels)
#        print(assignment)
        
        energy = evaluate_energy(nodes, edges,assignment)
        if energy<min_energy:
            min_energy=energy
            min_assignment=assignment.copy()
    
    return (min_assignment, min_energy)


# For the rest your are on your own... :)
#%%
# For exercise 1.4
def dynamic_programming(nodes,edges):
    F=[]
    r=[]
    F.append(np.zeros(len(nodes[0].costs)))
    r.append(np.zeros(len(nodes[0].costs)))
    for i in range(1,len(nodes)):
        F.append(np.zeros(len(nodes[i].costs)))
        r.append(np.zeros(len(nodes[i].costs)))
        for s in range(np.size(F[i-1])):
            F_S=[]
            for t in range(np.size(F[i-1])):
                F_S.append(F[i-1][t]+nodes[i-1].costs[t]+edges[i-1].costs[(t,s)])
            F[i][s]=min(F_S)
            r[i][s]=np.argmin(F_S)
    
    intermediates=[F,r]
    return(intermediates)
    
def backtrack(nodes,edges,F,r):
    energy_ls=[]
    
    for s in range(len(nodes[-1].costs)):
        energy_ls.append(F[-1][s]+nodes[-1].costs [s])
    assignment=[0]*len(nodes)
    assignment[-1]=np.argmin(energy_ls)
    for n in reversed(range(1,len(nodes)-1)):
        assignment[n-1]=int(r[n][int(assignment[n])])
    
    return(assignment)
    
#%%
# For exercise 1.5
def compute_min_marginals(nodes,edges):
    energy=[]
    for n in range(len(nodes)):
        
        F=[]
        
        #r=[]
        F.append(np.zeros(len(nodes[0].costs)))
        #r.append(np.zeros(len(nodes[0].costs)))
        for i in range(1,n+1):
            
            F.append(np.zeros(len(nodes[i].costs)))
         #   r.append(np.zeros(len(nodes[i].costs)))
            for s in range(np.size(F[i-1])):
                F_S=[]
                for t in range(np.size(F[i-1])):
                    F_S.append(F[i-1][t]+nodes[i-1].costs[t]+edges[i-1].costs[(t,s)])
                F[i][s]=min(F_S)
          #      r[i][s]=np.argmin(F_S)
        
        
        B=[]    
        #rb=[]
        B.append(np.zeros(len(nodes[-1].costs)))
        #rb.append(np.zeros(len(nodes[0].costs)))
        for i in reversed(range(n,len(nodes)-1)):
            
            B.append(np.zeros(len(nodes[i].costs)))
            #rb.append(np.zeros(len(nodes[i].costs)))
            for s in range(np.size(B[len(nodes)-i-2])):
                B_S=[]
                for t in range(np.size(B[len(nodes)-i-2])):
                    B_S.append(B[len(nodes)-i-2][t]+nodes[i+1].costs[t]+edges[i].costs[(s,t)])
                B[len(nodes)-i-1][s]=min(B_S)
                #rb[i][s]=np.argmin(B_S)
        energy_ls=[]
        for s in range(len(nodes[-1].costs)):
            energy_ls.append(B[-1][s]+F[-1][s]+nodes[n].costs [s])
    
        energy.append(energy_ls)
    #print(energy)
    return(energy)


#%%
    # For exercise 1.6
def dynamic_programming_tree(nodes,edges):
    F=[]
    
    for i in range(len(nodes)):
        F.append(copy.deepcopy(nodes[i].costs))
    
    #Calculates edges associated with each node. Position i of the list correspond to the edges incident to vertex 
    # i. Length of list of position i gives degree of i
    degree_nodes=[[] for _ in range(len(nodes))]
    for e in range(len(edges)):
        degree_nodes[edges[e].left].append(e)
        degree_nodes[edges[e].right].append(e)
        
    r=[np.zeros(len(nodes[i].costs)) for i in range(len(nodes))]
    order=[] #stores order of visited leaves and neighbour
    while  any(degree_nodes):
        edge_leave,leave,neighbour_leave=find_leave(degree_nodes,edges)
        degree_nodes[leave]=None
        degree_nodes[neighbour_leave].remove(edge_leave)
        
        for s in range(np.size(F[neighbour_leave])):
            F_S=[]
            for t in range(np.size(F[leave])):
                label_edge=(s,t)*(edges[edge_leave].left==neighbour_leave)+(t,s)*(edges[edge_leave].left==leave)
                F_S.append(F[leave][t]+edges[edge_leave].costs[label_edge])
            F[neighbour_leave][s]=F[neighbour_leave][s]+min(F_S)
            r[leave][s]=np.argmin(F_S)
        order.append((leave,neighbour_leave))
    


    intermediates=[F,r,order]
    return(intermediates)

def find_leave(degree_nodes,edges):
    leave=-1
    for i in degree_nodes:
        leave=leave+1
        if i!=None:
            if len(i)==1:
                neighbour_leave=(edges[i[0]].left!=leave)*edges[i[0]].left+(edges[i[0]].right!=leave)*edges[i[0]].right
                edge_leave=i[0]
            
                return edge_leave,leave,neighbour_leave
    return None,None,None
        
def backtrack_tree(nodes,edges,F,r,order):
    assignment=[0]*len(nodes)
    energy_ls=[]
    leave=order[-1][-1]
    for t in range(np.size(F[leave])):
        energy_ls.append(F[leave][t])
    assignment[leave]=np.argmin(energy_ls)
    
    for i in reversed(range(len(order))):
        assignment[order[i][0]]=int(r[order[i][0]][int(assignment[order[i][1]])])

    
    return(assignment)

#%%

# For exercise 1.7




def seam__carving_energy(img):
    g=np.zeros(img.shape[0:2])

    for i in range(img.shape[-1]):
        g_=np.gradient(img[:,:,i])
        g=g+g_[0]**2+g_[1]**2
    g=np.sqrt(g)
    
    F=np.zeros(g.shape)
    r=np.zeros(g.shape)
    F[0,:]=g[0,:]
    
    for i in range(1,img.shape[0]):
        
        for s in range(img.shape[1]):
            F_S=[]
            for t in range(3):
                if s==0 and t==0:
                    F_S.append(np.inf)
                elif s==(img.shape[1]-1) and t==2:
                    F_S.append(np.inf)
                else:
                    F_S.append(g[i,s]+F[i-1,s-1+t])
            F[i][s]=min(F_S)
            r[i][s]=np.argmin(F_S)
    
    intermediates=[F,r]
    return(intermediates)
    
def backtrack_seam_carving(F,r):
    path=[]
    path.append(((F.shape[0]-1)*F.shape[1]+np.argmin(F[-1,:])))
    for i in reversed(range(F.shape[0]-1)):
        path.append(int(i*F.shape[1]+path[-1]%F.shape[1]-1+r[i+1,int(path[-1]%F.shape[1])]))
    
    return(path)
    
def remove_elements(img,path):
    new_img_chan0=np.delete(img[:,:,0],path)
    new_img_chan0=np.reshape(new_img_chan0,(img.shape[0],img.shape[1]-1))
    new_img_chan0= np.expand_dims(new_img_chan0, axis=2)

    
    new_img_chan1=np.delete(img[:,:,1],path)
    new_img_chan1=np.reshape(new_img_chan1,(img.shape[0],img.shape[1]-1))
    new_img_chan1= np.expand_dims(new_img_chan1, axis=2)

    
    new_img_chan2=np.delete(img[:,:,2],path)
    new_img_chan2=np.reshape(new_img_chan2,(img.shape[0],img.shape[1]-1))
    new_img_chan2= np.expand_dims(new_img_chan2, axis=2)
    
    new_img=np.concatenate((new_img_chan0,new_img_chan1,new_img_chan2),axis=2)
    return new_img


def seam_carving(img,it=48):
    
    for i in range(it):
        intermediates=seam__carving_energy(img)
        path=backtrack_seam_carving(*intermediates)
        img=remove_elements(img,path)
    return img
