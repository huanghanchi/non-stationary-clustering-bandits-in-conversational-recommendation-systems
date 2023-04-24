import matplotlib.pyplot as plt
import numpy as np
import queue
import random
def get_best_reward(items, theta):
	return np.max(np.dot(items, theta))
def feedback(i, x):
		r = np.dot(theta[i], x)
		y = r+np.random.normal(0, 0.1)
		br = get_best_reward(fixedItemSet,theta[i])
		return y, r, br
def CB(alpha,x,M):
    return alpha*np.sqrt(np.dot(np.dot(x.T,np.linalg.inv(M)),x))

def norm(x,M_):
    return np.sqrt(np.dot(np.dot(x.T,np.linalg.inv(M_)),x))    
def bt(t):
    return 5*int(np.log(t))
def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d))
    x = np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))
    return x
def _get_theta( num_users, m):
    delta=2/m
    x=[]
    for i in range(m):
        x.append(np.random.normal(-1+delta*i, 1, d))
    #x = np.array(x)
    thetam = np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))
    k = int(num_users / m)
    theta = {i:thetam[0] for i in range(k)}
    for j in range(1, m):
        theta.update({i:thetam[j] for i in range(k * j, k * (j + 1))})
    if len(theta.keys())<num_users:
        for i in range(len(theta.keys()),num_users,1):
            theta[i]=theta[len(theta.keys())-1]
    return theta
def checkValidGenerated(user,item,m,delta):
    for i in item:
        templist=[]
        for j in range(m):
            templist.append(user[j*(num_users//m)].dot(i))
        templist.sort()
        for i in range(len(templist)-1):
            if templist[i+1]-templist[i]<delta:
                return False
    return True
def keyRec(M_,Mtutil_):
    index=[]
    for k in range(num_keys):
        xtutil=(1/relationk[k])*np.sum(np.array([float(k in relationa[a])*fixedItemSet[a] for a in range(K)]),axis=0)
        tmp=np.linalg.norm(fixedItemSet.dot(np.linalg.inv(M_)).dot(np.linalg.inv(Mtutil_)).dot(xtutil))**2/(1+xtutil.T.dot(np.linalg.inv(M_)).dot(xtutil))
        index.append(tmp)
    return np.argmax(index)

num_users=300
num_keys=10
m=10
d=10
K=40
T=100000
lamb=0.1
lambdatutil=np.clip(2*(1-lamb)/lamb/(1-np.sqrt(lamb))**2,0,1)
stationary=False


fixedItemSet=np.load('conversationalFixedItemSet.npy')
theta=np.load('conversationalTheta.npy')


relationa=np.load('relationa.npy',allow_pickle=True)
relationk=np.load('relationk.npy')


key_fea=np.load('key_fea.npy')

import matplotlib.pyplot as plt
import numpy as np
import queue
import random
def get_best_reward(items, theta):
	return np.max(np.dot(items, theta))
def feedback(i, x):
		r = np.dot(theta[i], x)
		y = r+np.random.normal(0, 0.1)
		br = get_best_reward(fixedItemSet,theta[i])
		return y, r, br
def CB(alpha,x,M):
    return alpha*np.sqrt(np.dot(np.dot(x.T,np.linalg.inv(M)),x))

    
def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d))
    x = np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))
    return x
def _get_theta( num_users, m):
    delta=2/m
    x=[]
    for i in range(m):
        x.append(np.random.normal(-1+delta*i, 1, d))
    #x = np.array(x)
    thetam = np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))
    k = int(num_users / m)
    theta = {i:thetam[0] for i in range(k)}
    for j in range(1, m):
        theta.update({i:thetam[j] for i in range(k * j, k * (j + 1))})
    if len(theta.keys())<num_users:
        for i in range(len(theta.keys()),num_users,1):
            theta[i]=theta[len(theta.keys())-1]
    return theta
def checkValidGenerated(user,item,m,delta):
    for i in item:
        templist=[]
        for j in range(m):
            templist.append(user[j*(num_users//m)].dot(i))
        templist.sort()
        for i in range(len(templist)-1):
            if templist[i+1]-templist[i]<delta:
                return False
    return True
def norm(x,M_):
    return np.sqrt(np.dot(np.dot(x.T,np.linalg.inv(M_)),x))    
def bt(t):
    return 5*int(np.log(t))
def keyRec(M_,Mtutil_):
    index=[]
    for k in range(num_keys):
        xtutil=(1/relationk[k])*np.sum(np.array([float(k in relationa[a])*fixedItemSet[a] for a in range(K)]),axis=0)
        tmp=np.linalg.norm(fixedItemSet.dot(np.linalg.inv(M_)).dot(np.linalg.inv(Mtutil_)).dot(xtutil))**2/(1+xtutil.T.dot(np.linalg.inv(M_)).dot(xtutil))
        index.append(tmp)
    return np.argmax(index)

num_users=300
m=10
d=10
K=40
T=100000
stationary=False
#fixedItemSet = generate_items(num_items=K, d=d)
# thetam = np.concatenate((np.dot(np.concatenate((np.eye(m), np.zeros((m,d-m-1))), axis=1), ortho_group.rvs(d-1))/np.sqrt(2), np.ones((m,1))/np.sqrt(2)), axis=1)
#theta = _get_theta(num_users, m)
# print([np.linalg.norm(theta[0]-theta[i]) for i in range(num_users)])
#print("ok")
#while not checkValidGenerated(theta,fixedItemSet,m,0.051):
#    fixedItemSet = generate_items(num_items=K, d=d)
##    theta = _get_theta(num_users, m)
#print("ok")
#theta1=pd.DataFrame(theta)
#theta1.to_csv("theta.csv")
#fixedItemSet1=pd.DataFrame(fixedItemSet)
#fixedItemSet1.to_csv("item.csv")

fixedItemSet=np.load('conversationalFixedItemSet.npy')
theta=np.load('conversationalTheta.npy')


relationa=np.load('relationa.npy',allow_pickle=True)
relationk=np.load('relationk.npy')


key_fea=np.load('key_fea.npy')
for rnd in range(2,10):
    alpha=0.5
    alpha2=0.0000001

    lamb=0.1
    lambdatutil=np.clip(2*(1-lamb)/lamb/(1-np.sqrt(lamb))**2,0,1)
    V=[]
    b=[]
    w=[]
    lastChangedTwoGroups=[]
    accuRegret=np.zeros(T)
    class bandit:
        def __init__(self):
            self.V=np.diag(np.ones(d))
            self.b=np.zeros(d).T


    V=np.zeros([num_users,d,d])#(1-lambdatutil)*np.array([np.diag(np.ones(d)) for _ in range(num_users)])
    Vtutil=np.zeros([num_users,d,d])#lambdatutil*np.array([np.diag(np.ones(d)) for _ in range(num_users)])
    for i in range(num_users):
        for j in range(d):
            V[i][j][j]=1-lamb
            Vtutil[i][j][j]=lambdatutil
    b=np.zeros([num_users,d])
    btutil=np.zeros([num_users,d])
    w=[]
    for i in range(num_users):
        w.append(np.dot(np.linalg.inv(V[i]),b[i]))
    G=[bandit()]
    change=[0]*num_users
    for t in range(T):
        if t%50==0 and t>0:
            print("CGLB:",t,accuRegret[t-1])
        alpha=0.5#min(0.8,max(0.1,0.023*np.log((t+1)/m)))#0.2*max(0.2,0.05*np.log((t+1)/m))
        #set changePoint
        if stationary==False:
            if t in [3000,3200,10500,11000,19000,22000,26000,50000,80000,120000,120400]:
                for _ in range(3):
                    changedCluster=np.random.choice(list(range(m)), 1, p=np.ones(m)/m)[0]
                    if len(lastChangedTwoGroups)==0:
                        lastChangedTwoGroups.append(changedCluster)
                    else:
                        while changedCluster in lastChangedTwoGroups:
                            changedCluster=np.random.choice(list(range(m)), 1, p=np.ones(m)/m)[0]
                        if len(lastChangedTwoGroups)==1:
                            lastChangedTwoGroups.append(changedCluster)
                    for jj in range(changedCluster*num_users//m,int((changedCluster+0.4)*num_users//m),1):
                        theta[jj]=-1*theta[jj]
        else:
            pass
        currentU = np.random.choice(list(range(num_users)), 1, p=np.ones(num_users)/num_users)[0]
        if t>0 and bt(t+1)-bt(t)>0:
            nq=bt(t+1)-bt(t)
            while nq>0:
                currentkey=keyRec(V[currentU],Vtutil[currentU])
                r_key=key_fea[currentkey].dot(theta[currentU])
                tmp=np.sum(np.array([float(currentkey in relationa[a])*fixedItemSet[a] for a in range(K)]),axis=0)/sum([float(currentkey in relationa[a]) for a in range(K)])
                Vtutil[currentU]+=np.outer(tmp,tmp)
                btutil[currentU]+=(r_key*tmp)
                nq-=1
                
        r=[np.linalg.inv(V[currentU]).dot(b[currentU]).T.dot(fixedItemSet[k]) for k in range(K)]
        ConB=[CB(alpha,fixedItemSet[k],V[currentU]) for k in range(K)]
        n=[1]*K
        for k in range(K):
            for i,g in enumerate(G):
                if abs(np.linalg.inv(g.V).dot(g.b).T.dot(fixedItemSet[k]-r[k]))<CB(alpha,fixedItemSet[k],g.V)+CB(alpha,fixedItemSet[k],V[currentU]):
                    r[k]=(r[k]*n[k]+np.linalg.inv(g.V).dot(g.b).T.dot(fixedItemSet[k]))/(n[k]+1)
                    ConB[k]=(ConB[k]*n[k]+CB(alpha,fixedItemSet[k],g.V))/(n[k]+1)
                    n[k]+=1
            if n[k]==1:
                G.append(bandit())
                if change[currentU]==1:
                    V[currentU]=np.diag(np.ones(d))
                    b[currentU]=np.zeros(d).T
                change[currentU]=1

            if len(G)>210:
                random.shuffle(G)
                G=G[:210]
        index1=[]
        for k in range(K):
            index1.append(r[k]+ConB[k])
        thetatutil=np.linalg.inv(Vtutil[currentU]).dot(btutil[currentU])
        theta2=np.linalg.inv(V[currentU]).dot(b[currentU]+(1-lamb)*thetatutil)
        index2=[]
        for k in range(K):
            index2.append(theta2.T.dot(fixedItemSet[k])+lamb*0.5*norm(fixedItemSet[k],V[currentU])+(1-lamb)*0.1*norm(fixedItemSet[k].T.dot(np.linalg.inv(V[currentU])),Vtutil[currentU]))            
        currentItem= np.argmax(np.array(index1)+np.array(index2))

        currentX=fixedItemSet[currentItem]
        currentY, expectedReward, maxReward=feedback(currentU, currentX)
        if t==1:
            accuRegret[t]=maxReward-expectedReward
        else:
            accuRegret[t]=accuRegret[t-1]+maxReward-expectedReward
        #UPDATE currentU's info
        V[currentU]=V[currentU]+np.dot(currentX.reshape((d,1)),currentX.reshape((1,d)))
        b[currentU]=b[currentU]+currentY*currentX
        for g in G:
            if abs(np.linalg.inv(g.V).dot(g.b).T.dot(fixedItemSet[currentItem]-r[currentItem]))<CB(alpha2,fixedItemSet[currentItem],g.V)+CB(alpha2,fixedItemSet[currentItem],V[currentU]):
                g.V+= np.dot(currentX.reshape((d,1)),currentX.reshape((1,d)))
                g.b+= currentY*currentX
    string="cglbcon_syn_conversation"+str(rnd)+".txt"
    with open(string, 'w') as f:
        for item in accuRegret:
            f.write("%s\n" % item)