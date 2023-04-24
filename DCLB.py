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

res=[]
for rnd in range(1,10):
    import queue
    import copy
    tau=[0]*num_users
    gamma0=0.01  #0.2
    gamma1=0.02
    delta=0.4
    numSample=60
    stationary=False
    alpha=0.5
    alpha2=0.000001
    def CD(w,b,Y):
        former=0
        latter=0
        tempXlist= Y[0]
        tempYlist=Y[1]
        tmpM1,tmpM2=np.diag(np.ones(d)),np.diag(np.ones(d))
        tmpb1,tmpb2=np.zeros(d).T,np.zeros(d).T
        for x,y in zip(tempXlist[:w//2],tempYlist[:w//2]):
            tmpM1+=np.dot(x.reshape((d,1)),x.reshape((1,d)))
            tmpb1+=y*x
        for x,y in zip(tempXlist[w//2:],tempYlist[w//2:]):
            tmpM2+=np.dot(x.reshape((d,1)),x.reshape((1,d)))
            tmpb2+=y*x
        tmp=0
        for k in range(K):
            if abs(np.dot(np.linalg.inv(tmpM1).dot(np.array(tmpb1))-np.linalg.inv(tmpM2).dot(np.array(tmpb2)),fixedItemSet[k]))>CB(alpha,fixedItemSet[k],tmpM1)+CB(alpha,fixedItemSet[k],tmpM2):
                tmp+=1
        if tmp>=2:
            return True
        else:
            return False    
    #INIT
    M=np.zeros([num_users,d,d])#(1-lambdatutil)*np.array([np.diag(np.ones(d)) for _ in range(num_users)])
    Mtutil=np.zeros([num_users,d,d])#lambdatutil*np.array([np.diag(np.ones(d)) for _ in range(num_users)])
    for i in range(num_users):
        for j in range(d):
            M[i][j][j]=1-lamb
            Mtutil[i][j][j]=lambdatutil
    b=np.zeros([num_users,d])
    btutil=np.zeros([num_users,d])
    w=[]
    for i in range(num_users):
        w.append(np.dot(np.linalg.inv(M[i]),b[i]))
    HD=10000
    changedUser=[]
    expectedRegret=[]
    accuRegret=np.zeros(T)
    lastChangedTwoGroups=[]

    Mmark=M.copy()
    bmark=b.copy()
    Nmatrix=np.zeros([num_users,K+1])
    #playing number of user i for item k
    ZmatrixX={}
    for i in range(num_users):
        ZmatrixX[i]=[]
    cntForZmatrixX=np.zeros(num_users).astype(int)
    ZmatrixY={}
    for i in range(num_users):
        ZmatrixY[i]=[]
    cntForZmatrixY=np.zeros(num_users).astype(int)
    for t in range(0,T,1):
        if t%50==0 and t>0:
            print("DCLB:",t,accuRegret[t-1])
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
                    for jj in range(changedCluster*num_users//m,changedCluster*num_users//m+1,1):
                        theta[jj]=-1*theta[jj]
        else:
            pass
        for i in range(num_users):
            w[i]=np.dot(np.linalg.inv(M[i]),b[i])
        #receive user
        currentU = np.random.choice(list(range(num_users)), 1, p=np.ones(num_users)/num_users)[0]
        if t>0 and bt(t+1)-bt(t)>0:
            nq=bt(t+1)-bt(t)
            while nq>0:
                currentkey=keyRec(M[currentU],Mtutil[currentU])
                r_key=key_fea[currentkey].dot(theta[currentU])
                tmp=np.sum(np.array([float(currentkey in relationa[a])*fixedItemSet[a] for a in range(K)]),axis=0)/sum([float(currentkey in relationa[a]) for a in range(K)])
                Mtutil[currentU]+=np.outer(tmp,tmp)
                btutil[currentU]+=(r_key*tmp)
                nq-=1
        if (t-max(tau)<num_users*K*d) or((currentU in changedUser) and [CB(alpha,fixedItemSet[k],M[i])>=gamma0/4 for k in range(K)]==[1]*K):
            clusters=[[currentU]]*K
            avgW=[w[currentU]]*K
            avgCB=[CB(alpha,fixedItemSet[k],M[currentU]) for k in range(K)]
        else:
            clusters=[]
            avgW=[]
            avgCB=[]

            for k in range(K):
                tempCluster=[currentU]
                tempW=w[currentU]
                tempCB=CB(alpha,fixedItemSet[k],M[currentU])
                for i in range(num_users):
                    #print(t,currentU,i,abs(CB(alpha,fixedItemSet[k],M[i])-CB(alpha,fixedItemSet[k],M[currentU])))
                    #if t%100==0:
                    #   pass
                    #print(t,currentU,i,abs(w[i].dot(fixedItemSet[k])-w[currentU].dot(fixedItemSet[k])),abs(CB(alpha,fixedItemSet[k],M[i])+CB(alpha,fixedItemSet[k],M[currentU])))
                    if i!=currentU and Nmatrix[i][k]>10 and 0<abs(w[i].dot(fixedItemSet[k])-w[currentU].dot(fixedItemSet[k]))<abs(CB(alpha2,fixedItemSet[k],M[i])+CB(alpha2,fixedItemSet[k],M[currentU])):
                        #print(abs(CB(alpha,fixedItemSet[k],M[i])+CB(alpha,fixedItemSet[k],M[currentU])))
                        #print(t,currentU,i)
                        if (i not in changedUser) or (CB(alpha,fixedItemSet[k],M[i])<gamma0/4):
                            tempCluster.append(i)
                            tempW+=w[i]
                            tempCB+=CB(alpha,fixedItemSet[k],M[i])
                clusters.append(tempCluster)
                avgW.append(tempW/len(tempCluster))
                avgCB.append(tempCB/len(tempCluster))


      
        index1=[]
        for k in range(K):
            index1.append(avgW[k].T.dot(fixedItemSet[k])+avgCB[k])
        thetatutil=np.linalg.inv(Mtutil[currentU]).dot(btutil[currentU])
        theta2=np.linalg.inv(M[currentU]).dot(b[currentU]+(1-lamb)*thetatutil)
        index2=[]
        for k in range(K):
            index2.append(theta2.T.dot(fixedItemSet[k])+lamb*0.5*norm(fixedItemSet[k],M[currentU])+(1-lamb)*0.1*norm(fixedItemSet[k].T.dot(np.linalg.inv(M[currentU])),Mtutil[currentU]))            
        currentItem= np.argmax(np.array(index1)+np.array(index2))
        #print(t,currentU,,len(clusters[0]),len(clusters[1]),len(clusters[2]),len(clusters[3]),len(clusters[4]),w[currentU].dot(fixedItemSet[currentItem]),theta[currentU].dot(fixedItemSet[currentItem]),CB(alpha,fixedItemSet[currentItem],M[currentU]))

        currentX=fixedItemSet[currentItem]
        currentY, expectedReward, maxReward=feedback(currentU, currentX)
        if t==1:
            accuRegret[t]=maxReward-expectedReward
        else:
            accuRegret[t]=accuRegret[t-1]+maxReward-expectedReward
        #UPDATE currentU's info
        Nmatrix[currentU][K]+=1
        Nmatrix[currentU]+=1
        M[currentU]=M[currentU]+np.dot(currentX.reshape((d,1)),currentX.reshape((1,d)))
        b[currentU]=b[currentU]+currentY*currentX
        if t==max(tau)+HD:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Mmark=M
            bmark=b
        if cntForZmatrixX[currentU]<numSample:
            ZmatrixX[currentU].append(currentX)
        else:
            ZmatrixX[currentU][cntForZmatrixX[currentU]%numSample]=currentX
        cntForZmatrixX[currentU]+=1
        if cntForZmatrixY[currentU]<numSample:
            ZmatrixY[currentU].append(currentY)
        else:
            ZmatrixY[currentU][cntForZmatrixY[currentU]%numSample]=currentY
        cntForZmatrixY[currentU]+=1
        
        mark2=0
        #for i in range(K):
        #   if t>0 and lastClusterLength[i]!=len(clusters[i]):
        #      mark2=1
        #     break
        #if (t>tau[currentU]+HD and Nmatrix[currentU]>=numSample and CD(numSample,delta,Zmatrix[currentU])==True) or (t>tau[currentU]+HD and mark2):
        #print(list(Zmatrix[currentU].queue))
        if (Nmatrix[currentU][currentItem]>=numSample and CD(numSample,delta,[ZmatrixX[currentU],ZmatrixY[currentU]])==True):
            print("--------------------------------------------------------------------------")
            tau[currentU]=t
            ZmatrixX[currentU]=[]
            ZmatrixX[currentU].append(currentX)
            cntForZmatrixX[currentU]=1
            ZmatrixY[currentU]=[]
            ZmatrixY[currentU].append(currentY)
            cntForZmatrixY[currentU]=1            
            changedUser.append(currentU)
            if len(changedUser)>5:
                changedUser.pop(0)
            for i in range(K+1):
                Nmatrix[currentU][i]=1
            for i in range(K):
                for j in clusters[i]:
                    M[j]=Mmark[j]
                    b[j]=bmark[j]
            M[currentU]=(1-lamb)*np.diag(np.ones(d))+lamb*np.dot(currentX.reshape((d,1)),currentX.reshape((1,d)))
            b[currentU]=lamb*currentY*currentX
        elif CB(alpha,fixedItemSet[currentItem],M[currentU])>=gamma0/4:
            pass
        else:
            for i in clusters[currentItem]:
                if CB(alpha,fixedItemSet[currentItem],M[i])<gamma0/4:
                    Nmatrix[i][K]+=1
                    M[i]+=lamb*np.dot(currentX.reshape((d,1)),currentX.reshape((1,d)))
                    b[i]+=lamb*currentY*currentX
    res.append([numSample,accuRegret[-1]])
    string="dclbcon_syn_conversation"+str(rnd)+".txt"
    with open(string, 'w') as f:
        for item in accuRegret:
            f.write("%s\n" % item)