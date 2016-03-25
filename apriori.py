# -*- coding: utf-8 -*-


import pandas as pd

#创建1-候选频繁项集（所有的事务）
def createC1(data):
    C1 = []
    for tran in data:
        for item in tran:
            if [item] not in C1:
                C1.append([item])           
    C1.sort()
    return map(frozenset, C1) 

#扫描整个数据集，根据支持度从Ck生成Lk和Lk中项的支持度
def scan(data, Ck, min_support):
    dct = {}
    #收集Ck里每个项集的支持度
    for tran in data:
        for item in Ck:
            if item.issubset(tran):
                if not dct.has_key(item): 
                    dct[item]=1
                else: 
                    dct[item] += 1
    numItems = float(len(data))
    retList = []
    supportData = {}
    #判断是否满足最小支持度
    for key in dct:
        support = dct[key]/numItems
        if support >= min_support:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

#连接步，从Lk-1生成Ck
def createCk(Lk, k): 
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2: 
                retList.append(Lk[i] | Lk[j]) 
    return retList

#apriori算法
def apriori(dataSet, min_support):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
#    print len(D)
    L1, supportData = scan(D, C1, min_support)
    #L存储所有1，2，3...频繁项集
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = createCk(L[k-2], k)
        Lk, supK = scan(D, Ck, min_support)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

#根据支持度的数据生成关联规则
def getRules(L, supportData, minConf):
    ruleList = []
    for i in range(1, len(L)):
        for itemSet in L[i]:
#            print L[i]
            H1 = [frozenset([item]) for item in itemSet]
#            print H1
            if (i > 1):
                rulesFromConseq(itemSet, H1, supportData, ruleList, minConf)
            else:
                calcConf(itemSet, H1, supportData, ruleList, minConf)
    return ruleList         

#计算置信度
def calcConf(itemSet, H, supportData, ruleList, minConf):
    prunedH = [] 
    for conseq in H:
        conf = supportData[itemSet]/supportData[itemSet-conseq] 
#        print itemSet
#        print conseq
#        if itemSet==frozenset(['R1','R3','R5']):
#                if conseq==frozenset(['R1']):
#                    print 'here'
#        print supportData[itemSet]
#        print supportData[itemSet-conseq] 
#        print conf
        if conf >= minConf: 
            print itemSet-conseq,'-->',conseq,'conf:',conf
            ruleList.append((itemSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#对一个频繁项集，找出其中的所有规则
def rulesFromConseq(itemSet, H, supportData, ruleList, minConf):
    m = len(H[0])
    calcConf(itemSet, H, supportData, ruleList, minConf)
    if (len(itemSet) > (m+1)):
        Hmp1 = createCk(H, m+1)
        if (len(Hmp1) > 1):  
            rulesFromConseq(itemSet, Hmp1, supportData, ruleList, minConf)           
        
#利用pandas里的read_csv导入数据，再取出values，利用tolist()转换成列表
dataframe = pd.read_csv("C:/Users/zhangm215/Desktop/data/apriori.csv")
change_part = dataframe['part'].values
part_list = change_part.tolist()
#print len(part_list)

#把list里的每个字符串转换成字符列表
new_list = []
for trac in part_list:
    tmp = []
    for i in range(0,len(trac),2):
        tmp.append(trac[i:i+2])
    new_list.append(tmp)
part_list = new_list

L, supportData = apriori(part_list, 0.04)
#print float(supportData[frozenset(['R1','R3','R5'])])/supportData[frozenset(['R3','R5'])]
#print L
#for each in L:
#    for a in each:
#        a = map(list, a)
rule = getRules(L, supportData, minConf = 0.8)   
#print rule
print len(rule)

#a = ['abe','bd','bc','abd','ac','bc','ac','abce','abc']
#L, supportData = apriori(a, 0)
##print L
#rule = getRules(L, supportData, minConf = 0) 
    
    
    
    
    
    
    
    
