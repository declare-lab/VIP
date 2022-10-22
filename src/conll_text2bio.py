import torch

def getonebatchresult(sen,target,preds):
    #typedic = {"org": "ORG", "location": "LOC", "person": "PER", "mix": "MISC"}
    typedic = {"org": "ORG", "money": "MONEY", "country": "GPE", "time": "TIME", "law": "LAW", "fact": "FAC",
               "thing": "EVENT", "measure": "QUANTITY",
               "order": "ORDINAL", "art": "WORK_OF_ART", "location": "LOC", "language": "LANGUAGE", "person": "PER",
               "product": "PRODUCT", "num": "CARDINAL", "national": "NORP", "date": "DATE", "per": "PERCENT", "mix": "MISC"}
    sennum = len(sen)
    restar = []
    respred = []
    for i in range(sennum):
        thissen, thistar, thispred = sen[i], target[i], preds[i]

        thissenlow = thissen.lower()

        sensplit = thissen.split(' ')
        sensplitlow = thissenlow.split(' ')

        tarres = ['O' for j in range(len(sensplit))]
        predres = ['O' for j in range(len(sensplit))]

        if thistar == 'end' and thispred == 'end':
            restar.append(tarres)
            respred.append(predres)
            continue

        if len(thistar) > 0 and thistar[-1] == ';':
            thistar = thistar[:-1]

        tarsplit1 = thistar.split(';')

        if thistar != 'end':
            for j in range(len(tarsplit1)):
                tarsplit2 = tarsplit1[j].split('!')
                if len(tarsplit2) != 2:
                    continue
                entity = tarsplit2[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit2[1].strip(' ')
                if type not in typedic:
                    continue
                if thissenlow.find(entitylow) == -1:
                    continue
                trueindex = -100
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    continue
                for k in range(trueindex, trueindex + len(entitysplit)):
                    if k == trueindex:
                        tarres[k] = 'B-' + typedic[type]
                    else:
                        tarres[k] = 'I-' + typedic[type]

        if len(thispred) > 0 and thispred[-1] == ';':
            thispred = thispred[:-1]

        tarsplit3 = thispred.split(';')

        if thispred != "end":
            for j in range(len(tarsplit3)):
                tarsplit4 = tarsplit3[j].split('!')
                if len(tarsplit4) != 2:
                    continue
                entity = tarsplit4[0].strip(' ')
                entitylow = entity.lower()
                type = tarsplit4[1].strip(' ')
                if type not in typedic:
                    continue
                if thissenlow.find(entitylow) == -1:
                    continue
                trueindex = -100
                entitysplit = entitylow.split(' ')
                for k in range(len(sensplit)):
                    if sensplitlow[k] == entitysplit[0] or entitysplit[0] in sensplitlow[k]:
                        iftrue = True
                        for l in range(1, len(entitysplit)):
                            if sensplitlow[k + l] != entitysplit[l] and (entitysplit[0] not in sensplitlow[k]):
                                iftrue = False
                                break
                        if iftrue:
                            trueindex = k
                            break
                if trueindex == -100:
                    continue
                else:
                    for k in range(trueindex, trueindex + len(entitysplit)):
                        if k == trueindex:
                            predres[k] = 'B-' + typedic[type]
                        else:
                            predres[k] = 'I-' + typedic[type]
        restar.append(tarres)
        respred.append(predres)
    return restar, respred