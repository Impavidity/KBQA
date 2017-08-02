import re
from collections import defaultdict
from fuzzywuzzy import process
import logging

logger = logging.getLogger()
logger.disabled = True

idtostr = defaultdict(list)
with open("data/FB5M.name.txt") as f:
    for line in f.readlines():
        line = line.split('\t')
        key, value = line[0], line[2]
        key = key.split('.')[1][:-1]
        value = value[1:-1]
        idtostr[key].append(value)


exact_match = False
count_exact_match = 0
total = 0
for item in ["train.txt","valid.txt","test.txt"]:
    fout = open("data/annotated_fb_entity_"+item, "w")
    flog = open("data/logging_"+item, "w")
    with open("data/annotated_fb_data_"+item) as f:
        print(item," : ")
        for line_num, line in enumerate(f.readlines()):
            if line_num % 1000 == 0:
                print("process %d" % (line_num + 1))
            total += 1
            exact_match = False
            line = line.split('\t')
            key, sent_ori = line[0], line[3]
            sent = re.findall(r"\w+|[^\w\s]", sent_ori, re.UNICODE)
            label = ["O"] * len(sent)
            key = key.split('/')[2]
            try:
                candi = idtostr.get(key, [])
                if candi == []:
                    fout.write(" ".join(sent) + '\t' + " ".join(label) + '\n')
                    continue
                v, score = process.extractOne(" ".join(sent), candi)
            except:
                print(line_num, line, sent, idtostr.get(key, []))
                exit()
            value = re.findall(r"\w+|[^\w\s]", v, re.UNICODE)
            # exact match here
            try:
                result = re.search(" "+" ".join(value), " "+" ".join(sent))
            except:
                result = None
            if result != None:
                exact_match = True
            if exact_match:
                count_exact_match += 1
                l = len(value)
                for i in range(len(sent)):
                    if sent[i:i+l] == value:
                        for j in range(i, i+l):
                            label[j] = 'I'
                        break
            else:
                for w in value:
                    result = process.extractOne(w, sent, score_cutoff=85)
                    if result == None:
                        continue
                    else:
                        word = result[0]
                        label[sent.index(word)] = 'I'

                if len(sent)>1 and (sent[1] == 'was' or sent[1] == 'is') and label[1] == 'I':
                    label[1] = 'O'

                start = end = -1
                for l in range(len(label)):
                    if label[l] == 'I':
                        start = l
                        break
                if start != -1:
                    for l in range(len(label)):
                        if label[len(label)-l-1] == 'I':
                            end = len(label)-l
                            break
                if start != -1 and end != -1:
                    for l in range(start, end):
                        label[l] = 'I'

                flog.write(str(line_num) + "\t" + " ".join(line) + "\t" + " ".join(value) + "\t" + " ".join(sent) + '\n\t' + " ".join(label)+"\n")

            fout.write(" ".join(sent) + '\t' + " ".join(label) + '\n')
    print("total = ",total,"exact_match = ",count_exact_match)

