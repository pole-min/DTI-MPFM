#### Description ：
# This function is separated from other .py functions.This GenerateVocab.py file aims to generate the protein
# embedding dictionary that includes two files,vocab.txt and vocab.csv.Vocab.txt is used to preprocess the
# protein sequence into subsequences.Vocab.csv is used to get the final protein dictionary.
seq_list1 = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z','J']
seq_list2 = []
seq_list3 = []
for i in seq_list1:
    for j in seq_list1:
        seq_list2.append(i+j)
for i in seq_list1:
    for j in seq_list2:
        seq_list3.append(i+j)
import pandas as pd
csv1 = pd.read_csv('data/train/protein_train.csv', encoding='utf-8')
final2 = ['aaa']
final3 = ['bbb']
for i in range(len(csv1)):
    for j in seq_list2:
        if csv1['Target_Sequence'][i].count(j)>=7:
            count = 0
            for q in final2:
                if j==q:
                    count=1
                else:
                    pass
            if count!=1:
                final2.append(j)
        else:
            pass
    print(i)
for i in range(len(csv1)):
    for j in seq_list3:
        if csv1['Target_Sequence'][i].count(j)>=7:
            count = 0
            for q in final3:
                if j==q:
                    count=1
                else:
                    pass
            if count!=1:
                final3.append(j)
        else:
            pass
    print(i)
print(final2)
print(len(final2))
print(final3)
print(len(final3))
file = open('vocab.txt','w',encoding='utf-8')
for i in final2:
    file.write(i[0])
    file.write(' ')
    file.write(i[1])
    file.write('\n')
for i in final3:
    file.write(i[0:2])
    file.write(' ')
    file.write(i[2])
    file.write('\n')
file2 = open('vocab.csv','w',encoding='utf-8')
for i in seq_list1:
    file2.write(i)
    file2.write('\n')
for i in final2:
    file2.write(i)
    file2.write('\n')
for i in final3:
    file2.write(i)
    file2.write('\n')