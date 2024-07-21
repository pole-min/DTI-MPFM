import pandas as pd
csv1 = pd.read_csv('data/train/morgan_train.csv')
csv2 = pd.read_csv('data/valid/morgan_valid.csv')
csv3 = pd.read_csv('data/test/morgan_test.csv')
final = []
for k in range(len(csv1)):
    l = [i for i in csv1['SMILES'][k]]
    final += l
for k in range(len(csv2)):
    l = [i for i in csv2['SMILES'][k]]
    final += l
for k in range(len(csv3)):
    l = [i for i in csv3['SMILES'][k]]
    final += l
kk = list(set(final))
kk_dict = {k:v+1 for v,k in enumerate(kk)}
def encod_SMILES(seq,kk_dict):
    if pd.isnull(seq):
        return [0]
    else:
        return [kk_dict[a] for a in seq]
a,b,c = 0,0,0
# drug_df = pd.read_csv('data/train/train.csv', index_col='ProteinID')
# drug_df['encoded_sequence'] = drug_df.SMILES.map(lambda a: encod_SMILES(a, kk_dict))
# for i in drug_df['encoded_sequence']:
#     if len(i)> 90 and len(i)<= 100:
#         a+=1
#     if len(i)>80 and len(i)<=90:
#         b+=1
#     if len(i)<=80:
#         c+=1

# drug_df = pd.read_csv('data/valid/valid.csv', index_col='ProteinID')
# drug_df['encoded_sequence'] = drug_df.SMILES.map(lambda a: encod_SMILES(a, kk_dict))
# for i in drug_df['encoded_sequence']:
#     if len(i)> 90 and len(i)<= 100:
#         a+=1
#     if len(i)>80 and len(i)<=90:
#         b+=1
#     if len(i)<=80:
#         c+=1
drug_df = pd.read_csv('data/test/test.csv', index_col='ProteinID')
drug_df['encoded_sequence'] = drug_df.SMILES.map(lambda a: encod_SMILES(a, kk_dict))
for i in drug_df['encoded_sequence']:
    if len(i)> 90 and len(i)<= 100:
        a+=1
    if len(i)>80 and len(i)<=90:
        b+=1
    if len(i)<=80:
        c+=1
number = 1972
print(c/number)
print((c+b)/number)
print((c+b+a)/number)