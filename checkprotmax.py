from subword_nmt.apply_bpe import BPE
import pandas as pd
a = pd.read_csv('vocab.csv')
values = a['Values'].values
prot_dict = dict(zip(values,range(1,len(values)+1)))
print(prot_dict)
vocab_txt = open('vocab.txt')
bpe = BPE(vocab_txt,merges=-1,separator='')
def encodeSeq(seq,protein_dict):
    firststep = bpe.process_line(seq).split()
    return [protein_dict[a] for a in firststep]
a,b,c = 0,0,0
protein_df = pd.read_csv('data/train/train.csv', index_col='ProteinID')
protein_df['encoded_sequence'] = protein_df.Target_Sequence.map(lambda a: encodeSeq(a, prot_dict))

for i in protein_df['encoded_sequence']:
    if len(i)> 700 and len(i)<= 800:
        a+=1
    if len(i)>600 and len(i)<=700:
        b+=1
    if len(i)<=600:
        c+=1
protein_df = pd.read_csv('data/valid/valid.csv', index_col='ProteinID')
protein_df['encoded_sequence'] = protein_df.Target_Sequence.map(lambda a: encodeSeq(a, prot_dict))

for i in protein_df['encoded_sequence']:
    if len(i)> 700 and len(i)<= 800:
        a+=1
    if len(i)>600 and len(i)<=700:
        b+=1
    if len(i)<=600:
        c+=1
protein_df = pd.read_csv('data/test/test.csv', index_col='ProteinID')
protein_df['encoded_sequence'] = protein_df.Target_Sequence.map(lambda a: encodeSeq(a, prot_dict))

for i in protein_df['encoded_sequence']:
    if len(i)> 700 and len(i)<= 800:
        a+=1
    if len(i)>600 and len(i)<=700:
        b+=1
    if len(i)<=600:
        c+=1









number = 13142
print(c/number)
print((c+b)/number)
print((c+b+a)/number)