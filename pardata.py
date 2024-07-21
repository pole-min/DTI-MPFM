import json

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from subword_nmt.apply_bpe import BPE
import tensorflow as tf
drug_col = 'DrugID'
protein_col = 'ProteinID'
label_col = 'Label'
# seq_list = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
############generate protein embedding dictionary
vocab_csv = pd.read_csv('vocab.csv')
values = vocab_csv['Values'].values
protein_dict = dict(zip(values,range(1,len(values)+1)))
print(protein_dict)
#seq_dict = {k : v+1 for v,k in enumerate(seq_list)}
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
print('drug dict : ',kk_dict)
print('drug dict length : ',len(kk_dict))
def encod_SMILES(seq,kk_dict):
    if pd.isnull(seq):
        return [0]
    else:
        return [kk_dict[a] for a in seq]
vocab_txt = open('vocab.txt')
bpe = BPE(vocab_txt,merges=-1,separator='')
def encodeSeq(seq,protein_dict):
    firststep = bpe.process_line(seq).split()
    return [protein_dict[a] for a in firststep]

#read training data and operate on traning data so that we can get the final training data
def parse_data(dti_dir,drug_dir,protein_dir,prot_vec='Convolution',prot_len=800,drug_vec='Convolution',drug_len=2048,drug_len2=100):

    print("parsing data : {0},{1},{2}".format(dti_dir,drug_dir,protein_dir))
    dti_df = pd.read_csv(dti_dir)

    kge_model = 'rescal'

    embeddings_key = ''
    if kge_model == 'complex' or kge_model == 'analogy':
        embeddings_key = 'ent_re_embeddings.weight'
    else:
        embeddings_key = 'ent_embeddings.weight'
    with open(f'data/kge/{kge_model}/entity_kge.vec', 'r') as f:
        entity_kge = f.readline()
        entity_kge = json.loads(entity_kge)
        entity_kge = entity_kge[embeddings_key]
        pass
    entity_kge_df = pd.DataFrame({'entity_kge': entity_kge})
    entity_all_df = pd.read_csv('./data/entity2id.txt', sep='\t', header=None, names=['entity', 'entity_id'])
    entity_all_df['entity_kge'] = entity_kge_df
    entity_all_df = entity_all_df.drop(['entity_id'], axis=1)

    drug_kge = entity_all_df[entity_all_df['entity'].str.contains('D')]
    protein_kge = entity_all_df[entity_all_df['entity'].str.contains('P')]

    drug_kge.columns = ['DrugID', 'drug_kge']
    protein_kge.columns = ['ProteinID', 'protein_kge']

    drug_kge.set_index('DrugID', inplace=True)
    protein_kge.set_index('ProteinID', inplace=True)

    dti_df = pd.merge(dti_df, drug_kge, left_on=drug_col, right_index=True)
    dti_df = pd.merge(dti_df, protein_kge, left_on=protein_col, right_index=True)

    drug_df = pd.read_csv(drug_dir,index_col=drug_col)
    drug_df['drug_embedding'] = drug_df.SMILES.map(lambda a: encod_SMILES(a,kk_dict))

    protein_df = pd.read_csv(protein_dir,index_col=protein_col)
    protein_df['encoded_sequence'] = protein_df.Target_Sequence.map(lambda a: encodeSeq(a,protein_dict))

    dti_df = pd.merge(dti_df,drug_df,left_on=drug_col,right_index=True)

    dti_df = pd.merge(dti_df,protein_df,left_on=protein_col,right_index=True)
    #drug_feature = dti_df['morgan_fp'].values
    # drug_feature = np.stack(dti_df[drug_vec].map(lambda fp:fp.split('\t')))
    c_train = dti_df['morgan_fp'].values
    l = []
    for i in c_train:
      temp = [int(k) for k in i]
      l.append(temp)
    drug_feature = np.array(l)

    drug_feature2 = sequence.pad_sequences(dti_df['drug_embedding'].values,drug_len2,padding='post')
    drug_feature3 = tf.convert_to_tensor([np.array(i) for i in dti_df['drug_kge'].values])
    # drug_feature3 = dti_df['drug_kge'].values

    protein_feature = sequence.pad_sequences(dti_df['encoded_sequence'].values, prot_len,padding='post')
    protein_feature2 = sequence.pad_sequences(dti_df['encoded_sequence'].values,prot_len,padding='post')
    protein_feature3 = tf.convert_to_tensor([np.array(i) for i in dti_df['protein_kge'].values])

    # protein_feature3 = dti_df['protein_kge'].values
    #protein_feature2 = tf.cast(protein_feature2,dtype=tf.float32)
    label = [int(i) for i in dti_df[label_col].values]
    label = np.array(label)
    print('\t Positive data :   \t')
    print(sum(dti_df[label_col]))
    print('\t Negative data :   \t')
    print(dti_df.shape[0] - sum(dti_df[label_col]))
    #print('\t Positive data : %d' ,(sum(dti_df[label_col])))
    #print('\t Negative data : %d' ,(dti_df.shape[0]- sum(dti_df[label_col])))
    return {
        "protein_feature" : protein_feature,
        "protein_feature2" : protein_feature2,
        "protein_feature3" : protein_feature3,
        "drug_feature" : drug_feature,
        "drug_feature2": drug_feature2,
        "drug_feature3": drug_feature3,
        "Label" : label,
    }
