import os
import pandas as pd
from Bio import SeqIO

import torch
from transformers import AutoModel
from bacformer.pp import preprocess_genome_assembly, protein_seqs_to_bacformer_inputs

data_path = "./ptn"
def build_all_into_one_dataframe(faa_files):
    records = []
    for faa in faa_files:
        genome_id = os.path.splitext(os.path.basename(faa))[0]  # filename without .faa
        for record in SeqIO.parse(faa, "fasta"):
            gene_id = record.id
            gene_seq = str(record.seq)
            records.append([genome_id, gene_id, gene_seq])
    return pd.DataFrame(records, columns=["genome_id", "gene_id", "gene_seq"])
# Example usage:
faa_files = [
    f"./ptn/{f}" for f in os.listdir(data_path) if f.endswith('.faa')
]
# df = build_all_into_one_dataframe(faa_files)
# # df.to_csv('all_proteins.csv')

# def split_df_into_per_genome_df(input_df):
#     for gnm in input_df['genome_id'].unique()

# genome_027067 = df.loc[df['genome_id'] == 'GUT_GENOME027067']


def convert_genome_faa_to_protein_df(filepath):
    records = []
    with open(filepath) as f:
        for record in SeqIO.parse(f, 'fasta'):
            genome_id = os.path.splitext(os.path.basename(filepath))[0]  # filename without .faa
            gene_id = record.id
            gene_seq = str(record.seq)
            records.append([genome_id, gene_id, gene_seq])
    return pd.DataFrame(records, columns=["genome_id", "gene_id", "gene_seq"])


# for fp in faa_files:
#     d = convert_genome_faa_to_protein_df(fp)
#     print(d.head())

d = convert_genome_faa_to_protein_df(faa_files[-1])
# print(d.head())
# print("asdfsfdfs", d.gene_seq.map(len).max())
def load_model():
    device = "cuda:0"
    model = AutoModel.from_pretrained(
        "macwiatrak/bacformer-masked-complete-genomes", trust_remote_code=True
    ).to(device).eval().to(torch.bfloat16)
    return model, device


def do_embedding(data, device, model):
    # print(len(data['gene_seq']))
    max_seq = data.gene_seq.map(len).max() + 1
    inputs = protein_seqs_to_bacformer_inputs(
        data['gene_seq'],
        device=device,
        batch_size=128,  # the batch size for computing the protein embeddings
        max_n_proteins=6000,  # the maximum number of proteins Bacformer was trained with
        max_prot_seq_len=max_seq
    )
    for k, v in inputs.items():
        print(k, v.shape)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        output_as_df = pd.DataFrame(outputs['last_hidden_state'].cpu().float().numpy())
        print(outputs["last_hidden_state"].shape)
        print(pd.concat([data, output_as_df], axis=1))


mod, dev = load_model()

do_embedding(d, dev, mod)