import os
import random
from Bio import SeqIO
from datasets import Dataset

def load_fasta_proteins(filepath):
    """Load protein sequences from a FASTA file."""
    proteins = []
    for record in SeqIO.parse(filepath, "fasta"):
        seq = str(record.seq)
        if 20 <= len(seq) <= 1024:  # Optional length filter
            proteins.append(seq)
    return proteins

def mask_proteins(proteins, mask_token="<mask>", mask_prob=0.15):
    """Randomly mask ~15% of proteins."""
    masked_proteins = []
    labels = []
    for prot in proteins:
        if random.random() < mask_prob:
            masked_proteins.append(mask_token)
            labels.append(prot)
        else:
            masked_proteins.append(prot)
            labels.append(None)
    return masked_proteins, labels


def build_dataset_from_fasta_dir(fasta_dir, mask_token="<mask>", mask_prob=0.15):
    entries = []
    for fname in os.listdir(fasta_dir):
        if not fname.endswith(".faa"):
            continue
        path = os.path.join(fasta_dir, fname)
        proteins = load_fasta_proteins(path)
        if len(proteins) < 10:
            continue  # skip incomplete genomes
        masked, labels = mask_proteins(proteins, mask_token, mask_prob)
        entries.append({
            "genome_id": fname,
            "masked_proteins": masked,
            "labels": labels,
        })
    return Dataset.from_list(entries)


dataset = build_dataset_from_fasta_dir("/pollard/home/czhao/projects/2025-08-12/data_for_hackathon/toy")
print(dataset[0])

dataset.save_to_disk("masked_bacformer_dataset")
# Later:
#from datasets import load_from_disk
#dataset = load_from_disk("masked_bacformer_dataset")

import os
import torch
from bacformer.pp import protein_seqs_to_bacformer_inputs

def pick_device():
    # honor CUDA_VISIBLE_DEVICES: only consider devices PyTorch can see
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    if n == 1:
        return "cuda:0"
    # choose visible GPU with most free memory
    free = []
    for i in range(n):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free.append((free_bytes, i))
    _, best = max(free)
    return f"cuda:{best}"

class BacformerMLMCollator:
    def __init__(self, max_n_proteins=6000, batch_size=128, device=None):
        # if device is not provided, auto-pick
        self.device = device or pick_device()
        self.max_n_proteins = max_n_proteins
        self.batch_size = batch_size
    def __call__(self, batch):
        all_inputs = []
        for ex in batch:
            masked = ex["masked_proteins"]
            labels = ex["labels"]  # NOTE: see label note below
            inputs = protein_seqs_to_bacformer_inputs(
                protein_sequences=masked,
                device=self.device,
                batch_size=self.batch_size,
                max_n_proteins=self.max_n_proteins,
            )
            inputs["labels"] = labels   # pass through (see note)
            all_inputs.append(inputs)
        # merge keys across examples
        batched = {}
        for k in all_inputs[0]:
            if k == "labels":
                batched[k] = [d["labels"] for d in all_inputs]
            else:
                batched[k] = torch.cat([d[k] for d in all_inputs], dim=0)
        return batched


data_collator = BacformerMLMCollator()



from transformers import TrainingArguments, Trainer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "macwiatrak/bacformer-masked-MAG",
    trust_remote_code=True
)

args = TrainingArguments(
    output_dir="./finetuned_bacformer",
    do_train=True, do_eval=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    learning_rate=1e-4,
    remove_unused_columns=False,   # <- important so 'masked_proteins' survives to the collator
    eval_steps=100, save_steps=100, logging_steps=10,
    save_total_limit=2,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,           # your HF Dataset with 'masked_proteins' and 'labels'
    data_collator=BacformerMLMCollator(),
)
trainer.train()
trainer.save_model("./finetuned_bacformer")

