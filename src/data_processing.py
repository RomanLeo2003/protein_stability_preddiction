import pandas as pd
import torch
from protein_task import get_protein_task, get_feature_tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import EsmTokenizer

def find_first_diff_index(str1, str2):
    min_len = min(len(str1), len(str2))
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            return i

    if len(str1) != len(str2):
        return min_len
    
    return -1

class ProteinDataset(Dataset):
    def __init__(
        self, csv_filename: str, esm_tokenizer: str, max_length: int = 100
    ) -> None:
        super().__init__()
        df = pd.read_csv(csv_filename)
        self.max_length = max_length
        self.wt_seqs_list = df.wt_seq.to_list()
        self.mt_seqs_list = df.mt_seq.to_list()
        self.ddG_list = df.ddG.to_list()
        self.tokenizer = EsmTokenizer.from_pretrained(esm_tokenizer)

    def __len__(self):
        return len(self.wt_seqs_list)

    def __getitem__(self, index):
        wt_seq = self.wt_seqs_list[index]
        mt_seq = self.mt_seqs_list[index]
        mutation_position = find_first_diff_index(wt_seq, mt_seq)
        seq_length = len(wt_seq)
        wt_tokens = self.tokenizer(
            wt_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        wt_tokens["input_ids"] = wt_tokens["input_ids"].squeeze(0)
        wt_tokens["attention_mask"] = wt_tokens["attention_mask"].squeeze(0)
        mt_tokens = self.tokenizer(
            mt_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        mt_tokens["input_ids"] = mt_tokens["input_ids"].squeeze(0)
        mt_tokens["attention_mask"] = mt_tokens["attention_mask"].squeeze(0)
        ddG = self.ddG_list[index]
        return ((wt_tokens, mt_tokens), torch.Tensor([ddG]), seq_length, mutation_position)


def load_data_prostata(
    batch_size: int = 16,
    val_split=0.15,
    train_dataset_path: str = "/data/prostata_filtered.csv",
    path_to_tasks: str = "/data/prostata_test_task",
    test_dataset_path_ssym: str = "/data/ssym.csv",
    test_dataset_path_s669: str = "/data/s669.csv",
):
    df_train = pd.read_csv(train_dataset_path)
    target = torch.tensor(df_train["ddg"], dtype=torch.float32)

    all_tasks = []

    # Обработка тренировочных данных
    for idx in range(len(df_train)):
        task = get_protein_task(df_train, idx=idx, path=path_to_tasks)
        all_tasks.append(task)

    df_test_ssym = pd.read_csv(test_dataset_path_ssym)
    df_test_s669 = pd.read_csv(test_dataset_path_s669)
    df_test = pd.concat((df_test_ssym, df_test_s669), axis="rows", ignore_index=True)

    # Fake target for testing purposes
    test_target = torch.zeros(df_test.shape[0], dtype=torch.float32)

    test_all_tasks = []
    path_to_test_tasks = {
        "ssym": "/data/ssym_test_task",
        "s669": "/data/s669_test_task",
    }

    # Обработка тестовых данных
    for idx in range(len(df_test)):
        source = df_test.iloc[idx]["source"]
        task = get_protein_task(df_test, idx=idx, path=path_to_test_tasks[source])
        test_all_tasks.append(task)

    # Извлечение признаков для тренировочной выборки
    features = []
    for task in all_tasks:
        mutation = task.task["mutants"]
        mutation_key, _ = next(iter(mutation.items()))
        res_name, position, chain_id = mutation_key
        residue_name = "_".join((res_name, chain_id, str(position)))
        feature_index = task.protein_job["protein_wt"]["obs_positions"][residue_name]
        feature_tensor = get_feature_tensor(
            task, feature_names=["pair", "lddt_logits", "plddt"]
        )
        # Объединение признаков для wild-type и mutant
        features.append(
            torch.cat(
                (
                    feature_tensor["wt"][feature_index],
                    feature_tensor["mt"][feature_index],
                ),
                dim=0,
            )
        )

    # Извлечение признаков для тестовой выборки
    features_test = []
    for task in test_all_tasks:
        mutation = task.task["mutants"]
        mutation_key, _ = next(iter(mutation.items()))
        res_name, position, chain_id = mutation_key
        residue_name = "_".join((res_name, chain_id, str(position)))
        feature_index = task.protein_job["protein_wt"]["obs_positions"][residue_name]
        feature_tensor = get_feature_tensor(
            task, feature_names=["pair", "lddt_logits", "plddt"]
        )
        # Объединение признаков для wild-type и mutant
        features_test.append(
            torch.cat(
                (
                    feature_tensor["wt"][feature_index],
                    feature_tensor["mt"][feature_index],
                ),
                dim=0,
            )
        )

    # Преобразование данных в TensorDataset для тренировочных данных
    train_features = torch.stack(features, dim=0)
    train_targets = target[:, None]

    # Преобразование данных в TensorDataset для тестовых данных
    test_features = torch.stack(features_test, dim=0)
    test_targets = test_target[:, None]

    total_train_size = len(train_features)
    val_idx = int((1 - val_split) * total_train_size)

    train_features, val_features = train_features[:val_idx], train_features[val_idx:]
    train_targets, val_targets = train_targets[:val_idx], train_targets[val_idx:]
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    test_dataset = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_data(train_csv_filename, val_csv_filename, tokenizer_name, batch_size):
    # train_df = pd.read_csv()
    # val_df = pd.read_csv()
    train_dataset = ProteinDataset(train_csv_filename, tokenizer_name)
    val_dataset = ProteinDataset(val_csv_filename, tokenizer_name)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def load_test_data(test_csv_filename, tokenizer_name, batch_size):
    df = pd.read_csv(test_csv_filename)
    test_dataset = ProteinDataset(test_csv_filename, tokenizer_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, df