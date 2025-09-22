import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Paths
description_path = "battery_robocrys_descriptions.csv"
excel_path = "charge_discharge_pairs.xlsx"
checkpoint_path = "roberta-pretrained/checkpoint-22380"

# Load descriptions
desc_df = pd.read_csv(description_path)
desc_df["filename"] = desc_df["filename"].str.replace(".cif", "")
desc_map = dict(zip(desc_df["filename"], desc_df["description"]))

# Load and filter voltage data
voltage_df = pd.read_excel(excel_path)
voltage_df = voltage_df.dropna(subset=["average_voltage"])
voltage_df = voltage_df[(voltage_df["average_voltage"] <= 7) & (voltage_df["average_voltage"] >= -2)]

# Attach textual descriptions
voltage_df["desc_charge"] = voltage_df["id_charge"].map(desc_map)
voltage_df["desc_discharge"] = voltage_df["id_discharge"].map(desc_map)
voltage_df = voltage_df.dropna(subset=["desc_charge", "desc_discharge"])

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)


class DualRobertaDataset(Dataset):
    """Custom dataset for paired charge/discharge descriptions."""
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc_charge = self.tokenizer(
            row["desc_charge"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc_discharge = self.tokenizer(
            row["desc_discharge"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids_charge": enc_charge["input_ids"].squeeze(0),
            "attention_mask_charge": enc_charge["attention_mask"].squeeze(0),
            "input_ids_discharge": enc_discharge["input_ids"].squeeze(0),
            "attention_mask_discharge": enc_discharge["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["average_voltage"], dtype=torch.float)
        }


class DualRobertaRegressor(nn.Module):
    """Dual-branch RoBERTa model for voltage regression."""
    def __init__(self):
        super().__init__()
        self.roberta_charge = RobertaModel.from_pretrained(checkpoint_path)
        self.roberta_discharge = RobertaModel.from_pretrained(checkpoint_path)
        self.regressor = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        input_ids_charge,
        attention_mask_charge,
        input_ids_discharge,
        attention_mask_discharge,
        labels=None
    ):
        emb_charge = self.roberta_charge(
            input_ids=input_ids_charge,
            attention_mask=attention_mask_charge
        ).last_hidden_state[:, 0]
        emb_discharge = self.roberta_discharge(
            input_ids=input_ids_discharge,
            attention_mask=attention_mask_discharge
        ).last_hidden_state[:, 0]

        x = emb_charge + emb_discharge
        output = self.regressor(x).squeeze(1)

        if labels is not None:
            loss = nn.functional.mse_loss(output, labels)
            return {"loss": loss, "logits": output}
        return {"logits": output}


# Train/test split
train_df, test_df = train_test_split(
    voltage_df,
    test_size=0.1,
    random_state=42,
    stratify=voltage_df["working_ion"]
)
train_dataset = DualRobertaDataset(train_df, tokenizer)
test_dataset = DualRobertaDataset(test_df, tokenizer)

# Training setup
training_args = TrainingArguments(
    output_dir="./dual_roberta_voltage",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=40,
    logging_dir="./logs",
    save_strategy="no",
    report_to="none"
)

# Trainer
model = DualRobertaRegressor()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train
trainer.train()

# Evaluate
preds = trainer.predict(test_dataset)
y_true = test_df["average_voltage"].values
y_pred = preds.predictions.flatten()

print("Test R²:", r2_score(y_true, y_pred))
print("Test MAE:", mean_absolute_error(y_true, y_pred, squared=False))
