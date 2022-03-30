import datasets
import torch
import json
from datasets import Dataset
from transformers import BertTokenizerFast, DistilBertTokenizerFast
from transformers import EncoderDecoderModel
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, input_encodings, output_encodings):
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings

    def __getitem__(self, idx):
        item = {}

        item['input_ids'] = self.input_encodings[idx].ids
        item['attention_mask'] = self.input_encodings[idx].attention_mask
        item['decoder_input_ids'] = self.output_encodings[idx].ids
        item['decoder_attention_mask'] = self.output_encodings[idx].attention_mask
        item['labels'] = self.output_encodings[idx].ids.copy()

        if item['labels'] == tokenizer.pad_token_id:
             item['labels'] = -100

        return item

    def __len__(self):
        return len(self.output_encodings)


# class GestureDataset(torch.utils.data.Dataset):
#     def __init__(self, input_trj, output_text):
#         self.input_trj = input_trj
#         self.output_text = output_text
#
#     def __getitem__(self, idx):
#         item = {}
#         item["trj"] = self.input_trj[idx]
#         item["text"] = self.output_text[idx]
#
#         return item
#
#     def __len__(self):
#         return len(self.output_text)


def read_data_dir(dir_name, file_num, input_trjs, output_text):
    for i in range(0, file_num):
        current_file_name = dir_name + "/gestures_" + str(i) + ".json"
        read_json_file(current_file_name, input_trjs, output_text)
        print("Data loading: " + str(i + 1) + "/" + str(file_num) + " finished.")


def read_json_file(file_name, input_trjs, output_text):
    f = open(file_name)
    gesture_data = json.load(f)
    f.close()

    for object in gesture_data:
        letters = object['Letters']
        letter_string = ""

        for letter in letters:
            tmp = "[" + letter + "] "
            letter_string += tmp

        letter_string = letter_string[:-1]

        input_trjs.append(letter_string)
        output_text.append(object['target phrase'])


def process_data_to_model_inputs(batch):
      # tokenize the inputs and labels
      inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=encoder_max_length)
      outputs = tokenizer(batch["output"], padding="max_length", truncation=True, max_length=decoder_max_length)

      batch["input_ids"] = inputs.input_ids
      batch["attention_mask"] = inputs.attention_mask
      batch["decoder_input_ids"] = outputs.input_ids
      batch["decoder_attention_mask"] = outputs.attention_mask
      batch["labels"] = outputs.input_ids.copy()

      # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
      # We have to make sure that the PAD token is ignored
      batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

      return batch


if __name__ == "__main__":
    f = open("data/combined/gestures_all_v2.json")
    gesture_data = json.load(f)
    f.close()
    print("Data loading completed.")

    # # keep last 10000 for testing
    raw_input = gesture_data['input'][:-10000]
    raw_output = gesture_data['output'][:-10000]

    train_input, val_input, train_output, val_output = train_test_split(raw_input, raw_output, test_size=.2)

    train_data = {"input": train_input, "output": train_output}
    val_data = {"input": val_input, "output": val_output}

    val_dataset = Dataset.from_dict(val_data)
    train_dataset = Dataset.from_dict(train_data)
    print("Dataset contruction completed.")

    # read_json_file("gestures_1.json", input_trjs=input_trjs, output_text=output_text)

    batch_size = 8  # change to 16 for full training

    encoder_max_length = 512
    decoder_max_length = 128

    # tokenizer part
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    special_tokens_dict = {'additional_special_tokens': ['[a]', '[b]', '[c]', '[d]', '[e]', '[f]', '[g]', '[h]', '[i]', '[j]', '[k]', '[l]', '[m]', '[n]',
                                                         '[o]', '[p]', '[q]', '[r]', '[s]', '[t]', '[u]', '[v]', '[w]', '[x]', '[y]', '[z]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["input", "output"]
    )

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    val_dataset = val_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["input", "output"]
    )

    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    print("Dataset format setup completed.")

    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
    # bert2bert = EncoderDecoderModel.from_pretrained("trained_model/checkpoint-369000")

    # set special tokens, especially new ones
    bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
    bert2bert.config.eos_token_id = tokenizer.eos_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id
    bert2bert.encoder.resize_token_embeddings(len(tokenizer))

    # sensible parameters for beam search
    bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
    bert2bert.config.max_length = 10
    bert2bert.config.min_length = 1
    bert2bert.config.no_repeat_ngram_size = 3
    bert2bert.config.early_stopping = True
    bert2bert.config.length_penalty = 2.0
    bert2bert.config.num_beams = 4

    # load rouge for validation
    rouge = datasets.load_metric("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }


    # set training arguments - these params are not really tuned, feel free to change
    training_args = Seq2SeqTrainingArguments(
        output_dir="online_trained_model/",
        num_train_epochs=3,
        # evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        logging_steps=1000,  # set to 1000 for full training
        save_steps=1000,  # set to 500 for full training
        # eval_steps=8000,  # set to 8000 for full training
        warmup_steps=2000,  # set to 2000 for full training
        overwrite_output_dir=True,
        save_total_limit=3,
        fp16=True,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=bert2bert,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
