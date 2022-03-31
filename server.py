import socket
import random
import torch
import json
import datetime
from transformers import BertTokenizerFast, EncoderDecoderModel
import warnings

warnings.filterwarnings(action="ignore")

buffer = []
pending_string = ""

def extract_compressed_string(input_string):
    result = ""

    i = 0

    while i < len(input_string):
        letter = input_string[i]

        i += 1
        if not input_string[i].isnumeric():
            raise Exception('EXTRACT_COMPRESSED_STRING: The input string has format error.')

        num_string = ""
        while input_string[i].isnumeric():
            num_string += input_string[i]
            if i == len(input_string) - 1:
                break
            i += 1

        num = int(num_string)

        for j in range(int(num)):
            result += letter

        if i == len(input_string) - 1:
            break

    return result


def expand_to_500(input_string):            //This function is written by me and it is used to make the sentence whose length is less than 500 to 500
    if len(input_string) >= 500:
        raise Exception('EXPAND_TO_500: The length should not exceed 500.')

    result = input_string

    if len(result) == 1:
        result += result

    while len(result) < 250:
        tmp = ""

        for i in range(len(result) - 1):
            tmp += result[i]
            tmp += result[i]

        tmp += result[-1]
        result = tmp

    tmp = ""

    if len(result) == 250:
        for i in range(len(result)):
            tmp += result[i]
            tmp += result[i]

        result = tmp
    else:
        insertion_candidates = [k for k in range(len(result) - 1)]
        insertion_position = random.sample(insertion_candidates, k=(500 - len(result)))

        for i in range(len(result) - 1):
            tmp += result[i]
            if i in insertion_position:
                tmp += result[i]

        tmp += result[-1]
        result = tmp

    return result


def cut_to_500(input_string):               //This function is written by me and it is used to make the input string whose length is more than 500 to 500
    if len(input_string) <= 500:
        raise Exception('CUT_TO_500: The length should exceed 500.')

    result = input_string

    while len(result) > 1000:
        tmp = ""

        for i in range(len(result) - 1):
            if i % 2 == 0:
                tmp += result[i]

        tmp += result[-1]
        result = tmp

    deletion_candidates = [k for k in range(1, len(result) - 1)]
    deletion_position = random.sample(deletion_candidates, k=(len(result) - 500))

    tmp = result[0]

    for i in range(1, len(result) - 1):
        if i in deletion_position:
            continue
        else:
            tmp += result[i]

    tmp += result[-1]
    result = tmp

    return result


def format_process(standard_string):
    if len(standard_string) != 500:
        raise Exception('FORMAT_PROCESS: The length should be 500.')

    result = ""

    for letter in standard_string:
        result += "["
        result += letter
        result += "]"
        result += " "

    result = result[:-1]

    return result


def clean_message(input_string):            //This function is written by me and it is used to clean the message and replace all the write spaces
    split_content = input_string.split(" ##")

    result = ""

    for content in split_content:
        result += content

    i = 0
    while result[i] == " ":
        i += 1

    result = result[i:]

    i = len(result) - 1
    while result[i] == " ":
        i -= 1

    if i - len(result) + 1 != 0:
        result = result[:(i - len(result) + 1)]

    return result


def single_message_process(single_message):
    if single_message[0] == "$":
        isLastMessage = True
        input_message = single_message[1:]
    else:
        isLastMessage = False
        input_message = single_message

    print("New message received: " + input_message)

    input_message = extract_compressed_string(input_message)

    if len(input_message) > 500:
        input_message = cut_to_500(input_message)
        print("Length cut to: " + str(len(input_message)))

    if len(input_message) < 500:
        input_message = expand_to_500(input_message)
        print("Length expand to: " + str(len(input_message)))

    input_message = format_process(input_message)
    print("Decoding...")

    # uncomment following lines (174-175, 186-187) to log out model inference time
    # ct = datetime.datetime.now()
    # print("Timestamp before model inference:", ct)

    model_input = []
    model_input.append(input_message)

    encoder_inputs = tokenizer(model_input, padding="max_length", truncation=True, max_length=512,
                               return_tensors="pt")
    encoder_input_ids = encoder_inputs.input_ids.to("cuda")
    attention_mask = encoder_inputs.attention_mask.to("cuda")
    outputs = model.generate(encoder_input_ids, attention_mask=attention_mask, num_return_sequences=3)

    # ct = datetime.datetime.now()
    # print("Timestamp after model inference:", ct)

    output_messages = ""

    for output in outputs:
        print(output)
        output_str = tokenizer.batch_decode(output, skip_special_tokens=True)

        output_message = ""

        for token in output_str:
            output_message += token
            output_message += " "

        output_message = output_message[:-1]

        output_message = clean_message(output_message)
        output_messages += output_message
        output_messages += "@@@"

    if isLastMessage:
        output_messages += "$"

    output_messages += '\n'
    return output_messages


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("trained_model/checkpoint-369000")
    model = EncoderDecoderModel.from_pretrained("trained_model/checkpoint-369000")
    model.to("cuda")

    my_socket = socket.socket()
    my_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = 8000
    max_connection = 999
    ip = socket.gethostname()

    my_socket.bind(('', port))
    my_socket.listen(max_connection)

    print("New server start at IP: " + ip + " on port " + str(port))

    while True:
        print("wating for connect...")
        (client_socket, address) = my_socket.accept()

        print("New connection established.")
        while True:
            try:
                client_socket.settimeout(500)

                result = pending_string

                print("Waiting for message...")

                while True:
                    if len(result.split("###\n")) >= 2:
                        break
                    result += client_socket.recv(1024).decode()

                buffer = result.split("###\n")
                pending_string = buffer[-1]
                buffer = buffer[:-1]

                for single_message in buffer:
                    if single_message[0] == '0':
                        continue
                    output_messages = single_message_process(single_message)
                    client_socket.send(bytes(output_messages, 'utf-8'))
                    print("Messages sent: " + output_messages)

            except socket.timeout:
                print("Time out!")

            except ConnectionResetError:
                break
