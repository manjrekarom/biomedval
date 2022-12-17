import re


def convert_BIO_labels(filename):
    result_labels = []
    with open(filename, 'r', encoding='utf-8') as file:
        cnt = 0
        for i, line in enumerate(file):
            line = re.sub(r'\*(\w+)', r'\1*', line)
            print(f"** line: {i+1} **")
            print(line)
            tokens = re.sub(r'[!"#$%&\'()+,-.:;<=>?@[\\\]^_`{\|}~?]', ' ', line.strip()).split()
            print(f"** Tokens: **")
            print(tokens)
            seq_label = []
            start_entity = 0
            entity_type = 'O'
            for idx, token in enumerate(tokens):
                if token.endswith('*'):
                    start_entity += 1 if (start_entity == 0 or token[:-1] != entity_type) else -1
                    entity_type = token[:-1]
                else:
                    if start_entity == 0:
                        seq_label.append('O')
                        entity_type = 'O'
                    elif start_entity < 0:
                        raise "Something errors"
                    else:
                        if tokens[idx - 1].endswith('*'):
                            seq_label.append('B-' + entity_type.upper())
                        else:
                            seq_label.append('I-' + entity_type.upper())
            result_labels.append(seq_label)
            cnt += 1
#             if cnt % 100 == 0:
#                 print('Processed %d sentences' % cnt)
    return result_labels


predict_file = './data/test/bc5cdr_chem_ner_predict_output_copy.txt-1012000'
# actual_file = './data/bc5cdr_chem_ner_actual_output.txt'

result_labels = convert_BIO_labels(predict_file)
# actual_labels = convert_BIO_labels(actual_file)

print(result_labels[0])
# print(actual_labels[0])
print()

print(result_labels[1])
# print(actual_labels[1])
