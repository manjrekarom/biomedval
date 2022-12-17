import re


# preprocess text for metrics
def postprocess_text(preds, labels):
    preds = [pred.strip().lower() for pred in preds]
    labels = [label.strip().lower() for label in labels]

    return preds, labels


def convert_bio_labels(text, select_only=[]):
    line = re.sub(r'\*(\w+)', r'\1*', text)
    tokens = re.sub(r'[!"#$%&\'()+,-.:;<=>?@[\\\]^_`{\|}~?]', ' ', line.strip()).split()
    seq_label = []
    start_entity = 0
    entity_type = 'O'
    for idx, token in enumerate(tokens):
        if token.endswith('*'):
            if select_only and token[:-1] in select_only:
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
    return seq_label
