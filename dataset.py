import collections
import enum
import json
import re
from tqdm import tqdm


TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
    """Answer record.

    An Answer contains the type of the answer and possibly the text (for
    long) as well as the offset (for extractive).
    """

    def __new__(cls, type_, text=None, offset=None):
        return super(Answer, cls).__new__(cls, type_, text, offset)


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, unique_id, example_index, doc_span_index, token_to_orig_map,input_ids, input_mask, segment_ids,
                 start_position=None, end_position=None, answer_text="", answer_type=AnswerType.SHORT):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = answer_text
        self.answer_type = answer_type


def has_long_answer(annotation):
    return annotation["long_answer"]["start_token"] >= 0 and annotation["long_answer"]["end_token"] >= 0


def should_skip_context(example, idx):
    if example["long_answer_candidates"][idx]["top_level"]:
        return True
    if not get_candidate_text(example, idx).text.strip():
        return True
    return False


def get_first_annotation(example):
    positive_annotations = sorted([a for a in example["annotations"] if has_long_answer(a)], key=lambda a: a["long_answer"]["candidate_index"])

    for annotation in positive_annotations:
        if annotation["short_answers"]:
            idx = annotation["long_answer"]["candidate_index"]
            start_token = annotation["short_answers"][0]["start_token"]
            end_token = annotation["short_answers"][-1]["end_token"]
            return annotation, idx, (token_to_char_offset(example, idx, start_token), token_to_char_offset(e, idx, end_token) - 1)

    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        return a, idx, (-1, -1)

    return None, -1, (-1, -1)


def get_text_span(example, span):
    token_positions = []
    tokens = []
    for idx in range(span["start_token"], span["end_token"]):
        token = example["document_tokens"][idx]
        if not token["html_token"]:
            token_positions.append(idx)
            token = token["token"].replace(" ", "")
            tokens.append(token)
    return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(example, candidate_idx, token_idx):
    """Converts a token index to the char offset within the candidate."""
    candidate = example["long_answer_candidates"][candidate_idx]
    char_offset = 0
    for i in range(candidate["start_token"], token_idx):
        token = example["document_tokens"][i]
        if not token["html_token"]:
            char_offset += len(token["token"].replace(" ", "")) + 1

    return char_offset


def get_candidate_type(e, idx):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    c = e["long_answer_candidates"][idx]
    first_token = e["document_tokens"][c["start_token"]]["token"]
    if first_token == "<Table>":
        return "Table"
    if first_token == "<P>":
        return "Paragraph"
    if first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    if first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    print("Unknown candidate type found: %s", first_token)
    return "Other"


def add_candidate_types_and_positions(example, max_pos):
    """Adds type and position info to each candidate in the document."""
    counts = collections.defaultdict(int)
    for idx, candidat in candidates_iter(example):
        context_type = get_candidate_type(example, idx)
        if counts[context_type] < max_pos:
            counts[context_type] += 1
        candidat["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(example, idx):
    """Returns type and position info for the candidate at the given index."""
    if idx == -1:
        return "[NoLongAnswer]"

    return example["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(example, idx):
    """Returns a text representation of the candidate at the given index."""
    # No candidate at this index.
    if idx < 0 or idx >= len(example["long_answer_candidates"]):
        return TextSpan([], "")

    return get_text_span(example, example["long_answer_candidates"][idx])


def candidates_iter(example):
    """Yield's the candidates that should not be skipped in an example."""
    for idx, candidate in enumerate(example["long_answer_candidates"]):
        if should_skip_context(example, idx):
            continue
        yield idx, candidate


def load_dataset(file_path, max_context, max_position):
    examples = []
    with open(file_path, 'r') as json_file:
        for line in tqdm(json_file):
            json_data = json.loads(line,  object_pairs_hook=collections.OrderedDict)
            examples.append(create_example_from_jsonl(json_data, max_context, max_position))
    return examples


def create_example_from_jsonl(example, max_contexts, max_position):
    """Creates an NQ example from a given line of JSON."""
    add_candidate_types_and_positions(example, max_position)
    annotation, annotated_idx, annotated_sa = get_first_annotation(example)

    # annotated_idx: index of the first annotated context, -1 if null.
    # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
    question = {"input_text": example["question_text"]}
    answer = {
        "candidate_id": annotated_idx,
        "span_text": "",
        "span_start": -1,
        "span_end": -1,
        "input_text": "long",
    }

    # Yes/no answers are added in the input text.
    if annotation is not None:
       if annotation["yes_no_answer"] in ("YES", "NO"):
           answer["input_text"] = annotation["yes_no_answer"].lower()

    # Add a short answer if one was found.
    if annotated_sa != (-1, -1):
        answer["input_text"] = "short"
        span_text = get_candidate_text(e, annotated_idx).text
        answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
        answer["span_start"] = annotated_sa[0]
        answer["span_end"] = annotated_sa[1]
        expected_answer_text = get_text_span(example, {"start_token": annotation["short_answers"][0]["start_token"], "end_token": annotation["short_answers"][-1]["end_token"],}).text
        assert expected_answer_text == answer["span_text"], (expected_answer_text, answer["span_text"])

    # Add a long answer if one was found.
    elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
        answer["span_text"] = get_candidate_text(example, annotated_idx).text
        answer["span_start"] = 0
        answer["span_end"] = len(answer["span_text"])

    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_type_and_position(example, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (get_candidate_text(example, -1))
    for idx, _ in candidates_iter(example):
        context = {"id": idx, "type": get_candidate_type_and_position(example, idx)}
        context["text_map"], context["text"] = get_candidate_text(example, idx)
        context_idxs.append(idx)
        context_list.append(context)
        if len(context_list) >= max_contexts:
            break

    # Assemble example.
    example = {
        "id": str(example["example_id"]),
        "questions": [question],
        "answers": [answer],
        "has_correct_context": annotated_idx in context_idxs
    }

    single_map = []
    single_context = []
    offset = 0
    for context in context_list:
        single_map.extend([-1, -1])
        single_context.append("[ContextId=%d] %s" %(context["id"], context["type"]))
        offset += len(single_context[-1]) + 1
        if context["id"] == annotated_idx:
            answer["span_start"] += offset
            answer["span_end"] += offset

        # Many contexts are empty once the HTML tags have been stripped, so we
        # want to skip those.
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1

    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map
    if annotated_idx in context_idxs:
        expected = example["contexts"][answer["span_start"]:answer["span_end"]]

        # This is a sanity check to ensure that the calculated start and end
        # indices match the reported span text. If this assert fails, it is likely
        # a bug in the data preparation code above.
        assert expected == answer["span_text"], (expected, answer["span_text"])

    return example


def convert_examples_to_features(examples, tokenizer, is_training, output_fn):
    """Converts a list of NqExamples into InputFeatures."""
    num_spans_to_ids = collections.defaultdict(list)

    for example in examples:
        example_index = example.example_id
        features = convert_single_example(example, tokenizer, is_training)
        num_spans_to_ids[len(features)].append(example.qas_id)

        for feature in features:
            feature.example_index = example_index
            feature.unique_id = feature.example_index + feature.doc_span_index
            output_fn(feature)

    return num_spans_to_ids


def convert_single_example(example, tokenizer, max_query_len=100, max_seq_len=50, doc_stride=2, is_training=True):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    features = []
    for (i, token) in enumerate(example.doc_tokens):
         orig_to_tok_index.append(len(all_doc_tokens))
         sub_tokens = tokenize(tokenizer, token)
         tok_to_orig_index.extend([i] * len(sub_tokens))
         all_doc_tokens.extend(sub_tokens)

    # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
    # tokenized word tokens in the contexts. The word tokens might themselves
    # correspond to word tokens in a larger document, with the mapping given
    # by `doc_tokens_map`.
    if example.doc_tokens_map:
        tok_to_orig_index = [example.doc_tokens_map[index] for index in tok_to_orig_index]

    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
    if len(query_tokens) > max_query_len:
        query_tokens = query_tokens[-max_query_len:]

    # ANSWER
    tok_start_position = 0
    tok_end_position = 0
    if is_training:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_len - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of up to our max length with a stride of `doc_stride`.
    DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        length = min(length, max_tokens_for_doc)
        doc_spans.append(DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        start_position = None
        end_position = None
        answer_type = None
        answer_text = ""
        if is_training:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = tok_start_position >= doc_start and tok_end_position <= doc_end
            if (not contains_an_annotation) or example.answer.type == AnswerType.UNKNOWN:
                # If an example has unknown answer type or does not contain the answer
                # span, then we only include it with probability --include_unknowns.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.
                start_position = 0
                end_position = 0
                answer_type = AnswerType.UNKNOWN
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                answer_type = example.answer.type

            answer_text = " ".join(tokens[start_position:(end_position + 1)])

        feature = InputFeatures(
            unique_id=-1,
            example_index=-1,
            doc_span_index=doc_span_index,
            token_to_orig_map=token_to_orig_map,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            answer_text=answer_text,
            answer_type=answer_type
         )

        features.append(feature)

    return features

# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.
    Args:
        tokenizer: a tokenizer from bert.tokenization.FullTokenizer
        text: text to tokenize
        apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).
    Returns:
      tokenized text.
    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """
    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize
    tokens = []
    for token in text.split(" "):
       if _SPECIAL_TOKENS_RE.match(token):
           if token in tokenizer.vocab:
               tokens.append(token)
           else:
               tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
       else:
           tokens.extend(tokenize_fn(token))
    return tokens
