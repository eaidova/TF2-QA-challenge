import collections
import random

import enum
import json
import re

import tensorflow as tf
from tqdm import tqdm

from tokenization import whitespace_tokenize, FullTokenizer

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

            return annotation, idx, (token_to_char_offset(example, idx, start_token), token_to_char_offset(example, idx, end_token) - 1)

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
            examples.append(create_example_from_jsonl(line, max_context, max_position))
    return examples


def create_example_from_jsonl(line, max_contexts, max_position):
    """Creates an NQ example from a given line of JSON."""
    example = json.loads(line, object_pairs_hook=collections.OrderedDict)
    document_tokens = example["document_text"].split(" ")
    example["document_tokens"] = []
    for token in document_tokens: example["document_tokens"].append(
        {"token": token, "start_byte": -1, "end_byte": -1, "html_token": "<" in token})
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
        span_text = get_candidate_text(example, annotated_idx).text
        answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
        answer["span_start"] = annotated_sa[0]
        answer["span_end"] = annotated_sa[1]
        expected_answer_text = get_text_span(
            example,
            {"start_token": annotation["short_answers"][0]["start_token"],
             "end_token": annotation["short_answers"][-1]["end_token"]}
        ).text
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
    """Converts a single Example into a list of InputFeatures."""
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


class Example:
    """A single training/test example."""

    def __init__(self,
                 example_id,
                 qas_id,
                 questions,
                 doc_tokens,
                 doc_tokens_map=None,
                 answer=None,
                 start_position=None,
                 end_position=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.questions = questions
        self.doc_tokens = doc_tokens
        self.doc_tokens_map = doc_tokens_map
        self.answer = answer
        self.start_position = start_position
        self.end_position = end_position


def read_entry(entry, is_training):
    def is_whitespace(c):
        return c in " \t\r\n" or ord(c) == 0x202F

    examples = []
    contexts_id = entry["id"]
    contexts = entry["contexts"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in contexts:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    questions = []
    for i, question in enumerate(entry["questions"]):
        qas_id = "{}".format(contexts_id)
        question_text = question["input_text"]
        start_position = None
        end_position = None
        answer = None
        if is_training:
            answer_dict = entry["answers"][i]
            answer = make_answer(contexts, answer_dict)

            # For now, only handle extractive, yes, and no.
            if answer is None or answer.offset is None:
                continue
            start_position = char_to_word_offset[answer.offset]
            end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                whitespace_tokenize(answer.text))
            if actual_text.find(cleaned_answer_text) == -1:
                tf.compat.v1.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue

        questions.append(question_text)
        example = Example(
            example_id=int(contexts_id),
            qas_id=qas_id,
            questions=questions[:],
            doc_tokens=doc_tokens,
            doc_tokens_map=entry.get("contexts_map", None),
            answer=answer,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)
    return examples


class ConvertExamples2Features:
    def __init__(self,tokenizer, is_training, output_fn, collect_stat=False):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.output_fn = output_fn
        self.num_spans_to_ids = collections.defaultdict(list) if collect_stat else None

    def __call__(self,example):
        example_index = example.example_id
        features = convert_single_example(example, self.tokenizer, self.is_training)
        if self.num_spans_to_ids is not None:
            self.num_spans_to_ids[len(features)].append(example.qas_id)

        for feature in features:
            feature.example_index = example_index
            feature.unique_id = feature.example_index + feature.doc_span_index
            self.output_fn(feature)
        return len(features)


def examples_iter(input_file, is_training, max_contexts, max_position, tqdm=None):
    """Read a json file into a list of Example."""
    input_paths = tf.io.gfile.glob(input_file)

    def _open(path):
        return tf.io.gfile.GFile(path, "r")

    for path in input_paths:
        tf.compat.v1.logging.info("Reading: %s"% path)
        with _open(path) as input_file:
            if tqdm is not None:
                input_file = tqdm(input_file)
            for index, line in enumerate(input_file):
                entry = create_example_from_jsonl(line, max_contexts, max_position)
                yield read_entry(entry, is_training)


def read_examples(input_file, is_training, max_contexts, max_position, tqdm=None):
    """Read a json file into a list of Example."""
    input_paths = tf.io.gfile.glob(input_file)
    input_data = []

    def _open(path):
        return tf.io.gfile.GFile(path, "r")

    for path in input_paths:
        tf.compat.v1.logging.info("Reading: %s", path)
        with _open(path) as input_file:
            if tqdm is not None:
                input_file = tqdm(input_file)
            for index, line in enumerate(input_file):
                input_data.append(create_example_from_jsonl(line, max_contexts, max_position))

    examples = []
    for entry in input_data:
        examples.extend(read_entry(entry, is_training))
    return examples


def read_candidates_from_one_split(input_path):
    """Read candidates from a single jsonl file."""
    candidates_dict = {}
    print("Reading examples from: %s" % input_path)
    with tf.io.gfile.GFile(input_path, "r") as input_file:
        for index, line in enumerate(input_file):
            e = json.loads(line)
            candidates_dict[e["example_id"]] = e["long_answer_candidates"]
    return candidates_dict


def read_candidates(input_pattern):
    """Read candidates with real multiple processes."""
    input_paths = tf.io.gfile.glob(input_pattern)
    final_dict = {}
    for input_path in input_paths:
        final_dict.update(read_candidates_from_one_split(input_path))
    return final_dict


class EvalExample:
    """Eval data available for a single example."""
    def __init__(self, example_id, candidates):
        self.example_id = example_id
        self.candidates = candidates
        self.results = {}
        self.features = {}


def prepare_training_data(vocab_file, train_file, lower_case, output_file, max_contexts, max_position):
    examples_processed = 0
    num_examples_with_correct_context = 0
    creator_fn = CreateTFExampleFn(is_training=True, vocab_file=vocab_file, do_lower_case=lower_case)
    examples = load_dataset(train_file, max_contexts, max_position)

    instances = []
    for example in examples:
        for instance in creator_fn.process(example):
            instances.append(instance)
        if example["has_correct_context"]:
            num_examples_with_correct_context += 1
        if examples_processed % 100 == 0:
            tf.logging.info("Examples processed: %d", examples_processed)
        examples_processed += 1
    tf.logging.info("Examples with correct context retained: %d of %d",
                    num_examples_with_correct_context, examples_processed)

    random.shuffle(instances)
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for instance in instances:
            writer.write(instance)
    tf.logging.info("Number precomputed: %d", len(instances))

    return len(instances)


class CreateTFExampleFn:
    """Functor for creating tf.Examples."""

    def __init__(self, is_training, vocab_file, do_lower_case):
        self.is_training = is_training
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def process(self, example):
        """Coverts an example in a list of serialized tf examples."""
        entries = read_entry(example, self.is_training)
        input_features = []
        for entry in entries:
            input_features.extend(convert_single_example(entry, self.tokenizer, self.is_training))

        for input_feature in input_features:
            input_feature.example_index = int(example["id"])
            input_feature.unique_id = (
                    input_feature.example_index + input_feature.doc_span_index)

            def create_int_feature(values):
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(values)))

            features = collections.OrderedDict()
            features["unique_id"] = create_int_feature([input_feature.unique_id])
            features["input_ids"] = create_int_feature(input_feature.input_ids)
            features["input_mask"] = create_int_feature(input_feature.input_mask)
            features["segment_ids"] = create_int_feature(input_feature.segment_ids)

            if self.is_training:
                features["start_positions"] = create_int_feature(
                    [input_feature.start_position])
                features["end_positions"] = create_int_feature(
                    [input_feature.end_position])
                features["answer_types"] = create_int_feature(
                    [input_feature.answer_type])
            else:
                token_map = [-1] * len(input_feature.input_ids)
                for k, v in input_feature.token_to_orig_map.items():
                    token_map[k] = v
                features["token_map"] = create_int_feature(token_map)

            yield tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def make_answer(contexts, answer):
    """Makes an Answer object.

    Args:
      contexts: string containing the context
      answer: dictionary with `span_start` and `input_text` fields

    Returns:
      an Answer object. If the Answer type is YES or NO or LONG, the text
      of the answer is the long answer. If the answer type is UNKNOWN, the text of
      the answer is empty.
    """
    start = answer["span_start"]
    end = answer["span_end"]
    input_text = answer["input_text"]
    answer_types = {
        "yes": AnswerType.YES,
        "no": AnswerType.NO,
        "long": AnswerType.LONG,
        "short": AnswerType.SHORT
    }

    if (answer["candidate_id"] == -1 or start >= len(contexts) or
            end > len(contexts)):
        answer_type = AnswerType.UNKNOWN
        start = 0
        end = 1
    else:
        answer_type = answer_types.get(input_text.lower(), AnswerType.SHORT)

    return Answer(answer_type, text=contexts[start:end], offset=start)
