import collections
import gzip
import json

import numpy as np
import tensorflow as tf
from dataset import AnswerType, Answer, convert_single_example, create_example_from_jsonl
from modeling import BertModel, get_shape_list
from tokenization import FullTokenizer, whitespace_tokenize


class DummyObject:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)


FLAGS=DummyObject(skip_nested_contexts=True,
                  max_position=50,
                  max_contexts=48,
                  max_query_length=64,
                  max_seq_length=512,
                  doc_stride=128,
                  include_unknowns=-1.0,
                  n_best_size=20,
                  max_answer_length=30)


class NqExample:
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


def make_nq_answer(contexts, answer):
    """Makes an Answer object following NQ conventions.

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


def read_nq_entry(entry, is_training):
    """Converts a NQ entry into a list of NqExamples."""

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
            answer = make_nq_answer(contexts, answer_dict)

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
                tf.compat.v1.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                                             cleaned_answer_text)
                continue

        questions.append(question_text)
        example = NqExample(
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


class CreateTFExampleFn:
    """Functor for creating NQ tf.Examples."""

    def __init__(self, is_training):
        self.is_training = is_training
        self.tokenizer = FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    def process(self, example):
        """Coverts an NQ example in a list of serialized tf examples."""
        nq_examples = read_nq_entry(example, self.is_training)
        input_features = []
        for nq_example in nq_examples:
            input_features.extend(convert_single_example(nq_example, self.tokenizer, self.is_training))

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

            yield tf.train.Example(features=tf.train.Features(
                feature=features)).SerializeToString()


def nq_examples_iter(input_file, is_training,tqdm=None):
    """Read a NQ json file into a list of NqExample."""
    input_paths = tf.io.gfile.glob(input_file)
    input_data = []

    def _open(path):
        if path.endswith(".gz"):
            return gzip.GzipFile(fileobj=tf.io.gfile.GFile(path, "rb"))
        else:
            return tf.io.gfile.GFile(path, "r")

    for path in input_paths:
        #tf.compat.v1.logging.info
        print("Reading: %s"% path)
        with _open(path) as input_file:
            if tqdm is not None:
                input_file = tqdm(input_file)
            for index, line in enumerate(input_file):
                entry = create_example_from_jsonl(line)
                yield read_nq_entry(entry, is_training)


def read_nq_examples(input_file, is_training,tqdm=None):
    """Read a NQ json file into a list of NqExample."""
    input_paths = tf.io.gfile.glob(input_file)
    input_data = []

    def _open(path):
        if path.endswith(".gz"):
            return gzip.GzipFile(fileobj=tf.io.gfile.GFile(path, "rb"))
        else:
            return tf.io.gfile.GFile(path, "r")

    for path in input_paths:
        tf.compat.v1.logging.info("Reading: %s", path)
        with _open(path) as input_file:
            if tqdm is not None:
                input_file = tqdm(input_file)
            for index, line in enumerate(input_file):
                input_data.append(create_example_from_jsonl(line))
                # if index > 100:
                #     break

    examples = []
    for entry in input_data:
        examples.extend(read_nq_entry(entry, is_training))
    return examples


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
    """Creates a classification model."""
    pooled_output, sequence_output = BertModel(config=bert_config)(
        input_word_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=segment_ids)

    # Get the logits for the start and end predictions.
    final_hidden = sequence_output #get_sequence_output()

    final_hidden_shape = get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.compat.v1.get_variable(
        "cls/nq/output_weights", [2, hidden_size],
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.compat.v1.get_variable(
        "cls/nq/output_bias", [2], initializer=tf.compat.v1.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(a=logits, perm=[2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # Get the logits for the answer type prediction.
    answer_type_output_layer = pooled_output #model.get_pooled_output()
    answer_type_hidden_size = answer_type_output_layer.shape[-1] #.value

    num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
    answer_type_output_weights = tf.compat.v1.get_variable(
        "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

    answer_type_output_bias = tf.compat.v1.get_variable(
        "answer_type_output_bias", [num_answer_types],
        initializer=tf.compat.v1.zeros_initializer())

    answer_type_logits = tf.matmul(
        answer_type_output_layer, answer_type_output_weights, transpose_b=True)
    answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                        answer_type_output_bias)

    return (start_logits, end_logits, answer_type_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_id = features["unique_id"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits, answer_type_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            model_tf = tf.keras.Model()
            checkpoint_tf = tf.train.Checkpoint(model=model_tf)
            checkpoint_tf.restore(init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
        #       else:
        #         tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                      init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = get_shape_list(input_ids)[1]

            # Computes the loss for positions.
            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    input_tensor=tf.reduce_sum(input_tensor=one_hot_positions * log_probs, axis=-1))
                return loss

            # Computes the loss for labels.
            def compute_label_loss(logits, labels):
                one_hot_labels = tf.one_hot(
                    labels, depth=len(AnswerType), dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    input_tensor=tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            answer_types = features["answer_types"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)
            answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

            total_loss = (start_loss + end_loss + answer_type_loss) / 3.0

            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_id": unique_id,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "answer_type_logits": answer_type_logits,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                             (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_id": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)
        name_to_features["answer_types"] = tf.io.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


class FeatureWriter:
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_id"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            features["answer_types"] = create_int_feature([feature.answer_type])
        else:
            token_map = [-1] * len(feature.input_ids)
            for k, v in feature.token_to_orig_map.items():
                token_map[k] = v
            features["token_map"] = create_int_feature(token_map)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def read_candidates_from_one_split(input_path):
    """Read candidates from a single jsonl file."""
    candidates_dict = {}
    print("Reading examples from: %s" % input_path)
    if input_path.endswith(".gz"):
        with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:
            for index, line in enumerate(input_file):
                e = json.loads(line)
                candidates_dict[e["example_id"]] = e["long_answer_candidates"]

    else:
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


class ScoreSummary:
    def __init__(self):
        self.predicted_label = None
        self.short_span_score = None
        self.cls_token_score = None
        self.answer_type_logits = None


def top_k_indices(logits,n_best_size,token_map):
    indices = np.argsort(logits[1:])+1
    indices = indices[token_map[indices]!=-1]
    return indices[-n_best_size:]


def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation."""
    predictions = []
    n_best_size = FLAGS.n_best_size
    max_answer_length = FLAGS.max_answer_length
    i = 0
    for unique_id, result in example.results.items():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = np.array(example.features[unique_id]["token_map"]) #.int64_list.value
        start_indexes = top_k_indices(result.start_logits,n_best_size,token_map)
        if len(start_indexes)==0:
            continue
        end_indexes   = top_k_indices(result.end_logits,n_best_size,token_map)
        if len(end_indexes)==0:
            continue
        indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))
        indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]
        for start_index,end_index in indexes:
            summary = ScoreSummary()
            summary.short_span_score = (
                    result.start_logits[start_index] +
                    result.end_logits[end_index])
            summary.cls_token_score = (
                    result.start_logits[0] + result.end_logits[0])
            summary.answer_type_logits = result.answer_type_logits-result.answer_type_logits.mean()
            start_span = token_map[start_index]
            end_span = token_map[end_index] + 1

            # Span logits minus the cls logits seems to be close to the best.
            score = summary.short_span_score - summary.cls_token_score
            predictions.append((score, i, summary, start_span, end_span))
            i += 1 # to break ties

    # Default empty prediction.
    score = -10000.0
    short_span = Span(-1, -1)
    long_span  = Span(-1, -1)
    summary    = ScoreSummary()

    if predictions:
        score, _, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
        short_span = Span(start_span, end_span)
        for c in example.candidates:
            start = short_span.start_token_idx
            end = short_span.end_token_idx
            ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
            if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                long_span = Span(c["start_token"], c["end_token"])
                break

    summary.predicted_label = {
        "example_id": int(example.example_id),
        "long_answer": {
            "start_token": int(long_span.start_token_idx),
            "end_token": int(long_span.end_token_idx),
            "start_byte": -1,
            "end_byte": -1
        },
        "long_answer_score": float(score),
        "short_answers": [{
            "start_token": int(short_span.start_token_idx),
            "end_token": int(short_span.end_token_idx),
            "start_byte": -1,
            "end_byte": -1
        }],
        "short_answer_score": float(score),
        "yes_no_answer": "NONE",
        "answer_type_logits": summary.answer_type_logits.tolist(),
        "answer_type": int(np.argmax(summary.answer_type_logits))
    }

    return summary


def compute_pred_dict(candidates_dict, dev_features, raw_results,tqdm=None):
    """Computes official answer key from raw logits."""
    raw_results_by_id = [(int(res.unique_id),1, res) for res in raw_results]

    examples_by_id = [(int(k),0,v) for k, v in candidates_dict.items()]

    features_by_id = [(int(d['unique_id']),2,d) for d in dev_features]

    # Join examples with features and raw results.
    examples = []
    print('merging examples...')
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    print('done.')
    for idx, type_, datum in merged:
        if type_==0: #isinstance(datum, list):
            examples.append(EvalExample(idx, datum))
        elif type_==2: #"token_map" in datum:
            examples[-1].features[idx] = datum
        else:
            examples[-1].results[idx] = datum

    # Construct prediction objects.
    print('Computing predictions...')

    nq_pred_dict = {}
    if tqdm is not None:
        examples = tqdm(examples)
    for e in examples:
        summary = compute_predictions(e)
        nq_pred_dict[e.example_id] = summary.predicted_label

    return nq_pred_dict
