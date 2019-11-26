import collections

import numpy as np
import tensorflow as tf
from dataset import AnswerType, EvalExample
from modeling import BertModel, get_shape_list, get_assignment_map_from_checkpoint
from optimization import create_optimizer


Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
    pooled_output, sequence_output = BertModel(config=bert_config)(
        input_word_ids=input_ids, input_mask=input_mask, input_type_ids=segment_ids)

    # Get the logits for the start and end predictions.
    final_hidden = sequence_output

    final_hidden_shape = get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.compat.v1.get_variable(
        "cls/nq/output_weights", [2, hidden_size], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
    )

    output_bias = tf.compat.v1.get_variable(
        "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # Get the logits for the answer type prediction.
    answer_type_output_layer = pooled_output
    answer_type_hidden_size = answer_type_output_layer.shape[-1]

    num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
    answer_type_output_weights = tf.compat.v1.get_variable(
        "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

    answer_type_output_bias = tf.compat.v1.get_variable(
        "answer_type_output_bias", [num_answer_types],
        initializer=tf.zeros_initializer())

    answer_type_logits = tf.matmul(answer_type_output_layer, answer_type_output_weights, transpose_b=True)
    answer_type_logits = tf.nn.bias_add(answer_type_logits, answer_type_output_bias)

    return start_logits, end_logits, answer_type_logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings):

    def model_fn(features, labels, mode, params):
        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        start_logits, end_logits, answer_type_logits = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = get_shape_list(input_ids)[1]

            # Computes the loss for positions.
            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            # Computes the loss for labels.
            def compute_label_loss(logits, labels):
                one_hot_labels = tf.one_hot(
                    labels, depth=len(AnswerType), dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            answer_types = features["answer_types"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)
            answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

            total_loss = (start_loss + end_loss + answer_type_loss) / 3.0

            train_op = create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "answer_type_logits": answer_type_logits,
            }
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.compat.v1.FixedLenFeature([], tf.int64),
        "input_ids": tf.compat.v1.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.compat.v1.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.compat.v1.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.compat.v1.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.compat.v1.FixedLenFeature([], tf.int64)
        name_to_features["answer_types"] = tf.compat.v1.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.compat.v1.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.compat.v1.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params.get("batch_size", 32)

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


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


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


def compute_predictions(example, n_best_size, max_answer_length):
    """Converts an example into an object for evaluation."""
    predictions = []
    n_best_size = n_best_size
    max_answer_length = max_answer_length
    i = 0
    for unique_id, result in example.results.items():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = np.array(example.features[unique_id]["token_map"]) #.int64_list.value
        start_indexes = top_k_indices(result.start_logits,n_best_size,token_map)
        if not start_indexes:
            continue
        end_indexes = top_k_indices(result.end_logits,n_best_size,token_map)
        if not end_indexes:
            continue
        indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))
        indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0] < max_answer_length)]
        for start_index,end_index in indexes:
            summary = ScoreSummary()
            summary.short_span_score = (result.start_logits[start_index] + result.end_logits[end_index])
            summary.cls_token_score = (result.start_logits[0] + result.end_logits[0])
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
    long_span = Span(-1, -1)
    summary = ScoreSummary()

    if predictions:
        score, _, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
        short_span = Span(start_span, end_span)
        for c in example.candidates:
            start = short_span.start_token_idx
            end = short_span.end_token_idx
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


def predictions_to_dict(candidates_dict, dev_features, raw_results, n_best_size, max_answer_length, tqdm=None):
    """Computes official answer key from raw logits."""
    raw_results_by_id = [(int(res.unique_id), 1, res) for res in raw_results]

    examples_by_id = [(int(k), 0, v) for k, v in candidates_dict.items()]

    features_by_id = [(int(d['unique_id']), 2, d) for d in dev_features]

    # Join examples with features and raw results.
    examples = []
    print('merging examples...')
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    print('done.')
    for idx, type_, datum in merged:
        if type_ == 0:  #isinstance(datum, list):
            examples.append(EvalExample(idx, datum))
        elif type_ == 2:  #"token_map" in datum:
            examples[-1].features[idx] = datum
        else:
            examples[-1].results[idx] = datum

    # Construct prediction objects.
    print('Computing predictions...')

    nq_pred_dict = {}
    if tqdm is not None:
        examples = tqdm(examples)
    for e in examples:
        summary = compute_predictions(e, n_best_size, max_answer_length)
        nq_pred_dict[e.example_id] = summary.predicted_label

    return nq_pred_dict
