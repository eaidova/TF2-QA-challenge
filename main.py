import os
import json
from argparse import ArgumentParser
import tensorflow as tf
from bert_utils import (
    model_fn_builder,
    input_fn_builder,
    FeatureWriter,
    predictions_to_dict,
    RawResult
)
from dataset import convert_examples_to_features, read_examples, read_candidates
from modeling import BertConfig
from tokenization import FullTokenizer


flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")


def build_arguments_parser():
    parser = ArgumentParser(description='Deep Learning accuracy validation framework', allow_abbrev=False)
    parser.add_argument(
        '--bert_config_file',
        help="The config json file corresponding to the pre-trained BERT model. This specifies the model architecture."
    )
    parser.add_argument('--vocab_file', help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument('--output_dir', help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--train_precomputed_file', help="Precomputed tf records for training.", required=False)
    parser.add_argument('--predict_file', help='json input file for predictions', required=False)
    parser.add_argument('--output_prediction_file', help='json file for prediction result')
    parser.add_argument('--init_checkpoint', help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument('--do_lower_case', action='store_true',
                        help="Whether to lower case the input text. Should be added for uncased models and "
                             "not specified for cased models.")
    parser.add_argument('--max_seq_length',
                        help="The maximum total input sequence length after WordPiece tokenization. "
                             "Sequences longer than this will be truncated, and sequences shorter "
                             "than this will be padded.",
                        type=int,
                        required=False,
                        default=512)
    parser.add_argument('--doc_stride', required=False, default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks."
                        )
    parser.add_argument('--max_query_length', required=False, default=64,
                        help="The maximum number of tokens for the question. Questions longer than "
                             "this will be truncated to this length.")
    parser.add_argument('--max_answer_length', required=False, default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed "
                             "because the start and end predictions are not conditioned on one another.")
    parser.add_argument('--max_position', required=False, default=50, type=int,
                        help="Maximum context position for which to generate special tokens.")
    parser.add_argument('--max_context', required=False, default=48, type=int,
                        help="Maximum number of contexts to output for an example.")
    parser.add_argument('--n_best_size', required=False, type=int, default=20,
                        help="The total number of n-best predictions to generate "
                             "in the nbest_predictions.json output file")
    parser.add_argument('--do_train', required=False, action='store_true', help='run training')
    parser.add_argument('--do_predict', required=False, action='store_true', help='make prediction')
    parser.add_argument('--train_batch_size', required=False, type=int, default=32)
    parser.add_argument('--predict_batch_size', required=False, type=int, default=8)
    parser.add_argument('--learning_rate', required=False, type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training."
                        )
    parser.add_argument('--skip_nested_contexts', required=False, action='store_true',
                        help="Completely ignore context that are not top level nodes in the page.")

    return parser


def main():
    parser = build_arguments_parser()
    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    num_train_steps = None
    num_warmup_steps = None
    if args.do_train:
        num_train_features = args.train_num_precomputed
        num_train_steps = int(num_train_features / args.train_batch_size * args.num_train_epochs)

        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(model_fn=model_fn,)

    if args.do_train:
        tf.logging.info("***** Running training on precomputed features *****")
        tf.logging.info("  Num split examples = %d", num_train_features)
        tf.logging.info("  Batch size = %d", args.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_filenames = tf.gfile.Glob(args.train_precomputed_file)
        train_input_fn = input_fn_builder(
            input_file=train_filenames,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if args.do_predict:
        if not args.output_prediction_file:
            raise ValueError("--output_prediction_file must be defined in predict mode.")

        eval_examples = read_examples(
            input_file=args.predict_file,  max_contexts=args.max_context,
            max_position=args.max_position, is_training=False
        )
        eval_writer = FeatureWriter(
            filename=os.path.join(args.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        num_spans_to_ids = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()
        eval_filename = eval_writer.filename

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", args.predict_batch_size)
        for spans, ids in num_spans_to_ids.iteritems():
            tf.logging.info("  Num split into %d = %d", spans, len(ids))

        predict_input_fn = input_fn_builder(
            input_file=eval_filename,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=False)

        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]
            all_results.append(RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    answer_type_logits=answer_type_logits))

        candidates_dict = read_candidates(args.predict_file)
        eval_features = [
            tf.train.Example.FromString(r)
            for r in tf.python_io.tf_record_iterator(eval_filename)
        ]
        predictions = predictions_to_dict(candidates_dict, eval_features, [r._asdict() for r in all_results])
        predictions_json = {"predictions": predictions.values()}
        with tf.gfile.Open(args.output_prediction_file, "w") as f:
            json.dump(predictions_json, f, indent=4)
