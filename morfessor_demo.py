#!/usr/bin/env python

"""This is a demo of the Morfessor library using sample files from
Morpho Challenge 2005. To try out this data, download it from:

http://research.ics.aalto.fi/events/morphochallenge2005/datasets.shtml
"""

From __future__ import print_function, unicode_literals
import logging

import morfessor

# for logging
_logger = logging.getLogger(__name__)


def train_model(input_file, output_file=None):

    # setup input and model objects
    morf_io = morfessor.MorfessorIO()
    morf_model = morfessor.BaselineModel()

    # build a corpus from input file
    train_data = morf_io.read_corpus_file(input_file)

    # load data into model
    # optional param "count_modifier" can set frequency dampening;
    # default is each token counts
    morf_model.load_data(train_data)

    # train the model in batch form (online training also available)
    morf_model.train_batch()

    # optionally pickle model
    if output_file is not None:
        morf_io.write_binary_model_file(output_file, morf_model)

    return morf_model


def test_model(model, gold_standard_file):

    # load IO object
    morf_io = morfessor.MorfessorIO()

    # load gold standard annotations file
    gold_standard = morf_io.read_annotations_file(gold_standard_file)

    # build evaluator object and run evaluation against gold standard
    evaluator = morfessor.MorfessorEvaluation(gold_standard)
    results = evaluator.evaluate_model(model)

    return results


def main():

    # logging makes the console output a lot more interesting
    default_formatter = logging.Formatter('%(asctime)s - %(message)s',
                                          '%Y-%m-%d %H:%M:%S')
    logging.basicConfig(level=logging.INFO)
    main_logger = logging.StreamHandler()
    main_logger.setLevel(logging.INFO)
    main_logger.setFormatter(default_formatter)
    _logger.addHandler(main_logger)

    # train, write, and test model
    # model_finnish = train_model("wordlist.fin", "trainedmodel.fin")
    # results_finnish = test_model("goldstdsample.fin.txt", model_finnish)

    model_english = train_model("wordlist.eng", "trainedmodel.eng")
    model_turkish = train_model("wordlist.tur", "trainedmodel.tur")

    # print(results_finnish)

if __name__ == '__main__':
    main()
    
