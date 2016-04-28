"""
eval_segment.py

Supplementary file for:
INFIXER: A METHOD FOR SEGMENTING NON-CONCATENATIVE MORPHOLOGY IN TAGALOG
by Steven Butler

Evaluation and segmentation classes to be used with trained Infixer models.

The InfixerEvaluation class utilizes some of Morfessor's built-in evaluation
tools in order to assess the quality of a particular trained model. It can take
a gold standard file as input and return it with the model's segmentations for
comparison.
"""

from __future__ import print_function
from __future__ import unicode_literals

import logging
import os.path
import tempfile

from morfessor.evaluation import EvaluationConfig, MorfessorEvaluation, FORMAT_STRINGS
from morfessor.io import MorfessorIO

from model import AffixFilter

_logger = logging.getLogger(__name__)


class InfixerEvaluation(object):
    """An object for evaluating modified Morfessor Baseline segmenters.

    Public functions:
        evaluate_model
        return_evaluation
        output_modified_gold_standard
    """

    def __init__(self, morfessor_model, feature_dict, affix_list):
        """Initialize an evaluation object with a model, feature dict, and affix list.

        :param morfessor_model: a trained Morfessor Baseline object
        :param feature_dict: the output dictionary from ModelBuilder object
        :param affix_list:
        """

        # save input
        self._model = morfessor_model
        self._feature_dict = feature_dict

        self._affix_filter = AffixFilter(affix_list)

        # set up morfessor's IO class
        self._io_manager = MorfessorIO()

    def _update_compounds(self, word):
        """Return the appropriate form the word in an annotation file.

        For words in the supplied feature dictionary (i.e., in vocabulary words),
        the final transformed word form is returned if it exists, and otherwise
        the same word is returned. If the word is OOV, it is filtered using the
        supplied list of affixes and returned.
        """

        # NOTE: leave "redundant parentheses" in return statements,
        # they are to ensure the output is a tuple

        # TODO: substitute 'word' in dict.get('final_word_base, word) with another thing to ensure it works correctly

        if word in self._feature_dict:

            _logger.debug("IN DICT: {} -> {}".format(word, self._feature_dict[word].get('final_word_base', word)))
            return ("IV", self._feature_dict[word].get('final_word_base', word))

        else:
            _logger.debug("OOV: {} -> {}".format(word, self._affix_filter.filter_word(word)))
            return ("OOV", self._affix_filter.filter_word(word))

    def _read_annotations_file(self, file_name, construction_separator=' ', analysis_sep=','):
        """Convert annotation file to generator.

        This is based off a method of the same name in the MorfessorIO class. It
        was modified so that the compound (for our purposes, the word being
        segmented) would undergo the same filtering as the training set, to ensure
        continuity.

        Each line in the segmentation file is expected to have the format:
        <compound> <constr1> <constr2>... <constrN>, <constr1>...<constrN>, ...
        """

        with open(file_name, 'r') as f:
            file_data = f.read().split('\n')
            annotation_list = [line for line in file_data if line != '']

        annotations = {}
        _logger.info(
            "Reading gold standard annotations from '%s'..." % file_name)
        for line in annotation_list:

            compound, analyses_line = line.split(None, 1)

            # apply filtering transformations if needed
            compound_mod = self._update_compounds(compound)

            if compound_mod not in annotations:
                annotations[compound_mod] = []

            if analysis_sep is not None:
                for analysis in analyses_line.split(analysis_sep):
                    analysis = analysis.strip()
                    annotations[compound_mod].append(analysis.strip().split(construction_separator))
            else:
                annotations[compound_mod].append(analyses_line.split(construction_separator))

        _logger.info("Done.")
        return annotations

    def evaluate_model(self, gold_standard_file, num_samples=10, sample_size=40):
        """Call the morfessor evaluator.

        :param gold_standard_file: a file with words and gold standard segmentations
        :param num_samples: the number of samples to be taken
        :param sample_size: the size of the samples to be taken
        """

        annotations = self._read_annotations_file(gold_standard_file)
        _logger.info(annotations)

        eval_obj = MorfessorEvaluation(annotations)
        results = eval_obj.evaluate_model(self._model, configuration=EvaluationConfig(num_samples, sample_size))

        print(results.format(FORMAT_STRINGS['default']))

    def return_evaluation(self, gold_standard_file, num_samples=10, sample_size=20):
        """Call the morfessor evaluator.

        :param gold_standard_file: a file with words and gold standard segmentations
        :param num_samples: the number of samples to be taken
        :param sample_size: the size of the samples to be taken
        """

        annotations = self._read_annotations_file(gold_standard_file)
        eval_obj = MorfessorEvaluation(annotations)
        results = eval_obj.evaluate_model(self._model, configuration=EvaluationConfig(num_samples, sample_size))

        return results

    def _process_segment_file(self, infile):
        """Process a word file to be segmented to ensure compatibility with the model."""

        with open(infile, 'r') as f:
            data = f.read().split('\n')

        data_filtered = []
        for word in data:
            data_filtered.append(self._update_compounds(word))

        temp_dir = tempfile.TemporaryDirectory()
        container_file = os.path.join(temp_dir.name, 'temp_file.txt')

        with open(container_file, 'w') as f:
            f.write('\n'.join(data_filtered))
            _logger.debug('\n'.join(data_filtered))

        return container_file

    def _filter_input_wordlist(self, in_list):
        """Process a word file to be segmented to ensure compatibility with the model."""

        data = in_list

        data_filtered = []
        for word in data:

            # filter empty strings
            if word == '':
                continue
            else:
                vocab_status, word_filtered = self._update_compounds(word)
                data_filtered.append((word, vocab_status, word_filtered))

        return data_filtered

    def _filter_input_list(self, infile):
        """Process a word file to be segmented to ensure compatibility with the model."""

        with open(infile, 'r') as f:
            data = f.read().split('\n')

        data_filtered = []
        for word in data:

            # filter empty strings
            if word == '':
                continue
            else:
                vocab_status, word_filtered = self._update_compounds(word)
                data_filtered.append((word, vocab_status, word_filtered))

        return data_filtered

    def _filter_gold_standard(self, infile):
        """Process a word file to be segmented to ensure compatibility with the model."""

        with open(infile, 'r') as f:
            data = f.readlines()

        data_filtered = []
        for line in data:

            # split word and segmentation
            word, gold_segmentation = line.split(None, 1)

            # filter empty strings
            if word == '':
                continue
            else:
                vocab_status, word_filtered = self._update_compounds(word)
                data_filtered.append((word, vocab_status, word_filtered, gold_segmentation.strip()))

        return data_filtered

    def output_modified_gold_standard(self, gs_infile, outfile):
        """Process a gold standard file and return it with model-supplied segmentations.

        :param gs_infile: a gold standard file to be processed
        :param outfile: the file where results should be written to
        """

        words_filtered = self._filter_gold_standard(gs_infile)

        with open(outfile, 'w') as f:

            for word, vocab_status, word_filtered, gs_segments in words_filtered:

                out_str = '{} {}\n'.format(word_filtered, gs_segments)
                out_str.encode('utf-8')
                f.write(out_str)

        _logger.info("Modified gold standard written to {}".format(outfile))

    def return_modified_gold_standard(self, gs_infile):
        """Process a gold standard file and return it with model-supplied segmentations.

        :param gs_infile: a gold standard file to be processed
        :param outfile: the file where results should be written to
        """

        words_filtered = self._filter_gold_standard(gs_infile)

        out_str = ""

        for word, vocab_status, word_filtered, gs_segments in words_filtered:

            out_str += '{} {}\n'.format(word_filtered, gs_segments)

        out_str.encode('utf-8')
        return out_str


class InfixerSegmenter(InfixerEvaluation):
    """An object that segments input using a trained Infixer model.

    Public functions:
        segment_word
        segment_file
        segment_gold_standard
    """

    def __init__(self, morfessor_model, feature_dict, affix_list):

        InfixerEvaluation.__init__(self, morfessor_model, feature_dict, affix_list)

    def segment_word(self, word, separators=None):
        """Segment a given word using a trained morfessor model.

        :param word: the input word string
        :param separators: a list of punctuation used to separate words (for filtering)
        """

        # set a default here to avoid issues with mutables in argument list
        if separators is None:
            separators = ['-']

        const_separator = ' '
        viterbi_smooth = 0
        viterbi_max_len = 30

        constructions, _ = self._model.viterbi_segment(word, viterbi_smooth, viterbi_max_len)
        constructions_filtered = [item for item in constructions if item not in separators]
        return const_separator.join(constructions_filtered)

    def segment_list(self, in_list, separator=','):
        """Segment an input file and write to an outfile.

        :param infile: an input file containing words to be segmented
        :param outfile: a filename for writing output
        :param separator: the separator for output lines, defaults to ',' mimicking csv
        """

        words_filtered = self._filter_input_wordlist(in_list)

        out_list = []

        for word, vocab_status, word_filtered in words_filtered:

            out_tuple = (word, self.segment_word(word_filtered))

            out_list.append(out_tuple)

        return out_list

    def segment_file(self, infile, outfile, separator=','):
        """Segment an input file and write to an outfile.

        :param infile: an input file containing words to be segmented
        :param outfile: a filename for writing output
        :param separator: the separator for output lines, defaults to ',' mimicking csv
        """

        words_filtered = self._filter_input_list(infile)

        with open(outfile, 'w') as f:

            for word, vocab_status, word_filtered in words_filtered:

                items = {'word': word,
                         'status': vocab_status,
                         'filtered': word_filtered,
                         'segments': self.segment_word(word_filtered),
                         'sep': separator}

                out_str = '{word}{sep}{status}{sep}{filtered}{sep}{segments}\n'.format(**items)

                f.write(out_str)

            _logger.info('Segmentations written to {}'.format(outfile))

    def segment_gold_standard(self, gs_infile, outfile, separator=','):
        """Segment an input gold standard file and write to an outfile.

        :param gs_infile: a gold standard file to be processed
        :param outfile: the file where results should be written to
        :param separator: the separator for output lines, defaults to ',' mimicking csv
        """

        words_filtered = self._filter_gold_standard(gs_infile)

        with open(outfile, 'w') as f:

            for word, vocab_status, word_filtered, gs_segments in words_filtered:

                items = {'word': word,
                         'status': vocab_status,
                         'filtered': word_filtered,
                         'gs_segments': "GS: " + gs_segments,
                         'segments': self.segment_word(word_filtered),
                         'sep': separator}

                out_str = '{word}{sep}{status}{sep}{filtered}{sep}{gs_segments}{sep}{segments}\n'.format(**items)
                f.write(out_str)

        _logger.info('Segmentations (gold standard) written to {}'.format(outfile))
