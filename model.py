"""
model.py

Supplementary file for:
INFIXER: A METHOD FOR SEGMENTING NON-CONCATENATIVE MORPHOLOGY IN TAGALOG
by Steven Butler

This file contains the primary parts of the program: the
InfixerModel class, which ultimately acts as a modified
MorfessorBaseline object designed to handle an input list of known
non-concatenative morphemes, and the AffixFilter class, which handles
the processing and filtering of those affixes.

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import json
import logging
import os.path
import pickle
import re
import tempfile

from utilities import morfessor_main

_logger = logging.getLogger(__name__)


class AffixFilter(object):
    """An object than can processes and filter affix regexps.

    This processes affixes for use with InfixerModel,
    InfixerEvaluation, and InfixerSegmenter. Additionally, it filters
    the feature dictionaries used by InfixerModel so that all affix
    processing can be handled from one class, allowing for
    modifications of affix processing over time without worrying about
    the model's design.

    The regular expressions should be ordered in descending priority;
    once a match is encountered, the filtering happens and the loop
    moves to the next word.

    Public functions:
        filter_word
        filter_feature_dictionary

    """

    def __init__(self, affix_list):
        """Initialize an AffixFilter object using an order list of affix regular expressions.

        :param affix_list: a list of affix regular expressions for search and filtering
        """

        if isinstance(affix_list, list):
            self.affix_list = affix_list
            self.formatted_affixes = self._format_affixes(affix_list)
        else:
            raise ValueError("Input must be a list.")

    @staticmethod
    def _make_group_pattern(regex):
        """Build an appropriate group string for an affix regular expression.

        Morfessor's default force-split character is '-', so the rearranged morphemes
        are separated with that in order to ensure correct segmentation.

        :param regex: regular expressions for an affix
        :return: a string containing the correct replacement form for the input regex
        """

        if re.search('CON', regex.pattern):
            out_str = r'<redup>-<\g<2>>-\g<4>'

        elif re.search('REDUP', regex.pattern):
            out_str = r'<redup>-\g<2>'

        else:
            out_str = r'<\g<2>>-\g<1>\g<3>'

        return out_str

    def _format_affixes(self, affix_list):
        """Reformat input affix patterns into something usable by the class.

        :param affix_list: a list of regular expression defining affixes
        :return: a list of tuples containing a compiled regular expression
                 and a group pattern to preserve a word without that affix
        """

        # TODO: make this method applicable to more languages than Tagalog

        affix_tuples = list()

        for affix in affix_list:

            # compiled regular expression
            regex_affix = re.compile(affix)

            # replacement form, using customizable method
            regex_repl = self._make_group_pattern(regex_affix)

            # whether reduplication tag exists
            if re.search('REDUP', regex_affix.pattern):
                has_redup = True
            elif re.search('CON', regex_affix.pattern):
                has_redup = True
            else:
                has_redup = False

            # orthographic form of the infix/affix
            find_affix = re.search('.*?([a-z]{2}).*', regex_affix.pattern)
            affix_form = find_affix.group(1) if find_affix else None

            # return all as a 4-tuple
            affix_tuples.append((regex_affix, regex_repl, has_redup, affix_form))

        return affix_tuples

    def filter_word(self, word):
        """Return a copy of the word with all affixes filtered appropriately, if applicable.

        :param word: a word string to be filtered
        """

        word_filtered = word

        for affix_tuple in self.formatted_affixes:

            # unpack return from _format_affixes, discard unneeded parts
            affix_regex, affix_repl, *_ = affix_tuple

            if affix_regex.search(word):

                word_filtered = affix_regex.sub(affix_repl, word)

                out_msg = "'{}': match on affix '{}', returning: '{}'".format(word, affix_regex.pattern, word_filtered)
                _logger.debug(out_msg)

                break

        return word_filtered

    def filter_feature_dictionary(self, feature_dictionary):
        """Cycle through each word and remove the affixes if present.

        :param feature_dictionary: the feature dictionary from an InfixerModel object
        """

        for word in feature_dictionary:

            for affix_tuple in self.formatted_affixes:

                # unpack return from _format_affixes
                affix_regex, affix_repl, has_redup, affix_form = affix_tuple

                _logger.debug("Testing affix '{}' on word '{}'".format(
                    affix_regex.pattern, word))

                if affix_regex.search(word):

                    # the affixes are mutually exclusive, if a filter matches,
                    # the loop breaks and the word is sent to be appended
                    # TODO: make this more generally applicable

                    feature_dictionary[word]['test_infix'] = affix_form
                    feature_dictionary[word]['test_has_redup'] = has_redup
                    feature_dictionary[word]['test_transformed'] = affix_regex.sub(affix_repl, word)

                    break

        _logger.info("Feature dictionary updated with affix testing.")
        return feature_dictionary


class InfixerModel(object):
    """A modified Morfessor-type model for use with a list of affixes.

    This model is built around the already extant Python
    implementation of the Morfessor segmentation algorithm. Its
    purpose is to modify that model to better handle the presence of
    known non-concatenative morphological patterns. This process
    involves three training cycles (each involving storage in the
    _feature_dict data structure):

        1. init: train an unmodified Morfessor model, store data in
            _feature_dict

        2. test: train a Morfessor model using a modified corpus
            containing every possible instance of any of the input
            affixes

        3. final: train the final Morfessor model on a modified corpus
            containing the most probable instances of the input
            affixes

    Public methods:

    modeling:
        build_test_model
        build_final_model

    loading previous models:
        get_feature_dict_from_file
        load_init_json
        load_test_json
        load_final_json

    getting and writing output:
        feature_dictionary
        get_model
        write_feature_dict
        write_changed_tokens

    """

    def __init__(self, word_list_file, affix_list, dampening='none'):
        """Initialize an InfixerModel and run the initial training step.

        :param word_list_file: a text file containing one word per row
        :param affix_list: an ordered list of regexps
        :param dampening: 'none' (default), 'ones', or 'log'
        """

        self.cycle_name = 'INIT'
        self.dampening = dampening

        # handle affixes
        self._affix_filter = AffixFilter(affix_list)

        with open(word_list_file, 'r') as f:
            training_words = f.read().split('\n')

        # build initial morfessor.Baseline model, init other models
        _logger.info("INIT CYCLE: training Morfessor Baseline model")
        self.model_init = self._call_morfessor(training_words, save_file=None)
        self.model_test = None
        self.model_final = None

        # extract feature dictionary from initial model
        self._build_feature_dict(self.model_init)

    # ---------------------- feature dictionary extraction -------------------

    def _call_morfessor(self, train_words, save_file):
        """Call the Morfessor Baseline model main on the supplied word list.

        Because of certain issues with formatting, the input for the Morfessor
        model must be saved to a temporary file to be read into the model.

        :param train_words: an iterable containing word strings for retraining
        :param save_file: an optional save file to write the binary to
        :return: trained Morfessor model
        """

        temp_dir = tempfile.TemporaryDirectory()
        container_file = os.path.join(temp_dir.name, 'temp_file.txt')

        with open(container_file, 'w') as f:
            f.write('\n'.join(train_words))

        # NOTE: input file must always be a list! the call fails otherwise
        model = morfessor_main([container_file], self.dampening, self.cycle_name, save_file)

        return model

    def _extract_features(self, word_tuple, segment_sep='+'):
        """Extract relevant features from a Morfessor model's segmentation output."""

        # store word count, init_segments, and base form
        word_count, segment_list = word_tuple
        word_base = re.sub(r'\+', '', ''.join(segment_list))

        # append morpheme boundary markers (+)
        if len(segment_list) > 1:
            temp_seg = []

            for segment in segment_list[1:-1]:

                temp_s = segment_sep + segment + segment_sep
                temp_seg.append(temp_s)

            temp_seg.insert(0, segment_list[0] + segment_sep)
            temp_seg.append(segment_sep + segment_list[-1])
            segment_list = temp_seg

        # get hypothesized init_root, remove its '+'
        # TODO: account for morphs of the same length
        # TODO: account for problem prefixes like 'nakaka-'
        # DONE: remove <affix> forms as a choice (attempted below)
        real_segments = [seg for seg in segment_list if '<' not in seg]
        morph_list = sorted(real_segments, key=len)
        word_root = morph_list.pop().replace('+', '')

        return word_count, word_base, word_root, segment_list

    def _build_feature_dict(self, morfessor_model):
        """Extract a nested dictionary of features (init_root, init_segments, count) for each word.

        :param morfessor_model: a trained MorfessorBaseline model object
        """

        # get segmentations from morfessor.Baseline model
        segmentation_list = morfessor_model.get_segmentations()

        # set a default dict with default arg as an empty dictionary
        feature_dict = collections.defaultdict(dict)

        for word_tuple in segmentation_list:

            wc, word_base, root, segments = self._extract_features(word_tuple)

            # for the __init__ cycle
            if self.cycle_name == 'INIT':

                # construct the initial dictionary
                feature_dict[word_base] = dict(count=wc,
                                               init_word_base=word_base,
                                               init_segments=segments,
                                               init_root=root)

                # store dictionaries in class variables
                self._feature_dict = feature_dict

            # second run-through, add new values to _feature_dict
            elif self.cycle_name == 'TEST':

                self._feature_dict[word_base]['test_segments'] = segments
                self._feature_dict[word_base]['test_root'] = root

            elif self.cycle_name == 'FINAL':

                self._feature_dict[word_base]['final_segments'] = segments
                self._feature_dict[word_base]['final_root'] = root

        if self.cycle_name == 'INIT':
            _logger.info("Feature dictionary extracted.")
        elif self.cycle_name == 'TEST':
            _logger.info("Feature dictionary updated with test values.")
        elif self.cycle_name == 'FINAL':
            _logger.info("Feature dictionary updated with final values.")

    # ---------------------- internal getters ----------------------

    def _get_init_roots(self):
        """Get Counter for all values of init_root from the feature dictionary."""

        roots_list = [self._feature_dict[word]['init_root']
                      for word in self._feature_dict]

        return collections.Counter(roots_list)

    def _get_morphemes(self):
        """Get Counter for all morphemes in the init_segments value of the feature dictionary."""

        morph_counter = collections.Counter()

        for word in self._feature_dict:
            morph_counter.update(self._feature_dict[word]['init_segments'])

        return morph_counter

    # ---------------------- filtering and retraining ------------------------

    @staticmethod
    def _flatten_to_generator(iterable):
        """Return a flattened generator for an iterable of mixed iterables and non-iterables.

        :param iterable: an iterable with any combination of iterable and non-iterable components
        """

        for item in iterable:
            if isinstance(item, list) or isinstance(item, tuple):
                for sub_item in item:
                    yield sub_item
            else:
                yield item

    def _flatten_segments(self, model_segments):
        """Flatten init_segments from morfessor baseline model output.

        :param model_segments: output from morfessor.Baseline.get_segmentations()
        :return: a list of flattened segmentation tuples
        """

        segments_out = []

        for tup in model_segments:

            tup_flat = self._flatten_to_generator(tup)
            segments_out.append(list(tup_flat))

        return segments_out

    def _filter_affixes(self):
        """Rebuild the feature dictionary with filtered forms for relevant words."""

        feature_dict_filtered = self._affix_filter.filter_feature_dictionary(self._feature_dict)
        self._feature_dict = feature_dict_filtered

    def _get_retrain_words(self):
        """Get right words for retrain and a dict with transformed-to-original mapping.

        :return retrain_words: a list of the original training words, with transformed
                               versions substituted when they exist
        :return original_transformed_map: a dictionary of transformed_word:original_word
                                          pairs, so that the _feature_dict can be repaired
        """

        retrain_words = list()
        original_transformed_map = dict()

        # determine lookup based on phase so this method can be reused
        if self.cycle_name == 'FINAL':
            lookup_feature = 'final_word_base'
        else:
            lookup_feature = 'test_transformed'

        for word in self._feature_dict:

            # if transformed exists, append it to retrain_words and add a dict
            # entry
            if self._feature_dict[word].get(lookup_feature, None):

                transformed = self._feature_dict[word][lookup_feature]
                original_transformed_map[transformed] = word
                retrain_words.append(transformed)

            else:
                retrain_words.append(word)

        return retrain_words, original_transformed_map

    def _rebuild_original_words(self, transformed_original_dict):
        """Remove transformed mappings as values and store results under original base form.

        :param transformed_original_dict: dictionary of {transformed_word: original_word} pairs
        """

        counter = 0
        if self.cycle_name == 'FINAL':
            feature_keys = ['final_root', 'final_segments']
            cycle_name = 'FINAL CYCLE'
        else:
            # TODO: insert feature keys from third part of __init__ recall
            feature_keys = ['test_infix', 'test_has_redup',
                            'test_transformed', 'test_segments', 'test_root']
            cycle_name = 'TEST CYCLE'

        # temporary dictionary to prevent RuntimeErrors due to dictionary
        # changing size
        feature_dict_copy = self._feature_dict.copy()

        for word in feature_dict_copy:

            # checks to see if key is in _get_retrain_words output
            if word in transformed_original_dict.keys():

                base_form = transformed_original_dict[word]
                transformed_values = self._feature_dict.get(word)

                # moves the values from the second cycle into the correct
                # _feature_dict entry
                for val in feature_keys:

                    # prevent overwriting extant values with None
                    dict_val = transformed_values.get(val, None)

                    if dict_val is not None:
                        self._feature_dict[base_form][val] = dict_val

                # remove transformed word
                self._feature_dict.pop(word)

                counter += 1

        _logger.info('{}: {} word forms rebuilt in the feature dictionary.'.format(cycle_name, counter))

    def _set_retrain_values(self):
        """Set counts and probabilities for each root affected by affix filter.

        This method checks to see whether the hypothesized root from
        transformed version of the word base can be found in the
        original data set. The assumption is that if a root from a
        transformed word is found in the original data set, then there
        is a higher likelihood that the transformation was a good
        idea.

        The "percent" value is a mostly arbitrary stand-in for some
        better method of evaluating relative likelihood. The more
        frequent a root is found in the original data set, the more
        likely it is that it is not a mistaken form.

        """

        init_roots = self._get_init_roots()     # returns a Counter object
        roots_sum = sum(init_roots.values())    # for root percentages

        for word in self._feature_dict:

            test_root = self._feature_dict[word].get('test_root', None)
            test_transformed = self._feature_dict[
                word].get('test_transformed', None)

            # only words with a found affix have a test_transformed feature
            if test_transformed is not None:

                test_root_count = init_roots.get(test_root, 0)
                self._feature_dict[word]['test_root_count'] = test_root_count
                self._feature_dict[word][
                    'test_root_per'] = test_root_count / roots_sum

        _logger.debug(
            'Counts and percents set for roots after training with transformed forms.')

    def _assign_final_values(self, threshold=0):
        """Assigns the final versions of roots and segmentations using retrain values.

        This method uses the values assigned in _get_retrain_values to
        determine which segmentation will be kept in the final model.

        :param: threshold: default=0; lower bound on using transformed form

        """

        # TODO: consider putting in a feature that cancels any alteration

        changed_words = list()

        for word in self._feature_dict:

            _logger.debug("Processing word: '{}'".format(word))
            features = self._feature_dict[word]

            if features.get('test_root_count', None) is not None:

                if features['test_root_count'] > threshold:

                    new_word_base = features['test_transformed']
                    self._feature_dict[word]['final_word_base'] = new_word_base

                    changed_words.append((word, new_word_base))

                    _logger.debug("TRAINING WORD CHANGED: '{}' changed to '{}'".format(
                        word, new_word_base))

                else:
                    _logger.debug("UNCHANGED: '{}', COUNT: {}".format(
                        word, features['test_root_count']))

            else:
                # _logger.info("NO CHANGE NEEDED: '{}'".format(word))
                pass

        unchanged_words = len(list(self._feature_dict)) - len(changed_words)
        _logger.info("{} tokens changed, {} tokens unchanged".format(
            len(changed_words), unchanged_words))

        self.word_changes = changed_words

    def _clean_segments(self, separator_list):
        """Remove unwanted characters from segment lists."""

        segment_keys = ['test_segments', 'final_segments']

        for word in self._feature_dict:

            for seg_key in segment_keys:

                segments = self._feature_dict[word].get(seg_key, None)

                if segments is not None:

                    # ensure the checked item is a list
                    assert(isinstance(segments, list))

                    segments_clean = [item for item in segments if item not in separator_list]
                    self._feature_dict[word][seg_key] = segments_clean

    # -------------------------- model building -------------------------

    def _build_model(self, save_file):
        """Call the Morfessor Baseline for the appropriate retraining."""

        # get training word set and key for remapping after the cycle
        words_for_retraining, transformed_mapping = self._get_retrain_words()

        # train model
        model = self._call_morfessor(words_for_retraining, save_file)

        # rebuild the feature dictionary from the second run
        self._build_feature_dict(model)

        # repair self._feature_dict and correctly maps new values to right keys
        self._rebuild_original_words(transformed_mapping)

        return model

    def build_test_model(self, save_file=None):
        """Run the Morfessor Baseline model on the transformed data.

        This method creates a second model, modified from the first,
        where the transformed words are substituted in place of their
        original forms. After the model is built, it is run through
        feature extraction again, only this time the values found are
        saved as updates in the main feature dictionary
        (_feature_dict). The model is then repaired by using a
        correspondence dictionary between the transformed forms and
        the original forms.

        :param save_file:

        """

        # set tracker variable
        self.cycle_name = 'TEST'

        # process and filter affixes from feature dictionary
        self._filter_affixes()

        # train model
        _logger.info("TEST CYCLE: training Morfessor Baseline model")
        self.model_test = self._build_model(save_file)

        _logger.info("Test model built.")

    def build_final_model(self, threshold=0, save_file=None):
        """Build the final model using the transformed and tested word list.

        :param threshold: default=0; lower bound on using transformed form
        :param save_file:
        """

        # set tracker variable
        self.cycle_name = 'FINAL'

        # sets up test_root counts and percents
        self._set_retrain_values()

        # determines final training values based on threshold
        self._assign_final_values(threshold)

        # train model
        _logger.info("FINAL CYCLE: training Morfessor Baseline model")
        self.model_final = self._build_model(save_file)

        # clean up segment lists
        self._clean_segments(separator_list=['-'])

        _logger.info("Final model built.")

    # ---------------------------- loading methods ---------------------------

    @staticmethod
    def _load_json_dict(in_file):
        """Build a default dict from a JSON input file."""

        with open(in_file, 'r') as f:
            json_dict = json.load(f)

        json_defdict = collections.defaultdict(dict, json_dict)

        _logger.info("Feature dictionary loaded from '{}'".format(in_file))

        return json_defdict

    @classmethod
    def get_feature_dict_from_file(cls, in_file):

        return cls._load_json_dict(in_file)

    def load_init_json(self, in_file):

        self._feature_dict = self._load_json_dict(in_file)
        self.cycle_name = 'INIT'

    def load_test_json(self, in_file):

        self._feature_dict = self._load_json_dict(in_file)
        self.cycle_name = 'TEST'

    def load_final_json(self, in_file):

        self._feature_dict = self._load_json_dict(in_file)
        self.cycle_name = 'FINAL'

    # ---------------------------- output methods ----------------------------

    def feature_dictionary(self):
        """Return the feature dictionary for the input file."""

        return self._feature_dict

    def get_model(self, which_model):
        """Return a trained Morfessor model for the given cycle.

        :param which_model: 'init', 'test', or 'train'
        """

        if which_model.lower() == 'init':
            return self.model_init

        elif which_model.lower() == 'test':
            return self.model_test

        elif which_model.lower() == 'final':
            return self.model_final

    def write_feature_dict(self, out_file, output_format):
        """Write feature set to output format (JSON).

        :param out_file: the destination file; do not use file extension
        :param output_format: JSON or pickle
        """

        if output_format.lower() not in {'json', 'pickle'}:
            _logger.error(
                'ERROR: unrecognized output format: {}'.format(output_format))
            raise ValueError("output_format: {'json', 'pickle'}")

        elif output_format.lower() == 'json':

            with open(out_file + '.json', 'w') as f:
                json.dump(self._feature_dict, f)

        elif output_format.lower() == 'pickle':

            with open(out_file + '.pickle', 'w') as f:
                pickle.dump(self._feature_dict, f)

        out_name = os.path.basename(out_file)
        out_msg = "Feature set dictionary written to {}".format(
            out_name + '.' + output_format.lower())
        _logger.info(out_msg)

    def write_changed_tokens(self, out_file):
        """Write a text file of the final word set, with changes marked.

        :param out_file: the text file to be written to
        """

        with open(out_file, 'w') as f:
            for item in self.word_changes:
                f.write('{}\t{}\n'.format(item[0], item[1]))

        out_name = os.path.basename(out_file)
        _logger.info("Word changes list written to {}".format(out_name))
