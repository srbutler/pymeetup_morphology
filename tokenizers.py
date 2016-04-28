"""
tokenizers.py

Supplementary file for:
INFIXER: A METHOD FOR SEGMENTING NON-CONCATENATIVE MORPHOLOGY IN TAGALOG
by Steven Butler

A collection of classes for processing and tokenizing different data input formats,
specifically Wikipedia corpora output from WikiExtractor (https://github.com/attardi/wikiextractor)
and IARPA Babel Program transcript files (http://www.iarpa.gov/index.php/research-programs/babel).
The output from these classes is intended for use with the InfixerModel object in preprocessor."""

from __future__ import print_function
from __future__ import unicode_literals

import collections
import glob
import logging
import os.path
import re

from nltk.corpus import names
from nltk.tokenize import wordpunct_tokenize

_logger = logging.getLogger(__name__)


class GeneralTokenizer(object):
    """A superclass for deriving specialized tokenizers.

    In its pure form, it will do a simple word-and-punctuation form of
    tokenization taken directly from NLTK. Tokenization is done at
    initialization, and optional filtering and editing can be done afterwards by
    calling the following methods:

    clean_tokens
    filter_tokens

    Output from this class can be called with the following methods:

    get_tokens
    output_text
    output_file_buffer

    """

    def __init__(self, target):
        """Initialize a GeneralTokenizer object with a file or a directory of files.

        :param target: a file or directory of files containing text to be tokenized.
        """

        if os.path.isdir(target):

            self.target = target        # stored for evaluation in other methods
            self.dir = target
            self._tokens = self._get_dir_tokens(self.dir)

        elif os.path.isfile(target):

            self.target = target        # stored for evaluation in other methods
            self.dir, self.filename = os.path.split(os.path.abspath(target))
            self._tokens = self._get_file_tokens(self.filename)

        else:
            raise ValueError("File or directory not readable.")

        self._data_len = len(self._tokens)

    def __len__(self):
        """Return the current number of tokens stored in the object."""

        return self._data_len

    # methods for extracting tokens from files and/or directories of files

    def _extract_tokens(self, file_text):
        """Extract tokens from a file and return a Counter dictionary.

        This method is designed specifically so that it can be overridden
        easily while maintaining _get_file_tokens and _get_dir_tokens.
        """

        token_dict = collections.Counter()

        # does a simple word and punctuation tokenization on the text
        tokens = wordpunct_tokenize(file_text)

        for token in tokens:
            token_dict[token] += 1

        return token_dict

    def _get_file_tokens(self, filename):
        """Get all unique tokens from a text file.

        This method needs to have a return instead of direct assignment to
        _tokens so that it can be called directly or as a subroutine
        of _get_dir_tokens, as needed.
        """

        with open(filename, 'r') as infile:
            infile_text = infile.read()

        # extract and add unique tokens to out_set
        token_dict = self._extract_tokens(infile_text)

        # logging
        filename = os.path.basename(infile.name)
        token_count = len(token_dict)

        # log info for each file in a dir only if logging.DEBUG
        if os.path.isdir(self.target):
            _logger.debug("{} token types in {}".format(token_count, filename))
        else:
            _logger.info("{} token types in {}".format(token_count, filename))

        return token_dict

    def _get_dir_tokens(self, directory):
        """Get all unique tokens from a directory of text files.

        This method needs to have a return instead of direct assignment to
        _tokens so that _get_file_tokens can be called directly or as
        a subroutine, as needed.
        """

        tokens_all = collections.Counter()

        files = glob.glob(directory + "*")

        for f in files:
            tokens_file = self._get_file_tokens(f)
            tokens_all.update(tokens_file)

        n_out = len(list(tokens_all.keys()))

        # logging
        _logger.info('{} token types found in {} files'.format(n_out, len(files)))

        return tokens_all

    # methods for removing various types of unwanted data

    def clean_tokens(self, rm_dupes=True, rm_names=True, rm_non_words=True,
                     rm_non_latin=True, rm_uppercase=True):
        """Call methods for removing various types of unwanted data in batch fashion.

        :param rm_dupes: remove duplicate upper-case tokens, preserving case and counts
        :param rm_names: remove names present in NLTK's names corpus
        :param rm_non_words: remove digits, non-alphanumeric tokens, and all-caps words
        :param rm_non_latin: remove non-Latin extended unicode characters
        :param rm_uppercase: remove upper-case words
        """

        if rm_dupes:
            self._remove_duplicates()

        if rm_names:
            self._remove_names()

        if rm_non_words:
            self._remove_non_words()

        if rm_non_latin:
            self._remove_non_latin()

        if rm_uppercase:
            self._remove_uppercase()

    def _remove_duplicates(self):
        """Remove duplicate upper-case tokens, preserving case and counts."""

        dupes = {key: count for (key, count) in self._tokens.items()
                 if key in self._tokens and key.lower() in self._tokens}

        no_dupes = {key: count for (key, count) in self._tokens.items()
                    if key not in dupes}

        # use Counter.update() method to preserve counts for duplicates
        dupes_lower = collections.Counter()

        for (key, count) in self._tokens.items():
            dupes_lower[key.lower()] = count

        no_dupes.update(dupes_lower)

        # logging
        _logger.info('{} duplicate tokens removed'.format(len(dupes)))

        self._tokens = collections.Counter(no_dupes)

    def _remove_names(self):
        """Remove names present in NLTK's names corpus."""

        name_set = set(names.words())

        no_names = {key: count for (key, count) in self._tokens.items()
                    if key not in name_set}

        # logging
        num_removed = len(self._tokens) - len(no_names)
        _logger.info(('{} name tokens removed').format(num_removed))

        self._tokens = collections.Counter(no_names)

    def _remove_non_words(self):
        """Remove digits, non-alphanumeric tokens, and all-caps words."""

        # pre-filter count of self.tokens for later comparison and logging
        base_len = len(self._tokens)

        regex = re.compile(r'(^\w*\d+\w*$|^\W*$|^[A-Z]*$|^.*_.*$)')

        matches_out = {key: count for (key, count) in self._tokens.items()
                       if regex.search(key) is None}

        # logging
        num_removed = len(self._tokens) - base_len
        _logger.info('{} non-word tokens removed'.format(num_removed))

        self._tokens = collections.Counter(matches_out)

    def _remove_non_latin(self):
        """Remove non-Latin extended unicode characters."""

        regex = re.compile(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]')

        matches_out = {key: count for (key, count) in self._tokens.items()
                       if regex.search(key) is None}

        _logger.info('Non-latin tokens removed')

        self._tokens = collections.Counter(matches_out)

    def _remove_uppercase(self):
        """Remove upper-case words. Only run AFTER remove_duplicates."""

        tokens_caps = {key for key in self._tokens.keys()
                       if key[0].isupper()}

        no_caps = {key: count for (key, count) in self._tokens.items()
                   if key not in tokens_caps}

        _logger.info('Uppercase tokens removed')

        self._tokens = collections.Counter(no_caps)

    def filter_tokens(self, filter_source, length=10000):
        """Filter the tokens using a specified outside word list."""

        with open(filter_source, 'r') as f:
            data = f.read()

        # structured for a file of entries in the form 'word ###\n'
        regex = re.compile(r'(\w*) \d*')
        filter_set = set(regex.findall(data)[:length])

        tokens_filtered = {key: count for (key, count) in self._tokens.items()
                           if key.lower() not in filter_set}

        _logger.info('Tokens filtered using {}'.format(filter_source))

        self._tokens = collections.Counter(tokens_filtered)

    # methods for writing the token sets to various types of files in various
    # configurations

    def get_tokens(self, output_type='items'):
        """Return tokens as a list object.

        :param output_type: 'items' (default), 'elements', or 'counts'
        """

        # create the correct output for type, or print error to screen
        if output_type == 'items':
            out_list = self._tokens.keys()

        elif output_type == 'elements':
            out_list = self._tokens.elements()

        elif output_type == 'counts':
            out_list = ['{}\t{}'.format(key, count) for (key, count)
                        in self._tokens.items()]
        else:
            err_msg = "output_type: 'items' (default), 'elements', 'counts'"
            raise ValueError(err_msg)

        return out_list

    def output_text(self, outfile, output_type='items'):
        """Output tokens to a text file.

        :param outfile: the file where tokens should be written
        :param output_type: 'items' (default), 'elements', or 'counts'
        """

        # create the correct output for type, or print error to screen
        if output_type == 'items':
            out_list = self._tokens.keys()

        elif output_type == 'elements':
            out_list = self._tokens.elements()

        elif output_type == 'counts':
            out_list = ['{}\t{}'.format(key, count) for (key, count)
                        in self._tokens.items()]
        else:
            err_msg = "output_type: 'items' (default), 'elements', 'counts'"
            raise ValueError(err_msg)

        with open(outfile, 'w') as out_file:
            out_file.write('\n'.join(out_list))

        out_msg = "{} tokens written to {}".format(len(self._tokens), outfile)

        _logger.info(out_msg)
        print(out_msg)


class WikipediaTokenizer(GeneralTokenizer):
    """A class for tokenizing the output from WikiExtractor corpora parsing.

    This class allows quick tokenization of Wikipedia corpora (dumps.wikimedia.org)
    that have been parsed using WikiExtractor (https://github.com/attardi/wikiextractor).
    It inherits all of the methods of GeneralTokenizer, with only a change to the
    private _extract_tokens method.
    """

    def __init__(self, target):
        """Initialize a WikipediaTokenizer object with a file or a directory of files.

        :param target: a file or directory of files containing text to be tokenized.
        """

        GeneralTokenizer.__init__(self, target)

    # methods for extracting tokens from files and/or directories of files

    def _extract_tokens(self, file_text):
        """Extract tokens from a file and return a Counter dictionary."""

        token_dict = collections.Counter()

        # matches and removes beginning and end tags
        regex = re.compile(r'(<doc id.*>|<\/doc>)')
        data = regex.sub('', file_text)

        tokens = wordpunct_tokenize(data)

        for token in tokens:
            token_dict[token] += 1

        return token_dict


class BabelTokenizer(GeneralTokenizer):
    """A class for tokenizing IARPA Babel Program audio transcript files.

    It inherits all of the methods of GeneralTokenizer, with only a change to the
    private _extract_tokens method.
    """

    def __init__(self, target):
        """Initialize a BabelTokenizer object with a file or a directory of files.

        :param target: a file or directory of files containing text to be tokenized.
        """

        GeneralTokenizer.__init__(self, target)

    def _extract_tokens(self, file_text):
        """Extract tokens from a Babel file and return a Counter dictionary."""

        token_dict = collections.Counter()

        # matches and removes beginning and end tags
        regex = re.compile(r'\[\d*\.\d*\]\n(.*)')
        matches = regex.findall(file_text)

        tokens = set()
        for match in matches:
            wp_tokenized = wordpunct_tokenize(match)
            tokens.update(wp_tokenized)

        for token in tokens:
            token_dict[token] += 1

        return token_dict
