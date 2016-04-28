"""
utilities.py

Supplementary file for:
INFIXER: A METHOD FOR SEGMENTING NON-CONCATENATIVE MORPHOLOGY IN TAGALOG
by Steven Butler

Most of the contents of this file are variations on scripts
published as part of Morfessor on the Python Package Index. The license for
this code is reproduced below:

Morfessor
Copyright (total_cost) 2012, Sami Virpioja and Peter Smit
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function
from __future__ import unicode_literals

import logging
import math
import time

from morfessor.baseline import BaselineModel
from morfessor.exception import ArgumentException
from morfessor.io import MorfessorIO

_logger = logging.getLogger(__name__)


def morfessor_main(train_files, dampening, cycle, save_file=None):
    """Calls an implementation of the Morfessor model.

    :param dampening: 'none', 'ones', or 'log'
    :param train_files: input files for model training
    :param cycle: from {'init', 'test', 'final'}
    :param save_file: base name of output files (if needed)
    :return: trained morfessor.BaselineModel
    """

    # define input variables normally input at command line
    # all arguments are equal to their args.item equivalent in original
    # script's main()

    trainfiles = train_files    # input files for training
    progress = True             # show progress bar
    encoding = 'utf-8'          # if None, tries UTF-8 and/or local encoding
    cseparator = '\s+'          # separator for compound segmentation
    separator = None            # separator for atom segmentation
    lowercase = False           # makes all inputs lowercase
    forcesplit = ['-']          # list of chars to force a split on
    corpusweight = 1.0          # load annotation data for tuning the corpus weight param
    skips = False               # use random skips for frequently seen compounds to speed up training
    nosplit = None              # if the expression matches the two surrounding characters, do not allow splitting
    dampening = dampening       # 'none', 'ones', or 'log'
    algorithm = 'recursive'     # 'recursive' or 'viterbi'
    finish_threshold = 0.005    # train stops when the improvement of last iteration is smaller than this
    maxepochs = None            # ceiling on number of training epochs
    develannots = None          # boolean on whether to use dev-data file
    splitprob = None            # initialize new words by random split using given probability
    epochinterval = 10000       # epoch interval for online training
    algparams = ()              # set algorithm parameters; for this model, we are not using 'viterbi', nothing to set

    # Progress bar handling
    global show_progress_bar
    if progress:
        show_progress_bar = True
    else:
        show_progress_bar = False

    # build I/O and model
    io = MorfessorIO(encoding=encoding,
                     compound_separator=cseparator,
                     atom_separator=separator,
                     lowercase=lowercase)

    model = BaselineModel(forcesplit_list=forcesplit,
                          corpusweight=corpusweight,
                          use_skips=skips,
                          nosplit_re=nosplit)

    # Set frequency dampening function
    if dampening == 'none':
        dampfunc = None
    elif dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % dampening)

    # for use when building a new model or doing online training
    # this is the online+batch training model
    if len(trainfiles) > 0:

        time_start = time.time()

        data = io.read_corpus_files(trainfiles)
        epochs, total_cost = model.train_online(data, dampfunc, epochinterval,
                                                algorithm, algparams,
                                                splitprob, maxepochs)
        epochs, total_cost = model.train_batch(algorithm, algparams, develannots,
                                               finish_threshold, maxepochs)
        _logger.info("Epochs: %s" % epochs)

        time_end = time.time()
        _logger.info("Final cost: %s" % total_cost)
        _logger.info("Training time: %.3fs" % (time_end - time_start))

    else:
        _logger.warning("No training data files specified.")

    # if save file is present, write binary model to file
    if isinstance(save_file, str):

        outfile_bin = save_file + "_bin"
        io.write_binary_model_file(outfile_bin, model)

    # return model object for further manipulation
    return model
