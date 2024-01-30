from datasets.utils.logging import set_verbosity_error

set_verbosity_error()

from lme.data_processors.translation import *
from lme.data_processors.sandhi import *
from lme.data_processors.pali import *
from lme.data_processors.chinese import *
from lme.data_processors.sanskrit import *
from lme.data_processors.mitra import *
from lme.data_processors.sentence_alignment import *