from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    MT5600MModelMixin,
    MT51BModelMixin,
    MT53BModelMixin,
    MT513BModelMixin,
)
from lme.training_argument_mixins import MT5FinetuneArgsMixin, MT513BFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin
from lme.experiments.translation.pali import TranslationPali, TranslationPaliDevanagari
from lme.experiments.translation.chinese import TranslationChinese, TranslationChineseParagraph


class TranslationMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationMT5600MExperiment(MT5600MModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT51BExperiment(MT51BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT53BExperiment(MT53BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT513BExperiment(MT513BModelMixin, MT513BFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass

class TranslationPaliMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationPali, FinetuneExperimentBase):
    pass

class TranslationPaliMT5600MExperiment(MT5600MModelMixin, TranslationPaliMT5ExperimentBase):
    pass


class TranslationPaliDevanagariMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationPaliDevanagari, FinetuneExperimentBase):
    pass

class TranslationPaliDevanagariMT5600MExperiment(MT5600MModelMixin, TranslationPaliDevanagariMT5ExperimentBase):
    pass

class TranslationChineseMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationChinese, FinetuneExperimentBase):
    pass

class TranslationChineseMT5600MExperiment(MT5600MModelMixin, TranslationChineseMT5ExperimentBase):
    pass

class TranslationChineseParagraphMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationChineseParagraph, FinetuneExperimentBase):
    pass

class TranslationChineseParagraphMT5600MExperiment(MT5600MModelMixin, TranslationChineseParagraphMT5ExperimentBase):
    pass
