from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1BModelMixin, NLLB3BModelMixin
from lme.training_argument_mixins import NLLBFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin
from lme.experiments.translation.pali import TranslationPali, TranslationPaliDevanagari
from lme.experiments.translation.chinese import TranslationChinese, TranslationChineseParagraph

class TranslationNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationNLLB600MExperiment(NLLB600MModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB1BExperiment(NLLB1BModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB3BExperiment(NLLB3BModelMixin, TranslationNLLBExperimentBase):
    pass 

class TranslationPaliNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationPali, FinetuneExperimentBase):
    pass

class TranslationPaliNLLB600MExperiment(NLLB600MModelMixin, TranslationPaliNLLBExperimentBase):
    pass

class TranslationPaliDevanagariNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationPaliDevanagari, FinetuneExperimentBase):
    pass

class TranslationPaliDevanagariNLLB600MExperiment(NLLB600MModelMixin, TranslationPaliDevanagariNLLBExperimentBase):
    pass

class TranslationChineseNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationChinese, FinetuneExperimentBase):
    pass

class TranslationChineseNLLB600MExperiment(NLLB600MModelMixin, TranslationChineseNLLBExperimentBase):
    pass

class TranslationChineseParagraphNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationChineseParagraph, FinetuneExperimentBase):
    pass

class TranslationChineseParagraphNLLB600MExperiment(NLLB600MModelMixin, TranslationChineseParagraphNLLBExperimentBase):
    pass

