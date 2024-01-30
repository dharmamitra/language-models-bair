from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import WMT21ModelMixin
from lme.training_argument_mixins import WMT21FinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin
from lme.experiments.translation.pali import TranslationPali, TranslationPaliDevanagari
from lme.experiments.translation.chinese import TranslationChinese, TranslationChineseParagraph, TranslationChineseLiterary


class TranslationChineseWMT21ExperimentBase(WMT21FinetuneArgsMixin, TranslationChinese, FinetuneExperimentBase):
    pass

class TranslationChineseWMT21Experiment(WMT21ModelMixin, TranslationChineseWMT21ExperimentBase):
    pass

class TranslationChineseParagraphWMT21ExperimentBase(WMT21FinetuneArgsMixin, TranslationChineseParagraph, FinetuneExperimentBase):
    pass

class TranslationChineseParagraphWMT21Experiment(WMT21ModelMixin, TranslationChineseParagraphWMT21ExperimentBase):
    pass

class TranslationChineseLiteraryWMT21ExperimentBase(WMT21FinetuneArgsMixin, TranslationChineseLiterary, FinetuneExperimentBase):
    pass

class TranslationChineseLiteraryWMT21Experiment(WMT21ModelMixin, TranslationChineseLiteraryWMT21ExperimentBase):
    pass