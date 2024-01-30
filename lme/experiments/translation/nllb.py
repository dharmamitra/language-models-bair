from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import NLLB600MModelMixin, NLLB1BModelMixin, NLLB3BModelMixin
from lme.training_argument_mixins import NLLBFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin, TranslationMixinReverse
from lme.experiments.translation.pali import TranslationPali, TranslationPaliDevanagari
from lme.experiments.translation.chinese import TranslationChinese, TranslationEnglishChinese, TranslationChineseParagraph, TranslationChineseLiterary, TranslationChineseKorean, TranslationChineseModern
from lme.experiments.translation.tibetan import TranslationTibetanParagraph, TranslationTibetanEnglishParagraph
from lme.experiments.translation.sanskrit import TranslationSanskritParagraph, TranslationSanskitItihasa, TranslationSanskritItihasaTagged
from lme.experiments.translation.mitra import TranslationMitra

class TranslationNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass

class TranslationNLLBReverseExperimentBase(NLLBFinetuneArgsMixin, TranslationMixinReverse, FinetuneExperimentBase):
    pass

class TranslationNLLB600MExperiment(NLLB600MModelMixin, TranslationNLLBExperimentBase):
    pass


class TranslationNLLB1BExperiment(NLLB1BModelMixin, TranslationNLLBExperimentBase):
    pass



class TranslationNLLB3BExperiment(NLLB3BModelMixin, TranslationNLLBExperimentBase):
    pass 

class TranslationNLLB600MReverseExperiment(NLLB600MModelMixin, TranslationNLLBReverseExperimentBase):
    pass

class TranslationNLLB1BReverseExperiment(NLLB1BModelMixin, TranslationNLLBReverseExperimentBase):
    pass

class TranslationNLLB3BReverseExperiment(NLLB3BModelMixin, TranslationNLLBReverseExperimentBase):
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

class TranslationChineseNLLB1BExperiment(NLLB1BModelMixin, TranslationChineseNLLBExperimentBase):
    pass

class TranslationChineseNLLB3BExperiment(NLLB3BModelMixin, TranslationChineseNLLBExperimentBase):
    pass

class TranslationEnglishChineseNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationEnglishChinese, FinetuneExperimentBase):
    pass

class TranslationEnglishChineseNLLB600MExperiment(NLLB600MModelMixin, TranslationEnglishChineseNLLBExperimentBase):
    pass

class TranslationEnglishChineseNLLB1BExperiment(NLLB1BModelMixin, TranslationEnglishChineseNLLBExperimentBase):
    pass

class TranslationChineseParagraphNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationChineseParagraph, FinetuneExperimentBase):
    pass

class TranslationChineseParagraphNLLB600MExperiment(NLLB600MModelMixin, TranslationChineseParagraphNLLBExperimentBase):
    pass

class TranslationChineseParagraphNLLB1BExperiment(NLLB1BModelMixin, TranslationChineseParagraphNLLBExperimentBase):
    pass

class TranslationChineseParagraphNLLB3BExperiment(NLLB3BModelMixin, TranslationChineseParagraphNLLBExperimentBase):
    pass

class TranslationChineseLiteraryNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationChineseLiterary, FinetuneExperimentBase):
    pass

class TranslationChineseLiteraryNLLB600MExperiment(NLLB600MModelMixin, TranslationChineseLiteraryNLLBExperimentBase):
    pass

class TranslationChineseLiteraryNLLB1BExperiment(NLLB1BModelMixin, TranslationChineseLiteraryNLLBExperimentBase):
    pass

class TranslationChineseModernNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationChineseModern, FinetuneExperimentBase):
    pass

class TranslationChineseModernNLLB600MExperiment(NLLB600MModelMixin, TranslationChineseModernNLLBExperimentBase):
    pass

class TranslationChineseModernNLLB1BExperiment(NLLB1BModelMixin, TranslationChineseModernNLLBExperimentBase):
    pass

class TranslationChineseKoreanNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationChineseKorean, FinetuneExperimentBase):
    pass

class TranslationChineseKoreanNLLB600MExperiment(NLLB600MModelMixin, TranslationChineseKoreanNLLBExperimentBase):
    pass

class TranslationTibetanParagraphNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationTibetanParagraph, FinetuneExperimentBase):
    pass

class TranslationTibetanParagraphNLLB1BExperiment(NLLB1BModelMixin, TranslationTibetanParagraphNLLBExperimentBase):
    pass

class TranslationTibetanParagraphNLLB3BExperiment(NLLB3BModelMixin, TranslationTibetanParagraphNLLBExperimentBase):
    pass

class TranslationTibetanEnglishParagraphNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationTibetanEnglishParagraph, FinetuneExperimentBase):
    pass


class TranslationTibetanEnglishParagraphNLLB1BExperiment(NLLB1BModelMixin, TranslationTibetanEnglishParagraphNLLBExperimentBase):
    pass

class TranslationTibetanEnglishParagraphNLLB3BExperiment(NLLB3BModelMixin, TranslationTibetanEnglishParagraphNLLBExperimentBase):
    pass

class TranslationSanskritParagraphNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationSanskritParagraph, FinetuneExperimentBase):
    pass

class TranslationSanskritParagraphNLLB1BExperiment(NLLB1BModelMixin, TranslationSanskritParagraphNLLBExperimentBase):
    pass

class TranslationSanskritParagraphNLLB3BExperiment(NLLB3BModelMixin, TranslationSanskritParagraphNLLBExperimentBase):
    pass

class TranslationSanskritItihasaNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationSanskitItihasa, FinetuneExperimentBase):
    pass

class TranslationSanskritItihasaNLLB1BExperiment(NLLB1BModelMixin, TranslationSanskritItihasaNLLBExperimentBase):
    pass

class TranslationSanskritItihasaNLLB3BExperiment(NLLB3BModelMixin, TranslationSanskritItihasaNLLBExperimentBase):
    pass

class TranslationSanskritItihasaTaggedNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationSanskritItihasaTagged, FinetuneExperimentBase):
    pass

class TranslationSanskritItihasaTaggedNLLB1BExperiment(NLLB1BModelMixin, TranslationSanskritItihasaTaggedNLLBExperimentBase):
    pass

class TranslationSanskritItihasaTaggedNLLB3BExperiment(NLLB3BModelMixin, TranslationSanskritItihasaTaggedNLLBExperimentBase):
    pass

class TranslationMitraNLLBExperimentBase(NLLBFinetuneArgsMixin, TranslationMitra, FinetuneExperimentBase):
    pass

class TranslationMitraNLLB600MExperiment(NLLB600MModelMixin, TranslationMitraNLLBExperimentBase):
    pass

class TranslationMitraNLLB1BExperiment(NLLB1BModelMixin, TranslationMitraNLLBExperimentBase):
    pass

class TranslationMitraNLLB3BExperiment(NLLB3BModelMixin, TranslationMitraNLLBExperimentBase):
    pass
