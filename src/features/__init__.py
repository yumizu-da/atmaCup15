from typing import List

from .anime2vec import Anime2VecBlock
from .anime_stat import AnimeStatBlock
from .base import AbstractBaseBlock
from .comma_length import CommaLengthBlock
from .convert_aired import ConvertAiredBlock
from .convert_duration import ConvertDurationBlock
from .count_vectorizer import CountVectorizerBlock
from .countencoding import CountEncodingBlock, CountEncodingByAnimeBlock
from .groupby import GroupbyBlock
from .groupbyrate import GroupbycountBlock, GroupbyrateBlock
from .identity import IdentityBlock
from .kmeans import KmeansBlock
from .labelencoding import LabelEncodingBlock
from .node2ve_cosine import Node2vecCosineBlock
from .node2vec import Node2vecBlock
from .node2vec_average_embedder import Node2vecAverageEmbedderBlock
from .node2vec_hadamard_embedder import Node2vecHadamardEmbedderBlock
from .objects_to_numerical import ObjectsToNumericalBlock
from .onehotencording_list_in_cols import OnehotencordingListInColsBlock
from .original_workname import OriginalWorknameBlock

# from .svd import SVDBlock
from .target_encoding import TargetEncodingBlock
from .tfidf_vectorize import TfidfVectorizerBlock
from .word2vec import Word2vecBlock
from .word2vec_without_score import Word2vecWithoutScoreBlock

__all__: List[str] = [
    "AbstractBaseBlock",
    "IdentityBlock",
    "CountEncodingBlock",
    "LabelEncodingBlock",
    "ObjectsToNumericalBlock",
    "ConvertDurationBlock",
    "ConvertAiredBlock",
    "CommaLengthBlock",
    "OnehotencordingListInColsBlock",
    "CountEncodingByAnimeBlock",
    "AnimeStatBlock",
    "GroupbyBlock",
    "TargetEncodingBlock",
    "TfidfVectorizerBlock",
    # "SVDBlock",
    "GroupbyrateBlock",
    "GroupbycountBlock",
    "CountVectorizerBlock",
    "Node2vecBlock",
    "Word2vecBlock",
    "OriginalWorknameBlock",
    "Word2vecWithoutScoreBlock",
    "Anime2VecBlock",
    # "GroupbyDiffBlock",
    "KmeansBlock",
    "Node2vecCosineBlock",
    "Node2vecAverageEmbedderBlock",
    "Node2vecHadamardEmbedderBlock",
]
