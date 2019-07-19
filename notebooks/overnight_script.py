import sys

import torch
from functools import partial
import qelos as q
from allennlp.data.iterators import BucketIterator

from allennlp.data.token_indexers import SingleIdTokenIndexer, WordpieceIndexer, PretrainedBertIndexer, \
    TokenCharactersIndexer

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, JustSpacesWordSplitter, WordSplitter
from typing import Iterator, List, Callable, Dict, Optional, Any

import pandas as pd
from overrides import overrides

from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Vocabulary
from allennlp.data.fields import TextField, MetadataField
from allennlp.models import Model, SimpleSeq2Seq
from allennlp.modules import Embedding, TextFieldEmbedder, Seq2SeqEncoder, Attention, SimilarityFunction
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer
from allennlp.training.metrics import SequenceAccuracy


class OvernightReader(DatasetReader):
    def __init__(self, nl_tokenizer:Callable[[str],List[str]],
                       fl_tokenizer:Callable[[str],List[str]],
                       nl_token_indexer:TokenIndexer,
                       fl_token_indexer:TokenIndexer,
                 **kw):
        super(OvernightReader, self).__init__(**kw)
        self.nl_tokenizer, self.fl_tokenizer = nl_tokenizer, fl_tokenizer
        self.nl_token_indexer, self.fl_token_indexer= {q.IK("tokens"): nl_token_indexer}, {q.IK("tokens"): fl_token_indexer}

    @overrides
    def text_to_instance(self, nl_tokens:List[Token], fl_tokens:List[Token], id:str=None) -> Instance:
        nl_field = TextField(nl_tokens, self.nl_token_indexer)
        fl_field = TextField(fl_tokens, self.fl_token_indexer)
        id_field = MetadataField(id)
        fields = {
            q.FK("nl"): nl_field,
            q.FK("fl"): fl_field,
            q.FK("id"): id_field,
        }
        return Instance(fields)

    @overrides
    def _read(self, p:str)->Iterator[Instance]:
        df = pd.read_csv(p, sep="\t", header=None)
        # print(df)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.nl_tokenizer(row[0])],
                [Token(x) for x in self.fl_tokenizer(row[1])],
                i
            )


class Seq2Seq(SimpleSeq2Seq):

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True) -> None:
        super(Seq2Seq, self).__init__(vocab, source_embedder,encoder,max_decoding_steps,
                                      attention=attention, attention_function=attention_function,
                                      beam_size=beam_size,target_namespace=target_namespace,
                                      target_embedding_dim=target_embedding_dim,
                                      scheduled_sampling_ratio=scheduled_sampling_ratio,
                                      use_bleu=use_bleu)
        self._seqacc = SequenceAccuracy()

    @overrides
    def forward(self,  # type: ignore
                nl: Dict[str, torch.LongTensor],
                fl: Dict[str, torch.LongTensor] = None, id=None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        source_tokens, target_tokens = nl, fl
        state = self._encode(source_tokens)

        if target_tokens:
            state = self._init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if True: #not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            # shape: (batch_size, beam_size, max_sequence_length)
            top_k_predictions = output_dict["predictions"]
            # shape: (batch_size, max_predicted_sequence_length)
            best_predictions = top_k_predictions[:, 0, :]
            if target_tokens and self._bleu:
                self._bleu(best_predictions, target_tokens["tokens"])
            if target_tokens:
                seqacc_gold = target_tokens["tokens"][:, 1:]
                self._seqacc(best_predictions.unsqueeze(1)[:, :, :seqacc_gold.size(1)],
                             seqacc_gold,
                             mask=(seqacc_gold != 0).long())

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        all_metrics.update({"SeqAcc": self._seqacc.get_metric(reset=reset)})
        return all_metrics



def run(trainp="overnight/calendar_train_delex.tsv",
        testp="overnight/calendar_test_delex.tsv",
        batsize=8,
        embdim=50,
        encdim=50,
        maxtime=100,
        lr=.001,
        gpu=0,
        cuda=False, epochs=20):
    device = torch.device("cuda", gpu) if cuda else torch.device("cpu")
    tt = q.ticktock("script")
    tt.tick("loading data")
    def tokenizer(x:str, splitter:WordSplitter=None)->List[str]:
        return [xe.text for xe in splitter.split_words(x)]

    reader = OvernightReader(partial(tokenizer, splitter=JustSpacesWordSplitter()),
                             partial(tokenizer, splitter=JustSpacesWordSplitter()),
                             SingleIdTokenIndexer(namespace="nl_tokens"),
                             SingleIdTokenIndexer(namespace="fl_tokens"))
    trainds = reader.read(trainp)
    testds = reader.read(testp)
    tt.tock("data loaded")

    tt.tick("building vocabulary")
    vocab = Vocabulary.from_instances(trainds)
    tt.tock("vocabulary built")

    tt.tick("making iterator")
    iterator = BucketIterator(sorting_keys=[("nl", "num_tokens"), ("fl", "num_tokens")],
                              batch_size=batsize,
                              biggest_batch_first=True)
    iterator.index_with(vocab)
    batch = next(iter(iterator(trainds)))
    #print(batch["id"])
    #print(batch["nl"])
    tt.tock("made iterator")

    # region model
    nl_emb = Embedding(vocab.get_vocab_size(namespace="nl_tokens"),
                       embdim, padding_index=0)
    fl_emb = Embedding(vocab.get_vocab_size(namespace="fl_tokens"),
                       embdim, padding_index=0)
    nl_field_emb = BasicTextFieldEmbedder({"tokens": nl_emb})
    fl_field_emb = BasicTextFieldEmbedder({"tokens": fl_emb})

    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(embdim, encdim, bidirectional=True, batch_first=True))
    attention = DotProductAttention()

    smodel = Seq2Seq(vocab, nl_field_emb, encoder, maxtime,
                  target_embedding_dim=embdim,
                  attention=attention,
                  target_namespace='fl_tokens',
                  beam_size=1,
                  use_bleu=True)

    smodel_out = smodel(batch["nl"], batch["fl"])

    smodel.to(device)

    optim = torch.optim.Adam(smodel.parameters(), lr=lr)
    trainer = Trainer(model=smodel,
                      optimizer=optim,
                      iterator=iterator,
                      train_dataset=trainds,
                      validation_dataset=testds,
                      num_epochs=epochs,
                      cuda_device=gpu if cuda else -1)

    metrics = trainer.train()

    sys.exit()
    class MModel(Model):
        def __init__(self, nlemb:Embedding,
                           flemb:Embedding,
                            vocab:Vocabulary,
                     **kwargs):
            super(MModel, self).__init__(vocab, **kwargs)
            self.nlemb, self.flemb = nlemb, flemb

        @overrides
        def forward(self,
                    nl:Dict[str, torch.Tensor],
                    fl:Dict[str, torch.Tensor],
                    id:Any):
            nlemb = self.nlemb(nl["tokens"])
            flemb = self.flemb(fl["tokens"])
            print(nlemb.size())
            pass

    m = MModel(nl_emb, fl_emb, vocab)
    batch = next(iter(iterator(trainds)))
    out = m(**batch)








if __name__ == '__main__':
    q.argprun(run)

