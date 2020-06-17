

class BertInput():
    def __init__(self, Tokenizer):
        self.tokenizer = Tokenizer

    def encode_sents(self, sentences):
        """ Tokenize list of sentences according do the tokenizer provided 
        @params sents (list[sentences]):list of sentences , where each sentence is represented as a sting
        @params tokenizer (PretrainedTokenizer): Tokenizer that tokenizes text 
        @returns  
        """
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []

        # For every sentence...
        for sent in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = self.tokenizer.encode(sent)

            # Add the encoded sentence to the list.
            input_ids.append(encoded_sent)
        return input_ids

    def pad_sents(self, sents):
        """ Pad list of sentences according to the longest sentence in the batch.
        @param sents (list[list[int]]): list of sentences, where each sentence
                                        is represented as a list of words
        @param pad_token (int): padding token
        @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
            than the max length sentence are padded out with the pad_token, such that
            each sentences in the batch now has equal length.
            Output shape: (batch_size, max_sentence_length)
        """
        sents_padded = []

        max_len = max(len(s) for s in sents)
        batch_size = len(sents)

        for s in sents:
            padded = [self.tokenizer.pad_token_id] * max_len
            padded[:len(s)] = s
            sents_padded.append(padded)

        return sents_padded

    def mask_sents(self, input_ids):
        attention_masks = []

        # For each sentence...
        for sent in input_ids:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id != self.tokenizer.pad_token_id)
                        for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)
        return attention_masks
    def fit_transform(self, sents):
        input_ids = self.encode_sents(sents)
        input_ids = self.pad_sents(input_ids)
        mask = self.mask_sents(input_ids)
        return (input_ids, mask)
