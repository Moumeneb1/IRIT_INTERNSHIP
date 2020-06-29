import re
import pandas as pd
import treetaggerwrapper
import os
import sys
import xml.etree.ElementTree as et


class LinguisticFeatures:

    def __init__(self, df, text_column, path_OpinionLex="../../Data/Lexicon/OpinionLexiconFinal2018.xml",
                 path_EMOTAIX="../../Data/Lexicon/lexique_EMOTAIX.xml"):
        self.df = df
        self.text_column = text_column
        opinionLex, emotaix = self.sentiment_lexicon_to_df(
            path_OpinionLex), self.sentiment_lexicon_to_df(path_EMOTAIX)
        lex_df = pd.concat([opinionLex, emotaix]).reset_index(drop=True)
        self.lex_df = lex_df

    def sentiment_lexicon_to_df(self, path_lexicon):
        try:
            tree = et.parse(path_lexicon)
            root = tree.getroot()
        except(e):
            print("Oops!", sys.exc_info()[0], "occurred.")
            return
        lemmas, categories, subcategories, polarities, strengths, types, frozen_exp, nwords, group = (
            list() for i in range(9))
        for node in root:
            lemma = node.find('./lemma')
            lemma_compose = lemma.find(
                './lemma_compose').attrib if lemma.find('./lemma_compose') else None
            sense = node.find('./sense').attrib
            category = sense['category'] if 'category' in sense.keys(
            ) else None
            subcategory = sense['subcategory'] if 'subcategory' in sense.keys(
            ) else None
            polarity = sense['polarity'] if 'polarity' in sense.keys(
            ) else None
            polarity = +1 if polarity == 'pos' else -1 if polarity == 'neg' else 0
            strength = sense['strenght'] if 'strenght' in sense.keys(
            ) else None
            strength = 1 if strength == '?' else int(
                strength) if strength is not None else 0
            type_ = sense['type'] if 'type' in sense.keys() else None
            frozen = (True if 'frozen' in lemma_compose.keys()
                      else False) if lemma_compose is not None else None
            nword = len(lemma.text.split())
            grp = 'word' if nword == 1 else 'frozen' if frozen == True else 'nonfrozen'

            lemmas.append(lemma.text)
            categories.append(category)
            subcategories.append(subcategory)
            polarities.append(polarity)
            strengths.append(strength)
            types.append(type_)
            frozen_exp.append(frozen)
            nwords.append(nword)
            group.append(grp)

            lex_df = pd.DataFrame({'lemma': lemmas, 'category': categories, 'subcategory': subcategories,
                                   'polarity': polarities, 'strength': strengths, 'type': types, 'frozen_exp': frozen_exp,
                                   'from_lexicon': os.path.splitext(os.path.basename(path_lexicon))[0], 'nwords': nwords,
                                   'group': group})
        lex_df.lemma = lex_df.lemma.apply(lambda x: " ".join(x.split()))
        return lex_df

    def number_verbs(self, list_pos_tweet):
        """
        counts number of verbs        
        :param text: lis_pos_tweet      
        :returns: number of verbs  
        """
        n = sum([bool(re.search(r'VER', l)) for l in list_pos_tweet])
        return n

    def number_proper_nouns(self, list_pos_tweet):
        """count  number of proper_nouns         
        :param list_pos_tweet: list_pos_tweet      
        :returns: number of pronoun nouns  
        """
        n = sum([bool(l == 'NAM') for l in list_pos_tweet])
        return n

    def number_imperative_verb(self, list_pos_tweet):
        """ returns number of proper_nouns         
        :param list_pos_tweet: list_pos_tweet      
        :returns: number of pronoun nouns  
        """

        n = sum([bool(l == 'VER:impe') for l in list_pos_tweet])
        return n

    def duplicates(self, lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    def has_expression(self, exp, string_ls, group='word'):
        '''
        group: 'word', 'frozen', 'nonfrozen'    
        '''
        if group == 'word':
            result = exp in string_ls
        else:
            exp_ls = exp.split()
            all_in = all(i in string_ls for i in exp_ls)
            if all_in:
                exp_index = [self.duplicates(string_ls, w) for w in exp_ls]
                if group == 'frozen':
                    result = all(a[-1]+1 == b[0]
                                 for a, b in zip(exp_index, exp_index[1:]))
                else:
                    result = any(a[-1] < b[0]
                                 for a, b in zip(exp_index, exp_index[1:]))
            else:
                result = False
        return(result)

    def has_intensifier(self, list_lex_exp):
        has_int = list_lex_exp.apply(
            lambda lod: 'intensifieur' in list(map(lambda d: d['category'], lod)))
        return has_int

    def avg_polarity(self, list_lex_exp):
        avg_pol = list_lex_exp.apply(lambda lod: sum(
            [x['polarity']*x['strength'] for x in lod])/len(lod) if len(lod) >= 1 else 0)
        return(avg_pol)

    def fit_transform(self):

        # Parsing (lemmatisation and pos-tagging)
        tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')
        df['tags'] = self.df[self.text_column].apply(
            lambda x: treetaggerwrapper.make_tags(tagger.tag_text(x)))
        df['lemma'] = df.tags.apply(lambda x: [(t.lemma).lower() if isinstance(
            t, treetaggerwrapper.Tag) else '' for t in x])
        df['text_lemma'] = df.apply(lambda row: " ".join(row.lemma), axis=1)
        df['pos'] = df.tags.apply(lambda x: [t.pos if isinstance(
            t, treetaggerwrapper.Tag) else '' for t in x])

        # surface based features
        df['number_verbs'] = df.pos.apply(lambda x: self.number_verbs(x))
        df['number_proper_nouns'] = df.pos.apply(
            lambda x: self.number_proper_nouns(x))
        df['number_imperative_verb'] = df.pos.apply(
            lambda x: self.number_imperative_verb(x))

        # sentiment features
        lex_dict = self.lex_df.to_dict('index')
        list_lex_exp = df.lemma.apply(lambda x: [lex_dict[key] if self.has_expression(exp=lex_dict[key]['lemma'],
                                                                                      string_ls=x, group=lex_dict[key]['group']) == True else None for key in lex_dict.keys()])
        list_lex_exp = list_lex_exp.apply(lambda x: list(filter(None, x)))
        has_int = self.has_intensifier(list_lex_exp)
        avg_pol = self.avg_polarity(list_lex_exp)

        df['has_intensifier'] = has_int
        df['avg_polarity'] = avg_pol
        return df
