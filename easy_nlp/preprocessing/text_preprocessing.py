# PREPROCESS
import re


class TextPreprocessing:

    def __init__(self, df, text_column):
        self.df = df
        self.text_column = text_column

    def remove_url(text):
        """
        Remove URL from text  
        :param texte: text to remove URL from    
        :returns: texte without URL 
        """
        text = re.sub(r'http(\S)+', '', text)
        text = re.sub(r'http ...', '', text)
        return text

    def remove_rt(text):
        """
        Remove RT mention from text   
        :param text: text to remove rt mention from    
        :returns: texte without rt mention 
        """
        return re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', '', text)

    def remove_at(text):
        """
        Remove at mention from text   
        :param text: text to remove ata mention from    
        :returns: texte without at mention 
        """
        return re.sub(r'@\S+', '', text)

    def remove_extraspace(text):
        """
        Remove extra space from text    
        :param text: text to remove rt mention from    
        :returns: texte without rt mention 
        """
        return re.sub(r' +', ' ', text)

    def replace_and(text):
        """
        Replace &amp which represents and in html with and  
        :param text: text to replace from 
        :returns: texte and replaced  
        """
        return re.sub(r'&amp;?', 'and', text)

    def replace_lt_lg(text):
        """
        replace lt and lg that stands for lower and upper in html with proper signs    
        :param text: text to replace lt and lg in     
        :returns: texte without &lt and &lg 
        """
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        return text

    def lower(text):
        """
        lowercase text     
        :param text: text to lowercase    
        :returns: texte lowercased 
        """
        return text.lower()

    def lower_then_k(text, k=3):
        """
        asserts text length      
        :param text: text to mesure length     
        :returns: null if text length lower then k, text instead 
        """
        return len(text) > k

    def remove_numero(text):
        """
        replace numbers from and replace them with numero       
        :param text: text     
        :returns: texte without numbers  
        """
        return re.sub(r'\d+', 'numero', text)

    def remove_punctuations(text):
        """
        remove ponctuations       
        :param text: text    
        :returns: text without punctuations 
        """

        return re.sub('["$#%()*+,-@./:;?![\]^_`{|}~\n\t’\']', ' ', text)

    def fit_transform(self, inplace=False, pretraitement=[
            remove_url,
            remove_rt,
            remove_at,
            replace_and,
            replace_lt_lg,
            remove_numero,
            remove_punctuations,
            remove_extraspace]):
        """
        apply a list if transformations to text       
        :inplace : If true returns text on same column, if not on new column named preprocesed_column  
        :returns: df passed on __init__ with processed_text  
        """

        processed_column = self.text_column if inplace else 'processed_text'
        self.df[processed_column] = self.df[self.text_column]

        for func in pretraitement:
            self.df[processed_column] = self.df[processed_column].apply(
                lambda x: func(x))
