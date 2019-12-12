from sklearn.metrics import mean_squared_error
from nltk import word_tokenize
from gensim.models import KeyedVectors, Word2Vec


class data_pipeline(INPUT_FILE):
    def __init__(self):
        self.input_file = INPUT_FILE
        self.input_sentences = []
        self.embedding = []
        self.mean_embedding = []
        self.target = []

    def extract_sentences(self):
        df = pd.read_csv(self.input_file, delimiter='\t', header=None, names=['sentence', 'label'])
        df_sentences = df.sentence.values
        sentences = df_sentences[1:]
        for sent in sentences:
            self.input_sentences.append(sent.split(' '))

    def find_target(self):
        df = pd.read_csv(self.input_file, delimiter='\t', header=None, names=['sentence', 'label'])
        df_target = df.label.values
        self.target = df_target[1:]

    def create_embedding(self):
        model = Word2Vec(self.input_sentences,min_count=1,size=300)
        for sentence in self.input_sentences:
            sentence_embedding = []
            for word in sentence:
                sentence_embedding.append(model[word])
            self.embedding.append(sentence_embedding)
        #return embedding
    def find_mean_embedding(self):
        for row in self.embedding:
            arr = np.mean(row, axis=0)
            temp = []
            temp.append(arr)
            self.mean_embedding.append(temp)
        #return mean_embedding
    
