from utils import read1D, read2D

class TextBlob:
    def _preprocess(self, text):
        return text.lower().split()
    def __init__(self, text):
        self.text = self._preprocess(text)

class TextNetwork:
    def compute_vocab_size(self, src_blobs, tgt_blobs):
        num_words = 0
        for blob in src_blobs:
            for word in blob:
                num_words = max(num_words, word)
        for blob in tgt_blobs:
            for word in blob:
                num_words = max(num_words, word)
        return num_words+1
    def __init__(self, src_blobs, tgt_blobs, edges, subreddits):
        """
            src_blobs: List of lists. Each list contains words associated with a particular source subreddit
            tgt_blobs: List of lists. Each list contains words associated with a particular target subreddit,user pair
            edges: List of lists. List i contains the list of src_blob indices that target blob i is connected to
            #subreddits: list of the subreddit associated with each tgt_blob in tgt_blobs
        """
        self.src_blobs = src_blobs
        self.tgt_blobs = tgt_blobs
        self.edges = edges
        self.subreddits = subreddits
        self.vocab_size = self.compute_vocab_size(src_blobs, tgt_blobs)
        self.num_src_subreddits = len(src_blobs)
        self.num_tgt_subreddits = max(subreddits) + 1
    @classmethod
    def load(cls, dir_path):
        src_blobs_path = "{}/src_blobs.txt".format(dir_path)
        tgt_blobs_path = "{}/tgt_blobs.txt".format(dir_path)
        edges_path = "{}/edges.txt".format(dir_path)
        subreddits_path = "{}/subreddits.txt".format(dir_path)
        src_blobs = read2D(src_blobs_path)
        tgt_blobs = read2D(tgt_blobs_path)
        edges = read2D(edges_path)
        subreddits = read1D(subreddits_path)
        return TextNetwork(src_blobs, tgt_blobs, edges, subreddits)
        
