from configparser import ConfigParser

class PipelineConfig:
    """" Class to handle configuration settings for the data processing pipeline. """

    def __init__(self, filename):
        self.pipline_cfile = None

        self.path_to_data = None            # String: Path to the data directory
        self.path_to_evaluation = None      # String: Path to the evaluation directory
        self.topics = []                    # List: Topics to include in the analysis
        self.model = None                   # String: Model type to use
        self.pca_variance_threshold = None  # Float: Variance threshold for PCA   
        self.use_pca = None                 # Boolean: Whether to apply PCA or not
        self.kfolds = None                  # Integer: Number of folds for cross-validation 
        self.mode = None                    # String: Mode of data preparation ("row", "user", "user_path")

        self.read_config(filename)

    
    def read_config(self, filename):
        self.pipline_cfile = ConfigParser()
        self.pipline_cfile.read(filenames=filename)

        self.read_data_path()
        self.read_ground_truth()
        self.read_settings()
        self.read_data()

    def read_data_path(self):
        self.path_to_data = self.pipline_cfile.get('DATA PATH', 'path_to_data')
        self.path_to_evaluation = self.pipline_cfile.get('DATA PATH', 'path_to_evaluation')

    def read_ground_truth(self):
        raw_topics = self.pipline_cfile.get('GROUND TRUTH', 'topics')
        self.topics = [t.strip() for t in raw_topics.split(',') if t.strip()]

    def read_settings(self):
        self.pca_variance_threshold = self.pipline_cfile.getfloat('SETTINGS', 'pca_variance_threshold')
        self.model = self.pipline_cfile.get('SETTINGS', 'model')
        self.use_pca = self.pipline_cfile.getboolean('SETTINGS', 'use_pca')
        self.kfolds = self.pipline_cfile.getint('SETTINGS', 'kfolds')

    def read_data(self):
        self.mode = self.pipline_cfile.get('DATA', 'mode')