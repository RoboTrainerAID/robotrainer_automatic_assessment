from configparser import ConfigParser

class PipelineConfig:
    """" Class to handle configuration settings for the data processing pipeline. """

    def __init__(self, filename):
        self.pipline_cfile = None

        self.path_to_data = None    # String: Path to the data directory
        self.topics = []            # List: Topics to include in the analysis    

        self.read_config(filename)

    
    def read_config(self, filename):
        self.pipline_cfile = ConfigParser()
        self.pipline_cfile.read(filenames=filename)

        self.read_data_path()
        self.read_ground_truth()

    def read_data_path(self):
        self.path_to_data = self.pipline_cfile.get('DATA PATH', 'path_to_data')

    def read_ground_truth(self):
        raw_topics = self.pipline_cfile.get('GROUND TRUTH', 'topics')
        self.topics = [t.strip() for t in raw_topics.split(',') if t.strip()]