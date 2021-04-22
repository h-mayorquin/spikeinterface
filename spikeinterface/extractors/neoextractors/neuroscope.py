import neo

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class NeuroScopeRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from neuroscope
    Ref: http://neuroscope.sourceforge.net
    
    Based on neo.rawio.NeuroScopeRawIO
    
    Parameters
    ----------
    file_path: str
        The xml  file.
    stream_id: str or None
    """ 
    mode = 'file'
    NeoRawIOClass = 'NeuroScopeRawIO'
    
    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename' : str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)