import numpy as np


class JerseySplitter():
    def __init__(self):
        
        pass


    def split_tracklet(self, tracklet, next_available_id):
        
        jerseys = tracklet.pred_attributes.get('jerseys', [])

        print(jerseys)

        switch_points = self.detect_id_switch(jerseys)


    
    def detect_id_switch(self, jerseys):

        for jersey_number in jerseys:
            if self.is_valid_number(jersey_number):
                print(jersey_number)



    def _is_valid(self, value):
        """Check if jersey value is valid."""
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return True