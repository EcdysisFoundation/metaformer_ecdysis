import pandas as pd
import unittest, logging
import json

from metrics import stats_to_json

def get_expected_data() -> dict:
    """ Return expected result of the example """
    return {
        "version": "2.0",
        "data":[
            {
                "precision":0.38,
                "recall":1.0,
                "total":5000,
                "f1":0.56,
                "morphospecie_id":10
            },
            {
                "precision":0.40,
                "recall":0.99,
                "total":3600,
                "f1":0.57,
                "morphospecie_id":11
            }
            ]
    }

def get_test_data() -> pd.DataFrame:
    """ Return pd.DataFrame with test data"""
    # data which have the same values on the required fields
    # doesn't have to be consistent 
    data = [
        {
            "name": 10,
            "TP": 145,
            "FP": 32,
            "TN": 3377,
            "FN": 13,
            "Precision":0.38,
            "Recall":1.0,
            "F1":0.56,
            "Total samples": 5000
        },
        {
            "name": 11,
            "TP": 6,
            "FP": 6,
            "TN": 3552,
            "FN": 3,
            "Precision":0.40,
            "Recall":0.99,
            "F1":0.57,
            "Total samples": 3600
        }
    ]

    return pd.DataFrame(data)


class TestGetJsonStats(unittest.TestCase):
    def test_get_json_stats(self):
        # Arrange
        df = get_test_data()
        id_column = "morphospecie_id"
        version = "2.0"
        
        result = json.dumps(stats_to_json(df, id_column, version),indent=4)
        expected_result = json.dumps(get_expected_data(),indent=4)
        logging.warning(f"Expected data is:\n{expected_result}")
        logging.warning(f"Result is:\n{result }")
        self.assertEqual(result,expected_result)

if __name__ == '__main__':
    unittest.main()
