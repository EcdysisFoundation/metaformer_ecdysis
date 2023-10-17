import pandas as pd
from dataset_generation.split import get_count_per_class_split

def test_count():
    # Sample input
    splits = {
        '99': {
            'test': [
                '/path/to/image1.jpg',
                '/path/to/image2.jpg'
            ],
            'train': [
                '/path/to/image3.jpg',
                '/path/to/image4.jpg',
                '/path/to/image5.jpg'
            ],
            'val': [
                '/path/to/image6.jpg',
                '/path/to/image7.jpg',
                '/path/to/image8.jpg',
                '/path/to/image9.jpg'
            ]
        },
        '100': {
            'test': [
                '/path/to/image10.jpg',
                '/path/to/image11.jpg',
                '/path/to/image12.jpg'
            ],
            'train': [
                '/path/to/image13.jpg',
                '/path/to/image14.jpg'
            ],
            'val': [
                '/path/to/image15.jpg',
                '/path/to/image16.jpg',
                '/path/to/image17.jpg'
            ]
        }
    }

    taxon_map = pd.DataFrame({
        'id': [99, 100,101],
        'name': ['Class A', 'Class B','Class C'],
        'taxon_id': [123, 456,999]
    })

    # Expected output
    expected_counts_df = pd.DataFrame({
        'id': [99, 100],
        'name': ['Class A', 'Class B'],
        'test': [2, 3],
        'train': [3, 2],
        'val': [4, 3]
    })

    # Test
    counts_df = get_count_per_class_split(splits)
    counts_df["id"] = counts_df["id"].astype(int)
    counts_df = taxon_map[['id','name']].merge(counts_df, left_on='id', right_on='id', how='right')
    logging.warning(f"Actual:\n{counts_df.head()}")
    logging.warning(f"Expected:\n{expected_counts_df.head()}")
    assert counts_df.equals(expected_counts_df)

if __name__ == '__main__':
    test_count()