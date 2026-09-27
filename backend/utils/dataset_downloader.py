import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATASETS = {
    'intent': {
        'url': 'https://raw.githubusercontent.com/singla007/MTSamples/master/mtsamples.csv',
        'raw_path': 'datasets/raw/intent/mtsamples.csv'
    },
    'diagnosis': {
        'url': 'https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv',
        'raw_path': 'datasets/raw/diagnosis/Symptom2Disease.csv'
    },
    'risk': {
        'url': 'https://raw.githubusercontent.com/rishabhpatel9/Healthcare-Triage-Assistant/main/data/raw/synthetic_medical_triage.csv',
        'raw_path': 'datasets/raw/risk/synthetic_medical_triage.csv'
    }
}

def download_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    
    info = DATASETS[name]
    url = info['url']
    raw_path = info['raw_path']
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    print(f"Downloading {name} dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, raw_path)
        print(f"Successfully downloaded and saved to {raw_path}")
        return True
    except Exception as e:
        print(f"Failed to download {name}: {str(e)}")
        return False

if __name__ == "__main__":
    # Download the Intent dataset first (as requested)
    download_dataset('intent')
