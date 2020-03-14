'''
download data file stored on my personal google drive 

# modified based on this thread: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
'''

import requests
import zipfile
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":

    file_id = '1nmQfjsSs3tRVHvS3xqu4w8Aj_T6hVcZJ'
    destination = './data.zip'

    #os.makedirs(destination)

    if os.path.exists('./data/'): 
        print('./data/ already exists, exiting')
        exit() 
    
    else: 
        if not os.path.exists('./data.zip'):
            print('downloading zip file from google drive...')

            download_file_from_google_drive(file_id, destination)

    print('unzipping data file...')
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall('./')

    print('removing zip file ... ')
    os.remove('./data.zip')