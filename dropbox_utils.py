import pandas as pd
import dropbox

class DropboxClient:
    def __init__(token):
        self.dbx = dropbox.Dropbox(token)
        self.upload_filename = 'upload.json'
        self.download_filename = 'download.json'

    def to_dropbox(data, dropbox_path):
        to_json(data, self.upload_filename)
        db_bytes = bytes(self.upload_filename, 'utf8')
        self.dbx.files_upload(
            f=db_bytes,
            path=dropbox_path,
            mode=dropbox.files.WriteMode.overwrite
        )

    def from_dropbox(dropbox_path):
        with open(self.download_filename, "wb") as f:
            metadata, res = self.dbx.files_download(path=dropbox_path)
            f.write(res.content)
        return read_json(self.download_filename)

def to_json(data, filepath):
    with open(filepath, 'wt') as json_file:
        json.dump(data, json_file)

def read_json(filepath):
    with open(filepath, 'rt') as json_file:
        return json.load(json_file)