import json

import dropbox


class DropboxClient:
    def __init__(self, token):
        self.dbx = dropbox.Dropbox(token)

    def to_dropbox(self, data_json, dropbox_path):
        db_bytes = bytes(json.dumps(data_json), "utf8")
        self.dbx.files_upload(
            f=db_bytes, path=dropbox_path, mode=dropbox.files.WriteMode.overwrite
        )

    def from_dropbox(self, dropbox_path):
        _, res = self.dbx.files_download(path=dropbox_path)
        return json.loads(res.content)
