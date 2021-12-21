from ast import literal_eval

import pandas as pd
import dropbox

def to_dropbox(dataframe, path, token):
    dbx = dropbox.Dropbox(token)
    df_string = dataframe.to_csv(index=False)
    db_bytes = bytes(df_string, 'utf8')
    dbx.files_upload(
        f=db_bytes,
        path=path,
        mode=dropbox.files.WriteMode.overwrite
    )

def from_dropbox(dropbox_path, local_path, token):
    dbx = dropbox.Dropbox(token)
    with open(local_path, "wb") as f:
        metadata, res = dbx.files_download(path=dropbox_path)
        f.write(res.content)
    return pd.read_csv(local_path, converters={'premise': literal_eval, 'hypothesis': literal_eval})