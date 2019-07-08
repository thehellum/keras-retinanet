import filetype

def get_type(path):
    kind = filetype.guess(path)
    if kind is None:
        print('Cannot guess file type!')
        return

    print('File MIME type: %s' % kind.mime)
    return kind.mime.split("/")[0] 
