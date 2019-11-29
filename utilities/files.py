def validate_path(path, op):
    try:
        open(path, op)
        return True
    except:
        return False
