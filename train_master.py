import segmentation
from glob import glob as ls
from shutil import move as mv

master = segmentation.Segmentation('datasets/dataset-main')
master.Load('master-main')

for session_path in ls('sessions/*.csv'):
    session_file = session_path.split('/')[1]
    master.LearningSession(session_path)
    mv (session_path, './sessions/sessions.old')

master.Store()