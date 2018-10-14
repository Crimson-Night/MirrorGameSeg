import segmentation
from glob import glob as ls
from shutil import move as mv

master = segmentation.Segmentation('datasets/dataset-main')
print ('Loading master\'s cache')
master.Load('master-main')

total_sessions = len(ls('sessions/*.csv'))
for (filenum, session_path) in enumerate(ls('sessions/*.csv')):
    print ('Learning session ' + str(filenum + 1) + '/' + str(total_sessions))
    session_file = session_path.split('/')[1]
    master.LearningSession(session_path)
    mv (session_path, './sessions/sessions.old')

print ('Storing training to master\'s cache')
master.Store()