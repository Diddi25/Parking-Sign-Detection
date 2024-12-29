from roboflow import Roboflow
rf = Roboflow(api_key="this is private")
project = rf.workspace("aiproject-1rkj1").project("ai-main-project")
version = project.version(3)
dataset = version.download("yolov7")
                